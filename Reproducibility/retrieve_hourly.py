import io, urllib.request, json, struct, zipfile, hashlib, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys, os, threading
import pyarrow.parquet as pq
import pyarrow.compute as pc

BASE='https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/'
ROOT=Path(__file__).parent/'hourly_trace'/'source_partitions'; ROOT.mkdir(exist_ok=True)
class Remote(io.RawIOBase):
    def __init__(self,url,size,offset=0): self.url=url; self.size=size; self.offset=offset; self.pos=0; self.bytes_read=0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,p,whence=0):
        self.pos=p if whence==0 else self.pos+p if whence==1 else self.size+p
        return self.pos
    def read(self,n=-1):
        if n<0: n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        start=self.offset+self.pos;end=start+n-1
        for retry in range(4):
            try:
                req=urllib.request.Request(self.url+f'?paper3_range={start}_{end}',headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=40) as r:
                    data=r.read()
                    if r.status!=206 or len(data)!=n or r.headers.get('Content-Range','').split('/')[0]!=f'bytes {start}-{end}': raise IOError((r.status,len(data),n,r.headers.get('Content-Range')))
                self.pos+=n;self.bytes_read+=n;return data
            except Exception:
                if retry==3:raise
    def member(self,info):
        self.seek(info.header_offset);h=self.read(30)
        vals=struct.unpack('<4s5H3L2H',h)
        if info.compress_type!=0: raise ValueError('Parquet ZIP member must be stored')
        return Remote(self.url,info.file_size,info.header_offset+30+vals[-2]+vals[-1])

def archive(table,size):
    r=Remote(BASE+table+'.zip',size);z=zipfile.ZipFile(r)
    return r,z

def inspect():
    r,z=archive('asi_opensource_pod_hourly',351803513445)
    infos=z.infolist()
    (ROOT/'archive_members.json').write_text(json.dumps([dict(name=i.filename,offset=i.header_offset,size=i.file_size,compression=i.compress_type) for i in infos],indent=2))
    info=next(i for i in infos if 'day=109/hour=00/' in i.filename and i.filename.endswith('.parquet'))
    f=r.member(info);pf=pq.ParquetFile(f,pre_buffer=False)
    print('MEMBER',info.filename,info.file_size,'rows',pf.metadata.num_rows,'groups',pf.num_row_groups,flush=True)
    print(pf.schema_arrow,flush=True)
    t=pf.read(columns=['cluster_id','job_type_public','priority_class','gpu_request','state_public'],use_threads=True)
    df=t.to_pandas();df=df[df.gpu_request>0]
    print(df.groupby(['cluster_id','job_type_public']).gpu_request.agg(['size','sum']).to_string(),flush=True)
    print('bytes',f.bytes_read,flush=True)
    df.to_csv(ROOT/'selection_hour.csv',index=False)

def download():
    """Preserve all observations of a fixed identifier panel in 48 consecutive hours."""
    import pyarrow as pa
    cols=['pod_id','cluster_id','state_public','priority_class','job_type_public','gpu_request','used_gpu_hours']
    members=json.loads((ROOT/'archive_members.json').read_text())
    manifest=json.loads((ROOT.parent/'source_manifest.json').read_text())
    expected={(row['day'],row['hour']):row['sha256'] for row in manifest['partitions']}
    selected=[]
    for d in range(109,111):
        for h in range(24):
            name=f'asi_opensource_pod_hourly/day={d}/hour={h:02d}/part-000.parquet'
            info=next(q for q in members if q['name']==name)
            selected.append((d,h,info))
    url=BASE+'asi_opensource_pod_hourly.zip'
    logs=json.loads((ROOT/'download_log.json').read_text()) if (ROOT/'download_log.json').exists() else [];lock=threading.Lock()
    with ThreadPoolExecutor(max_workers=24) as pool:
        def one(item):
            d,h,info=item
            target=ROOT/f'day{d}_hour{h:02d}.parquet'
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==expected[d,h]:return
            raw=Remote(url,351803513445);zi=zipfile.ZipInfo(info['name']);zi.header_offset=info['offset'];zi.file_size=info['size'];zi.compress_type=info['compression']
            f=raw.member(zi)
            f.seek(f.size-8);last=f.read(8);footerlen=struct.unpack('<L',last[:4])[0]
            f.seek(f.size-8-footerlen);footer=f.read(footerlen+8)
            path=ROOT/f'tmp{os.getpid()}_{d}_{h}.parquet'
            with path.open('wb') as out:
                out.write(b'PAR1');out.truncate(f.size);out.seek(f.size-len(footer));out.write(footer)
            pf=pq.ParquetFile(path)
            chunks=[]
            for g in range(pf.num_row_groups):
                rg=pf.metadata.row_group(g)
                ranges=[]
                for k in range(rg.num_columns):
                    c=rg.column(k)
                    if c.path_in_schema in cols:
                        start=min(q for q in [c.dictionary_page_offset,c.data_page_offset] if q is not None and q>=0)
                        ranges.append((start,start+c.total_compressed_size))
                ranges.sort(); merged=[]
                for a,b in ranges:
                    if merged and a-merged[-1][1]<16384:merged[-1]=(merged[-1][0],b)
                    else:merged.append((a,b))
                chunks.extend(merged)
            def get_chunk(ab):
                a,b=ab;r=Remote(url,b-a,f.offset+a);return a,r.read(b-a)
            futures=[pool.submit(get_chunk,ab) for ab in chunks]
            with path.open('r+b') as out:
                for task in as_completed(futures):
                    a,data=task.result();out.seek(a);out.write(data)
            table=pq.read_table(path,columns=cols)
            # A fixed source-hash panel, with no selection on workload outcome.
            mask=pc.and_(pc.starts_with(table['pod_id'],'00'),pc.greater(table['gpu_request'],0))
            sample=table.filter(mask)
            pending=path.with_suffix('.pending')
            pq.write_table(sample,pending,compression='zstd')
            checksum=hashlib.sha256(pending.read_bytes()).hexdigest()
            if checksum!=expected[d,h]:
                pending.unlink()
                path.unlink()
                raise ValueError(f'Extract checksum mismatch for day {d} hour {h}. Check the source and PyArrow version.')
            pending.replace(target)
            record=dict(day=d,hour=h,source_rows=pf.metadata.num_rows,panel_rows=sample.num_rows,member=info['name'],source_bytes=sum(b-a for a,b in chunks),sha256=checksum)
            with lock:
                logs.append(record)
                (ROOT/'download_log.json').write_text(json.dumps(sorted(logs,key=lambda x:(x['day'],x['hour'])),indent=2))
            path.unlink()
            print('DOWNLOADED',d,h,'rows',sample.num_rows,'bytes',record['source_bytes'],flush=True)
        with ThreadPoolExecutor(max_workers=4) as hours:
            list(hours.map(one,selected))

if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='download':download()
elif __name__=='__main__':inspect()
