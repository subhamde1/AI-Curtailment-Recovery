import io,urllib.request,struct,json,hashlib,time
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

OUT=Path(__file__).parent
expected=json.loads((OUT/'trace_metadata.json').read_text())
URL='https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/asi_opensource_job_execution_summary.zip'
SZ=1188294827
cache=OUT/'trace_ranges';cache.mkdir(exist_ok=True)
def get(a,b):
    fn=cache/f'{a}_{b}.bin'
    if fn.exists():return fn.read_bytes()
    for retry in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity'}),timeout=40) as r:
                data=r.read()
                if r.status!=206 or r.headers.get('Content-Range','').split('/')[0]!=f'bytes {a}-{b}' or len(data)!=b-a+1:
                    raise ValueError((r.status,len(data),a,b,r.headers.get('Content-Range')))
                fn.write_bytes(data);return data
        except Exception:
            if retry==2:raise
head=get(0,79);a=struct.unpack('<4s5H3L2H',head[:30]);OFFSET=30+a[-2]+a[-1]
print('offset',OFFSET,flush=True)
class RangeFile(io.RawIOBase):
    def __init__(self):self.pos=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SZ+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=SZ-self.pos
        n=min(n,SZ-self.pos)
        if n==0:return b''
        data=get(OFFSET+self.pos,OFFSET+self.pos+n-1);self.pos+=n;return data
f=RangeFile();pf=pq.ParquetFile(f,pre_buffer=False)
print(pf.schema_arrow,flush=True)
print('rows',pf.metadata.num_rows,'rowgroups',pf.num_row_groups,flush=True)
info={'source':URL,'zip_member':'asi_opensource_job_execution_summary/part-000.parquet','member_offset':OFFSET,'member_bytes':SZ,'total_rows':pf.metadata.num_rows,'row_groups':pf.num_row_groups,'schema':str(pf.schema_arrow),'selection_rule':'row groups 0, floor((G-1)/2), G-1, selected before examining values'}
ids=sorted({0,(pf.num_row_groups-1)//2,pf.num_row_groups-1});info['selected_rowgroups']=ids
assert info['selected_rowgroups']==expected['selected_rowgroups']
cols=['pod_id','gpu_request','duration_hours','job_type_public','priority_class','model_type_public']
tabs=[]
for i in ids:
    t=pf.read_row_group(i,columns=cols,use_threads=False);tabs.append(t);print('read group',i,t.num_rows,flush=True)
tab=pa.concat_tables(tabs)
sample=OUT/'trace_sample.parquet'
pending=sample.with_suffix('.pending')
pq.write_table(tab,pending,compression='zstd')
info['sha256_sample']=hashlib.sha256(pending.read_bytes()).hexdigest()
if info['sha256_sample']!=expected['sha256_sample']:
    pending.unlink()
    raise SystemExit('Execution-summary sample checksum differs from the archived input. Check the source and PyArrow version.')
pending.replace(sample)
df=tab.to_pandas();keep=np.isfinite(df.gpu_request)&np.isfinite(df.duration_hours)&(df.gpu_request>0)&(df.duration_hours>0)
info['rows_read']=len(df);info['positive_finite_rows']=int(keep.sum())
clean=df[keep].copy();clean['requested_gpu_hours']=clean.gpu_request*clean.duration_hours
info['categories']={str(k):{'count':int(len(g)),'duration_median':float(g.duration_hours.median()),'gpu_hours_sum':float(g.requested_gpu_hours.sum()),'gpu_request_median':float(g.gpu_request.median())} for k,g in clean.groupby('job_type_public')}
info['priority_counts']={str(k):int(v) for k,v in clean.priority_class.value_counts().items()}
info['duration_quantiles']={str(k):float(v) for k,v in clean.duration_hours.quantile([.1,.25,.5,.75,.9,.95,.99]).items()}
info['gpu_request_quantiles']={str(k):float(v) for k,v in clean.gpu_request.quantile([.1,.25,.5,.75,.9,.95,.99]).items()}
print(json.dumps(info,indent=2),flush=True)
