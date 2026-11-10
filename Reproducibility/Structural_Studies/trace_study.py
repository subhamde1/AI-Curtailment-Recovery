import json,itertools,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix,csr_matrix
from studies import chain
OUT=Path(__file__).parent
df=pq.read_table(OUT/'trace_sample.parquet').to_pandas()
# Fixed rule: three lowest SHA256 pod identifiers in each of two declared classes,
# restricted to positive finite spans of at most two hours. No outcome filtering.
df=df[(df.duration_hours>0)&(df.duration_hours<=2)&(df.gpu_request>0)].copy()
df['key']=df.pod_id.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
chosen=pd.concat([df[df.job_type_public==c].sort_values('key').head(3) for c in ['training','offline_inference']])
W=float((chosen.duration_hours*chosen.gpu_request).sum())
catalog=[]
for _,v in chosen.iterrows():catalog.append({'source_hash':v['key'],'class':v.job_type_public,'duration_hours':float(v.duration_hours),'gpu_request':float(v.gpu_request),'work_coefficient':float(v.duration_hours*v.gpu_request/W),'rate':float(v.gpu_request/W)})
(OUT/'trace_catalog.json').write_text(json.dumps({'normalizer_gpu_hours':W,'jobs':catalog},indent=2))

def solve(K,slack,z,mode='frontier',r=.5,b=.9):
    P=chain(.25,.4);D=[int(np.ceil(j['duration_hours']))+slack for j in catalog];N=K+max(D)
    nodes={}
    for t in range(1,N+1):
        for tail in itertools.product((0,1),repeat=max(t-K,0)):
            hist=(1,)*min(t,K)+tail
            nodes[hist]=float(np.prod([P[hist[i-1],hist[i]] for i in range(K,t)]))
    leaves=[n for n in nodes if len(n)==N]
    jobs=[(o,i,o+1,o+D[i],catalog[i]['work_coefficient'],catalog[i]['rate']) for o in range(1,K+1) for i in range(6)]
    vr={};c=[];lb=[];ub=[];ig=[]
    def add(key,cost=0,low=0,high=np.inf,integer=0):
        vr[key]=len(c);c.append(cost);lb.append(low);ub.append(high);ig.append(integer)
    add('r',-1 if mode=='frontier' else 0,0 if mode=='frontier' else r,1 if mode=='frontier' else r)
    add('z',1 if mode=='reserve' else 0,0 if mode=='reserve' else z,np.inf if mode=='reserve' else z)
    for n in nodes:
        for o,i,a,d,w,rate in jobs:
            if a<=len(n)<=d:
                for ch in (0,1):add((n,o,i,ch))
    for ell in leaves:add(('y',ell),integer=1,high=1)
    rows=[];lo=[];hi=[]
    def row(dd,l=-np.inf,h=np.inf):rows.append({vr[k]:v for k,v in dd.items()});lo.append(l);hi.append(h)
    for n in nodes:
        t=len(n)
        for ch in (0,1):
            dd={(n,o,i,ch):1 for o,i,a,d,w,rate in jobs if (n,o,i,ch) in vr}
            if ch:dd['z']=-1
            row(dd,h=0 if ch else 1-n[-1])
        for o,i,a,d,w,rate in jobs:
            if not a<=t<=d:continue
            row({(n,o,i,ch):1 for ch in (0,1)},h=rate)
            dd={(n[:tt],o,i,ch):1 for tt in range(a,t+1) for ch in (0,1)};dd['r']=-w;row(dd,h=0)
    for ell in leaves:
        for o,i,a,d,w,rate in jobs:
            dd={(ell[:tt],o,i,ch):1 for tt in range(a,d+1) for ch in (0,1)}
            dd['r']=-w;dd[('y',ell)]=-w;row(dd,l=-w)
    row({('y',ell):nodes[ell] for ell in leaves},l=b)
    A=lil_matrix((len(rows),len(vr)))
    for ii,dd in enumerate(rows):
        for jj,x in dd.items():A[ii,jj]=x
    st=time.perf_counter();res=milp(c,integrality=ig,bounds=Bounds(lb,ub),constraints=LinearConstraint(csr_matrix(A),lo,hi),options={'presolve':False,'time_limit':60,'mip_rel_gap':1e-9})
    out={'K':K,'slack':slack,'z_declared':z,'status':int(res.status),'seconds':time.perf_counter()-st,'variables':len(vr)}
    if res.x is not None:
        rr=float(res.x[vr['r']]);zz=float(res.x[vr['z']]);succ=0
        for ell in leaves:
            if all(sum(res.x[vr[(ell[:t],o,i,ch)]] for t in range(a,d+1) for ch in (0,1))>=w*rr-1e-7 for o,i,a,d,w,rate in jobs):succ+=nodes[ell]
        ax=A@res.x
        viol=max(0,float(np.max(np.array(lo)-ax)),float(np.max(ax-np.array(hi))))
        out.update(r=rr,z=zz,success=succ,gap=float(res.mip_gap),dual=float(res.mip_dual_bound),max_violation=viol)
    return out

if __name__=='__main__':
    out=[]
    for slack,z in itertools.product((0,1,2),(.1,.25,.4)):
        for k in (1,2,3):
            rr=solve(k,slack,z);rr['reserve_at_half']=solve(k,slack,z,'reserve');out.append(rr);print(rr,flush=True)
            (OUT/'trace_results.json').write_text(json.dumps(out,indent=2))
