"""Trace-calibrated hourly recovery cohorts and exact causal finite-tree control.

Observed marks and temporal ordering are preserved. Grid calls, interruption
permissions, recovery deadlines, headroom and channel prices are counterfactual.
Reliability is conditional on each fixed observed workload panel, not an estimate
of real production reliability. Source used_gpu_hours is an occupation proxy.
"""
from pathlib import Path
from collections import defaultdict
from itertools import product
import json,time,sys
import numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint,linprog
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'hourly_trace';DATA.mkdir(exist_ok=True)

def chain(p,rho):return np.array([[1-p*(1-rho),p*(1-rho)],[1-p-(1-p)*rho,p+(1-p)*rho]])

def catalog(frame,start,K,W,slack=1,overhead=0.,rate_factor=1.,permission='LP',remote=True):
    f=frame[(frame.t>=start)&(frame.t<start+K)&frame.job_type_public.isin(['training','offline_inference'])&(frame.used_gpu_hours>0)].copy()
    if permission=='LP':f=f[f.priority_class=='LP']
    sums=f.groupby('t').used_gpu_hours.sum().reindex(range(start,start+K),fill_value=0)
    rmax=min(1.,float(sums.min()/W))
    jobs=[];rates=f.groupby('pod_id').gpu_request.min()*rate_factor/W
    for q in f.itertuples():
        t=int(q.t-start);D=(2 if q.job_type_public=='training' else 1)+slack
        jobs.append(dict(pod=q.pod_id,cls=q.job_type_public,origin=t,release=t+1,deadline=t+D,
                    w0=0.,beta=float(q.used_gpu_hours/sums[q.t])*(1+overhead if q.job_type_public=='training' else 1),
                    rate=float(rates[q.pod_id]),channels=[0,1] if remote or q.job_type_public!='training' else [0]))
    return jobs,rmax

def make_tree(K,H,p=.25,rho=.4,start=0):
    P=chain(p,rho);remaining=H-start
    forced=max(1,K-start)
    nd={}
    for length in range(1,remaining+1):
        for tail in product((0,1),repeat=max(length-forced,0)):
            n=(1,)*min(length,forced)+tail
            nd[n]=float(np.prod([P[n[k-1],n[k]] for k in range(forced,length)]))
    leaves=[n for n in nd if len(n)==remaining]
    return nd,leaves

def solve(jobs,K,h,z,rmax,r=None,mode='frontier',b=.9,p=.25,rho=.4,fee=2.,foresight=False,selected=None,start=0,keep=False,time_limit=90):
    H=max(j['deadline'] for j in jobs)+1
    nodes,leaves=make_tree(K,H,p,rho,start);vr={};costs=[];low=[];high=[];integer=[]
    def add(key,c=0.,lo=0.,hi=np.inf,i=0):
        vr[key]=len(costs);costs.append(c);low.append(lo);high.append(hi);integer.append(i)
    add('r',-1 if mode=='frontier' else 0,0 if mode=='frontier' else r,rmax if mode=='frontier' else r)
    add('z',1 if mode=='reserve' else 0,0 if mode=='reserve' else z,np.inf if mode=='reserve' else z)
    contexts=[(ell,nodes[ell]) for ell in leaves] if foresight else [(None,1.)]
    for ctx,lp in contexts:
        nn=[ctx[:t] for t in range(1,len(ctx)+1)] if foresight else list(nodes)
        for n in nn:
            t=start+len(n)-1;prob=lp if foresight else nodes[n]
            for j,q in enumerate(jobs):
                if max(start,q['release'])<=t<=q['deadline']:
                    for k in q['channels']:add((ctx,n,j,k),prob*(1. if k==0 else 4.) if mode=='cost' else 0)
    for ell in leaves:
        fixed=b==1. or selected is not None and ell in selected
        add(('y',ell),-fee*nodes[ell] if mode=='cost' else 0,1. if fixed else 0,1.,0 if b==1 or selected is not None else 1)
    rows=[];rhs=[]
    def row(co,cap):rows.append({vr[k]:v for k,v in co.items() if v});rhs.append(cap)
    for ctx,lp in contexts:
        nn=[ctx[:t] for t in range(1,len(ctx)+1)] if foresight else list(nodes)
        for n in nn:
            t=start+len(n)-1;active=[j for j,q in enumerate(jobs) if (ctx,n,j,q['channels'][0]) in vr]
            for k in (0,1):
                d={(ctx,n,j,k):1 for j in active if (ctx,n,j,k) in vr}
                if k==1:d['z']=-1
                row(d,0 if k else h*(1-n[-1]))
            bypod=defaultdict(dict)
            for j in active:
                q=jobs[j]
                for k in q['channels']:bypod[q['pod']][ctx,n,j,k]=1
                d={(ctx,n[:s-start+1],j,k):1 for s in range(max(start,q['release']),t+1) for k in q['channels']}
                d['r']=-q['beta'];row(d,q['w0'])
            for pod,d in bypod.items():row(d,min(jobs[j]['rate'] for j in active if jobs[j]['pod']==pod))
    for ell in leaves:
        ctx=ell if foresight else None
        for j,q in enumerate(jobs):
            d={(ctx,ell[:t-start+1],j,k):-1 for t in range(max(start,q['release']),q['deadline']+1) for k in q['channels']}
            M=q['w0']+q['beta']*rmax;d['r']=q['beta'];d['y',ell]=M
            row(d,M-q['w0'])
    row({('y',ell):-nodes[ell] for ell in leaves},-b)
    ii=[];jj=[];vv=[]
    for n,d in enumerate(rows):
        for col,val in d.items():ii.append(n);jj.append(col);vv.append(val)
    A=coo_matrix((vv,(ii,jj)),shape=(len(rows),len(vr))).tocsc()
    st=time.perf_counter()
    res=milp(costs,integrality=integer,bounds=Bounds(low,high),constraints=LinearConstraint(A,-np.inf,rhs),options={'presolve':False,'mip_rel_gap':1e-9,'time_limit':time_limit})
    out=dict(status=int(res.status),seconds=time.perf_counter()-st,variables=len(vr),constraints=len(rows),leaves=len(leaves),mode=mode,target=b)
    if res.x is None:return out
    x=res.x;rr=float(x[vr['r']]);zz=float(x[vr['z']]);actual=0.;successful=[]
    for ell in leaves:
        ctx=ell if foresight else None
        ok=all(sum(x[vr[ctx,ell[:t-start+1],j,k]] for t in range(max(start,q['release']),q['deadline']+1) for k in q['channels'])>=q['w0']+q['beta']*rr-1e-8 for j,q in enumerate(jobs))
        if ok:actual+=nodes[ell];successful.append(ell)
    flag=sum(nodes[ell]*x[vr['y',ell]] for ell in leaves)
    value=-res.fun if mode=='frontier' else res.fun+(fee if mode=='cost' else 0)
    raw_bound=res.mip_dual_bound if res.mip_dual_bound is not None else res.fun
    bound=-raw_bound if mode=='frontier' else raw_bound+(fee if mode=='cost' else 0)
    out.update(r=rr,z=zz,value=float(value),solver_bound=float(bound),gap=float(res.mip_gap or 0.),actual_success=float(actual),flag_success=float(flag),max_violation=float(max(0,np.max(A@x-np.asarray(rhs)))),success_leaves=[''.join(map(str,n)) for n in successful])
    if mode=='cost':
        processing=sum(c*x[i] for i,c in enumerate(costs) if i<vr['y',leaves[0]])
        out['processing_cost']=float(processing);out['actual_penalized_cost']=float(processing+fee*(1-actual))
    if keep and not foresight:
        out['actions']=[dict(history=''.join(map(str,n)),time=start+len(n)-1,job=j,channel=k,service=float(x[i])) for key,i in vr.items() if isinstance(key,tuple) and len(key)==4 for ctx,n,j,k in [key] if x[i]>1e-10]
        out['budgets']=[dict(history=''.join(map(str,n)),success=sum(nodes[ell] for ell in successful if ell[:len(n)]==n)/prob) for n,prob in nodes.items()]
    return out

def all_stress_calendar(jobs,K,h,z,r,rmax):
    # A zero-ordinary-capacity fixed tail is feasible under every clearing path.
    out=solve(jobs,K,0.,z,rmax,r,mode='cost',b=1.,fee=0.,p=.25,rho=.4,keep=True)
    if 'actions' in out:
        out['fixed_calendar']=[a for a in out.pop('actions') if set(a['history'])=={'1'}]
    return out

def run():
    f=pd.read_csv(DATA/'panel.csv');meta=json.loads((DATA/'calibration.json').read_text());W=meta['normalizer_GPU_hours'];h=.5
    starts=[24+3*q for q in range(8)];rows=[]
    specs=[('base',{}),('slack_zero',dict(slack=0)),('slack_two',dict(slack=2)),('reserve_low',dict(z=.1)),('reserve_high',dict(z=.4)),('overhead_20',dict(overhead=.2)),('half_rate',dict(rate_factor=.5)),('training_local',dict(remote=False)),('reliability_95',dict(b=.95)),('persistent',dict(p=.25,rho=.75)),('frequent',dict(p=.4,rho=.4)),('all_training_batch',dict(permission='all'))]
    for label,changes in specs:
        for start in starts:
            for K in ((1,2,3) if label=='base' else (3,)):
                a=dict(slack=1,overhead=0.,rate_factor=1.,permission='LP',remote=True);a.update({k:v for k,v in changes.items() if k in a})
                jobs,rmax=catalog(f,start,K,W,**a);z=changes.get('z',.25);b=changes.get('b',.9);p=changes.get('p',.25);rho=changes.get('rho',.4)
                q=solve(jobs,K,h,z,rmax,b=b,p=p,rho=rho)
                reserve=solve(jobs,K,h,z,rmax,.25,'reserve',b,p,rho) if rmax>=.25 else dict(status=2,reason='observed eligible work below fixed relief')
                row=dict(label=label,start=start,K=K,jobs=len(jobs),pods=len({j['pod'] for j in jobs}),rmax=rmax,inputs=dict(**a,z=z,b=b,p=p,rho=rho),frontier=q,reserve_at_quarter=reserve)
                if label=='base' and K==3:
                    r_eval=min(.2,.8*q.get('r',0));r_eval=max(0.,r_eval)
                    exact=solve(jobs,K,h,z,rmax,r_eval,'cost',b,p,rho,keep=True)
                    pi=solve(jobs,K,h,z,rmax,r_eval,'cost',b,p,rho,foresight=True)
                    pir=solve(jobs,K,h,z,rmax,.25,'reserve',b,p,rho,foresight=True)
                    cal=all_stress_calendar(jobs,K,h,z,r_eval,rmax)
                    nd,ll=make_tree(K,max(j['deadline'] for j in jobs)+1,p,rho)
                    def runlen(n):return max(map(len,''.join(map(str,n)).split('0')))
                    selected=ll
                    for L in range(K,K+max(j['deadline'] for j in jobs)+2):
                        ss=[ell for ell in ll if runlen(ell)<=L]
                        if sum(nd[ell] for ell in ss)>=b-1e-12:selected=ss;break
                    fixed=solve(jobs,K,h,z,rmax,r_eval,'cost',b,p,rho,fee=0.,selected=selected)
                    if fixed.get('value') is not None:fixed['actual_penalized_cost']=fixed['processing_cost']+2*(1-fixed['actual_success'])
                    # Execute slots 0 and 1 of the causal policy; carry actual debt into slot 2.
                    current=[]
                    for j,job in enumerate(jobs):
                        qj=dict(job);served=sum(x['service'] for x in exact.get('actions',[]) if x['job']==j and x['time']<2)
                        if job['origin']<2:
                            debt=max(0,job['beta']*r_eval-served)
                            if debt<=1e-8:continue
                            qj.update(w0=debt,beta=0.,release=2)
                        current.append(qj)
                    maximum=min(1.,float(f[(f.t==start+2)&(f.priority_class=='LP')&f.job_type_public.isin(['training','offline_inference'])].used_gpu_hours.sum()/W))
                    now=solve(current,K,h,z,maximum,b=b,p=p,rho=rho,start=2)
                    static=solve([j for j in current if j['origin']==2],K,h,z,maximum,b=b,p=p,rho=rho,start=2)
                    row.update(cost_at=r_eval,dynamic=exact,foresight_cost=pi,foresight_reserve=pir,calendar=cal,selected=fixed,current_frontier=now,debt_free_frontier=static,inherited_debt=sum(j['w0'] for j in current))
                rows.append(row);(DATA/'results.json').write_text(json.dumps(rows,indent=2))
                print(label,start,K,'frontier',q.get('r'),'reserve',reserve.get('z'),'status',q['status'],reserve['status'],flush=True)
    return rows

if __name__=='__main__':run()
