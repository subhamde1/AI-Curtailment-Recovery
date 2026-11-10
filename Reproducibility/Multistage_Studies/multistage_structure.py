"""Exact two-class service corridors and independent shared-history LP checks.

All class deadlines are fixed within an audit. Arrivals are proportional shards
within a class, with a common combined rate calendar and channel eligibility.
No storage, endogenous losses, or integer processing is introduced.
"""
from itertools import combinations, product
from pathlib import Path
import json, time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parent

def matrix(rows,n):
    ii=[];jj=[];vv=[]
    for i,row in enumerate(rows):
        for j,v in row.items():
            if v: ii.append(i);jj.append(j);vv.append(v)
    return coo_matrix((vv,(ii,jj)),shape=(len(rows),n)).tocsr()

def slot_rank(cap,rates,elig):
    """Network min cut over the channel side; shared job ceilings counted once."""
    out=[]
    for S in ((0,),(1,),(0,1)):
        vals=[]
        for bits in product((0,1),repeat=len(cap)):
            Q={k for k,v in enumerate(bits) if v}
            vals.append(sum(cap[k] for k in Q)+sum(rates[j] for j in S if not set(elig[j]).issubset(Q)))
        out.append(min(vals))
    return np.array(out,dtype=float)

def normalize(h):
    if min(h)<-1e-10:return None
    h=np.maximum(h,0)
    a=min(h[0],h[2]);b=min(h[1],h[2])
    return np.array([a,b,min(h[2],a+b)])

def rank_backward(nodes,z,selected=None):
    leaves=[n for n,v in nodes.items() if not v['children']]
    selected=set(leaves if selected is None else selected)
    kept={n for leaf in selected for n in nodes if leaf[:len(n)]==n}
    F={};R={};witness={}
    for n in sorted(kept,key=len,reverse=True):
        v=nodes[n];children=[c for c in v['children'] if c in kept]
        if not children:
            tail=np.zeros(3)
        else:
            hs=[]
            for c in children:
                if F[c] is None:break
                a=np.array(nodes[c]['arrival'],float)
                hs.append(F[c]-np.r_[a,a.sum()])
            else:
                h=np.min(hs,axis=0)
                for j,d in enumerate(v['deadlines']):
                    if v['t']>=d:h[j]=0
                tail=normalize(h)
                if tail is not None:
                    R[n]=tail;F[n]=tail+slot_rank([v['local'],z],v['rates'],v['elig']);continue
            tail=None
        R[n]=tail
        F[n]=None if tail is None else tail+slot_rank([v['local'],z],v['rates'],v['elig'])
    return F,R

def robust_lp(nodes,initial,z=None,selected=None):
    """Independent balance-variable LP, with no rank recursion."""
    leaves=[n for n,v in nodes.items() if not v['children']]
    selected=set(leaves if selected is None else selected)
    kept={n for leaf in selected for n in nodes if leaf[:len(n)]==n}
    keys=[];bounds=[]
    for n in kept:
        for j in (0,1):
            for typ in ('before','after'):
                keys.append((n,j,typ));bounds.append((0,None))
            for k in nodes[n]['elig'][j]:keys.append((n,j,k));bounds.append((0,None))
    ix={q:i for i,q in enumerate(keys)};nv=len(keys);cost=np.zeros(nv+(z is None))
    if z is None:zi=nv;bounds.append((0,None));cost[zi]=1
    rows=[];rhs=[];eq=[];be=[]
    for n in kept:
        v=nodes[n]
        for j in (0,1):
            service={ix[n,j,k]:1 for k in v['elig'][j]}
            rows.append(service);rhs.append(v['rates'][j])
            eq.append({ix[n,j,'before']:1,ix[n,j,'after']:-1,**{i:-q for i,q in service.items()}});be.append(0)
            if len(n)==1:eq.append({ix[n,j,'before']:1});be.append(initial[j])
            else:
                eq.append({ix[n,j,'before']:1,ix[n[:-1],j,'after']:-1});be.append(v['arrival'][j])
            if v['t']>=v['deadlines'][j] or not any(c in kept for c in v['children']):
                eq.append({ix[n,j,'after']:1});be.append(0)
        for k in (0,1):
            row={ix[n,j,k]:1 for j in (0,1) if (n,j,k) in ix}
            if k==1 and z is None:row[zi]=-1
            rows.append(row);rhs.append(v['local'] if k==0 else (z if z is not None else 0))
    ans=linprog(cost,A_ub=matrix(rows,len(cost)),b_ub=rhs,A_eq=matrix(eq,len(cost)),b_eq=be,bounds=bounds,method='highs')
    return ans

def reserve_rank(nodes,initial,selected=None):
    def ok(z):
        F,_=rank_backward(nodes,z,selected);r=F[(0,)]
        return r is not None and np.all(np.r_[initial,sum(initial)]<=r+1e-11)
    if ok(0):return 0.
    if not ok(100):return None
    lo=0.;hi=100.
    for _ in range(50):
        mid=(lo+hi)/2
        if ok(mid):hi=mid
        else:lo=mid
    return hi

def random_tree(rng,H=4,B=3):
    nodes={};ds=[int(rng.integers(2,H+1)),H]
    for t in range(H):
        words=[(0,)] if t==0 else [(0,)+w for w in product(range(B),repeat=t)]
        for n in words:
            rates=rng.uniform(.2,1.2,2)
            rates=np.array([rates[j] if t+1<=ds[j] else 0 for j in (0,1)])
            arr=np.zeros(2) if t==0 else rng.uniform(0,.12,2)
            arr=np.array([arr[j] if t+1<=ds[j] else 0 for j in (0,1)])
            elig=[([0] if rng.random()<.25 else [0,1]) for _ in (0,1)]
            nodes[n]=dict(t=t+1,deadlines=ds,rates=rates.tolist(),arrival=arr.tolist(),local=float(rng.uniform(0,1.3)),elig=elig,children=[] if t==H-1 else [n+(b,) for b in range(B)])
    return nodes

def verify():
    rng=np.random.default_rng(20261009);checks=[];maxerr=0.;cases=0
    start=time.perf_counter()
    for k in range(100):
        nodes=random_tree(rng,H=3+(k%2),B=2+(k%3==0));initial=rng.uniform(0,1.3,2)
        z=float(rng.uniform(.05,.8));F,R=rank_backward(nodes,z)
        predicted=F[(0,)] is not None and np.all(np.r_[initial,initial.sum()]<=F[(0,)]+1e-8)
        ans=robust_lp(nodes,initial,z)
        assert predicted==ans.success,(k,initial,F[(0,)],ans.message)
        cases+=1
        if k<35:
            a=robust_lp(nodes,initial,None);b=reserve_rank(nodes,initial)
            assert a.success==(b is not None)
            if a.success:maxerr=max(maxerr,abs(a.fun-b));assert abs(a.fun-b)<1e-7
        if k<20:
            leaves=[n for n,v in nodes.items() if not v['children']]
            chosen=[l for l in leaves if rng.random()<.5] or leaves[:1]
            Fs,_=rank_backward(nodes,z,chosen);pred=Fs[(0,)] is not None and np.all(np.r_[initial,sum(initial)]<=Fs[(0,)]+1e-8)
            assert pred==robust_lp(nodes,initial,z,chosen).success;cases+=1
    # Three-class counterexample: all seven subset support inequalities accept
    # (1,1,1), while shared-history service cannot complete both B and C now.
    c=np.zeros(3);ub=np.array([[0,1,1],[-1,-1,0],[-1,0,-1]])
    counter=linprog(c,A_ub=ub,b_ub=[1,-1,-1],bounds=[(0,0),(0,1),(0,1)],method='highs')
    assert counter.status==2
    report=dict(seed=20261009,feasibility_LP_checks=cases,reserve_LP_checks=35,selected_subtree_checks=20,max_reserve_error=maxerr,three_class_counterexample_infeasible=True,seconds=time.perf_counter()-start)
    (ROOT/'structure_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':verify()
