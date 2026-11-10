"""Two-class, many-scenario reserve cuts versus an independently assembled LP.

All futures are revealed after the common first allocation. No chance constraint.
The two extreme calendars and every dominated calendar are included by design.
"""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'earlier_studies'))
from revision_algorithms import reserve_identity

def cuts(cs,jobs,z):
    W=np.array([j['work'] for j in jobs]);A=np.array([j['rate'] for j in jobs])
    ds=np.array([j['deadline'] for j in jobs]);u=np.minimum(W,A)
    ll=[]
    for ss in ((0,),(1,),(0,1)):
        at=np.array([sum(A[j] for j in ss if t<=ds[j]) for t in range(1,cs.shape[1])])
        ll.append(np.maximum(0,W[list(ss)].sum()-np.minimum(cs[:,1:]+z,at).sum(axis=1)))
    vals=np.array(ll).T;L=vals.max(axis=0);cap=min(cs[0,0]+z,u.sum())
    ok=L[0]<=u[0]+1e-12 and L[1]<=u[1]+1e-12 and max(L[0]+L[1],L[2])<=cap+1e-12
    return bool(ok),vals

def run():
    rng=np.random.default_rng(20261009);jobs=[dict(release=0,deadline=1,work=.8,rate=1.),dict(release=0,deadline=3,work=.8,rate=.5)]
    rows=[]
    for S in (8,32,128,512):
        cs=np.array([[1,0,2,2],[1,2,0,0]]+[[1]+(np.array([0,2,2] if j%2==0 else [2,0,0])+rng.uniform(0,1,3)).tolist() for j in range(S-2)])
        st=time.perf_counter();full=reserve_identity(cs.tolist(),jobs,reveal=2);fullseconds=time.perf_counter()-st
        st=time.perf_counter();lo=0.;hi=1.
        for k in range(44):
            mid=(lo+hi)/2
            if cuts(cs,jobs,mid)[0]:hi=mid
            else:lo=mid
        cutseconds=time.perf_counter()-st
        ok,vals=cuts(cs,jobs,hi)
        witness=sorted(set(np.argmax(vals[:,:2],axis=0).tolist()))
        # If a joint or individual cut is binding, a singleton can suffice.
        if len(witness)==0:witness=[int(np.argmax(vals[:,2]))]
        witness_value=reserve_identity(cs.tolist(),jobs,reveal=2,selected=witness)
        assert full['status']=='optimal' and abs(full['reserve']-hi)<1e-8
        assert abs(witness_value['reserve']-full['reserve'])<1e-8
        row=dict(scenarios=S,reserve=full['reserve'],cut_lower=max(0.,lo-1e-10),cut_upper=hi+1e-10,numerical_padding=1e-10,witness=witness,
                 full_seconds=fullseconds,cut_seconds=cutseconds,full_primal_variables=full['primal_variables'],
                 witness_primal_variables=witness_value['primal_variables'],normalized_dual=full['normalized_dual'])
        rows.append(row);print(row,flush=True)
    out=dict(seed=20261009,assumptions='two classes; common first slot; full future revelation; identical channel access; no storage',cases=rows)
    (Path(__file__).parent/'conflict_results.json').write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':run()
