"""Recover min-operation conflict witnesses and independently replay them."""
from fractions import Fraction
from pathlib import Path
import json,time
import numpy as np
from multistage_structure import random_tree,slot_rank,robust_lp,rank_backward

ROOT=Path(__file__).resolve().parent

def certificates(nodes,z):
    F={};W={};bad={}
    leaves=[n for n,v in nodes.items() if not v['children']]
    for n in sorted(nodes,key=len,reverse=True):
        v=nodes[n];cs=v['children']
        if not cs:F[n]=slot_rank([v['local'],z],v['rates'],v['elig']);W[n]=[{n}]*3;continue
        failed=[c for c in cs if F[c] is None]
        if failed:F[n]=None;bad[n]=bad[failed[0]];continue
        raw=[];sets=[]
        for s in range(3):
            choices=[]
            for c in cs:
                a=nodes[c]['arrival'];aa=a[s] if s<2 else sum(a)
                choices.append((F[c][s]-aa,W[c][s]))
            val,ss=min(choices,key=lambda q:q[0]);raw.append(val);sets.append(ss)
        if min(raw)<-1e-10:
            i=int(np.argmin(raw));F[n]=None;bad[n]=sets[i];continue
        for j,d in enumerate(v['deadlines']):
            if v['t']>=d:raw[j]=0.;sets[j]=set()
        r=[];rw=[]
        for j in (0,1):
            i=j if raw[j]<=raw[2] else 2;r.append(raw[i]);rw.append(sets[i])
        if raw[2]<=sum(r)+1e-12:rt=raw[2];wt=sets[2]
        else:rt=sum(r);wt=rw[0]|rw[1]
        F[n]=slot_rank([v['local'],z],v['rates'],v['elig'])+np.array([r[0],r[1],rt])
        descendant=next(l for l in leaves if l[:len(n)]==n)
        W[n]=[ss or {descendant} for ss in [rw[0],rw[1],wt]]
    return F,W,bad

def verify_reserve_endpoints():
    """Replay the open and closed feasible-reserve intervals in Appendix M.2."""
    root, left, right = (0,), (0, 0), (0, 1)
    initial = [1, 0]
    workloads = np.array([1, 0, 1])
    cases = 0
    for reserve in map(Fraction, ("0", "1", "9/8", "15/8", "2", "17/8")):
        # At each fixed reserve, the sole usable channel can be represented by
        # the LP's first capacity field. The second channel is unavailable.
        nodes = {
            root: dict(t=1, children=[left, right], rates=[0, 0], local=0),
            left: dict(t=2, children=[], rates=[1, 0],
                       local=float(reserve if reserve > 1 else 0)),
            right: dict(t=2, children=[], rates=[1, 0], local=float(reserve / 2)),
        }
        for node in nodes.values():
            node.update(deadlines=[2, 2], arrival=[0, 0], elig=[[0], [0]])
        for selected, expected in (
            ({left}, reserve > 1),
            ({right}, reserve >= 2),
            ({left, right}, reserve >= 2),
        ):
            ranks, _ = rank_backward(nodes, 0, selected)
            assert bool(np.all(workloads <= ranks[root])) == expected
            assert robust_lp(nodes, initial, 0, selected).success == expected
            cases += 1
    return {"fixed_reserve_cases": cases, "left_infimum": 1,
            "left_attains_infimum": False, "right_minimum": 2, "full_tree_minimum": 2}


def run():
    rng=np.random.default_rng(9042026);cases=0;witness_cases=0;largest=0;scale=[];start=time.perf_counter()
    for k in range(80):
        nodes=random_tree(rng,H=4,B=3+(k%2));z=float(rng.uniform(.05,.7));initial=rng.uniform(0,1.8,2)
        F,W,bad=certificates(nodes,z);root=(0,);leaves=[n for n,v in nodes.items() if not v['children']]
        if F[root] is None:S=bad[root]
        else:
            scores=np.r_[initial,initial.sum()]-F[root]
            if max(scores)<1e-8:continue
            S=W[root][int(np.argmax(scores))]
        S=S or {leaves[0]};m=3;assert len(S)<=2**m
        ans=robust_lp(nodes,initial,z,S);assert not ans.success
        largest=max(largest,len(S));witness_cases+=1
    for B in (2,4,8):
        nodes=random_tree(rng,H=4,B=B);initial=[.4,.5];z=.5
        # Fixed-reserve separation is timed, not the 50-query reserve bisection.
        t0=time.perf_counter();F,_=rank_backward(nodes,z);rt=time.perf_counter()-t0
        t0=time.perf_counter();ans=robust_lp(nodes,initial,z);lt=time.perf_counter()-t0
        pred=F[(0,)] is not None and np.all(np.r_[initial,sum(initial)]<=F[(0,)]+1e-8);assert pred==ans.success
        scale.append(dict(branches=B,nodes=len(nodes),leaves=B**3,rank_seconds=rt,LP_seconds=lt,LP_variables=len(ans.x) if ans.x is not None else None,feasible=bool(pred)))
    out=dict(seed=9042026,multistage_witnesses_checked=witness_cases,largest_retained_witness=largest,max_branching_observations=3,witness_bound=8,independent_selected_subtree_LP_infeasibility=True,reserve_endpoint_check=verify_reserve_endpoints(),scale=scale,seconds=time.perf_counter()-start)
    (ROOT/'witness_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':run()
