"""Independent full-tree LP audit of the single-job slope recursion."""
from pathlib import Path
from itertools import product
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from lagrangian_bounds import job_value_and_service, optimize_bound
from compute_extensions import transition, prices, causal_cost

ROOT=Path(__file__).resolve().parent


def independent_job_lp(job,initial,P,mu):
    """Build a one-job causal tree independently of the DP and revision LP helper."""
    release,deadline=job['release'],job['deadline']
    length=deadline-release+1
    leaves=[(initial,)+v for v in product((0,1),repeat=length-1)]
    nodes=sorted({v[:t+1] for v in leaves for t in range(length)},key=lambda v:(len(v),v))
    probability={(initial,):1.}
    for n in nodes:
        if len(n)>1:probability[n]=probability[n[:-1]]*P[n[-2],n[-1]]
    edges=[(n,k) for n in nodes for k in ((0,1) if n[-1]==0 else (1,))]
    ix={e:i for i,e in enumerate(edges)}
    objective=[probability[n]*(prices(release+len(n)-1,n[-1])[k]+mu[release+len(n)-1,n[-1],k]) for n,k in edges]
    rows=[];rhs=[]
    for n in nodes:
        rows.append({ix[n,k]:1. for k in (0,1) if (n,k) in ix});rhs.append(job['rate'])
    for leaf in leaves:
        rows.append({ix[leaf[:t+1],k]:-1. for t in range(length) for k in (0,1) if (leaf[:t+1],k) in ix});rhs.append(-job['work'])
    rr=[];cc=[];vv=[]
    for i,row in enumerate(rows):
        for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
    A=coo_matrix((vv,(rr,cc)),shape=(len(rows),len(edges))).tocsr()
    answer=linprog(objective,A_ub=A,b_ub=rhs,bounds=(0,None),method='highs')
    assert answer.success
    # All prices are nonnegative, so completion >= work has the same optimal
    # value as useful exactly-completed service; excess can be clipped.
    return float(answer.fun)


def main():
    rng=np.random.default_rng(20261015);errors=[];cases=[]
    for i in range(30):
        D=int(rng.integers(1,7));release=int(rng.integers(0,4));H=release+D
        rate=float(rng.uniform(.1,.8));work=float(rng.uniform(.01,D*rate))
        job=dict(work=work,rate=rate,release=release,deadline=H-1)
        P=transition(float(rng.uniform(.1,.6)),float(rng.uniform(0,.9)))
        mu=rng.uniform(0,2,size=(H,2,2));mu[:,1,0]=0
        for e in (0,1):
            dp,service=job_value_and_service(job,np.eye(2)[e],1.,P,mu,1.,1.)
            lp=independent_job_lp(job,e,P,mu)
            error=abs(dp-lp);assert error<1e-8
            assert abs(service.sum()-work)<1e-8
            errors.append(error)
            cases.append(dict(case=i,state=e,job=job,P=P.tolist(),multipliers=mu.tolist(),dp=dp,tree_lp=lp,error=error))
    small=[]
    base=[dict(id=('base',0),release=0,deadline=1,work=.6,rate=.4),
          dict(id=('base',1),release=0,deadline=3,work=.8,rate=.3)]
    for rho in (0.,.6,.85):
        P=transition(.2,rho)
        bound=optimize_bound(base,3,.6,6,P,1.2,1.2,current=1,small=True,iterations=80)
        exact=causal_cost(base,3,.6,P,1.2,1.2)
        if isinstance(exact,dict):exact=exact.get('value',exact.get('cost'))
        assert bound['lower']<=exact+1e-8
        small.append(dict(rho=rho,lower=bound['lower'],exact=float(exact),gap=float(exact-bound['lower'])))
    out=dict(single_job_tree_checks=len(errors),maximum_error=max(errors),small_joint_checks=small)
    (ROOT/'revision_data/decomposition_checks.json').write_text(json.dumps(out,indent=2))
    (ROOT/'revision_data/decomposition_cases.json').write_text(json.dumps(cases,indent=2))
    print(out)


if __name__=='__main__':main()
