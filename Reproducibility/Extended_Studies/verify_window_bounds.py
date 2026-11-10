"""Validate observation-window bounds against independent full-tree LPs."""
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from window_certificate import class_jobs, prices, solve


def full_tree(jobs,H):
    paths=[(0,)+tail for tail in product((0,1),repeat=H-1)]
    nodes=sorted({leaf[:t+1] for leaf in paths for t in range(H)},key=lambda v:(len(v),v))
    probabilities={(0,):1.}
    for node in nodes[1:]:
        probabilities[node]=probabilities[node[:-1]]*((.92,.08),(.32,.68))[node[-2]][node[-1]]
    jobs_at=defaultdict(list)
    variables={}
    for node in nodes:
        t=len(node)-1
        for j,job in enumerate(jobs):
            if (job['release']<=t<=job['deadline'] and
                (job['id'][0]=='base' or node[job['id'][1]]==1)):
                jobs_at[node].append(j)
                for k in (0,1):
                    variables[node,j,k]=len(variables)
    upper,limits,equality,targets=[],[],[],[]
    for node in nodes:
        for k,limit in enumerate((1.2*(1-node[-1]),1.6)):
            upper.append([variables[node,j,k] for j in jobs_at[node]])
            limits.append(limit)
        for j in jobs_at[node]:
            job=jobs[j]
            upper.append([variables[node,j,k] for k in (0,1)])
            limits.append(job['rate'])
            if len(node)-1==job['deadline']:
                equality.append([variables[node[:t+1],j,k]
                                 for t in range(job['release'],job['deadline']+1) for k in (0,1)])
                targets.append(job['work'])

    def matrix(rows):
        A=lil_matrix((len(rows),len(variables)))
        for i,row in enumerate(rows):
            A[i,row]=1.
        return A.tocsr()

    c=[probabilities[n]*prices(len(n)-1,n[-1])[k] for n,j,k in variables]
    result=linprog(c,A_ub=matrix(upper),b_ub=limits,A_eq=matrix(equality),b_eq=targets,method='highs')
    assert result.success,result.message
    return float(result.fun)


def main():
    rng=np.random.default_rng(247011)
    records=[]
    for case in range(12):
        H=7+case%2
        jobs=[]
        for j in range(6+case%4):
            start=int(rng.integers(0,H-1));duration=int(rng.integers(2,min(5,H-start)+1))
            jobs.append(dict(id=('base',j),release=start,deadline=start+duration-1,
                             work=float(rng.uniform(.12,.24)),rate=float(rng.uniform(.12,.20))))
        for origin in range(1,H-6):
            jobs.extend(class_jobs(origin,.4))
        exact=full_tree(jobs,H)
        windows=[solve(w,jobs,H) for w in (2,3,H)]
        for q in windows:
            assert q['projected_lower']<=exact+1e-8<=q['policy_upper']+2e-8
        for low,high in zip(windows,windows[1:]):
            assert low['projected_lower']<=high['projected_lower']+1e-8
            assert low['policy_upper']>=high['policy_upper']-1e-8
        last=windows[-1]
        error=max(abs(last['projected_lower']-exact),abs(last['policy_upper']-exact))
        assert error<1e-8
        records.append(dict(case=case,horizon=H,full_tree_cost=exact,error_at_full_window=error,windows=windows))
    report=dict(independent_full_tree_cases=len(records),window_comparisons=3*len(records),
                max_full_window_error=max(r['error_at_full_window'] for r in records),cases=records)
    (Path(__file__).resolve().parent/'window_bounds_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':
    main()
