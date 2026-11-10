"""All complete matched one-to-three-call windows on chronological holdout days."""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

HERE=Path(__file__).resolve().parent
TRACE=HERE.parent/'hourly_trace'
sys.path.insert(0,str(HERE.parent))
from hourly_trace_study import catalog


def load_panel():
    base=pd.read_csv(TRACE/'panel.csv',float_precision='round_trip')
    assert set(base.t)==set(range(48))
    frames=[base]
    manifest=json.loads((HERE/'additional_trace_manifest.json').read_text())
    added=manifest['days']
    assert added in ([111],[111,112])
    assert [(r['day'],r['hour']) for r in manifest['partitions']]==[(d,h) for d in added for h in range(24)]
    for r in manifest['partitions']:
        day,hour=r['day'],r['hour']
        source=HERE/'additional_trace_partitions'/f'day{day}_hour{hour:02d}.parquet'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==r['sha256']
        frame=pq.read_table(source).to_pandas()
        assert len(frame)==r['selected_rows']
        frame['t']=24*(day-109)+hour
        frames.append(frame)
    panel=pd.concat(frames,ignore_index=True)
    assert set(panel.t)==set(range(24*(2+len(added)))) and not panel.duplicated(['t','pod_id']).any()
    assert panel.pod_id.str.startswith('00').all()
    assert np.isfinite(panel.gpu_request).all() and (panel.gpu_request>0).all()
    assert np.isfinite(panel.used_gpu_hours).all() and (panel.used_gpu_hours>=0).all()
    return panel,[110]+added


def sparse(rows,size):
    ii,jj,vv=[],[],[]
    for i,row in enumerate(rows):
        for j,x in row.items():
            ii.append(i);jj.append(j);vv.append(x)
    return coo_matrix((vv,(ii,jj)),shape=(len(rows),size)).tocsr()


def all_stress(jobs,K,rmax,independent_cohorts=False):
    """Direct one-path LP, independent of the original scenario-tree MILP."""
    if not jobs or rmax<=0:
        return dict(frontier=0.,reserve=None,reason='an origin has no eligible occupation')
    clearing=max(j['deadline'] for j in jobs)-K+1
    assert clearing>=0 and Fraction(11,20)**clearing > Fraction(1,10)
    edges=[(i,t) for i,j in enumerate(jobs) for t in range(j['release'],j['deadline']+1)]
    r,z=len(edges),len(edges)+1
    size=len(edges)+2
    work=[{r:-j['beta']} for j in jobs]
    capacity,rate=defaultdict(dict),defaultdict(dict)
    caps={}
    for col,(i,t) in enumerate(edges):
        job=jobs[i]
        work[i][col]=1.
        capacity[t][col]=1.
        key=(i if independent_cohorts else job['pod'],t)
        rate[key][col]=1.
        caps[key]=min(caps.get(key,np.inf),job['rate'])
    for row in capacity.values():
        row[z]=-1.
    ub=list(capacity.values())+list(rate.values())
    rhs=[0.]*len(capacity)+[caps[key] for key in rate]
    U=sparse(ub,size);E=sparse(work,size)

    def solve(which,rbound,zbound):
        c=np.zeros(size);c[which]=-1. if which==r else 1.
        result=linprog(c,A_ub=U,b_ub=rhs,A_eq=E,b_eq=np.zeros(len(jobs)),
                       bounds=[(0,None)]*len(edges)+[rbound,zbound],method='highs')
        assert result.status in (0,2),result.message
        if not result.success:
            return None,0.
        errors=[float(np.max(np.abs(E@result.x))),float(np.max(U@result.x-rhs)),float(-result.x.min())]
        dual=sum(m*b for m,b in zip(result.ineqlin.marginals,rhs))
        for index,(lo,hi) in ((r,rbound),(z,zbound)):
            if lo is not None:dual+=lo*result.lower.marginals[index]
            if hi is not None:dual+=hi*result.upper.marginals[index]
        assert abs(dual-result.fun)<1e-8 and max(errors)<1e-8
        return float(result.x[which]),max(0.,*errors)

    front,err1=solve(r,(0,rmax),(.25,.25))
    reserve,err2=(None,0.) if rmax<.25 else solve(z,(.25,.25),(0,None))
    assert front is not None
    return dict(frontier=front,reserve=reserve,clearing_slots=clearing,
                all_stress_mass=float(Fraction(11,20)**clearing),max_residual=max(err1,err2))


def main():
    panel,holdout_days=load_panel()
    W=json.loads((TRACE/'calibration.json').read_text())['normalizer_GPU_hours']
    before=json.loads((TRACE/'results.json').read_text())
    old={(q['start'],q['K']):q for q in before if q['label']=='base'}
    rows=[]
    for day in holdout_days:
        for hour in range(22):
            start=(day-109)*24+hour
            for K in (1,2,3):
                jobs,rmax=catalog(panel,start,K,W)
                out=all_stress(jobs,K,rmax)
                rows.append(dict(day=day,hour=hour,calls=K,design='base',jobs=len(jobs),
                                 pods=len({j['pod'] for j in jobs}),eligible_relief_cap=rmax,**out))
                if (start,K) in old:
                    previous=old[start,K]
                    assert abs(out['frontier']-previous['frontier']['r'])<1e-8
                    assert abs(out['reserve']-previous['reserve_at_quarter']['z'])<1e-8
            base,rmax=catalog(panel,start,3,W)
            cases=[('independent_cohorts',base,rmax,True),
                   ('overhead_and_rate',*catalog(panel,start,3,W,overhead=.2,rate_factor=.75),False)]
            for label,jobs,cap,relax in cases:
                rows.append(dict(day=day,hour=hour,calls=3,design=label,jobs=len(jobs),
                                 pods=len({j['pod'] for j in jobs}),eligible_relief_cap=cap,
                                 **all_stress(jobs,3,cap,independent_cohorts=relax)))
    frame=pd.DataFrame(rows)
    base=frame[frame.design=='base']
    paired=base.pivot(index=['day','hour'],columns='calls',values='frontier')
    assert (paired[1]>=paired[2]-1e-9).all() and (paired[2]>=paired[3]-1e-9).all()
    variants=frame[frame.calls==3].pivot(index=['day','hour'],columns='design',values='frontier')
    assert (variants.independent_cohorts>=variants.base-1e-9).all()
    assert (variants.overhead_and_rate<=variants.base+1e-9).all()
    daily=[]
    for (day,k),sub in base.groupby(['day','calls']):
        daily.append(dict(day=int(day),calls=int(k),origins=len(sub),mean_frontier=float(sub.frontier.mean()),
                          min_frontier=float(sub.frontier.min()),max_frontier=float(sub.frontier.max()),
                          finite_reserve=int(sub.reserve.notna().sum()),mean_finite_reserve=float(sub.reserve.mean())))
    daily_panel=[]
    for d in [109]+holdout_days:
        observed=panel[panel.t//24+109==d]
        daily_panel.append(dict(day=d,rows=len(observed),unique_pods=int(observed.pod_id.nunique())))
    reserve_pairs=frame[frame.calls==3].pivot(index=['day','hour'],columns='design',values='reserve')
    finite_pair=reserve_pairs[['base','overhead_and_rate']].dropna()
    assert len(finite_pair)>0 and (finite_pair.overhead_and_rate>=finite_pair.base-1e-9).all()
    report=dict(training_day=109,holdout_days=holdout_days,training_normalizer=W,observed_hours=24*(1+len(holdout_days)),
                observed_rows=len(panel),unique_pods=int(panel.pod_id.nunique()),daily_panel=daily_panel,
                matched_origins=len(paired),base_models=len(base),all_models=len(frame),daily=daily,
                strict_single_to_three_declines=int((paired[1]>paired[3]+1e-9).sum()),
                mean_single_to_three_decline=float((paired[1]-paired[3]).mean()),
                lineage_relaxation_strict_origins=int((variants.independent_cohorts>variants.base+1e-9).sum()),
                max_lineage_relaxation=float((variants.independent_cohorts-variants.base).max()),
                joint_stress_mean_frontier=float(variants.overhead_and_rate.mean()),
                three_call_mean_frontier=float(variants.base.mean()),
                three_call_mean_reserve=float(finite_pair.base.mean()),
                joint_stress_mean_reserve=float(finite_pair.overhead_and_rate.mean()),
                joint_stress_finite_reserve=int(frame.loc[frame.design=='overhead_and_rate','reserve'].notna().sum()),
                largest_lp_residual=float(frame.max_residual.max()),original_matches=len(old))
    frame.to_csv(HERE/'holdout_results.csv',index=False,float_format='%.17g')
    (HERE/'holdout_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
