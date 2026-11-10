"""Matched K=0,1,2,3 study with fresh-process peak RSS and timed online calls."""
from pathlib import Path
import argparse,csv,json,math,resource,subprocess,sys,time
import numpy as np
from scipy.optimize import linprog
import revision_algorithms as ra
from compute_extensions import transition

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'final_data';DATA.mkdir(exist_ok=True)
LP_SECONDS=1.;WALL_SECONDS=.050;PATHS=32;DELTA=.05/12

def worker(n,K):
    catalog=next(x for x in json.loads((ROOT/'extension_results.json').read_text())['large'] if x['base_jobs']==n)
    base=[dict(j,id=tuple(j['id'])) for j in catalog['base_catalog']];T=catalog['T'];H=catalog['H'];P=transition(.2,.6)
    paths=json.loads((ROOT/f'revision_data/larger_policy_paths_{H}.json').read_text())[:PATHS]
    calls=[];solver_limit_hits=[];old_calendar=ra.calendar_lookahead
    def limited_lp(c,rows,rhs,equalities=(),targets=(),bounds=None):
        A=ra.matrix(rows,len(c)) if rows else None;E=ra.matrix(equalities,len(c)) if equalities else None
        tic=time.perf_counter()
        ans=linprog(c,A_ub=A,b_ub=rhs if rows else None,A_eq=E,b_eq=targets if equalities else None,
            bounds=bounds or [(0,None)]*len(c),method='highs',options={'time_limit':LP_SECONDS,
                'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        ans['solve_seconds']=time.perf_counter()-tic
        ans['matrix_bytes']=sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in (A,E) if x is not None)
        if ans.success:
            dual=0.
            if rows:dual+=float(np.dot(ans.ineqlin.marginals,rhs))
            if equalities:dual+=float(np.dot(ans.eqlin.marginals,targets))
            if bounds is not None:
                dual+=sum(ans.lower.marginals[i]*lo for i,(lo,hi) in enumerate(bounds) if lo is not None)
                dual+=sum(ans.upper.marginals[i]*hi for i,(lo,hi) in enumerate(bounds) if hi is not None)
            assert abs(ans.fun-dual)<1e-7*max(1.,abs(ans.fun));ans['dual_value']=dual
        elif ans.status==1:solver_limit_hits.append(dict(variables=len(c),message=ans.message))
        return ans
    def timed_calendar(*args,**kwargs):
        tic=time.perf_counter();q=old_calendar(*args,**kwargs);wall=time.perf_counter()-tic
        calls.append(dict(wall_seconds=wall,solve_seconds=q['seconds'],variables=q['variables'],matrix_bytes=q['matrix_bytes']))
        return q
    ra.solve_lp=limited_lp;ra.calendar_lookahead=timed_calendar
    start=time.perf_counter();root=None;records=[];errors=[]
    try:root=ra.calendar_lookahead(base,0,H,0,P,1.2,1.6,T,.4,K=K)
    except RuntimeError as e:errors.append(dict(stage='root',reason=str(e)))
    # No cache: each record executes every online call, preserving timing scope.
    if root is not None:
        for path in paths:
            try:
                q=ra.operate_lookahead(base,path['events'],T,.4,P,1.2,1.6,K=K)
                records.append(dict(path=path['path'],events=path['events'],cost=q['cost'],failures=q['failures']))
            except RuntimeError as e:errors.append(dict(stage='execution',path=path['path'],reason=str(e)))
    lower=next(x['lower'] for x in json.loads((ROOT/'revision_data/larger_bounds.json').read_text()) if x['base_jobs']==n)
    B=2.7*(sum(j['work'] for j in base)+(T-1)*1.1*.4)
    wall=np.array([q['wall_seconds'] for q in calls]);solve=np.array([q['solve_seconds'] for q in calls])
    result=dict(base_jobs=n,H=H,T=T,K=K,paths_requested=PATHS,paths_completed=len(records),
        complete=root is not None and not errors,lower=lower,B=B,delta=DELTA,
        lp_time_limit=LP_SECONDS,wall_budget=WALL_SECONDS,online_calls=len(calls),
        solver_budget_failures=len(solver_limit_hits),errors=errors,
        wall_budget_overshoots=int(sum(wall>WALL_SECONDS)),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        root_wall_ms=1000*calls[0]['wall_seconds'] if root is not None else None,
        root_solve_ms=1000*calls[0]['solve_seconds'] if root is not None else None,
        root_variables=root['variables'] if root is not None else None,
        max_variables=max((x['variables'] for x in calls),default=0),
        max_matrix_mib=max((x['matrix_bytes'] for x in calls),default=0)/2**20,
        wall_p50_ms=float(np.median(wall)*1000) if len(wall) else None,
        wall_p95_ms=float(np.quantile(wall,.95)*1000) if len(wall) else None,
        wall_max_ms=float(max(wall)*1000) if len(wall) else None,
        solve_p95_ms=float(np.quantile(solve,.95)*1000) if len(solve) else None,
        elapsed_seconds=time.perf_counter()-start,records=records,calls=calls)
    if root is not None:result.update(U=root['value'],deterministic_width=root['value']-lower)
    if result['complete']:
        costs=np.array([q['cost'] for q in records]);mean=float(costs.mean());SE=float(costs.std(ddof=1)/np.sqrt(len(costs)))
        statistical=mean+B*math.sqrt(math.log(1/DELTA)/(2*len(costs)))
        upper=min(root['value'],statistical)
        result.update(sample_mean=mean,sample_SE=SE,statistical_upper=statistical,
            upper_95_simultaneous=upper,high_confidence_width=upper-lower,deadline_failures=sum(q['failures'] for q in records))
    (DATA/f'lookahead_{n}_{K}.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:result.get(k) for k in ('base_jobs','K','complete','U','sample_mean','wall_p95_ms','peak_rss_mib','solver_budget_failures')},flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',nargs=2,type=int);a=p.parse_args()
    if a.worker:return worker(*a.worker)
    for n in (20,50,100):
        for K in range(4):subprocess.run([sys.executable,__file__,'--worker',str(n),str(K)],cwd=ROOT,check=True)
    results=[json.loads((DATA/f'lookahead_{n}_{K}.json').read_text()) for n in (20,50,100) for K in range(4)]
    choices=[]
    for n in (20,50,100):
        rows=[x for x in results if x['base_jobs']==n];U0=rows[0].get('U')
        for q0,q1 in zip(rows,rows[1:]):
            if q0['complete'] and q1['complete']:assert q0['U']>=q1['U']-1e-7
        candidates=[x for x in rows if x['complete'] and x['U']<=.98*U0 and x['wall_p95_ms']<=50]
        choices.append(dict(base_jobs=n,selected_K=min(x['K'] for x in candidates) if candidates else None,
                            rule='smallest K attaining 2% deterministic upper improvement and observed p95 <=50 ms'))
    (DATA/'lookahead_depth.json').write_text(json.dumps(results,indent=2)+'\n')
    (DATA/'lookahead_depth_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    flat=[{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in results]
    keys=list(dict.fromkeys(k for x in flat for k in x))
    with (DATA/'lookahead_depth.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    max_pair=max(sum(sorted([j['rate'] for j in x['base_catalog']]+[.22,.25,.2]*2,reverse=True)[:2])
                 for x in json.loads((ROOT/'extension_results.json').read_text())['large'])
    (DATA/'pair_group_diagnostic.json').write_text(json.dumps(dict(max_pair_rate=max_pair,
        local_normal_cap=1.2,protected_cap=1.6,price_domain='nonnegative resource multipliers',
        outcome='two-job grouping has no resource coupling; separable optima are jointly feasible'),indent=2)+'\n')
    print('Choices:',choices,flush=True)

if __name__=='__main__':main()
