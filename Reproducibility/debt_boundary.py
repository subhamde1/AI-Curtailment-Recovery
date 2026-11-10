"""Paired new accreditation after an executed prefix at fixed relief 0.30.

This boundary diagnostic retains the 0.20 cost comparison in the main study.
It changes only the fixed existing relief to show when debt ceases to be slack.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pandas as pd
from hourly_trace_study import catalog,solve
R=Path(__file__).resolve().parent;D=R/'hourly_trace'

def run():
 f=pd.read_csv(D/'panel.csv');W=json.loads((D/'calibration.json').read_text())['normalizer_GPU_hours'];rows=[]
 for start in [24+3*q for q in range(8)]:
  jobs,rm=catalog(f,start,3,W);r=.3;exact=solve(jobs,3,.5,.25,rm,r,'cost',keep=True);assert exact['status']==0
  current=[]
  for k,v in enumerate(jobs):
   q=dict(v);served=sum(a['service'] for a in exact['actions'] if a['job']==k and a['time']<2)
   if v['origin']<2:
    debt=max(0,v['beta']*r-served)
    if debt<=1e-8:continue
    q.update(w0=debt,beta=0.,release=2)
   current.append(q)
  nowcap=min(1.,float(f[(f.t==start+2)&(f.priority_class=='LP')&f.job_type_public.isin(['training','offline_inference'])].used_gpu_hours.sum()/W))
  now=solve(current,3,.5,.25,nowcap,start=2);static=solve([q for q in current if q['origin']==2],3,.5,.25,nowcap,start=2)
  rows.append(dict(start=start,existing_relief=r,policy=exact,inherited_debt=sum(q['w0'] for q in current),current_frontier=now,debt_free_frontier=static))
  print(start,now['r'],static['r'],static['r']-now['r'],flush=True)
 (D/'debt_boundary_results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':run()
