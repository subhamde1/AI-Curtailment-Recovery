"""Independent source-mark replay and all-stress LP check of the debt boundary."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collections import defaultdict
import pandas as pd
from verify_hourly import leaf_lp
from hourly_trace_study import make_tree
D=Path(__file__).resolve().parent/'hourly_trace'

def run():
 f=pd.read_csv(D/'panel.csv');W=json.loads((D/'calibration.json').read_text())['normalizer_GPU_hours'];rows=json.loads((D/'debt_boundary_results.json').read_text());err=0.;viol=0.;checks=0
 for row in rows:
  start=row['start'];part=f[(f.t>=start)&(f.t<start+3)&(f.priority_class=='LP')&f.job_type_public.isin(['training','offline_inference'])&(f.used_gpu_hours>0)].copy()
  totals=part.groupby('t').used_gpu_hours.sum();rates=part.groupby('pod_id').gpu_request.min()/W;marks=[]
  for v in part.itertuples():
   o=int(v.t-start);marks.append(dict(pod=v.pod_id,origin=o,release=o+1,deadline=o+(3 if v.job_type_public=='training' else 2),beta=float(v.used_gpu_hours/totals[v.t]),w0=0.,rate=float(rates[v.pod_id]),channels=[0,1]))
  pol=row['policy'];actions=pol['actions'];H=max(v['deadline'] for v in marks)+1;nd,ll=make_tree(3,H);group=defaultdict(list)
  for a in actions:
   n=tuple(map(int,a['history']));assert a['time']==len(n)-1;group[n].append(a)
   m=marks[a['job']];assert m['release']<=a['time']<=m['deadline']
  cost=0.;actual=0.
  for n,aa in group.items():
   for k in [0,1]:viol=max(viol,sum(a['service'] for a in aa if a['channel']==k)-(.25 if k else .5*(1-n[-1])))
   for pod in {marks[a['job']]['pod'] for a in aa}:viol=max(viol,sum(a['service'] for a in aa if marks[a['job']]['pod']==pod)-rates[pod])
   cost+=nd[n]*sum(a['service']*(1 if a['channel']==0 else 4) for a in aa)
  for ell in ll:
   done=[sum(a['service'] for n,aa in group.items() if n==ell[:len(n)] for a in aa if a['job']==j) for j in range(len(marks))]
   viol=max(viol,max(x-.3*m['beta'] for x,m in zip(done,marks)))
   if all(x>=.3*m['beta']-1e-8 for x,m in zip(done,marks)):actual+=nd[ell]
  assert actual>=.9-1e-8 and abs(cost+2*(1-actual)-pol['actual_penalized_cost'])<1e-7
  now=[]
  for j,m in enumerate(marks):
   q=dict(m)
   if m['origin']<2:
    rem=max(0,.3*m['beta']-sum(a['service'] for a in actions if a['job']==j and a['time']<2))
    if rem<=1e-8:continue
    q.update(beta=0.,w0=rem,release=2)
   q.update(release=q['release']-2,deadline=q['deadline']-2);now.append(q)
  assert abs(sum(q['w0'] for q in now)-row['inherited_debt'])<1e-9
  cap=min(1.,float(totals[start+2]/W));hh=max(q['deadline'] for q in now)+1;ell=(1,)*hh
  for qs,name in [(now,'current_frontier'),([q for q in now if q['origin']==2],'debt_free_frontier')]:
   opt=leaf_lp(qs,ell,.5,.25,cap);err=max(err,abs(opt-row[name]['r']));checks+=1
   assert row[name]['status']==0 and row[name]['max_violation']<1e-7 and row[name]['actual_success']>=.9-1e-8
  assert row['debt_free_frontier']['r']>=row['current_frontier']['r']-1e-8
 assert err<1e-7 and viol<1e-7
 report=dict(policies_replayed=len(rows),independent_all_stress_LPs=checks,max_frontier_error=err,max_replay_violation=max(0,viol))
 (D/'debt_boundary_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':run()
