"""Independent leaf LPs, exported policy replay, and conditional-budget checks."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hourly_trace_study import catalog,make_tree
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'hourly_trace'

def leaf_lp(jobs,ell,h,z,rmax,r=None,mode='frontier'):
    # Separate representation: equality completions, slot/channel and pod capacities.
    idx={};obj=[-1. if mode=='frontier' else 0.,1. if mode=='reserve' else 0.]
    bounds=[(0,rmax) if mode=='frontier' else (r,r),(0,None) if mode=='reserve' else (z,z)]
    for j,q in enumerate(jobs):
        for t in range(q['release'],q['deadline']+1):
            for k in q['channels']:idx[j,t,k]=len(obj);obj.append(0.);bounds.append((0,None))
    eq=[];ev=[];ub=[];uv=[]
    for j,q in enumerate(jobs):
        d={0:-q['beta']};d.update({i:1. for (jj,t,k),i in idx.items() if jj==j});eq.append(d);ev.append(q['w0'])
    for t in range(len(ell)):
        for k in (0,1):
            d={i:1. for (j,tt,kk),i in idx.items() if tt==t and kk==k}
            if k==1:d[1]=-1.
            ub.append(d);uv.append(0. if k else h*(1-ell[t]))
        for pod in {q['pod'] for q in jobs}:
            relevant=[j for j,q in enumerate(jobs) if q['pod']==pod]
            d={i:1. for (j,tt,k),i in idx.items() if tt==t and j in relevant}
            if d:ub.append(d);uv.append(min(jobs[j]['rate'] for j in relevant))
    def matrix(rows):
        ii=[];jj=[];vv=[]
        for n,d in enumerate(rows):
            for i,v in d.items():ii.append(n);jj.append(i);vv.append(v)
        return coo_matrix((vv,(ii,jj)),shape=(len(rows),len(obj))).tocsr()
    res=linprog(obj,A_ub=matrix(ub),b_ub=uv,A_eq=matrix(eq),b_eq=ev,bounds=bounds,method='highs')
    assert res.success,(mode,res.message)
    return -res.fun if mode=='frontier' else res.fun

def run():
    f=pd.read_csv(DATA/'panel.csv');W=json.loads((DATA/'calibration.json').read_text())['normalizer_GPU_hours'];rows=json.loads((DATA/'results.json').read_text())
    successful=[r for row in rows for r in [row['frontier'],row['reserve_at_quarter']] if 'value' in r]
    assert all(r['status']==0 and r['gap']<=1e-8 and r['max_violation']<=1e-7 and r['actual_success']>=r['target']-1e-7 for r in successful)
    errors=[];replay_count=0;leafchecks=0;maxprimal=0.;budgeterror=0.
    for row in rows:
        if row['label']!='base' or row['K']!=3:continue
        a=row['inputs'];jobs,rmax=catalog(f,row['start'],3,W,**{k:a[k] for k in ['slack','overhead','rate_factor','permission','remote']})
        nd,ll=make_tree(3,max(j['deadline'] for j in jobs)+1,a['p'],a['rho'])
        rleaf=[leaf_lp(jobs,ell,.5,a['z'],rmax) for ell in ll];zleaf=[leaf_lp(jobs,ell,.5,a['z'],rmax,.25,'reserve') for ell in ll];leafchecks+=2*len(ll)
        frontier=max(v for v in rleaf if sum(nd[e] for e,x in zip(ll,rleaf) if x>=v-1e-8)>=a['b']-1e-10)
        reserve=min(v for v in zleaf if sum(nd[e] for e,x in zip(ll,zleaf) if x<=v+1e-8)>=a['b']-1e-10)
        errors.append(max(0.,row['frontier']['r']-frontier));errors.append(abs(reserve-row['foresight_reserve']['z']))
        for mode in ['dynamic','foresight_cost','calendar','selected','current_frontier','debt_free_frontier']:
            q=row[mode];assert q['status']==0 and q['actual_success']>=q['target']-1e-7 and q['max_violation']<=1e-7,(row['start'],mode,q)
        q=row['dynamic'];acts=q['actions'];r=row['cost_at'];z=a['z'];history={n:[] for n in nd}
        for act in acts:history[tuple(map(int,act['history']))].append(act)
        actual=0.;proc=0.
        for n,aa in history.items():
            t=len(n)-1
            for k in (0,1):maxprimal=max(maxprimal,sum(v['service'] for v in aa if v['channel']==k)-(z if k else .5*(1-n[-1])))
            for pod in {jobs[v['job']]['pod'] for v in aa}:
                maxprimal=max(maxprimal,sum(v['service'] for v in aa if jobs[v['job']]['pod']==pod)-min(j['rate'] for j in jobs if j['pod']==pod))
            for v in aa:
                j=jobs[v['job']];assert j['release']<=t<=j['deadline'] and v['channel'] in j['channels']
                proc+=nd[n]*v['service']*(1 if v['channel']==0 else 4)
        suc=[]
        for ell in ll:
            ok=True
            for j,jq in enumerate(jobs):
                total=sum(v['service'] for n,aa in history.items() if n==ell[:len(n)] for v in aa if v['job']==j)
                maxprimal=max(maxprimal,total-jq['beta']*r-jq['w0'])
                ok=ok and total>=jq['beta']*r+jq['w0']-1e-8
            if ok:actual+=nd[ell];suc.append(ell)
        assert abs(actual-q['actual_success'])<1e-9
        assert abs(proc+2*(1-actual)-q['actual_penalized_cost'])<1e-7
        bb={tuple(map(int,v['history'])):v['success'] for v in q['budgets']}
        for n,prob in nd.items():
            expected=sum(nd[ell] for ell in suc if ell[:len(n)]==n)/prob
            budgeterror=max(budgeterror,abs(expected-bb[n]))
            kids=[v for v in nd if len(v)==len(n)+1 and v[:-1]==n]
            if kids:budgeterror=max(budgeterror,abs(bb[n]-sum(nd[k]*bb[k]/prob for k in kids)))
        assert maxprimal<=1e-7 and budgeterror<=1e-9
        # Fixed all-stress calendar has one node action per slot and is feasible everywhere.
        cal=row['calendar']['fixed_calendar']
        for t in range(max(j['deadline'] for j in jobs)+1):
            assert sum(v['service'] for v in cal if v['time']==t and v['channel']==0)<1e-8
            assert sum(v['service'] for v in cal if v['time']==t and v['channel']==1)<=z+1e-7
        for j,jq in enumerate(jobs):assert sum(v['service'] for v in cal if v['job']==j)>=jq['beta']*r-1e-8
        assert row['foresight_cost']['value']<=q['value']+1e-7
        assert q['value']<=min(row['calendar']['value'],row['selected']['actual_penalized_cost'])+1e-7
        assert row['debt_free_frontier']['r']>=row['current_frontier']['r']-1e-7
        replay_count+=1
    report=dict(primary_optima_checked=len(successful),heldout_policies_replayed=replay_count,independent_leaf_LPs=leafchecks,max_leaf_comparison_error=max(errors),max_action_violation=max(0,maxprimal),max_budget_error=budgeterror)
    assert report['max_leaf_comparison_error']<1e-7
    (DATA/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':run()
