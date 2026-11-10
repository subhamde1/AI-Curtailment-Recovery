import json,itertools,sys
from pathlib import Path
import numpy as np
from studies import tree
OUT=Path(__file__).parent
rows=json.load(open(OUT/'dynamic_results.json'))
checks=[]
def independent_frontier(a,static=False):
    T,D=a['T'],a['D'];N=max(T+D,2);nd=tree(N,a['p'],a['rho'],a['e0']);vals=[]
    u,v=(0,0) if static else (a['u'],a['v'])
    for ell,prob in nd.items():
        if len(ell)!=N:continue
        rr=1
        for aa in range(1,N+1):
            for bb in range(aa,N+1):
                cap=sum(a['h']*(1-ell[t-1])+a['z'] for t in range(aa,bb+1))
                debt=(u if aa==1 else 0)+(v if aa==1 and bb>=2 else 0)
                n=sum(ell[t-1] for t in range(1,T+1) if aa<=t+1 and t+D<=bb)
                if cap+1e-9<debt:rr=-np.inf
                elif n:rr=min(rr,(cap-debt)/n)
        vals.append((rr,prob))
    cand=sorted({max(0,r) for r,p in vals if r>=-1e-9}|{0})
    ok=[r for r in cand if sum(p for rr,p in vals if rr>=r-1e-8)>=a['b']-1e-8]
    return max(ok) if ok else None
for row in rows:
    a=row['inputs']
    if a['rate']==10:
        for st,lab in [(False,'frontier'),(True,'static')]:
            x=independent_frontier(a,st);y=row[lab].get('value')
            assert x is None and y is None or abs(x-y)<1e-7,(row['name'],lab,x,y)
            checks.append((row['name'],lab,x,y))
    for lab in ['dynamic','hard_optimum','selected','foresight']:
        v=row[lab]
        if v['status']==0:
            assert v['success']>=a['b']-1e-7
            assert abs(v['value']-v['dual'])<1e-6
    c=row['dynamic'].get('value');f=row['foresight'].get('value');sel=row['selected'].get('value')
    if c is not None:
        assert c>=f-1e-7
        if sel is not None:assert c<=sel+1e-7
        floor=row.get('floor')
        if floor and floor.get('floor') is not None:assert row['dynamic']['early']>=floor['floor']-1e-7
    assert row['pathfront']['value']>=row['frontier']['value']-1e-7
groups=json.load(open(OUT/'group_results.json'))
for row in groups:
    # The singleton-to-pair and singleton-to-triple/full comparisons are valid;
    # the displayed pair and triple partitions are not nested with each other.
    assert row['single']['value']<=row['pairs']['value']+1e-7
    assert row['single']['value']<=row['triple']['value']+1e-7
    assert row['pairs']['value']<=row['full']['value']+1e-7
    assert row['triple']['value']<=row['full']['value']+1e-7
    assert row['foresight']['value']<=row['full']['value']+1e-7
    assert row['full']['value']<=row['calendar']['value']+1e-7
    for k in ['single','pairs','triple','full','foresight']:
        assert row[k]['max_violation']<1e-7
        assert abs(row[k]['value']-row[k]['dual'])<1e-7
summary={'independent_interval_frontier_checks':len(checks),'dynamic_cases':len(rows),'group_cases':len(groups),'maximum_group_primal_dual_gap':max(abs(row[k]['value']-row[k]['dual']) for row in groups for k in ['single','pairs','triple','full','foresight']),'maximum_dynamic_mip_gap_absolute':max(abs(row[k]['value']-row[k]['dual']) for row in rows for k in ['dynamic','hard_optimum','selected','foresight'] if row[k]['status']==0)}
(OUT/'verification.json').write_text(json.dumps(summary,indent=2));print(summary)
