"""Independent numerical checks of analytic identities; these are not proofs.

Run with Python 3.12, NumPy and SciPy. All randomized cases use a fixed seed.
The manuscript supplies the proofs and states the domains of each result.
"""
from pathlib import Path
import itertools, json, hashlib
import numpy as np
from scipy.optimize import linprog, minimize_scalar

ROOT = Path(__file__).parent
rng = np.random.default_rng(20261008)
counts = {}
def record(name, n):
    counts[name] = n

def flow_value(jobs, capacity):
    # jobs are (release, deadline, amount), indexed from zero.
    edges = [(j,t) for j,(a,d,w) in enumerate(jobs) for t in range(a,d+1)]
    rows = [[int(jj==j) for jj,t in edges] for j in range(len(jobs))]
    rhs = [w for a,d,w in jobs]
    rows += [[int(tt==t) for j,tt in edges] for t in range(len(capacity))]
    rhs += list(capacity)
    res = linprog(-np.ones(len(edges)), A_ub=rows, b_ub=rhs,
                  bounds=(0,None), method='highs')
    assert res.success
    return -res.fun

def cuts(jobs, capacity):
    return {(a,b):sum(w for r,d,w in jobs if a<=r and d<=b)-sum(capacity[a:b+1])
            for a in range(len(capacity)) for b in range(a,len(capacity))}

def edf_backup(jobs, capacity):
    rem = np.array([w for a,d,w in jobs]); backup=0.
    for t,cap in enumerate(capacity):
        for j in sorted(range(len(jobs)), key=lambda j:(jobs[j][1],j)):
            a,d,w=jobs[j]
            if a<=t<=d:
                x=min(rem[j],cap); rem[j]-=x; cap-=x
        for j,(a,d,w) in enumerate(jobs):
            if d==t: backup+=rem[j]; rem[j]=0.
    return backup

for _ in range(250):
    horizon=int(rng.integers(2,8)); n=int(rng.integers(1,7))
    jobs=[]
    for j in range(n):
        a=int(rng.integers(horizon)); d=int(rng.integers(a,horizon))
        jobs.append((a,d,float(rng.integers(0,11))/5))
    cap=rng.integers(0,16,horizon)/5
    deficit=sum(w for a,d,w in jobs)-flow_value(jobs,cap)
    cc=cuts(jobs,cap); dp=np.zeros(horizon+1)
    for b in range(horizon):
        dp[b+1]=max(dp[b],max(dp[a]+cc[a,b] for a in range(b+1)))
    assert abs(deficit-dp[-1])<1e-8
    assert abs(deficit-edf_backup(jobs,cap))<1e-8
    assert (deficit<1e-8)==(max(cc.values())<1e-8)
record('interval_flow_EDF_and_disjoint_deficit_cases',250)

# New tail-margin theorem: optimize the additional singleton-deadline work
# directly in a flow LP, independently of the interval expression.
for _ in range(180):
    h=int(rng.integers(2,8)); jobs=[]
    for j in range(int(rng.integers(1,6))):
        a=int(rng.integers(h)); d=int(rng.integers(a,h))
        jobs.append((a,d,float(rng.integers(1,9))/5))
    cap=rng.integers(0,16,h)/5; cc=cuts(jobs,cap)
    edges=[(j,t) for j,(a,d,w) in enumerate(jobs) for t in range(a,d+1)]
    eq=[[0]+[int(jj==j) for jj,t in edges] for j in range(len(jobs))]
    ub=[[int(t==0)]+[int(tt==t) for j,tt in edges] for t in range(h)]
    res=linprog([-1]+[0]*len(edges),A_ub=ub,b_ub=cap,A_eq=eq,
                b_eq=[w for a,d,w in jobs],bounds=(0,None),method='highs')
    if max(cc.values())>1e-8: assert res.status==2
    else:
        assert res.success
        assert abs(-res.fun-min(-cc[0,b] for b in range(h)))<1e-8
record('tail_margin_against_independent_flow_LP',180)

# Bounded-run formula: every finite admissible word must obey the proposed
# sharp cut value. Necessity of the infinite-cycle bound is analytic.
checked=0
for L in range(1,5):
  for D in range(1,6):
    for h,z in [(1.,0.),(1.,.2),(.3,.7)]:
      m=min((h+(L+1)*z)/L,
            min(((k+D-1)*z+h*((D+k)//(L+1)))/k for k in range(1,L+1)))
      for e in itertools.product((0,1),repeat=9):
        if '1'*(L+1) in ''.join(map(str,e)):continue
        jobs=[(t+1,t+D,m) for t in range(9-D) if e[t]]
        if not jobs:continue
        cap=np.array([h*(1-x)+z for x in e])
        assert max(cuts(jobs,cap).values())<1e-8
        checked+=1
record('bounded_run_finite_path_checks',checked)

def reserve_lp(work, rates, deadlines, tails, c0):
    H=len(tails[0])+1; J=len(work)
    keys=[('z',)]+[(0,j,0) for j in range(J)]
    keys += [(b+1,j,t) for b in range(len(tails)) for j in range(J)
             for t in range(1,deadlines[j])]
    ids={k:i for i,k in enumerate(keys)}; rows=[]; rhs=[]
    def row(dd,b):
        ar=np.zeros(len(keys))
        for k,v in dd.items():ar[ids[k]]=v
        rows.append(ar);rhs.append(b)
    row({**{(0,j,0):1 for j in range(J)},('z',):-1},c0)
    for b,tail in enumerate(tails,1):
        for t,c in enumerate(tail,1):
            row({**{(b,j,t):1 for j in range(J) if (b,j,t) in ids},('z',):-1},c)
        for j in range(J):
            row({(0,j,0):-1,**{(b,j,t):-1 for t in range(1,deadlines[j])}},-work[j])
    bounds=[(0,None)]+[(0,min(work[j],rates[j])) for j in range(J)]
    bounds += [(0,rates[j]) for b in range(len(tails)) for j in range(J)
               for t in range(1,deadlines[j])]
    res=linprog([1]+[0]*(len(keys)-1),A_ub=rows,b_ub=rhs,bounds=bounds,method='highs')
    return res

for r in np.linspace(0,1,41):
    res=reserve_lp([r/2,r,r],[1,.5,1.5],[2,4,4],[[0,2.5,2.5],[2.5,0,0]],.5)
    assert res.success and abs(res.fun-max(3*r-2,0)/8)<1e-8
record('three_job_exact_reserve_grid',41)
for _ in range(120):
    H=int(rng.integers(2,7)); ds=rng.integers(2,H+1,2)
    rates=rng.uniform(.1,1.5,2); work=rng.uniform(.05,1,2)*rates*ds
    c0=float(rng.uniform(0,1.5)); tails=rng.uniform(0,2,(int(rng.integers(2,7)),H-1))
    def feasible(z):
        lower=[]
        for ss in [(0,),(1,),(0,1)]:
            lower.append(max(0,max(sum(work[j] for j in ss)-
               sum(min(c+z,sum(rates[j] for j in ss if t<ds[j]))
                   for t,c in enumerate(tail,1)) for tail in tails)))
        u=np.minimum(work,rates)
        return lower[0]<=u[0]+1e-10 and lower[1]<=u[1]+1e-10 and max(sum(lower[:2]),lower[2])<=min(c0+z,sum(u))+1e-10
    lo=0.;hi=20.
    for k in range(65):
        mid=(lo+hi)/2
        if feasible(mid):hi=mid
        else:lo=mid
    res=reserve_lp(work,rates,ds,tails,c0)
    assert res.success and abs(res.fun-hi)<1e-7
record('two_class_many_scenario_cut_vs_LP',120)

# Sparse sharp construction: constraints x_j+z >=1 on each retained scenario.
for K in range(1,10):
    for omit in [None]+list(range(K)):
        rows=[[1]*K+[-1]];rhs=[K-1]
        for j in range(K):
            if j==omit:continue
            row=[0]*(K+1);row[j]=-1;row[-1]=-1;rows.append(row);rhs.append(-1)
        res=linprog([0]*K+[1],A_ub=rows,b_ub=rhs,bounds=[(0,1)]*K+[(0,None)],method='highs')
        assert res.success and abs(res.fun-(1/(K+1) if omit is None else 0))<1e-9
record('sparse_witness_sharpness_and_proper_subsets',sum(1+K for K in range(1,10)))

for _ in range(240):
    pi=float(rng.uniform(.01,1)); h=float(rng.uniform(.1,2)); n=float(rng.uniform(.1,3)); A=float(rng.uniform(-1,5))
    y0=0 if A<=0 else A/(1+pi) if A<=(1+pi)*h else (A+(1-pi)*h)/2
    y=min(n,y0)
    f=lambda yy: max(A-yy,0)**2+(1-pi)*max(yy-h,0)**2+pi*yy**2
    opt=minimize_scalar(f,bounds=(0,n),method='bounded',options={'xatol':1e-13})
    assert f(y)<=opt.fun+1e-9
record('quadratic_closed_form_objective_checks',240)

meta=json.loads((ROOT/'trace_metadata.json').read_text())
sample = ROOT / 'trace_sample.parquet'
if sample.is_file():
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == meta['sha256_sample']
else:
    print('Verifying archived trace results. Reconstruct the execution-summary sample to check its hash.')
trace=json.loads((ROOT/'trace_results.json').read_text())
for row in trace:
    for mode,v in [('frontier',row),('reserve',row['reserve_at_half'])]:
        assert v['status']==0 and v['success']>=.9-1e-8 and v['max_violation']<1e-8
        assert abs((v['r'] if mode=='frontier' else v['z'])-(-v['dual'] if mode=='frontier' else v['dual']))<1e-7
record('trace_models_status_success_residual_and_gap',2*len(trace))

out={'seed':20261008,'checks':counts,'all_checks_passed':True,
     'scope':'Numerical cross-checks supplement the analytic proof review. Original large-catalog simulations and their historical timings are retained, not rerun by this script.'}
(ROOT/'mathematical_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
