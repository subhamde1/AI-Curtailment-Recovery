"""Exact rational lower bound and repaired feasible upper for the 100-job study.

Binary floating source work, rates, capacities, prices and leaf weights are
treated as exact rationals. Leaf weights are normalized once; node masses are
their descendant sums. All-support success makes the probability row redundant.
The lower certificate is a bounded-variable Lagrangian inequality for explicit
nonpositive multipliers. The upper policy completes residual work exactly at
each deadline and is mixed with a strictly slack protected uniform calendar.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import json,sys,time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import vstack
from certified_models import model

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent

def rational(x):return F.from_float(float(x))

def run():
    start=time.perf_counter();sys.path.insert(0,str(PRIOR/'earlier_studies'))
    from compute_extensions import transition,prices,class_jobs
    d=json.loads((PRIOR/'earlier_studies/extension_results.json').read_text());base=next(v['base_catalog'] for v in d['large'] if v['base_jobs']==100)
    spec=json.loads((ROOT/'study_specification.json').read_text())['large_catalog'];P=transition(.2,.6);nodes={}
    for t in range(1,49):
        words=[(0,)*t] if t<10 else ([(0,)*9+(a,)*(t-9) for a in (0,1)] if t<24 else [(0,)*9+(a,)*14+(b,)*(t-23) for a,b in product((0,1),repeat=2)])
        for n in words:
            prob=1 if t<10 else P[0,n[9]]*(P[n[9],n[23]] if t>=24 else 1)
            nodes[n]=dict(t=t,state=n[-1],prob=float(prob),children=[n+(e,) for e in ((0,1) if t in (9,23) else (n[-1],))] if t<48 else [],local=1.2*(1-n[-1]),prices=list(prices(t-1,n[-1])))
    jobs=[dict(w,release=w['release']+1,deadline=w['deadline']+1,birth=()) for w in base]
    for n,v in nodes.items():
        origin=v['t']-1
        if origin<42 and v['state']:
            jobs.extend(dict(w,release=w['release']+1,deadline=w['deadline']+1,birth=n) for w in class_jobs(origin,.4))
    M=model(nodes,jobs,b=1,z=1.6,fee=0,mean_rows=False)
    # Delete only the redundant reliability row. y_l=1 is retained in bounds.
    AA=vstack([M['A'][1:],M['R']],format='csr');LL=np.r_[M['L'][1:],np.full(len(M['meta']),-np.inf)];UU=np.r_[M['U'][1:],M['resource_upper']]
    lo=np.isfinite(LL);up=np.isfinite(UU)
    A=vstack([AA[up],-AA[lo]],format='csr');b=np.r_[UU[up],-LL[lo]]
    upper=[]
    for ctx,n,j,k in M['edges']:upper.append(rational(jobs[j]['work']))
    upper += [F(1)]*len(M['yi']);lower=[rational(v) for v in M['lb']]
    leaves=list(M['yi']);raw={l:rational(nodes[l]['prob']) for l in leaves};total=sum(raw.values());prob={l:p/total for l,p in raw.items()}
    c=[F(0)]*M['N']
    for (ctx,n,j,k),i in M['ix'].items():c[i]=sum(p for l,p in prob.items() if l[:len(n)]==n)*rational(nodes[n]['prices'][k])
    ans=linprog(np.array([float(v) for v in c]),A_ub=A,b_ub=b,bounds=[(float(a),float(u)) for a,u in zip(lower,upper)],method='highs');assert ans.success
    # Independent exact-arithmetic Lagrangian lower certificate.
    multipliers=[min(F(0),rational(v)) for v in ans.ineqlin.marginals]
    reduced=c.copy();L=sum(a*v for a,v in zip(c,lower))
    for i,lam in enumerate(multipliers):
        if not lam:continue
        shifted=rational(b[i])
        for col,coef in zip(A.indices[A.indptr[i]:A.indptr[i+1]],A.data[A.indptr[i]:A.indptr[i+1]]):
            cf=rational(coef);shifted-=cf*lower[col];reduced[col]-=lam*cf
        L+=lam*shifted
    L+=sum(min(F(0),v)*(u-l) for v,u,l in zip(reduced,upper,lower))
    # Exact causal policy projection, followed by a tiny convex feasibility repair.
    projected=[F(0)]*M['N'];uniform=[F(0)]*M['N'];remaining={}
    for leaf,idx in M['yi'].items():projected[idx]=uniform[idx]=F(1)
    for n in sorted(nodes,key=len):
        t=nodes[n]['t']
        for j,w in enumerate(jobs):
            if (None,n,j,0) not in M['ix']:continue
            work=rational(w['work'])
            rem=work if t==w['release'] else remaining[n[:-1],j]
            il=M['ix'][None,n,j,0];ir=M['ix'][None,n,j,1]
            local=min(rem,max(F(0),rational(ans.x[il])))
            remote=rem-local if t==w['deadline'] else min(rem-local,max(F(0),rational(ans.x[ir])))
            projected[il]=local;projected[ir]=remote;remaining[n,j]=rem-local-remote
            uniform[ir]=work/(w['deadline']-w['release']+1)
    mixture=F(0);violations=0
    for i in range(A.shape[0]):
        pj=F(0);uj=F(0)
        for col,coef in zip(A.indices[A.indptr[i]:A.indptr[i+1]],A.data[A.indptr[i]:A.indptr[i+1]]):
            cf=rational(coef);pj+=cf*projected[col];uj+=cf*uniform[col]
        rhs=rational(b[i]);assert uj<=rhs,('uniform infeasible',i,float(uj-rhs))
        if pj>rhs:
            violations+=1;assert uj<rhs
            mixture=max(mixture,(pj-rhs)/(pj-uj))
    # Outward dyadic rounding of the repair coefficient.
    power=2**48;mixture=F((mixture.numerator*power+mixture.denominator-1)//mixture.denominator,power)
    feasible=[(1-mixture)*p+mixture*u for p,u in zip(projected,uniform)]
    for i in range(A.shape[0]):
        lhs=sum(rational(coef)*feasible[col] for col,coef in zip(A.indices[A.indptr[i]:A.indptr[i+1]],A.data[A.indptr[i]:A.indptr[i+1]]));assert lhs<=rational(b[i])
    for x,l,u in zip(feasible,lower,upper):assert l<=x<=u
    U=sum(v*x for v,x in zip(c,feasible));assert L<=U
    scale=10**10;lb=F((L.numerator*scale)//L.denominator,scale);ub=F((U.numerator*scale+U.denominator-1)//U.denominator,scale)
    report=dict(lower=float(lb),upper=float(ub),certified_width=float(ub-lb),exact_unrounded_width=float(U-L),repair_weight=float(mixture),projected_rows_repaired=violations,inequalities_checked=A.shape[0],variables=M['N'],all_checks_exact_rational=True,probability_convention='Normalize exact binary-rational leaf weights and sum their descendant masses',seconds=time.perf_counter()-start)
    (ROOT/'exact_overlap_certificate.json').write_text(json.dumps(report,indent=2))
    cert=dict(lower_fraction=str(L),upper_fraction=str(U),repair_fraction=str(mixture),dual_nonzero=[(i,str(v)) for i,v in enumerate(multipliers) if v],policy_nonzero=[(list(n),j,k,str(feasible[i])) for (ctx,n,j,k),i in M['ix'].items() if feasible[i]])
    (ROOT/'exact_overlap_certificate_values.json').write_text(json.dumps(cert,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':run()
