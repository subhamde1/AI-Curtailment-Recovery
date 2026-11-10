"""Independent routing-LP audit of the service-range corollary.

The LP uses channel service variables directly, rather than the projected
three-inequality description used to derive the closed-form endpoints.
"""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
from scipy.optimize import linprog
from multistage_structure import slot_rank

ROOT=Path(__file__).resolve().parent

def verify():
    rng=np.random.default_rng(20261102)
    errors=[]; cases=0; solves=0
    for k in range(400):
        cap=rng.uniform(0.0,1.8,2)
        rate=rng.uniform(0.0,1.8,2)
        elig=[([0] if rng.random()<.25 else ([1] if rng.random()<.2 else [0,1])) for _ in range(2)]
        # Start from one physically feasible routing, then construct feasible
        # residual work without relying on the projection formula being tested.
        route=np.zeros((2,2))
        for ch in range(2):
            shares=rng.dirichlet([1,1,1])[:2]
            for j in range(2):
                if ch in elig[j]:route[j,ch]=cap[ch]*shares[j]
        for j in range(2):
            if route[j].sum()>rate[j]:route[j]*=rate[j]/route[j].sum()
        ra,rb=rng.uniform(0,1.5,2)
        rab=rng.uniform(max(ra,rb),ra+rb)
        residual=rng.uniform(0,1,2)*[ra,rb]
        if residual.sum()>rab:residual*=rab/residual.sum()
        work=route.sum(axis=1)+residual
        q=slot_rank(cap,rate,elig)
        lo=np.maximum(work-[ra,rb],0)
        joint=max(work.sum()-rab,0)
        up=np.minimum(work,q[:2]);c=q[2]
        predicted=[(max(lo[0],joint-up[1]),min(up[0],c-lo[1])),
                   (max(lo[1],joint-up[0]),min(up[1],c-lo[0]))]
        # Variables: A on local/protected; B on local/protected.
        a=np.array([1.,1.,0.,0.]);b=np.array([0.,0.,1.,1.])
        Aub=np.array([a,b,[1,0,1,0],[0,1,0,1],a,b,-a,-b,-a-b])
        rhs=np.r_[rate,cap,work,-lo,-joint]
        bounds=[(0,None) if ch in elig[j] else (0,0) for j in range(2) for ch in range(2)]
        for j,obj in enumerate([a,b]):
            vals=[]
            for sign in [1,-1]:
                ans=linprog(sign*obj,A_ub=Aub,b_ub=rhs,bounds=bounds,method='highs')
                assert ans.success,ans.message
                vals.append(sign*ans.fun);solves+=1
            errors.extend(abs(vals[t]-predicted[j][t]) for t in [0,1])
        cases+=1
    assert max(errors)<1e-8,max(errors)
    exact=[]
    for s in [F(3,2),F(8,5),F(7,4),F(19,10),F(2)]:
        z=(2*s-3)/3
        l=s-(1+z);h=2*s-(2+2*z);u=min(s,1+z);c=1+z
        lower=max(l,h-u);upper=min(u,c-l)
        assert lower==upper==s/3
        exact.append({'s':str(s),'reserve':str(z),'range_A':[str(lower),str(upper)],'range_B':[str(lower),str(upper)]})
    report={'seed':20261102,'routing_cases':cases,'independent_LP_solves':solves,
            'maximum_endpoint_error':max(errors),'exact_witness_family_checks':exact,
            'scope':'Fixed selected success tree; ranges are coordinate projections, not a Cartesian feasible set. No approximation of the full cost frontier.'}
    (ROOT/'service_range_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':verify()
