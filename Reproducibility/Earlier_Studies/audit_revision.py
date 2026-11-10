"""Additional adversarial boundary and channel tests; original replay comparison."""
from pathlib import Path
import json
import numpy as np
from revision_algorithms import reserve_identity
from compute_extensions import reserve_tree,transition
from compute_results import run_floor,word_probability
from itertools import product

ROOT=Path(__file__).resolve().parent


def main():
    rng=np.random.default_rng(20261014);errors=[];channel_cases=[]
    calendars=[[.5,0,0,2.5,2.5,2.5],[.5,0,2.5,0,0,0]]
    for i in range(200):
        jobs=[]
        for j in range(int(rng.integers(3,8))):
            channels=tuple((0,) if j%3==0 else ((1,) if j%3==1 else (0,1)))
            deadline=int(rng.integers(1,6));rate=float(rng.uniform(.2,.8))
            work=.05 if channels==(0,) else float(rng.uniform(.05,rate*(deadline+1)*.7))
            jobs.append(dict(work=work,rate=rate,deadline=deadline,channels=channels))
        q=reserve_identity(calendars,jobs,reveal=int(rng.integers(1,4)))
        assert q["status"]=="optimal"
        errors.append(abs(q["reserve"]-q["normalized_dual"]))
        channel_cases.append(dict(case=i,jobs=jobs,**q))
    # Original scalar reserve closed form versus independent LPs, including kinks.
    reserves=[]
    for r in [0.,1e-10,.25,.5,2/3-1e-10,2/3,2/3+1e-10,.9,1.]:
        z,dual=reserve_tree(r)
        assert abs(z-max(0,(3*r-2)/8))<1e-8
        reserves.append(dict(r=r,primal=z,dual=dual))
    # Stationary path law and killed-chain probabilities independently enumerated.
    probabilities=[]
    for T,D,p,rho in product((1,3,5),(1,2,3),(.2,.5),(0.,.6)):
        N=T+D
        words=list(product((0,1),repeat=N))
        total=sum(word_probability(e,p,rho) for e in words)
        blackout=sum(word_probability(e,p,rho) for e in words
                     if any(all(e[s:s+D+1]) for s in range(T)))
        assert abs(total-1)<1e-11 and abs(blackout-run_floor(T,D,p,rho))<1e-11
        P=transition(p,rho)
        assert np.max(np.abs(np.array([1-p,p])@P-np.array([1-p,p])))<1e-12
        probabilities.append(dict(T=T,D=D,p=p,rho=rho,zero_atom=blackout))
    out=dict(channel_primal_dual_cases=len(errors),maximum_channel_error=max(errors),
             original_reserve_boundary_cases=len(reserves),probability_cases=len(probabilities))
    (ROOT/"revision_data/audit_checks.json").write_text(json.dumps(out,indent=2))
    (ROOT/"revision_data/channel_dual_cases.json").write_text(json.dumps(channel_cases,indent=2))
    (ROOT/"revision_data/reserve_boundary_cases.json").write_text(json.dumps(reserves,indent=2))
    (ROOT/"revision_data/probability_boundary_cases.json").write_text(json.dumps(probabilities,indent=2))
    print(out)


if __name__=="__main__":main()
