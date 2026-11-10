"""Frozen synthetic revision experiments. Run --part all to replay every block."""
from pathlib import Path
from itertools import product
import argparse, csv, json, math, platform, sys
from time import perf_counter
import numpy as np
from scipy.optimize import minimize_scalar
from revision_algorithms import (reserve_identity, make_tree, accreditation,
    calendar_lookahead, operate_lookahead, restored_lower, solve_lp)
from compute_extensions import (transition, prices, causal_cost, initial_certificate,
                                 class_jobs, calendar, make_jobs, mean_lower)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "revision_data"
DATA.mkdir(exist_ok=True)


def save(name, value):
    (DATA / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False))
    if isinstance(value, list) and value and isinstance(value[0], dict):
        keys = list(dict.fromkeys(k for row in value for k in row))
        with (DATA / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
            for row in value:
                writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def reserve_phase():
    calendars = [[.5, 0, 0, 2.5, 2.5, 2.5], [.5, 0, 2.5, 0, 0, 0]]
    rows = []
    for reveal, b_rate, a_deadline, a_work, b_work in product(
            (1, 2, 3), (.25, .5, .75, 1.), (1, 2, 3), (.25, .5, .75), (.75, 1., 1.25)):
        jobs = [dict(work=a_work, rate=1., deadline=a_deadline),
                dict(work=b_work, rate=b_rate, deadline=5),
                dict(work=1., rate=1.5, deadline=5)]
        causal = reserve_identity(calendars, jobs, reveal=reveal)
        leaf = [reserve_identity(calendars, jobs, reveal=1, selected=[k]) for k in (0, 1)]
        assert causal["status"] == "optimal" and all(q["status"] == "optimal" for q in leaf)
        pi = max(q["reserve"] for q in leaf)
        assert pi <= causal["reserve"] + 1e-8
        rows.append(dict(reveal_slot=reveal, b_rate=b_rate, a_deadline=a_deadline + 1,
                         a_work=a_work, b_work=b_work, c_work=1.,
                         causal=causal["reserve"], leafwise=pi, premium=causal["reserve"] - pi,
                         dual=causal["normalized_dual"],
                         zero_leaf_positive_causal=bool(pi < 1e-8 and causal["reserve"] > 1e-8)))
    save("reserve_phase", rows)
    summary = []
    for reveal in (1, 2, 3):
        selected = [q for q in rows if q["reveal_slot"] == reveal]
        positive = sum(q["premium"] > 1e-8 for q in selected)
        zero = sum(q["leafwise"] < 1e-8 for q in selected)
        conflict = sum(q["zero_leaf_positive_causal"] for q in selected)
        summary.append(dict(reveal_slot=reveal, cases=len(selected), positive_premium=positive,
                            zero_leaf_cases=zero, causal_conflicts=conflict,
                            mean_premium=float(np.mean([q["premium"] for q in selected])),
                            max_premium=max(q["premium"] for q in selected)))
    save("reserve_phase_summary", summary)
    return summary


def accreditation_experiment():
    P = transition(.2, .6)
    tree = make_tree([], 1, 4, 1, P, rmax=1.)
    rows, exhaustive_checks = [], 0
    for eps in (0., .1, .25, .4):
        mip = accreditation(tree, .8, .25, eps)
        best = None
        for bits in product((0, 1), repeat=len(tree["leaves"])):
            selected = [i for i, bit in enumerate(bits) if bit]
            probability = sum(tree["probability"][tree["leaves"][i]] for i in selected)
            if probability < 1 - eps - 1e-12: continue
            result = accreditation(tree, .8, .25, eps, selected=selected)
            exhaustive_checks += 1
            if result["frontier"] is not None:
                best = result["frontier"] if best is None else max(best, result["frontier"])
        assert best is not None and abs(best - mip["frontier"]) < 1e-8
        rows.append(dict(H=4, epsilon=eps, **mip, exhaustive=best))
    for H, T in ((6, 3), (7, 3), (8, 4)):
        tree = make_tree([], T, H, 1, P, rmax=1.)
        for eps in (0., .1, .25, .4):
            mip = accreditation(tree, .8, .4, eps)
            relaxed = accreditation(tree, .8, .4, eps, integer=False)
            assert mip["frontier"] is not None
            assert mip["frontier"] <= relaxed["frontier"] + 1e-7
            rows.append(dict(H=H, epsilon=eps, **mip, binary_relaxation=relaxed["frontier"]))
    save("accreditation", rows)
    save("accreditation_checks", dict(exhaustive_selected_set_LPs=exhaustive_checks,
                                     maximum_exhaustive_error=max(abs(q["frontier"]-q["exhaustive"]) for q in rows if "exhaustive" in q)))
    return rows


def structural_experiment():
    rng = np.random.default_rng(20261010)
    rows = []
    max_error = 0.
    for i in range(240):
        w = float(rng.uniform(.1, 2.))
        p = float(rng.uniform(0, 3.))
        v = float(rng.uniform(0, 1.))
        k0 = float(rng.uniform(.1, 2.))
        k = float(rng.uniform(.1, 3.))
        exponent = int(rng.integers(2, 5))
        h = float(rng.uniform(0, 1.5))
        B = float(rng.uniform(max(.05, .5*w), 2.))
        current = float(rng.uniform(max(0., w - B), w))
        lo, hi = max(0., w-current), min(w, B)
        assert lo <= hi
        probs = np.array([.2, .4, .4])
        stress_probs = np.array([.5, .3, .2])
        capacities = np.array([0., h/2, h])
        def derivative(y, pp=p, qq=probs, hh=capacities):
            return -pp-k0*(w-y)+v+k*sum(qq*np.maximum(y-hh, 0)**(exponent-1))
        def analytic(pp=p, qq=probs, hh=capacities, upper=hi):
            left, right = lo, upper
            if derivative(left, pp, qq, hh) >= 0: return left
            if derivative(right, pp, qq, hh) <= 0: return right
            for _ in range(70):
                middle=(left+right)/2
                if derivative(middle, pp, qq, hh) < 0: left=middle
                else: right=middle
            return (left+right)/2
        def objective(y):
            return p*(w-y)+k0*(w-y)**2/2+v*y+k/exponent*sum(probs*np.maximum(y-capacities,0)**exponent)
        x = analytic()
        numeric = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-12})
        candidates = [numeric.x, lo, hi]
        numerical_y = min(candidates, key=objective)
        error = abs(objective(x)-objective(numerical_y))
        max_error=max(max_error,error)
        assert error < 1e-8
        expensive = analytic(pp=p+.25)
        stress = analytic(qq=stress_probs)
        headroom = analytic(hh=capacities+.2)
        more_reserve = analytic(upper=min(w,B+.25))
        assert expensive >= x-1e-10
        assert stress <= x+1e-10
        assert headroom >= x-1e-10
        assert more_reserve >= x-1e-10
        rows.append(dict(case=i, work=w, current_price=p, future_base_price=v, current_curvature=k0,
                         future_curvature=k, exponent=exponent, normal_headroom=h, future_reserve=B,
                         lower=lo, upper=hi, residual=x, expensive_residual=expensive,
                         stress_residual=stress, more_capacity_residual=headroom,
                         more_reserve_residual=more_reserve, objective_error=error))
    # EDF fails for a fixed service quantity under individual ceilings.
    jobs = [dict(work=1., rate=1., deadline=1), dict(work=1.5, rate=.5, deadline=2)]
    feasible = reserve_identity([[1.,1.,1.]], jobs, reveal=1)
    assert feasible["reserve"] < 1e-9
    naive_remaining_B = 1.5-2*.5
    assert naive_remaining_B == .5
    save("structural", rows)
    summary=dict(cases=len(rows), maximum_objective_error=max_error,
                 comparative_statics_checks=4*len(rows),
                 edf_counterexample=dict(capacity=[1,1,1], work=[1,1.5], rate=[1,.5],
                 deadlines=[2,3], EDF_missed_B=.5,
                 feasible_A=[.5,.5,0], feasible_B=[.5,.5,.5]))
    save("structural_checks", summary)
    return summary


def all_words(T, D):
    N = T+D
    values = np.arange(2**N, dtype=np.uint64)
    E = ((values[:,None] >> np.arange(N, dtype=np.uint64)) & 1).astype(np.int8)
    return E


def frontier_array(E, T, D, h, z, debt=None, a=2.):
    N=E.shape[1]; npaths=len(E)
    prefix=np.column_stack([np.zeros(npaths,dtype=int), np.cumsum(E,axis=1)])
    result=np.full(npaths,a,dtype=float)
    debt=debt or [0.]*D
    for u in range(N):
        for v in range(u,N):
            cap=h*(v-u+1-prefix[:,v+1]+prefix[:,u])+z*(v-u+1)
            inherited=sum(debt[:v+1]) if u==0 else 0.
            result[cap<inherited-1e-10]=-1.
            left=max(0,u-1); right=min(T-1,v-D)
            if right<left: continue
            count=prefix[:,right+1]-prefix[:,left]
            mask=(count>0)&(result>=0)
            result[mask]=np.minimum(result[mask],(cap[mask]-inherited)/count[mask])
    return result


def markov_weights(E, p, rho, state):
    P=transition(p,rho)
    W=np.ones(len(E))
    W[E[:,0]!=state]=0.
    for t in range(1,E.shape[1]): W*=P[E[:,t-1],E[:,t]]
    assert abs(W.sum()-1)<1e-10
    return W


def quantile_frontier(cap, W, epsilon, a=2.):
    if W[cap<0].sum()>epsilon+1e-12: return None
    order=np.argsort(cap); vals=cap[order]; mass=W[order]
    unique, first=np.unique(vals,return_index=True)
    probability=np.add.reduceat(mass,first)
    strict_lower=np.r_[0.,np.cumsum(probability)[:-1]]
    eligible=unique[(unique>=0)&(strict_lower<=epsilon+1e-12)]
    return float(min(a,eligible[-1])) if len(eligible) else 0.


def frontier_surface():
    rows=[]
    for T in (6,10):
        for D in range(1,7):
            E=all_words(T,D)
            caps={z:frontier_array(E,T,D,1.,z) for z in (0.,.1,.25,.5)}
            for rho,state in product((0.,.6,.85),(0,1)):
                W=markov_weights(E,.2,rho,state)
                zero=float(W[caps[0.]<1e-12].sum())
                margin=(T+D-1)*.002
                for z,cap in caps.items():
                    nominal=quantile_frontier(cap,W,.1)
                    guarded=quantile_frontier(cap,W,max(0.,.1-margin)) if margin<=.1 else None
                    rows.append(dict(T=T,D=D,rho=rho,initial_state=state,z=z,p=.2,
                                     epsilon=.1,frontier=nominal,zero_atom=zero,
                                     transition_margin=margin,guarded_frontier=guarded,
                                     paths=len(E)//2,probability_method="exhaustive exact finite-support"))
    save("frontier_surface",rows)
    states=[]
    for D,state,z,location in product((2,4),(0,1),(0.,.1,.3),("empty","due_now","due_end")):
        T=4;E=all_words(T,D);b=[0.]*D
        if location=="due_now":b[0]=.25
        if location=="due_end":b[-1]=.25
        cap=frontier_array(E,T,D,1.,z,debt=b);W=markov_weights(E,.2,.6,state)
        states.append(dict(T=T,D=D,initial_state=state,z=z,debt=b,profile=location,
                           zero_failure=float(W[cap<0].sum()),frontier=quantile_frontier(cap,W,.1)))
    save("conditional_debt",states)
    # Independent interval calculations on randomly selected rows.
    rng=np.random.default_rng(20261010);checks=0
    for _ in range(80):
        T=4;D=int(rng.integers(1,5));E=all_words(T,D);z=float(rng.choice([0.,.1,.3]))
        b=list(rng.uniform(0,.2,size=D));cap=frontier_array(E,T,D,1.,z,debt=b)
        indices=rng.choice(len(E),size=8,replace=False)
        for i in indices:
            independent=initial_certificate(b,E[i],T,D,1.,z,a=2.)
            assert abs(independent-cap[i])<1e-8
            checks+=1
    save("surface_checks",dict(independent_state_interval_checks=checks,rows=len(rows)))
    return dict(surface_rows=len(rows),debt_rows=len(states),independent_checks=checks)


def duration_survival(age,state,law):
    if law=="geometric":return (1-(.32 if state else .08))**age
    if state:
        lengths,weights=([3,4],[.875,.125]) if law=="narrow" else ([1,9],[.734375,.265625])
    else:lengths,weights=[12,13],[.5,.5]
    return sum(p for length,p in zip(lengths,weights) if length>age)


def duration_weights(E,state,law):
    W=np.ones(len(E));W[E[:,0]!=state]=0.
    age=np.ones(len(E),dtype=int)
    for t in range(1,E.shape[1]):
        previous=E[:,t-1]
        continuation=np.zeros(len(E))
        for e in (0,1):
            for a in np.unique(age[previous==e]):
                denominator=duration_survival(int(a)-1,e,law)
                continuation[(previous==e)&(age==a)]=duration_survival(int(a),e,law)/denominator if denominator>0 else 0.
        same=(E[:,t]==previous)
        W*=np.where(same,continuation,1-continuation)
        age=np.where(same,age+1,1)
    assert abs(W.sum()-1)<1e-10
    return W


def duration_experiment():
    rows=[]
    for T,D in product((6,10),(3,6)):
        E=all_words(T,D)
        ordinary_cap=frontier_array(E,T,D,1.,0.)
        for state,z in product((0,1),(0.,.25)):
            cap=frontier_array(E,T,D,1.,z)
            for law in ("geometric","narrow","heavy"):
                W=duration_weights(E,state,law)
                rows.append(dict(T=T,D=D,initial_state=state,initial_age=1,z=z,law=law,
                                 mean_normal=12.5,mean_stress=3.125,long_run_share=.2,
                                 frontier=quantile_frontier(cap,W,.1),
                                 zero_atom=float(W[cap<1e-12].sum()),
                                 blackout_without_protection=float(W[ordinary_cap<1e-12].sum())))
    save("duration",rows)
    return rows


def small_bounds():
    T=3;H=6;r=.6;h=reserve=1.2
    base=[dict(id=("base",0),release=0,deadline=1,work=.6,rate=.4),
          dict(id=("base",1),release=0,deadline=3,work=.8,rate=.3)]
    rows=[]
    for rho in (0.,.6,.85):
        P=transition(.2,rho);exact=causal_cost(base,T,r,P,h,reserve)["value"]
        previous_U=math.inf;previous_L=-math.inf
        for K in range(H):
            U=calendar_lookahead(base,0,H,1,P,h,reserve,T,r,K=K,small=True)
            L=restored_lower(base,T,H,1,P,h,reserve,K,r)
            cache={};expected=0.;max_gap=0.
            for word in product((0,1),repeat=H-1):
                e=(1,)+word;pw=math.prod(P[e[t-1],e[t]] for t in range(1,H))
                policy=operate_lookahead(base,e,T,r,P,h,reserve,K=K,small=True,cache=cache)
                expected+=pw*policy["cost"]
            assert previous_L<=L["value"]+1e-7
            assert L["value"]<=exact+1e-7
            assert exact<=expected+1e-7
            assert expected<=U["value"]+1e-7
            assert U["value"]<=previous_U+1e-7
            previous_U=U["value"];previous_L=L["value"]
            rows.append(dict(rho=rho,K=K,upper=U["value"],lower=L["value"],exact=exact,
                             policy_expected=expected,gap=U["value"]-L["value"],
                             policy_gap=expected-exact,upper_variables=U["variables"],
                             lower_variables=L["variables"],upper_ms=1000*U["seconds"],
                             lower_ms=1000*L["seconds"],matrix_bytes=U["matrix_bytes"]))
        assert abs(previous_U-exact)<1e-8 and abs(previous_L-exact)<1e-8
        cache={};edf=0.
        for word in product((0,1),repeat=H-1):
            e=(1,)+word;pw=math.prod(P[e[t-1],e[t]] for t in range(1,H))
            edf+=pw*operate_lookahead(base,e,T,r,P,h,reserve,K=0,small=True,edf_shield=True,cache=cache)["cost"]
        for q in rows:
            if q["rho"]==rho:q["edf_shield_expected"]=edf
    save("small_bounds",rows)
    return rows


def physical_experiment():
    rows=[]
    for it_eta,pue in product((800.,1200.,1600.),(1.1,1.2,1.4)):
        eta=it_eta/pue;Delta=.25;unit=500.;P0=2.;Pb=6.;Ph=2.
        a=eta*Delta*(Pb-P0)/unit;h=eta*Delta*Ph/unit;z=.25
        L=3;D=4
        long=(h+(L+1)*z)/L
        boundary=[((k+D-1)*z+h*((D+k)//(L+1)))/k for k in range(1,L+1)]
        work=min(a,long,*boundary);mw=work*unit/(eta*Delta)
        rows.append(dict(IT_productivity=it_eta,marginal_PUE=pue,facility_productivity=eta,
                         baseline_MW=Pb,power_floor_MW=P0,extra_headroom_MW=Ph,
                         slot_hours=Delta,work_unit_accelerator_hours=unit,
                         protected_accelerators=500.,ordinary_work_per_slot=h,
                         robust_work=work,robust_relief_MW=mw,L=L,D=D,chi=1.))
    save("physical",rows)
    return rows


def battery_experiment():
    rows=[]
    D=2;z=.2
    for origins,Q,eta,rate in product((1,2,4,8,16,32),(0.,.5,1.,2.),(.8,1.),(.25,.5)):
        H=origins+D
        edges=[(j,t,k) for j in range(origins) for t in range(j+1,j+D+1) for k in (0,1)]
        ix={edge:i for i,edge in enumerate(edges)};rv=len(edges)
        c=np.zeros(rv+1);c[rv]=-1.
        ub=[];rhs=[];eq=[];targets=[]
        for t in range(1,H):
            for k,cap in ((0,rate),(1,z)):
                ub.append({ix[j,t,k]:1. for j in range(origins) if (j,t,k) in ix});rhs.append(cap)
        ub.append({i:1. for (j,t,k),i in ix.items() if k==0});rhs.append(eta*Q)
        for j in range(origins):
            row={ix[j,t,k]:1. for t in range(j+1,j+D+1) for k in (0,1)}
            row[rv]=-1.;eq.append(row);targets.append(0.)
        ans=solve_lp(c,ub,rhs,eq,targets,bounds=[(0,None)]*rv+[(0,2.)])
        assert ans.success
        upper=((origins+D-1)*z+eta*Q)/origins
        assert -ans.fun<=upper+1e-8
        rows.append(dict(origins=origins,D=D,z=z,initial_charge=Q,efficiency=eta,
                         discharge_rate=rate,terminal_charge=0.,frontier=float(-ans.fun),
                         total_cut_upper=upper,dual=float(-ans.dual_value)))
    save("battery_runs",rows)
    return dict(cases=len(rows),maximum_duality_error=max(abs(q["frontier"]-q["dual"]) for q in rows))


PARTS={"reserve":reserve_phase,"accreditation":accreditation_experiment,
       "structure":structural_experiment,"surfaces":frontier_surface,
       "duration":duration_experiment,"bounds":small_bounds,
       "physical":physical_experiment,"battery":battery_experiment}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--part",choices=list(PARTS)+["all"],default="all")
    args=parser.parse_args();selected=PARTS if args.part=="all" else {args.part:PARTS[args.part]}
    for name,fun in selected.items():
        start=perf_counter();out=fun()
        print(json.dumps(dict(part=name,seconds=perf_counter()-start,
                             summary=out if isinstance(out,dict) else {"rows":len(out)})),flush=True)
    save("environment",dict(python=sys.version,platform=platform.platform(),numpy=np.__version__))


if __name__=="__main__":main()
