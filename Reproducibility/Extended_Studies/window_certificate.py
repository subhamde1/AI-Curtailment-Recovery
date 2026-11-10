"""Deterministic observation-window bounds on the original Markov benchmark."""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import ceil, floor, isqrt
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent.parent / 'earlier_studies'
sys.path.insert(0, str(ROOT))
from compute_extensions import class_jobs, prices, transition

catalogue = next(row for row in json.loads((ROOT / 'extension_results.json').read_text())['large'] if row['base_jobs'] == 100)
horizon, origins = catalogue['H'], catalogue['T']
jobs = [dict(job) for job in catalogue['base_catalog']]
for origin in range(origins):
    if origin:  # The initial observed grid state is normal.
        jobs.extend(class_jobs(origin, .4))

P = transition(.2, .6)
probabilities = [np.array([1., 0.])]
for _ in range(1, horizon):
    probabilities.append(probabilities[-1] @ P)


def words(start, end):
    if start == 0:
        return ((0,) + tail for tail in product((0, 1), repeat=end))
    return product((0, 1), repeat=end-start+1)


def matrix(rows, size):
    row_ids, column_ids, values = [], [], []
    for i, row in enumerate(rows):
        for j, value in row.items():
            row_ids.append(i)
            column_ids.append(j)
            values.append(value)
    return coo_matrix((values, (row_ids, column_ids)), shape=(len(rows), size)).tocsr()


def certify(starts, variables, individual, balances, conditional, robust, lower, upper):
    """Round outward using exact rational bounds on costs and capacities."""
    transition_exact = ((F(23,25), F(2,25)), (F(8,25), F(17,25)))
    p = [(F(1), F(0))]
    for _ in range(1, horizon):
        p.append(tuple(sum(p[-1][i]*transition_exact[i][j] for i in (0,1)) for j in (0,1)))

    @lru_cache(None)
    def exact_mass(start, history):
        value = p[start][history[0]]
        for a, b in zip(history, history[1:]):
            value *= transition_exact[a][b]
        return value

    scale = 10**12
    sqrt_half_lower = F(isqrt(2*scale*scale)//2, scale)
    sqrt_half_upper = sqrt_half_lower + F(1,scale)
    cos = [F(1), F(1,2), F(-1,2), F(-1), F(-1,2), F(1,2)]
    sin_lo = [0, sqrt_half_lower, 1, sqrt_half_lower, 0, -sqrt_half_upper, -1, -sqrt_half_upper]
    sin_hi = [0, sqrt_half_upper, 1, sqrt_half_upper, 0, -sqrt_half_lower, -1, -sqrt_half_lower]
    targets = [F(str(job['work'])) if job['id'][0]=='base' else (F(14,100),F(18,100),F(12,100))[job['id'][2]] for job in jobs]
    rates = [F(str(job['rate'])) for job in jobs]
    caps = (F(6,5), F(8,5))
    keys = list(variables)
    cost_lower, cost_upper = [], []
    for j,t,hist,k in keys:
        prob = exact_mass(starts[j], hist)
        if k==0:
            lo = hi = F(7,10)+F(22,100)*cos[t%6]+F(2,10)*hist[-1]
        else:
            base = F(165,100)+F(7,10)*hist[-1]
            lo = base+F(35,100)*sin_lo[t%8]
            hi = base+F(35,100)*sin_hi[t%8]
        cost_lower.append(prob*lo)
        cost_upper.append(prob*hi)

    reduced = cost_lower.copy()
    dual_constant = F(0)
    conditional_keys = list(conditional)
    rows = list(individual)+list(conditional.values())
    for i, (row, multiplier) in enumerate(zip(rows, lower.ineqlin.marginals)):
        lam = min(F(0),F(float(multiplier)))
        if not lam:
            continue
        first = next(iter(row))
        if i < len(individual):
            rhs = rates[keys[first][0]]
            for col in row:
                reduced[col] -= lam
        else:
            t,suffix,k = conditional_keys[i-len(individual)]
            denominator = exact_mass(t-len(suffix)+1,suffix)
            rhs = caps[k]
            for col in row:
                j,_,hist,_ = keys[col]
                reduced[col] -= lam*exact_mass(starts[j],hist)/denominator
        dual_constant += lam*rhs
    for row, multiplier in zip(balances,lower.eqlin.marginals):
        nu = F(float(multiplier))
        dual_constant += nu*targets[keys[next(iter(row))][0]]
        for col in row:
            reduced[col] -= nu
    exact_lower = dual_constant + sum(min(0,r)*rates[key[0]] for r,key in zip(reduced,keys))

    # Restore exact completion on each job's local path. Mix the repaired
    # actions with a feasible protected-only schedule and verify all rows.
    x = [F(floor(max(0.,float(v))*scale),scale)*F(999999999,10**9) for v in upper.x]
    for row in balances:
        first = next(iter(row))
        j,_,_,_ = keys[first]
        deadline = jobs[j]['deadline']
        last = [col for col in row if keys[col][1]==deadline]
        remaining = targets[j]-sum(x[col] for col in row if col not in last)
        assert remaining >= 0
        for col in last[:-1]:
            x[col] = min(x[col],remaining)
            remaining -= x[col]
        x[last[-1]] = remaining
    eta = F(1,10**6)
    for i,(j,t,hist,k) in enumerate(keys):
        uniform = targets[j]/(jobs[j]['deadline']-jobs[j]['release']+1) if k==1 else F(0)
        x[i] = (1-eta)*x[i]+eta*uniform
    assert min(x)>=0
    assert all(sum(x[col] for col in row)==targets[keys[next(iter(row))][0]] for row in balances)
    assert all(sum(x[col] for col in row)<=rates[keys[next(iter(row))][0]] for row in individual)
    assert all(sum(x[col] for col in row)<=caps[k] for k,row in robust)
    exact_upper = sum(q*c for q,c in zip(x,cost_upper))
    assert 0 < exact_lower <= exact_upper
    places = 10**8
    outward = [F(floor(places*exact_lower),places),F(ceil(places*exact_upper),places)]
    assert outward[0] <= exact_lower <= exact_upper <= outward[1]
    return dict(checked_rational_lower=float(exact_lower),checked_rational_upper=float(exact_upper),
                outward_decimal_interval=list(map(float,outward)),
                exact_lower_fraction=str(exact_lower),exact_upper_fraction=str(exact_upper),
                verified_exactly=True)


def solve(window, jobs=jobs, horizon=horizon, certificate=False):
    @lru_cache(None)
    def mass(start, history):
        value = probabilities[start][history[0]]
        for before, after in zip(history, history[1:]):
            value *= P[before, after]
        return value

    # Store each job's actions as a function of its observations from its
    # chosen starting slot. For call jobs the history includes their origin.
    starts = [max(0, job['release']-max(1, window-1)) for job in jobs]
    variables, costs = {}, []
    individual, rate_limits = [], []
    balances, requirements = [], []
    conditional_capacity = defaultdict(dict)
    active = defaultdict(list)
    for j, job in enumerate(jobs):
        start = starts[j]
        is_call = job['id'][0] == 'call'

        def present(history):
            return not is_call or history[job['id'][1]-start] == 1

        for t in range(job['release'], job['deadline']+1):
            active[t].append(j)
            conditioning_start = max(0, t-window+1)
            assert start <= conditioning_start
            for hist in words(start, t):
                if not present(hist):
                    continue
                p = mass(start, hist)
                suffix = hist[conditioning_start-start:]
                row = {}
                for channel in ((1,) if hist[-1] else (0, 1)):
                    col = len(costs)
                    variables[j, t, hist, channel] = col
                    costs.append(p * prices(t, hist[-1])[channel])
                    row[col] = 1.
                    conditional_capacity[t, suffix, channel][col] = p / mass(conditioning_start, suffix)
                individual.append(row)
                rate_limits.append(job['rate'])
        for hist in words(start, job['deadline']):
            if not present(hist):
                continue
            row = {}
            for t in range(job['release'], job['deadline']+1):
                prefix = hist[:t-start+1]
                for channel in (0, 1):
                    col = variables.get((j, t, prefix, channel))
                    if col is not None:
                        row[col] = 1.
            balances.append(row)
            requirements.append(job['work'])

    def limit(channel):
        return (1.2, 1.6)[channel]

    # Conditional expectations of any full causal policy satisfy the lower
    # model. Markov independence preserves its projected deadline balances.
    rows = list(conditional_capacity.values())
    rhs = [limit(channel) for _, _, channel in conditional_capacity]
    E = matrix(balances, len(costs))
    lower = linprog(costs, A_ub=matrix(individual+rows, len(costs)),
                    b_ub=rate_limits+rhs, A_eq=E, b_eq=requirements,
                    method='highs', options={'time_limit': 90})

    def all_capacity_rows():
        for t, eligible in active.items():
            begin = min(starts[j] for j in eligible)
            for history in words(begin, t):
                for channel in ((1,) if history[-1] else (0, 1)):
                    yield channel, {variables[j, t, history[starts[j]-begin:], channel]: 1.
                                    for j in eligible if (j, t, history[starts[j]-begin:], channel) in variables}

    # The upper model checks actual capacities at every relevant suffix.
    robust = list(all_capacity_rows())
    upper = linprog(costs, A_ub=matrix(individual+[r for _, r in robust], len(costs)),
                    b_ub=rate_limits+[limit(k) for k, _ in robust],
                    A_eq=E, b_eq=requirements, method='highs', options={'time_limit': 90})
    assert lower.success and upper.success, (lower.message, upper.message)
    assert lower.fun <= upper.fun + 1e-8
    violations = [max(0., -upper.x.min()), float(np.max(np.abs(E@upper.x-requirements)))]
    for k, row in all_capacity_rows():
        violations.append(sum(upper.x[col] for col in row)-limit(k))
    for row, cap in zip(individual, rate_limits):
        violations.append(sum(upper.x[col] for col in row)-cap)
    result = dict(conditioning_window=window, decisions=len(costs),
                pathwise_capacity_rows=len(robust), deadline_rows=len(balances),
                projected_lower=float(lower.fun), policy_upper=float(upper.fun),
                pilot_interval_width=float(upper.fun-lower.fun),
                max_policy_residual=max(violations))
    if certificate:
        result.update(certify(starts,variables,individual,balances,conditional_capacity,robust,lower,upper))
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows',nargs='+',type=int,default=[2,3,4,5])
    args=parser.parse_args()
    records=[]
    for window in args.windows:
        result=solve(window,certificate=window>=3)
        records.append(result)
        print(json.dumps({k:v for k,v in result.items() if not k.endswith('_fraction')}),flush=True)
    (Path(__file__).parent/'window_certificate_results.json').write_text(json.dumps(records,indent=2)+'\n')
