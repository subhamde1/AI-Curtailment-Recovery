"""Finite-resource algorithms used by the proposal-grounded revision.

The original scripts and outputs are retained. These formulations independently
build their constraints; no analytical reserve formula is substituted into them.
All indices in this executable code are zero based; manuscript slots are one based.
"""
from collections import defaultdict
from itertools import product
from time import perf_counter
import math
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from compute_extensions import transition, prices, tree_jobs, class_jobs


def matrix(rows, n):
    rr, cc, vv = [], [], []
    for i, row in enumerate(rows):
        for j, v in row.items():
            if v:
                rr.append(i); cc.append(j); vv.append(v)
    return coo_matrix((vv, (rr, cc)), shape=(len(rows), n)).tocsr()


def solve_lp(c, rows, rhs, equalities=(), targets=(), bounds=None):
    n = len(c)
    A = matrix(rows, n) if rows else None
    E = matrix(equalities, n) if equalities else None
    start = perf_counter()
    ans = linprog(c, A_ub=A, b_ub=rhs if rows else None,
                  A_eq=E, b_eq=targets if equalities else None,
                  bounds=bounds or [(0, None)] * n, method="highs",
                  options={"primal_feasibility_tolerance": 1e-9,
                           "dual_feasibility_tolerance": 1e-9})
    elapsed = perf_counter() - start
    if ans.success:
        dual = 0.
        if rows: dual += float(np.dot(ans.ineqlin.marginals, rhs))
        if equalities: dual += float(np.dot(ans.eqlin.marginals, targets))
        if bounds is not None:
            dual += sum(ans.lower.marginals[i] * lo for i, (lo, hi) in enumerate(bounds) if lo is not None)
            dual += sum(ans.upper.marginals[i] * hi for i, (lo, hi) in enumerate(bounds) if hi is not None)
        assert abs(ans.fun - dual) <= 1e-7 * max(1., abs(ans.fun))
        ans["dual_value"] = dual
    ans["solve_seconds"] = elapsed
    ans["matrix_bytes"] = sum(a.data.nbytes + a.indices.nbytes + a.indptr.nbytes
                              for a in (A, E) if a is not None)
    return ans


def reserve_identity(calendars, jobs, reveal=2, selected=None):
    """Independent two-channel reserve primal and normalized analytical dual."""
    S = list(range(len(calendars))) if selected is None else list(selected)
    H = len(calendars[0])
    paths = {}
    capacities = {}
    for ell in S:
        path = []
        for t in range(H):
            n = (t, "common" if t < reveal - 1 else ell)
            if n in capacities:
                assert abs(capacities[n] - calendars[ell][t]) < 1e-12
            capacities[n] = calendars[ell][t]
            path.append(n)
        paths[ell] = path
    nodes = list(capacities)
    edges = [(n, j, k) for n in nodes for j, o in enumerate(jobs)
             if o.get("release", 0) <= n[0] <= o["deadline"]
             for k in o.get("channels", (0, 1))]
    ix = {edge: i for i, edge in enumerate(edges)}
    rates = {(n, j): o["rate"] for n in nodes for j, o in enumerate(jobs)
             if o.get("release", 0) <= n[0] <= o["deadline"]}
    completions = {}
    for ell, path in paths.items():
        for j, o in enumerate(jobs):
            v = path[o["deadline"]]
            completions[j, v] = [(path[t], j) for t in range(o.get("release", 0), o["deadline"] + 1)]
    z = len(edges)
    c = np.zeros(z + 1); c[z] = 1.
    rows, rhs = [], []
    for n in nodes:
        for k in (0, 1):
            row = {ix[e]: 1. for e in edges if e[0] == n and e[2] == k}
            if k == 1: row[z] = -1.
            rows.append(row); rhs.append(capacities[n] if k == 0 else 0.)
    for nj, rate in rates.items():
        rows.append({ix[nj + (k,)]: 1. for k in (0, 1) if nj + (k,) in ix})
        rhs.append(rate)
    for (j, v), path in completions.items():
        rows.append({ix[nj + (k,)]: -1. for nj in path for k in (0, 1) if nj + (k,) in ix})
        rhs.append(-jobs[j]["work"])
    primal = solve_lp(c, rows, rhs)
    if not primal.success:
        return {"status": "infeasible", "solver_status": int(primal.status)}
    # This dual is built from the theorem's multipliers rather than primal matrices.
    lambdas = {q: i for i, q in enumerate(completions)}
    offset = len(lambdas)
    alphas = {(n, k): offset + i for i, (n, k) in enumerate((n, k) for n in nodes for k in (0, 1))}
    offset += len(alphas)
    mus = {nj: offset + i for i, nj in enumerate(rates)}
    dc = np.zeros(offset + len(mus))
    for (j, v), i in lambdas.items(): dc[i] = -jobs[j]["work"]
    for (n, k), i in alphas.items(): dc[i] = capacities[n] if k == 0 else 0.
    for nj, i in mus.items(): dc[i] = rates[nj]
    dr, db = [], []
    for n, j, k in edges:
        row = {alphas[n, k]: -1., mus[n, j]: -1.}
        for req, path in completions.items():
            if (n, j) in path: row[lambdas[req]] = 1.
        dr.append(row); db.append(0.)
    dr.append({alphas[n, 1]: 1. for n in nodes}); db.append(1.)
    dual = solve_lp(dc, dr, db)
    assert dual.success
    assert abs(primal.fun + dual.fun) < 1e-8
    return {"status": "optimal", "reserve": float(primal.fun),
            "normalized_dual": float(-dual.fun), "nodes": len(nodes),
            "primal_variables": len(c), "dual_variables": len(dc),
            "seconds": primal.solve_seconds + dual.solve_seconds}


def make_tree(base, T, H, z0, P, rmax=1.):
    leaves = [(z0,) + word for word in product((0, 1), repeat=H - 1)]
    nodes = sorted({ell[:t + 1] for ell in leaves for t in range(H)}, key=lambda n: (len(n), n))
    prob = {(z0,): 1.}
    for n in nodes:
        if len(n) > 1: prob[n] = prob[n[:-1]] * P[n[-2], n[-1]]
    jobs = {}
    for i, job in enumerate(base):
        jobs[("base", i)] = dict(job, birth=(), w0=job["work"], beta=0.)
    for birth in nodes:
        origin = len(birth) - 1
        if origin < T and birth[-1]:
            for job in tree_jobs(origin, 1.):
                jid = ("call", birth, job["id"][-1])
                jobs[jid] = dict(job, birth=birth, w0=0., beta=job["work"])
    return dict(leaves=leaves, nodes=nodes, probability=prob, jobs=jobs,
                H=H, T=T, rmax=rmax)


def accreditation(tree, h, reserve, epsilon, selected=None, integer=True):
    """Exact deterministic causal chance frontier, or one fixed-success-set LP."""
    leaves = tree["leaves"]
    if selected is not None:
        leaves = [leaves[i] for i in selected]
    nodes = sorted({ell[:t + 1] for ell in leaves for t in range(tree["H"])}, key=lambda n: (len(n), n))
    jobs = tree["jobs"]
    edges = [(n, jid, k) for n in nodes for jid, j in jobs.items()
             if j["release"] <= len(n) - 1 <= j["deadline"]
             and (not j["birth"] or n[:len(j["birth"])] == j["birth"])
             for k in (0, 1)]
    ix = {edge: i for i, edge in enumerate(edges)}
    rv = len(edges)
    yvars = {ell: rv + 1 + i for i, ell in enumerate(leaves)} if selected is None else {}
    nv = rv + 1 + len(yvars)
    rows, rhs = [], []
    bycap, byrate = defaultdict(dict), defaultdict(dict)
    for e, i in ix.items():
        n, jid, k = e
        bycap[n, k][i] = 1.; byrate[n, jid][i] = 1.
    for (n, k), row in bycap.items():
        rows.append(row); rhs.append(h * (1 - n[-1]) if k == 0 else reserve)
    for (n, jid), row in byrate.items():
        rows.append(row); rhs.append(jobs[jid]["rate"])
        # Useful service along every prefix, even on failed leaves.
        j = jobs[jid]
        used = {ix[n[:t + 1], jid, k]: 1.
                for t in range(j["release"], len(n)) for k in (0, 1)}
        used[rv] = -j["beta"]
        rows.append(used); rhs.append(j["w0"])
    for ell in leaves:
        for jid, j in jobs.items():
            if j["birth"] and ell[:len(j["birth"])] != j["birth"]: continue
            row = {ix[ell[:t + 1], jid, k]: -1.
                   for t in range(j["release"], j["deadline"] + 1) for k in (0, 1)}
            row[rv] = j["beta"]
            if yvars:
                big_m = j["w0"] + j["beta"] * tree["rmax"]
                row[yvars[ell]] = big_m
                rhs.append(big_m - j["w0"])
            else: rhs.append(-j["w0"])
            rows.append(row)
    if yvars:
        rows.append({i: -tree["probability"][ell] for ell, i in yvars.items()})
        rhs.append(-(1. - epsilon))
    c = np.zeros(nv); c[rv] = -1.
    upper = np.full(nv, np.inf); upper[rv] = tree["rmax"]
    for i in yvars.values(): upper[i] = 1.
    if yvars:
        integ = np.zeros(nv)
        if integer:
            for i in yvars.values(): integ[i] = 1.
        A = matrix(rows, nv)
        start = perf_counter()
        ans = milp(c, integrality=integ, bounds=Bounds(np.zeros(nv), upper),
                   constraints=LinearConstraint(A, -np.inf, rhs),
                   options={"mip_rel_gap": 1e-9, "time_limit": 120.})
        seconds = perf_counter() - start
    else:
        ans = solve_lp(c, rows, rhs, bounds=[(0, tree["rmax"]) if i == rv else (0, None) for i in range(nv)])
        seconds = ans.solve_seconds
    if not ans.success and ans.x is None:
        return dict(status=int(ans.status), frontier=None, seconds=seconds, variables=nv, leaves=len(leaves))
    success_probability = sum(tree["probability"][ell] for ell, i in yvars.items() if ans.x[i] >= .5) if yvars else sum(tree["probability"][ell] for ell in leaves)
    dual_bound = getattr(ans, "mip_dual_bound", None)
    bound = -float(ans.fun if dual_bound is None else dual_bound)
    return dict(status=int(ans.status), frontier=float(ans.x[rv]),
                upper_bound=bound, probability=float(success_probability),
                seconds=seconds, variables=nv, constraints=len(rows), leaves=len(leaves))


def calendar_lookahead(active, t, H, current, P, h, reserve, T, r, K=0,
                       small=False, edf_shield=False):
    """Full-support K-future-observation tree followed by fixed shadow tails."""
    cut = min(H - 1, t + K)
    prefixes = [(current,) + w for w in product((0, 1), repeat=cut - t)]
    nodes = sorted({(prefix[:min(s - t + 1, len(prefix))], s)
                    for prefix in prefixes for s in range(t, H)},
                   key=lambda n: (n[1], n[0]))
    prob = {}
    for p, s in nodes:
        prob[p, s] = math.prod(P[p[i - 1], p[i]] for i in range(1, len(p)))
    jobs = {}
    for i, job in enumerate(active):
        if job["work"] > 1e-10 and job["deadline"] >= t:
            jobs[("active", i)] = dict(job, birth=())
    job_factory = tree_jobs if small else class_jobs
    for origin in range(t, T):
        births = {p[:origin - t + 1] for p in prefixes} if origin <= cut else set(prefixes)
        for birth in births:
            if origin <= cut and not birth[-1]: continue
            for j in job_factory(origin, r):
                jobs[("future", origin, j["id"][-1], birth)] = dict(j, birth=birth)
    edges = []
    for n in nodes:
        p, s = n
        for jid, j in jobs.items():
            if max(t, j["release"]) <= s <= j["deadline"] and (not j["birth"] or p[:len(j["birth"])] == j["birth"]):
                edges.extend([(n, jid, 0), (n, jid, 1)])
    if not edges:
        return dict(value=0., actions={}, variables=0, seconds=0., matrix_bytes=0)
    ix = {e: i for i, e in enumerate(edges)}
    c = []
    bycap, byrate = defaultdict(dict), defaultdict(dict)
    for e, i in ix.items():
        n, jid, k = e
        p, s = n
        if s <= cut: price = prices(s, p[-1])[k]
        else:
            belief = np.eye(2)[p[-1]] @ np.linalg.matrix_power(P, s - cut)
            price = sum(belief[a] * prices(s, a)[k] for a in (0, 1))
        c.append(prob[n] * price)
        bycap[n, k][i] = 1.; byrate[n, jid][i] = 1.
    rows, rhs, eq, target = [], [], [], []
    for (n, k), row in bycap.items():
        p, s = n
        cap = h * (1 - p[-1]) if k == 0 and s <= cut else (reserve if k == 1 else 0.)
        rows.append(row); rhs.append(cap)
    for (n, jid), row in byrate.items():
        rows.append(row); rhs.append(jobs[jid]["rate"])
    seen = set()
    for prefix in prefixes:
        for jid, j in jobs.items():
            if j["birth"] and prefix[:len(j["birth"])] != j["birth"]: continue
            deadline_node = (prefix[:min(j["deadline"] - t + 1, len(prefix))], j["deadline"])
            key = (jid, deadline_node)
            if key in seen: continue
            seen.add(key)
            row = {ix[(prefix[:min(s - t + 1, len(prefix))], s), jid, k]: 1.
                   for s in range(max(t, j["release"]), j["deadline"] + 1) for k in (0, 1)}
            eq.append(row); target.append(j["work"])
    ans = solve_lp(c, rows, rhs, eq, target)
    if not ans.success:
        raise RuntimeError(("lookahead calendar infeasible", t, K, ans.message))
    if edf_shield:
        # Lexicographic EDF allocation, constrained to retain the robust calendar.
        for jid in sorted((q for q in jobs if q[0] == "active"), key=lambda q: (jobs[q]["deadline"], repr(jobs[q]["id"]))):
            terms = {ix[((current,), t), jid, k]: 1. for k in (0, 1)
                     if (((current,), t), jid, k) in ix}
            if not terms: continue
            objective = np.zeros(len(c))
            for i in terms: objective[i] = -1.
            step = solve_lp(objective, rows, rhs, eq, target)
            assert step.success
            eq.append(terms); target.append(float(-step.fun))
        ans = solve_lp(c, rows, rhs, eq, target)
        assert ans.success
    actions = {}
    for ((p, s), jid, k), q in zip(edges, ans.x):
        if s == t and jid[0] == "active" and q > 1e-10:
            actions.setdefault(jobs[jid]["id"], np.zeros(2))[k] += q
    return dict(value=float(ans.fun), dual=float(ans.dual_value), actions=actions,
                variables=len(c), nodes=len(nodes), seconds=float(ans.solve_seconds),
                matrix_bytes=int(ans.matrix_bytes))


def operate_lookahead(base, events, T, r, P, h, reserve, K=0, small=False,
                      edf_shield=False, cache=None):
    active = [dict(j) for j in base]
    cost, failures, times, variables, bytes_used = 0., 0, [], 0, 0
    for t, e in enumerate(events):
        key = (tuple(events[:t + 1]), K, edf_shield)
        plan = cache.get(key) if cache is not None else None
        if plan is None:
            plan = calendar_lookahead(active, t, len(events), int(e), P, h, reserve, T, r,
                                      K=K, small=small, edf_shield=edf_shield)
            if cache is not None: cache[key] = plan
        times.append(plan["seconds"]); variables = max(variables, plan["variables"])
        bytes_used = max(bytes_used, plan["matrix_bytes"])
        channel = np.zeros(2)
        for j in active:
            x = plan["actions"].get(j["id"], np.zeros(2))
            assert x.sum() <= min(j["work"], j["rate"]) + 1e-7
            j["work"] -= x.sum(); channel += x
            if j["deadline"] == t and j["work"] > 1e-7: failures += 1
        assert channel[0] <= h * (1 - e) + 1e-7 and channel[1] <= reserve + 1e-7
        cost += float(channel @ prices(t, int(e)))
        active = [j for j in active if j["work"] > 1e-9 and j["deadline"] > t]
        if t < T and e: active.extend((tree_jobs if small else class_jobs)(t, r))
    assert not active and failures == 0
    return dict(cost=cost, failures=failures, times=times, max_variables=variables,
                matrix_bytes=bytes_used)


def restored_lower(base, T, H, current, P, h, reserve, K, r, small=True, multipliers=None):
    """Restore nonanticipativity for the first K slots; later decisions have foresight."""
    leaves = [(current,) + word for word in product((0, 1), repeat=H - 1)]
    prob = {ell: math.prod(P[ell[t - 1], ell[t]] for t in range(1, H)) for ell in leaves}
    paths, jobs, states, weights = {}, {}, {}, {}
    job_factory = tree_jobs if small else class_jobs
    for ell in leaves:
        path = []
        for t in range(H):
            n = (t, ell[:t + 1] if t < K else ell)
            path.append(n); states[n] = ell[t]
            weights[n] = sum(prob[v] for v in leaves if v[:t + 1] == ell[:t + 1]) if t < K else prob[ell]
        paths[ell] = path
        jobs[ell] = [(("base", i), dict(j)) for i, j in enumerate(base)]
        for origin in range(T):
            if ell[origin]:
                for j in job_factory(origin, r):
                    jobs[ell].append((("call", ell[:origin + 1], j["id"][-1]), j))
    edgelist, desc = set(), {}
    for ell in leaves:
        for jid, j in jobs[ell]:
            desc[jid] = j
            for t in range(j["release"], j["deadline"] + 1):
                for k in (0, 1): edgelist.add((paths[ell][t], jid, k))
    edges = sorted(edgelist, key=repr); ix = {e: i for i, e in enumerate(edges)}
    c = [weights[n] * (prices(n[0], states[n])[k] +
         (multipliers[n[0], states[n], k] if multipliers is not None else 0.))
         for n, jid, k in edges]
    bycap, byrate = defaultdict(dict), defaultdict(dict)
    for e, i in ix.items():
        n, jid, k = e
        bycap[n, k][i] = 1.; byrate[n, jid][i] = 1.
    rows, rhs, eq, targets = [], [], [], []
    for (n, k), row in bycap.items():
        rows.append(row); rhs.append(h * (1 - states[n]) if k == 0 else reserve)
    for (n, jid), row in byrate.items():
        rows.append(row); rhs.append(desc[jid]["rate"])
    seen = set()
    for ell in leaves:
        for jid, j in jobs[ell]:
            key = (jid, paths[ell][j["deadline"]])
            if key in seen: continue
            seen.add(key)
            eq.append({ix[paths[ell][t], jid, k]: 1.
                       for t in range(j["release"], j["deadline"] + 1) for k in (0, 1)})
            targets.append(j["work"])
    ans = solve_lp(c, rows, rhs, eq, targets)
    assert ans.success
    return dict(value=float(ans.fun), dual=float(ans.dual_value),
                seconds=float(ans.solve_seconds), variables=len(c),
                matrix_bytes=int(ans.matrix_bytes))
