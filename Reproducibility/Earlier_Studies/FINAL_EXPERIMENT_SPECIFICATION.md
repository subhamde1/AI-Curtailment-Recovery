# Integrated final development studies

Frozen before computing the comparisons below, 8 October 2026. Every input is
synthetic. The preceding numerical studies remain separate frozen records.

## Joint reliability and operating cost

Reuse the two initial jobs of the six-slot heterogeneous benchmark, with
current state changed explicitly to normal: release 0, deadline 1, work 0.6,
rate 0.4; release 0, deadline 2, work 0.8, rate 0.3. Grid law p=0.2, rho=0.6;
local headroom 1.2; T=3, H=6; original event classes (D,beta,rate)=(2,0.45,0.30)
and (3,0.65,0.40), both channels eligible, commitment ceiling 1.2. Full support
contains 32 leaves. R=(0.25,0.50,0.75,1.00), z=(0,0.15,0.30,0.45,0.60,0.90),
epsilon=(0,0.05,0.10,0.20). Operating-cost optima use flat failure fees 0 and 4.
This is 192 cost cases, 24 accreditation and 16 minimum-reserve cases.

Completion constraints cover all audited jobs. The rate ceiling is combined
across channels. Useful-service, channel, release and rate constraints apply on
all physical leaves. A deadline failure is never reclassified as success; after
the first missed audited deadline optional recovery stops and the modeled flat
failure fee is charged once. No post-failure catch-up or unspecified protection
is counted. The study excludes downstream remedial production costs.

Compare the exact chance optimum with a predeclared selected-tree policy: the
smallest maximum stress-run cap whose event probability reaches 1-epsilon.
This success set is selected from the exogenous law before cost optimization.
Its full physical-tree LP minimizes processing cost; actual success is then
enumerated and the actual failure fee added. The all-support calendar is
compared wherever feasible, using complete-support execution, not simulation.
Store node actions, selected flags, first missed deadlines and leaf costs.

## Two-job reserve boundary

One common first slot followed by two fully revealed ordinary calendars; two
released divisible jobs with fixed deadlines and combined ceilings, both
channels eligible. Analytic projection uses all three job-subset tail cuts.
The principal example has calendars (1,0,2,2) and (1,2,0,0), jobs
(work,rate,zero-based deadline)=(0.8,1,1),(0.8,0.5,3). Vary the urgent deadline,
the second job's rate, headroom only at the root, and headroom only in already
abundant normal tails. The protected channel adds z to every slot. A separate
seed 20261028 draws 200 additional two-job/four-to-seven-slot cases; compare
the projected reserve with independently constructed causal and leaf LPs.

## Lookahead on common larger workloads

Retain all original 20/50/100-job catalogs, resource envelopes and prices.
Compare K=0,1,2,3 on the first 32 of the same 128 stored independent paths per
catalog. All four methods use exactly that common subset. Prior 128-path
results remain intact. Each (catalog,K) runs in a fresh Python subprocess;
record whole-process peak RSS, sparse matrix memory, complete model plus
solver wall times, initial upper certificate and all realized costs. The LP
solver time limit is 1 second per optimization. A separate 50 ms wall-time
assessment budget counts overshoots, including model construction; it is not
silently described as an enforced solver limit. Report any incomplete solve,
and do not certify an incomplete plan as optimal. Twelve expected-policy upper
certificates allocate delta=0.05/12; use the bounded-cost Hoeffding margin and
the minimum with the deterministic calendar upper. Sample SE is descriptive.

An illustrative operating rule seeks the smallest tested K reducing the
initial calendar upper by at least 2% of U0 while the observed 95th percentile
wall latency is within 50 ms. These are disclosed decision thresholds, not
claims of an optimal budget rule or a future worst-case timing guarantee. An
online implementation must retain a feasible calendar for deadline fallback.

Pairwise decomposition is assessed first analytically: if both single-job
channel ceilings fit simultaneously, aggregating the pair does not retain a
binding capacity constraint and cannot strengthen the evaluated single-job
Lagrangian certificate. The original catalog rates and resource envelopes are
checked against this condition; do not run costly grouping without a possible
gain.

## Coordinated AI workload sensitivity

Observed stress state, p=0.2, rho=0.6, local headroom 1.2, T=2, H=6,
epsilon=0.10, r ceiling 1.2 and comparison obligation r_ref=0.6. Known jobs:
training work 0.25 released at 0, deadline 3, rate 0.20; batch work 0.20,
deadline 5, rate 0.25. Both can use both channels. Each call creates three
illustrative categories: short-window deferrable inference beta=0.15, D=1,
rate=0.20; checkpointable training beta=0.50*(1+0.20)=0.60, D=2, rate=0.35;
asynchronous batch beta=0.35, D=4, rate=0.22. Both channels eligible. z=0.40.
The labels are permissions and timing abstractions, not a production taxonomy;
interactive inference unable to tolerate even one slot is outside the pool.

Thirteen coordinated variations: baseline; patient training D=4; inference
D=2; inference ceiling 0.08; training overhead 0.80; inference local-only;
training local-only; z=0.25; z=0.60; urgent mix beta=(0.60,0.30,0.15), inference
rate 0.75 and patient D=(3,4); the same urgent mix with training D=4; the urgent
mix with inference D=2; urgent mix/training D=4 with inference local-only.
Full 32-leaf support is common to every variation. Report the frontier at the
declared reserve, minimum reserve for r_ref, and executable chance-policy cost
with failure fee 2 at r_eval=min(r_ref,r_star). Show r_eval explicitly; do not
compare unequal commitments as if their costs arose from equal work. A missing
finite reserve is a structural infeasibility, not a large cost. Report exact
optimizer lower and executed-policy upper separately even when their interval
closes. Record all class primitives and all supported leaf actions/outcomes.

## Numerical and timing conventions

Use HiGHS LP and MILP, finite probability enumeration, gap tolerance 1e-9,
physical residual tolerance 1e-7 and 30-second small-MILP time limit. A solver
time limit is reported explicitly and does not imply infeasibility. All time
and RSS values depend on the supplied runtime and hardware. Sources and JSON
record full precision; displayed tables round for readability.

Execution note: the final small-tree MILPs use HiGHS presolve disabled and 1e-9 feasibility tolerances. The executed policy, actual success, solver bound, and all physical residuals are checked independently of a success flag. This numerical setting is recorded in `final_algorithms.py` and Appendix K; the frozen designs and comparison policies above are unchanged.
