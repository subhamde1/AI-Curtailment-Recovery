# Frozen revision experiment specification

Prepared 8 October 2026 before comparative revision outputs were calculated.
The existing synthetic experiments, figures, inputs and recorded numerical
results are retained. New experiments use separate files and seed streams.

All new data are synthetic. No GPU production trace is represented as containing
observed deadlines, curtailment permissions, power curves or recovery overhead.
The dimensional case is an explicitly hypothetical stress study.

1. Accreditation: exact finite-tree LP and selected-leaf MILP, with nonanticipative
   node allocations, combined job ceilings and prefix useful-service constraints.
   Compare the MILP with exhaustive success sets on eight-leaf instances. Report
   incumbent, optimizer bound, selected probability, run time and model size.
2. Reserve: independently formulated normalized channel dual versus min-reserve
   primal; original three-job curve retained. Sweep reveal slots 1, 2, 3 before
   the first different capacity, B ceilings 0.25, 0.5, 0.75, 1, A deadlines 2,
   3, 4, A work weights 0.25, 0.5, 0.75 and B weights 0.75, 1, 1.25; C weight 1.
   The two six-slot ordinary calendars are (0.5,0,0,2.5,2.5,2.5) and
   (0.5,0,2.5,0,0,0). All jobs release in slot 1. Report both reserve levels,
   their difference, and the incidence of zero-leaf-reserve/positive-causal-reserve
   within this declared grid, without a population-prevalence interpretation.
3. Recovery structure: randomized convex one-cohort tests, fixed feasible intervals,
   numerical optimizer checks, and an explicit finite-rate EDF counterexample.
   Seed 20261010. Test all support states and projection/kink boundaries.
4. Full commitment surfaces: exact path probabilities, p=0.2 fixed,
   rho in (0,0.6,0.85), T in (6,10), D=1,...,6, z in (0,0.1,0.25,0.5),
   observed initial state in (normal,stress), h=1, chi=1, a=2, epsilon=0.1.
   Report the full positive-commitment frontier separately from the zero atom.
   Add transition-error accreditation using delta=0.002 and known initial state.
5. Duration robustness: geometric, narrowly dispersed, and heavy-tail synthetic
   alternating-run laws with the same mean normal duration 12.5 slots and mean
   stress duration 3.125 slots; fresh observed state, known age 1. Hold the
   long-run stress share 0.2 fixed, not first-order correlation. Exact finite
   support propagation; no calibration claim.
6. Bound hierarchy: all-support K-lookahead calendar upper values and restored
   nonanticipativity lower values on the original six-slot small cases, K=0,...,5.
   Preserve all original policies and table outputs. Check nested bounds against
   the independently built exact causal LP and evaluate receding policies on
   every supported path. Do not assert monotonic realized policy costs.
7. Larger bounds: a causal single-job Lagrangian lower certificate with nonnegative
   state/slot/channel capacity prices, retaining individual rate and availability
   restrictions; a fixed projected supergradient search, retaining its best
   valid bound. Evaluate on original larger catalogs. Separate optimization
   certificates, sample means and statistical intervals.
8. Scaling: 50, 200 and 500 base jobs; horizons 24,48,72; three catalog seeds
   20261011, 20261012, 20261013; eight common evaluation paths per catalog.
   Freeze work, deadline and reserve normalization to maintain comparable load.
   Report all-support calendar feasibility, median/95th percentile solve times,
   variable counts, sampled deadline checks, and certificate width. Base job
   releases are uniform over 0,...,T-1, windows uniform over 2,...,6, work uniform
   on (0.12,0.24), and ceilings uniform on (0.12,0.20); T=H-6. Keep h=1.2,
   r=0.4, and protected rate z=1.35(0.18 n/T+1.1r). Infeasible initial
   calendars, if any, are reported rather than repaired after seeing outcomes.
   No sampled
   failure count is treated as a reliability proof.
9. Power study: one work unit=500 accelerator-hours, quarter-hour slots,
   fixed power floor 2 MW, reference import 6 MW, extra headroom 2 MW, remote
   protected rate 500 accelerators. Marginal IT productivity in (800,1200,1600)
   accelerator-hours/MWh and marginal PUE in (1.1,1.2,1.4); all are assumed,
   not measured. Convert the exact L=3,D=4 robust work frontier to MW.
10. Battery: no charging over the audited stress/clearing episode; z=0.2,
    D=2, run origins in (1,2,4,8,16,32), initial charge in (0,0.5,1,2),
    discharge efficiencies (0.8,1) and per-slot discharge ceilings (0.25,0.5).
    Compute exact finite-run commitment LPs; retain original battery cost cases.

Tolerance: 1e-8 objective equality, 1e-7 feasibility unless explicitly tightened.
All-support claims follow proofs plus pathwise constraints. Numerical tests are
corroboration and implementation audits, not substitutes for theorem proofs.
Hardware timing is descriptive and may change on replay.

The original larger catalogs additionally receive 128 fresh, independent common
evaluation paths for each of K=0 and K=1, using seed 20261020+H. Report
Hoeffding expected-policy upper bounds with failure probability 0.05/6 per
configuration/policy and the deterministic causal-decomposition lower bound.
The deterministic U values remain separate. All six statistical certificates
have joint confidence at least 95 percent. The policy-cost bound is 2.7 times
total upper shadow work; it is not an observed maximum. Original 24-path and
1,024-path studies and their original outputs remain unchanged.
