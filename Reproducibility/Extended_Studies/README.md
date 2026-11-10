# Reproduce the extended studies

From the repository root, run `python reproducibility/run_replay.py --extended-studies`.
Use a working copy to retain the frozen results and package checksums.
Recomputing the holdout requires the provider inputs and `--trace-data`.
See [Alibaba data and retrieval](../../README.md#alibaba-data-and-retrieval).

| Manuscript result | Program and output |
| --- | --- |
| Theorem 1A and the weighted pivot corridor | `verify_symmetric_corridor.py` compares rank, feasibility, minimum service and balancing against independent scenario-tree LPs. It also checks the separate weighted cut. The JSON records counts and tolerances. |
| Theorem 4A and Appendix M.5 | `verify_depth_sharpness.py` builds independent cumulative-service LPs at depths one through five. It checks reserve on the full and critical trees, the extrema of each current class total, and zero reserve after each of the 62 critical-leaf deletions. |
| Theorem 31A and Table 47 | `verify_window_bounds.py` builds independent full-tree LPs for smaller cases. `window_certificate.py` solves the lower and upper LPs for the original 48-slot, 100-base-job catalogue and checks rational bounds on the wider-window results. The certificate JSON distinguishes numerical solver results from the outward-checked interval. |
| Table 48 and Figure 31 | `holdout_validation.py` reads the original 48 hours and the 24 appended hours, confirms the all-stress path is mandatory, and checks 24 base results against the earlier MILPs. It saves all 220 cases as CSV and aggregates them as JSON. `plot_holdout.py` redraws the figure from the CSV. |

`verify_extended_release.py` checks the published results and exact-fraction
enclosure against the archived records without solving LPs. The replay
checks local extract hashes when run with `--trace-data`.
The long certificate builds LPs of up to 161,382 service variables. Use the
pinned dependencies and allow several minutes. The solver can select
alternative optimal schedules. The supplied certificates report the
outward bounds from the checked run.

Day 109 sets the normalizer. The 22 complete matched starts on each of days
110 and 111 are held out. The new partitions preserve all matching rows
under the original fixed identifier-prefix and positive-request rule.
`additional_trace_manifest.json` lists source members, offsets, rows,
bytes of requested column ranges, and SHA-256 hashes. The optional
`retrieve_additional_hours.py` retrieves the day-111 Parquet extracts from
the official archive, using HTTP byte ranges. The archived-result checks
work offline. Partition byte hashes are tied to the Parquet encoding and
may change with the PyArrow version used for retrieval.

The v2026 fields describe workloads. The replay uses declared deadlines,
recovery work, processing-rate proxies and grid scenarios. The held-out
results do not measure facility power, actual curtailments or achieved
service reliability in deployment.
