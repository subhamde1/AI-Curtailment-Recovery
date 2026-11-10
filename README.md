# The Cost of Repeatable Flexibility

Replication and verification files for Subham De, *The Cost of Repeatable
Flexibility: Stochastic Scheduling of AI Data Centers under Grid Curtailment*.
This repository snapshot corresponds to the above titled manuscript. Cite the article when referring to the study or its results.

The manuscript PDF links to this repository, which provides the code, archived results, and instructions for reproducing and verifying the reported analyses. The manuscript source hash, Table 23 data hash, and frozen values for Tables
47 and 48 are recorded in
[`published\\\_reference.json`](reproducibility/extended_studies/published_reference.json).

## Rights and access

The author-owned materials are all rights reserved. The
[`LICENSE`](LICENSE) file is a copyright notice and grants no license.
Obtain written permission from the author before running the code or reusing
author-owned materials where permission is required. Use the corresponding
author contact in the article. The commands below document the workflow for
authorized verification.

Public GitHub access is subject to the
[GitHub Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#5-license-grant-to-other-users).
Alibaba data are governed by the source provider’s terms.

## Verify the archived results

Use Python 3.12 in an isolated environment. From the repository root, run
the following commands in a working copy. The SHA-256 check uses
`sha256sum` on Linux, WSL or Git Bash. On macOS use
`shasum -a 256 -c SHA256SUMS.txt`.

```bash
sha256sum -c SHA256SUMS.txt
python -m pip install -r requirements.txt
python reproducibility/run\\\_replay.py
```

The replay runs independent numerical and rational-arithmetic checks of the
mathematical examples, adaptive cost accounting, selected-subtree and reserve
witnesses, matched policies, service ranges, and the connected-overlap
certificate. It checks the saved Markov enclosure and holdout summary
against the frozen published values. This default replay works offline
after installation. The proofs of general results are in the article.

## Recompute selected studies

|Command from the repository root|Scope|
|-|-|
|`python reproducibility/run\\\_replay.py --extended-studies`|Rebuild the exchangeable and weighted-corridor checks, depth-sharpness checks through depth five, and original-law 48-slot Markov certificate.|
|`python reproducibility/run\\\_replay.py --new-studies --multistage-studies`|Recompute the two-class conflict benchmark, matched multistage designs, and controls.|
|`python reproducibility/run\\\_replay.py --legacy-quick`|Run the partial replay of the earlier experiments.|
|`python reproducibility/earlier\\\_studies/run\\\_all.py`|Run the longer earlier-study workflow, including larger catalogs, scaling and lookahead profiles.|

Run recomputations in a working copy because programs write outputs. Fresh
solves can select different optimal allocations and give slightly different
certificate widths. The Markov check uses the reported interval
`\\\[18.77934618, 18.81152705]`. Hardware-dependent timings in Table 23 and
Figure 17 use the frozen records under `reproducibility/publication\\\_snapshot/`.
The four-leaf certificate concerns the separate two-observation model.

## Alibaba data and retrieval

The empirical inputs come from Alibaba Cluster Trace GPU v2026, documented
by Suyi Li et al., *Heterogeneity at Hyperscale: Characterization and
Scheduling of Large Production AI Clusters at Alibaba* (OSDI 2026). Alibaba
provides its traces for research and study. Obtain the source records
directly from the provider. The repository supplies the selected member index,
checksums, reconstruction scripts, and archived derived results.
[Trace release](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026) ·
[Download guide](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/data_download.md) ·
[Source-use terms](https://github.com/alibaba/clusterdata#our-motivation).

The hourly panel selects all positive requests with source pod identifiers
beginning with `00` across days 109–111. Day 109 sets the normalizer.
Days 110 and 111 supply the held-out origins. The execution-summary study
selects row groups 0, 172 and 345 from its separate archive. The retrieval
scripts use HTTP byte ranges and may transfer several gigabytes of source
columns. Run these commands from the repository root with the pinned PyArrow
version and access to the source server.

```bash
python reproducibility/retrieve\\\_hourly.py download
python reproducibility/extended\\\_studies/retrieve\\\_additional\\\_hours.py
python reproducibility/structural\\\_studies/trace\\\_read.py
python reproducibility/run\\\_replay.py --trace-data --extended-studies
```

`--trace-data` checks the local extracts and replays the saved hourly and
inherited-debt policies. With `--extended-studies` it also recomputes all
220 holdout cases. Add `--new-studies` to recompute the original hourly and
debt experiments. For the execution-template counterfactuals, run
`python reproducibility/structural\\\_studies/trace\\\_study.py` after retrieving
the sample. `studies.py` and `group\\\_study.py` in that directory recompute the
dynamic-policy and grouped-certificate experiments. Workload measurements
inform the study. Grid calls, deadlines,
interruption costs and processing permissions are declared model inputs.

## Find the supporting records

Paths below are relative to `reproducibility/`. Section, theorem, table, and
figure numbers are the displayed numbers in the master manuscript.

|Manuscript location|Programs and records|
|-|-|
|Sections 4–5, Theorems 1–4A|`multistage\\\_studies/`, `witness\\\_depth2/`, and the corridor and depth checks in `extended\\\_studies/`.|
|Sections 6–12.19 and Appendices G–K|`earlier\\\_studies/` contains the algorithms, declared catalogs, full-precision records and experiment specifications.|
|Proposition 5 and equations (116a)–(116b)|`verify\\\_adaptive\\\_certificate.py` and `adaptive\\\_certificate\\\_verification.json`.|
|Table 23 and Figure 17|`publication\\\_snapshot/scaling.json` and `render\\\_publication\\\_scaling.py`.|
|Sections 12.20–12.22 and Appendix L|`structural\\\_studies/`, `hourly\\\_trace/`, hourly study and debt-boundary scripts.|
|Section 12.23.1 and Appendix M.3|`multistage\\\_studies/exact\\\_overlap\\\_certificate.py` and its rational certificate.|
|Section 12.24, Theorem 31A and Table 47|`extended\\\_studies/window\\\_certificate.py`, `verify\\\_window\\\_bounds.py`, and the interval records.|
|Table 48 and Figure 31|`extended\\\_studies/holdout\\\_results.csv`, `holdout\\\_summary.json`, `holdout\\\_validation.py`, and `plot\\\_holdout.py`.|

After rebuilding the hourly panel, redraw the 31 figures from study records
using `python reproducibility/render\\\_figure\\\_sources.py --output-dir redrawn\\\_figures`.
Use `python reproducibility/render\\\_publication\\\_scaling.py --output scaling.png`
to redraw Figure 17 from its frozen timing data.

