import json,math,platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
W=Path(__file__).parent;fig=W/'figures';fig.mkdir(exist_ok=True)
dyn=json.load(open(W/'dynamic_results.json')); grp=json.load(open(W/'group_results.json'));tr=json.load(open(W/'trace_results.json'));meta=json.load(open(W/'trace_metadata.json'));cat=json.load(open(W/'trace_catalog.json'))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':10,'legend.fontsize':9,'savefig.bbox':'tight'})

f,ax=plt.subplots(1,2,figsize=(7,3))
vv=np.linspace(0,.9,181)
for z,q,color in [(.35,.25,'#a44728'),(.4,.4,'#183b63'),(.5,.5,'#34765c')]:
    fl=np.maximum(vv-q,0); feasible=fl<=z-.1+1e-9
    ax[0].plot(vv[feasible],fl[feasible],color=color,label=f'z = {z:g}')
    ax[0].plot(vv[~feasible],fl[~feasible],color=color,linestyle=':',alpha=.7)
ax[0].set(xlabel='Younger debt v',ylabel='Minimum early service',title='A  Debt and protected capacity');ax[0].legend(frameon=False)
ax[1].plot([0,.7],[0,0],color='#183b63');ax[1].plot([.7,1],[.6,.6],color='#183b63')
ax[1].scatter([.7],[0],s=35,color='#183b63',zorder=3);ax[1].scatter([.7],[.6],s=35,facecolor='white',edgecolor='#183b63',zorder=3)
ax[1].set(xlabel='Inherited success requirement b',ylabel='Minimum early service',title='B  A reliability threshold',ylim=(-.03,.7));f.tight_layout();f.savefig(fig/'Figure22_Early_Service.png',dpi=240);plt.close(f)

base=grp[4];f,ax=plt.subplots(figsize=(6.5,3.4));labs=['Singleton bound','Pair bound','Exact component','Calendar policy'];vals=[25*base[k]['value'] for k in ['single','pairs','full','calendar']]
bars=ax.bar(labs,vals,color=['#92a6bf','#92a6bf','#34765c','#b56d42'],width=.65)
for p,v in zip(bars,vals):ax.text(p.get_x()+p.get_width()/2,v+1,f'{v:.3f}',ha='center',fontsize=10)
ax.set(ylabel='Expected cost for 100 jobs',ylim=(0,165));ax.tick_params(axis='x',labelsize=9);f.tight_layout();f.savefig(fig/'Figure23_Component_Certificates.png',dpi=240);plt.close(f)

f,ax=plt.subplots(figsize=(6.4,3.5))
for slack,col,mark in [(0,'#a44728','o'),(1,'#183b63','s'),(2,'#34765c','^')]:
    vv=[v for v in tr if v['slack']==slack and v['z_declared']==.25];ax.plot([v['K'] for v in vv],[v['r'] for v in vv],marker=mark,color=col,label=f'Extra deadline slack = {slack} hour(s)')
ax.set(xlabel='Successive called origins in the declared episode',ylabel='Reliable commitment r',xticks=[1,2,3],ylim=(0,1));ax.legend(frameon=False);f.tight_layout();f.savefig(fig/'Figure24_Trace_Repeatability.png',dpi=240);plt.close(f)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(z) for z in rr)+' |' for rr in rows])
def num(v):return 'Infeasible' if v is None else f'{0. if abs(v)<5e-7 else v:.6f}'
parts=[]
parts.append(r'''ZZZPART_COMPUTATIONZZZ

## 10.20 The economic value of carrying debt and reliability forward

The additional study uses matched obligations to separate four quantities: the reserve premium of causal information, the error from ignoring inherited debt, the cost of a specified implementable policy, and the remaining distance between valid lower and upper certificates. These quantities cannot be inferred from one another. In particular, an unrestricted recovery pool can have no information premium in feasibility while still having a positive operating-cost premium for lack of foresight.

The base episode has three admitted call origins, recovery window two, normal headroom one, protected rate 0.4, recovery work 0.4 per call, and an observed stressed initial slot. Inherited work is 0.1 due immediately and 0.5 due next slot. The Markov parameters are $p=0.25$ and $\rho=0.4$, and the success target is 0.9. Local prices over the five-slot clearing horizon are $(1.6,1.1,0.8,1.8,1.0)$; protected processing costs four per unit and an audit failure incurs a separate fee of two. Work is continuously divisible. The original call and processing accounting is unchanged. Each variation changes only its displayed primitive; all competing policies within a row face the same inputs.

The dynamic policy solves the complete causal chance-cost problem and carries the conditional probability of its actual successful leaves into each observed successor. This is an exact finite-tree implementation of (V4), with a measurable current action at every shared history. The selected-run policy fixes the smallest maximum-run threshold whose probability meets the target, then optimizes physical service subject to those guaranteed leaves. The all-support comparator is the exact hard-deadline causal optimum, which is a stronger benchmark than an arbitrary feasible calendar. Full foresight relaxes only information. All policies retain physical constraints on unsuccessful paths and the same fee convention.

For accreditation, the static comparator is specified precisely: it sets inherited debt to zero while preserving the event law, resources and remaining audit. Its error is therefore nonnegative here by Theorem 12's debt order. It is not a universal statement about every policy that might be called static. The pathwise comparator permits separate service on each leaf but keeps the same probability target. Table 33 reports selected variations; Appendix L declares the complete sixteen-case design.

Table 33. Accreditation from the actual state and from a debt-free static state.
''')
names=['base','normal_now','independent','persistent','low_reserve','high_reserve','low_debt','high_debt','urgent_mix','rate_limited','slack_one','slack_three']
rows=[]
for nm in names:
    v=next(v for v in dyn if v['name']==nm);rows.append([nm.replace('_',' '),num(v['frontier']['value']),num(v['static']['value']),num(v['static']['value']-v['frontier']['value']),num(v['reserve']['value'])])
parts.append(table(['Variation','Causal commitment','Static commitment','Static excess','Reserve for r = 0.4'],rows))
parts.append(r'''
In the base state, the reliable commitment is 0.5, whereas the debt-free calculation reports 0.6. Increasing younger debt to 0.7 lowers the state-aware frontier to 0.4 while leaving the static answer at 0.6. A one-slot recovery window cannot support the fixed evaluation commitment 0.4 with the declared reserve; its frontier is 0.2. Costs for that fixed service are therefore reported as infeasible rather than compared with a smaller promise. Pathwise and causal accreditation coincide in the unrestricted rows, as EDF's feasibility theorem predicts. Their required-reserve difference is zero. The rate-limited row also has zero information premium in this particular design; the strictly positive heterogeneous premium remains established by Proposition 5 and Theorem 13a's sharp construction.

Table 34. Expected operating cost at the same commitment and resources.
''')
names2=['base','normal_now','low_frequency','independent','persistent','low_reserve','low_reliability','slack_three']
rows=[]
for nm in names2:
    v=next(v for v in dyn if v['name']==nm);rows.append([nm.replace('_',' '),num(v['foresight'].get('value')),num(v['dynamic'].get('value')),num(v['selected'].get('value')),num(v['hard_optimum'].get('value'))])
parts.append(table(['Variation','Foresight','Dynamic causal','Selected run','All support'],rows))
parts.append(r'''
The dynamic rule does not improve every row. In the base episode it chooses all-support success and costs 3.394890. Under independent grid states it costs 2.236250, compared with 2.313750 for the hard-deadline optimum at the identical physical service level. With protected rate 0.35, the chance-certified cost is 3.726836 while all-support delivery is infeasible. This comparison concerns a different accepted reliability criterion, not a computational saving at the hard criterion. At the 0.8 target, the optimized actual success probability is 0.833625. Its next-state requirements are one following a normal observation and 0.6975 following stress; their weighted average is $0.45+0.55(0.6975)=0.833625$. Requiring 0.8 again on both successors would discard that feasible allocation of the original allowance.

Theorem 15b gives a direct operating intervention. In the base episode its tail quantile is 0.4 and the early-service floor is 0.1. With protected rate 0.35, the quantile falls to 0.25 and the floor rises to 0.25, exactly exhausting the current capacity remaining after urgent service. Raising protected rate to 0.5 removes the floor for younger debt 0.5. These are feasibility effects; the economic optimum may process more. Figure 22 separates the debt effect from the reliability breakpoint of the final-slot specialization.

![Figure 22. Minimum early service. Panel A uses the base repeated-call episode and varies younger debt; dotted segments exceed current capacity after urgent service and are infeasible. Panel B uses the explicitly separate one-step example in (V9), with younger debt 0.8, ordinary headroom 1, reserve 0.2 and stress probability 0.3.](figures/Figure22_Early_Service.png){width=6.3in}

The new computations use continuous allocation variables and binary terminal success indicators. Every feasible reported optimization terminates with status optimal; infeasible entries are identified by the solver. Thirty independently enumerated interval-frontier calculations, covering actual and debt-free states in the fifteen unrestricted designs, agree with their causal MILPs within $10^{-7}$. Actual completion probabilities are evaluated separately from the binary flags. All feasible new cost models have matching optimizer lower and upper values to the reported precision. These numerical checks support the finite formulations; the causal guarantee and service-floor identity follow from their proofs.

## 10.21 A stronger certificate when the group contains the bottleneck

The second matched experiment targets resource competition that smaller groups cannot represent. Four jobs are available at the start of a three-slot block. Each requires 0.6 work, has ceiling 0.4, and is due at the block's end. Ordinary capacity is 1.2 in a normal slot and zero in stress; protected capacity is 0.8 in every slot. Local prices are $(2.5,1,0.5)$ and the protected price is four. The grid has the same two-state Markov law, initially stationary. Every policy must complete every job on every supported path. There is no failure fee or random success-set selection in this all-support study.

The singleton certificate optimizes the Lagrangian prices over time and current grid state. Equivalently, its LP keeps each job's complete causal decisions while replacing global pathwise capacities by probability-weighted capacity constraints for each time/state pair. Group programs additionally retain internal pathwise capacities. The compared partitions contain four singletons, two pairs, a triple and singleton, or all four jobs. The pair and triple partitions are different branches of the grouping hierarchy; neither is a coarsening of the other. Both are coarser than singletons and finer than the complete group.

The feasible calendar uses the observed current capacity and protected-only future capacity, reoptimizing after each observation. All job ceilings and deadlines remain in its tail. Equal-price ties favor later protected processing by a declared $10^{-8}$ tie-breaking perturbation; reported costs exclude that perturbation. The proposed component policy instead solves the exact three-slot causal program. Complete enumeration evaluates both operating policies under identical inputs.

Table 35. Per-block causal lower bounds and executable policy costs.
''')
rows=[]
for v in grp[:9]:
    a=v['inputs'];rows.append([f"{a['p']:.2f}, {a['rho']:.2f}",f"{v['single']['value']:.6f}",f"{v['pairs']['value']:.6f}",f"{v['triple']['value']:.6f}",f"{v['full']['value']:.6f}",f"{v['calendar']['value']:.6f}"])
parts.append(table(['p, persistence','Singleton','Pairs','Triple plus one','Exact component','Calendar'],rows))
parts.append(r'''
At $(p,\rho)=(0.25,0.4)$, the singleton bound is 4.521429 and the exact causal cost is 4.792500. Pairing adds no restriction: two individual ceilings sum to the protected capacity, so a pair cannot internally exceed that bottleneck. The complete group retains the four jobs' competition and closes the certificate. At independent stress with $p=0.25$, a triple already raises the lower bound from 3.825000 to 4.075000, but a residual gap remains until the complete conflict is represented. The result supports grouping around a binding constraint, not an unconditional claim that any larger group helps.

Twenty-five consecutive blocks give 100 jobs over 75 slots. Their windows do not overlap across blocks, they carry no shared energy state, and the Markov law is action independent. Each block begins with the stationary marginal distribution even though successive blocks are correlated. Theorem 18b therefore makes the sum of conditional block optima exact; independence of blocks is unnecessary. At the base parameters, the singleton lower bound is 113.035714, the exact component policy costs 119.812500, and the calendar costs 146.653125. The policy saving is 26.840625, or 18.3021% of calendar cost. Replacing the singleton bound by the exact component bound alone reduces the interval for that same calendar by 6.776786. Replacing the calendar by the component policy closes both sides to solver tolerance. These are separate improvements, as Figure 23 makes visible.

![Figure 23. Bounds and costs for 100 jobs in 25 separated conflict components. The singleton and pair quantities are lower bounds; the component and calendar quantities are feasible policy costs. The component value is also an attained lower bound by Theorem 18b.](figures/Figure23_Component_Certificates.png){width=6.3in}

The long-horizon conclusion is analytic composition of solved short components, not an attempted enumeration of $2^{75}$ histories. Fourteen component configurations also vary individual ceilings, urgent workload, protected rate and ordinary capacity. Each LP has at most 112 service variables in its causal form; primal and dual costs differ by less than $3\times10^{-15}$ in these calculations. Appendix L gives the complete variations and replay definitions. This exact separated-component class does not replace or tighten the reported 36.5320 width for the original, differently overlapping 100-job catalog. That unresolved width remains an explicit limitation of the general policy certificate.

## 10.22 Production-trace execution scales and a repeated-call counterfactual

The Alibaba GPU v2026 release provides production observations suitable for grounding workload scales (S. Li et al. 2026; Alibaba 2026). We read the public execution-summary table, which contains GPU requests and observed execution durations. It does not contain arrival timestamps or a complete concurrency calendar. Consequently this study calibrates execution sizes and rate proxies only. Its release timing, interruption permissions, deadlines and grid calls are constructed counterfactual inputs. It is not a replay of observed production service under grid interruptions.

Before inspecting values, we select Parquet row groups 0, 172 and 345: the first, middle and last groups in a 346-group file. They contain 250,571 execution spans, all with finite positive GPU requests and durations. The entire table contains 40,522,321 rows. This deterministic coverage sample is reproducible but is not a probability sample; its proportions must not be interpreted as population estimates. Table 36 reports descriptive quantities of these selected rows only. Requested GPU-hours are the product of GPU request and duration, not measured marginal electricity use or validated computational work.

Table 36. Observed execution summaries in the selected production-trace rows.
''')
rows=[]
for k,g in meta['categories'].items():rows.append([k.replace('_',' '),f"{g['count']:,}",f"{g['duration_median']:.6f}",f"{g['gpu_request_median']:.3f}",f"{g['gpu_hours_sum']:.3f}"])
parts.append(table(['Public workload class','Spans','Median hours','Median requested GPUs','Requested GPU-hours'],rows))
parts.append(r'''
For a tractable heterogeneous certificate, we fix six templates without examining scheduling outcomes: among positive spans of at most two hours, take the three lowest SHA-256 pod-identifier hashes in training and the three lowest in offline inference. The restriction defines a short-workload illustration; long training runs and online inference are not represented. No manual labels are added. A production category or low-priority tag does not itself establish legal or technical interruption permission. All six templates are assumed preemptible and eligible for both channels solely for this counterfactual.

Let observed duration and requested GPUs be $d_j^{obs}$ and $g_j^{obs}$. One slot is one hour. With $W=\sum_jg_j^{obs}d_j^{obs}=3.069444444444444$ requested GPU-hours, a call creates template work $r\beta_j$ and the combined processing ceiling is $a_j$, where

$$\beta_j=\frac{g_j^{obs}d_j^{obs}}{W},\qquad a_j=\frac{g_j^{obs}}{W},\qquad D_j=\lceil d_j^{obs}\rceil+s.\qquad\text{(V12)}$$

The fractions satisfy $\sum_j\beta_j=1$. The ceiling is a modeled one-hour request-equivalent service proxy; it is not an estimated accelerator throughput curve. The deadline increment $s\in\{0,1,2\}$ is assumed permission beyond the rounded observed duration. Appendix L records every selected duration and request. Ordinary recovery capacity is one in a normal slot and zero in stress, while protected rate is $z\in\{0.1,0.25,0.4\}$. These are normalized processing declarations, not inferred MW limits.

The declared episode consists of $K\in\{1,2,3\}$ consecutive called origins, each creating the same template mix one slot later. That initial stress prefix is part of the specified experiment. Afterward, the grid follows (5), starting from stress, with $p=0.25$ and $\rho=0.4$ through every deadline. No additional cohorts are admitted during clearing. Thus the comparison isolates repeated recovery obligations with a stochastic tail; it does not estimate the empirical frequency of these episodes. A common causal tree, combined job ceilings and a 0.9 success target determine the maximum commitment in $[0,1]$. A second optimization holds commitment at 0.5 and minimizes protected rate.

Table 37. Trace-informed commitment and reserve at declared protected rate 0.25.
''')
rows=[]
for v in tr:
    if v['z_declared']==.25:rows.append([v['slack'],v['K'],f"{v['r']:.6f}",f"{v['reserve_at_half']['z']:.6f}"])
parts.append(table(['Extra deadline hours','Called origins','Reliable commitment','Reserve for r = 0.5'],rows))
parts.append(r'''
With one additional hour of deadline permission and protected rate 0.25, the isolated-call commitment is 0.662896. Two successive calls reduce it to 0.442000, and three reduce it to 0.370805. The corresponding minimum reserves for the identical fixed commitment 0.5 are 0.174208, 0.282805 and 0.337104. Extra deadline permission improves these values, but it does not make successive obligations equivalent to one isolated interruption. Figure 24 shows this distinction across all three deadline assumptions.

![Figure 24. Reliable commitment for the six observed execution templates under constructed repeated calls. Protected rate is 0.25 and required success is 0.9. The horizontal dimension changes the declared number of successive calls; the deadline curves change assumed permission, not measured production deadlines.](figures/Figure24_Trace_Repeatability.png){width=6.3in}

All 27 frontier and 27 fixed-commitment reserve programs terminate optimally. Maximum reported primal constraint violation is below $5\times10^{-14}$ for the frontier programs; realized completion mass is checked independently against the target. The source sample, deterministic template selection and full-precision outputs are recorded in the numerical supplement. The study supplies an execution-scale illustration and a transparent permission sensitivity. It does not establish production-calibrated arrival processes, marginal power, checkpoint losses, or field reliability. Those measurements are required before using these frontiers as commercial accreditation.

ZZZPART_APPENDIXZZZ

# Appendix L Additional structural studies and production-trace provenance

## L.1 Shared inputs and verification of the dynamic study

The sixteen configurations in Section 10.20 use the base parameters given there and the following single changes: initially normal grid; event share 0.15 or 0.4; persistence zero or 0.75; protected rate 0.35 or 0.5; success target 0.97 or 0.8; younger debt 0.2 or 0.7; urgent debt 0.3 and younger debt 0.3 with their sum fixed; a combined ceiling 0.3 for every job; and recovery window one or three. In the three-slot window, the sixth local price is 1.3. Existing debt retains its own deadlines one and two when the new-cohort window changes. The finite nonbinding ceiling in unrestricted numerical rows is ten, above every possible relevant slot allocation.

For every observed node and active job, the solver retains local and protected service variables, a combined ceiling, channel capacities and cumulative useful-service inequalities. A completion inequality is imposed at every required deadline on each selected leaf. Its relaxation constant is that job's maximum possible requirement on the commitment interval; it introduces no extra processing capacity. Binary terminal flags have unconditional probabilities. The objective includes all processing incurred on shared histories and the fee once per failed episode. Foresight duplicates the service variables by leaf and changes no physical input. Presolve is disabled in these MILPs, relative optimality tolerance is $10^{-9}$, and an independent interval calculation checks the unrestricted commitment frontiers. No floating-point optimizer certificate substitutes for the analytic proofs.

To reconstruct the exact early-service margin, enumerate every future grid path conditional on the observed initial state. For each interval in the remaining horizon, subtract the work of future jobs whose complete windows are confined to it from its total capacity. If any such cut is negative, assign the path margin minus infinity. Otherwise take the smallest prefix margin starting at the next slot. The largest nonnegative margin with tail mass at least the inherited target is (V7), with equality retained. This calculation uses no optimal scheduling actions from the MILP and therefore checks a different representation of the feasibility boundary.

## L.2 Conflict components and the resource-price relaxation

The nine main component configurations cross $p\in\{0.15,0.25,0.4\}$ with $\rho\in\{0,0.4,0.75\}$. Five additional configurations retain $(0.25,0.4)$ and respectively set all job ceilings to 0.35 or 0.5; change the first job to work 0.4 due at slot 2 and the fourth to work 0.8 due at slot 3; raise protected rate to one; or reduce ordinary capacity to one. Other jobs and prices retain the base values. The partitions are $\{1\},\{2\},\{3\},\{4\}$; $\{1,2\},\{3,4\}$; $\{1,2,3\},\{4\}$; and $\{1,2,3,4\}$.

For an explicit finite tree, replacing global node capacities by expected capacities conditional on time and current grid state is the primal counterpart of optimizing the time/state resource prices in (V10). Each job retains its node-specific causal actions and completion on every supported history. Internal group capacities are also node specific. Strong LP duality establishes the reported optimized lower certificates. The supplied verification compares primal and dual values and checks the valid partial orders between partitions. It does not incorrectly treat the pair and triple partitions as nested.

The 100-job experiment concatenates 25 identical three-slot windows. The exact policy observes the grid state at each new block, uses the corresponding conditional component policy and carries no work past the component deadline. The global exogenous chain continues without a reset. Stationarity makes each block's unconditional cost identical, while action independence and absence of cross-component constraints justify summing these costs. The 75-slot result is obtained by this proved decomposition; it is not presented as a measured 75-slot full-tree solve or as a new timing result for the earlier overlapping catalog.

## L.3 Trace selection and the six execution templates

The source is Alibaba Cluster Trace GPU v2026, execution-summary archive `asi_opensource_job_execution_summary.zip`, member `asi_opensource_job_execution_summary/part-000.parquet`, retrieved on 8 October 2026. The archive member is stored without ZIP compression and contains 1,188,294,827 bytes. The public download documentation, schema and accompanying OSDI paper are cited in the references. The declared row-group rule selects 0, 172 and 345 from 346 groups before inspecting values. The sample contains 250,571 rows. Its saved Parquet checksum and the hashes identifying the selected templates are in `trace_metadata.json` and `trace_catalog.json` in the numerical supplement; they support replay of the stated sample rather than a claim of random representativeness.

Table 38. Complete observed inputs for the six counterfactual templates.
''')
rows=[]
for i,j in enumerate(cat['jobs'],1):rows.append([i,j['class'].replace('_',' '),f"{j['duration_hours']:.12f}",f"{j['gpu_request']:.6f}",f"{j['work_coefficient']:.12f}",f"{j['rate']:.12f}"])
parts.append(table(['Template','Class','Duration hours','GPU request','Work fraction','Rate proxy'],rows))
parts.append(r'''
The selection rule uses SHA-256 of each existing anonymized pod identifier, sorts ascending within training and offline inference, and chooses the first three positive spans no longer than two hours in each class. The observed duration is not reclassified as a contractual deadline: rounding and extra slack in (V12) are explicit modeling choices. Likewise, the request-duration product is a request-equivalent execution proxy. All six templates are assumed resumable without extra checkpoint loss, with identical local and protected eligibility. Transfer delay, nonlinearity of compute throughput and marginal facility power are outside this illustration. The repeated initial stress prefix and the absence of new clearing admissions are declared experiment inputs, not inferred production behavior.

The numerical supplement contains the new study programs, their full-precision outputs, observed template catalog, source-access script, verification checks and figure-generation script. `studies.py`, `group_study.py` and `trace_study.py` generate the dynamic, component and trace experiments respectively. `verify_new.py` checks the first two studies; the trace program records physical residuals, actual success and solver gaps for every case. `trace_read.py` reproduces the sample from the public source. The original source workflows in Appendices G–K concern the retained earlier experiments and remain distinct from these additions.

The added studies were executed with Python VERSIONPY, NumPy VERSIONNP and SciPy VERSIONSCIPY using the bundled HiGHS optimizer. All allocations are continuous; only success indicators are integral. Reported optimality means optimality of the stated finite model at its numerical tolerances. No general heterogeneous cost-threshold theorem, arbitrary multistage sparse-root theorem, or production reliability estimate is inferred from these experiments.

ZZZPART_NEWREFSZZZ

Alibaba. 2026. “Alibaba Cluster Trace GPU v2026.” Dataset, execution-summary table, schema and download documentation. Accessed October 8, 2026. <https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026>.

Caprara, Adriano, Ying Yu, Fei Teng, Adrià Junyent-Ferré, Eduard Bullich-Massagué, and Mònica Aragüés-Peñalba. 2026. “Data Center Workload Flexibility for Power System Demand Response: Evidence from Alibaba Traces.” *International Journal of Electrical Power & Energy Systems* 178: 111940. <https://doi.org/10.1016/j.ijepes.2026.111940>.

Helly, Eduard. 1923. “Über Mengen konvexer Körper mit gemeinschaftlichen Punkten.” *Jahresbericht der Deutschen Mathematiker-Vereinigung* 32: 175–176.

Huynh, Vu Anh, Leonid Kogan, and Emilio Frazzoli. 2014. “A Martingale Approach and Time-Consistent Sampling-Based Algorithms for Risk Management in Stochastic Optimal Control.” *53rd IEEE Conference on Decision and Control*, 1858–1865. Preprint: <https://arxiv.org/abs/1312.7602>.

Li, Suyi, Lingyun Yang, Haoxuan Yu, Sheng Yao, Tianyuan Wu, Xiaoxiao Jiang, Hanfeng Lu, Kangjin Wang, Chenhao Wang, Shenglin Xu, Lun Wang, Qingyang Duan, Shenghao Liang, Xiu Lin, Wenchao Wu, Yinghao Yu, Guodong Yang, Liping Zhang, and Wei Wang. 2026. “Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba.” *20th USENIX Symposium on Operating Systems Design and Implementation*. <https://www.usenix.org/system/files/osdi26-li-suyi.pdf>.

Liu, Wenyu, Yuejun Yan, Peng Wang, Zhaoyang Wang, and Zhaohao Ding. 2024. “Flexibility Quantification for Energy-Aware Data Center Job Scheduling: A Chance Constrained Bi-Level Model.” *2024 IEEE/IAS Industrial and Commercial Power System Asia*. <https://doi.org/10.1109/ICPSAsia61913.2024.10761863>.

Schmid, Niklas, Marta Fochesato, Sarah H. Q. Li, Tobias Sutter, and John Lygeros. 2025. “Computing Optimal Joint Chance Constrained Control Policies.” *IEEE Transactions on Automatic Control* 70 (7): 4904–4911. <https://doi.org/10.1109/TAC.2025.3546078>.

Wang, Andrew J., and Brian C. Williams. 2015. “Chance-Constrained Scheduling via Conflict-Directed Risk Allocation.” *Proceedings of the AAAI Conference on Artificial Intelligence* 29 (1). <https://doi.org/10.1609/aaai.v29i1.9693>.

ZZZPART_ENDZZZ
''')
out='\n\n'.join(parts).replace('VERSIONPY',platform.python_version()).replace('VERSIONNP',np.__version__).replace('VERSIONSCIPY',scipy.__version__)
(W/'results_additions.md').write_text(out)
print('generated results prose and figures')
