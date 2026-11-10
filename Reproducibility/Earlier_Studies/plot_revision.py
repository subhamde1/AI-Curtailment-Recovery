"""Publication figures from the complete revision records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;DATA=ROOT/"revision_data";FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300})
COLORS=["#173F5F","#A34A28","#367C68","#75628F"]
def read(name):return json.loads((DATA/(name+".json")).read_text())
def finish(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+".png"));plt.close(fig)

phase=read("reserve_phase");summary=read("reserve_phase_summary")
fig,axes=plt.subplots(1,2,figsize=(7.1,3.3))
grid=np.zeros((3,4))
for ai,d in enumerate((2,3,4)):
    for bi,b in enumerate((.25,.5,.75,1.)):
        grid[ai,bi]=np.mean([max(0,q["premium"]) for q in phase if q["reveal_slot"]==3 and q["a_deadline"]==d and q["b_rate"]==b])
im=axes[0].imshow(grid,origin="lower",cmap="Blues",aspect="auto")
axes[0].set(xticks=range(4),xticklabels=["0.25","0.50","0.75","1.00"],
            yticks=range(3),yticklabels=["2","3","4"],xlabel="B processing ceiling",ylabel="A deadline slot")
for i in range(3):
    for j in range(4):axes[0].text(j,i,f"{grid[i,j]:.3f}",ha="center",va="center",color="white" if grid[i,j]>.02 else "black",fontsize=9)
fig.colorbar(im,ax=axes[0],label="Mean added reserve per slot",shrink=.85)
axes[1].bar([q["reveal_slot"] for q in summary],[100*q["causal_conflicts"]/q["zero_leaf_cases"] for q in summary],color=COLORS[0])
axes[1].set(xlabel="Branch revealed at slot",ylabel="Conflict among zero-leaf-reserve cases (%)",
            xticks=(1,2,3),ylim=(0,18))
axes[1].grid(axis="y",alpha=.15);finish(fig,"revision_reserve_phase")

surface=read("frontier_surface")
fig,axes=plt.subplots(1,2,figsize=(7.1,3.4),sharey=True)
for ax,state in zip(axes,(0,1)):
    for color,z in zip(COLORS,(0.,.1,.25,.5)):
        qs=[q for q in surface if q["T"]==10 and q["rho"]==.6 and q["initial_state"]==state and q["z"]==z]
        ax.plot([q["D"] for q in qs],[q["frontier"] for q in qs],"o-",color=color,label=f"z = {z:g}",ms=4)
    selected=[q for q in surface if q["T"]==10 and q["rho"]==.6 and q["initial_state"]==state and q["z"]==.25]
    ax.plot([q["D"] for q in selected],[q["guarded_frontier"] for q in selected],"s--",color=COLORS[2],label="z = 0.25, guarded",ms=3)
    ax.set(xlabel="Deadline window D (slots)",xticks=range(1,7),ylim=(-.04,2.04),
           title="Observed normal" if state==0 else "Observed stress")
    ax.grid(alpha=.15)
axes[0].set_ylabel("Exact reliable commitment r");axes[1].legend(fontsize=8,loc="upper left")
finish(fig,"revision_frontier_surface")

duration=read("duration")
fig,axes=plt.subplots(1,2,figsize=(7.1,3.3))
for color,law,label in zip(COLORS,("geometric","narrow","heavy"),("Geometric","Narrow durations","Heavy-tail durations")):
    qs=[q for q in duration if q["T"]==10 and q["initial_state"]==1 and q["z"]==.25 and q["law"]==law]
    axes[0].plot([q["D"] for q in qs],[q["frontier"] for q in qs],"o-",color=color,label=label)
    axes[1].plot([q["D"] for q in qs],[q["blackout_without_protection"] for q in qs],"o-",color=color)
axes[0].set(xlabel="Deadline D (slots)",ylabel="Exact commitment with z = 0.25",xticks=(3,6),ylim=(0,2.1))
axes[1].set(xlabel="Deadline D (slots)",ylabel="Ordinary-only blackout probability",xticks=(3,6),ylim=(0,1))
axes[0].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.15)
finish(fig,"revision_duration")

bounds=read("small_bounds")
fig,axes=plt.subplots(1,3,figsize=(8.0,3.1))
for ax,rho in zip(axes,(0.,.6,.85)):
    qs=[q for q in bounds if q["rho"]==rho];K=[q["K"] for q in qs]
    ax.plot(K,[q["upper"] for q in qs],"o-",color=COLORS[0],label="Calendar upper")
    ax.plot(K,[q["lower"] for q in qs],"s-",color=COLORS[1],label="Restored lower")
    ax.axhline(qs[0]["exact"],color=COLORS[2],ls="--",label="Exact causal")
    ax.fill_between(K,[q["lower"] for q in qs],[q["upper"] for q in qs],color=COLORS[0],alpha=.09)
    ax.set(xlabel="Lookahead K",title=f"Persistence {rho:g}",xticks=range(6));ax.grid(alpha=.15)
axes[0].set_ylabel("Expected cost");axes[2].legend(fontsize=7)
finish(fig,"revision_bound_hierarchy")

large=read("larger_bounds")
fig,axes=plt.subplots(1,2,figsize=(7.1,3.3))
for ax,K in zip(axes,(0,1)):
    qs=[q for q in large if q["K"]==K];x=np.arange(len(qs))
    for i,q in enumerate(qs):
        ax.vlines(i,q["lower"],q["U"],color=COLORS[0],lw=4,alpha=.25)
        ax.vlines(i+.12,q["lower"],q["upper_95_simultaneous"],color=COLORS[1],lw=4)
    ax.plot(x,[q["sample_mean"] for q in qs],"o",color=COLORS[2],label="Policy sample mean")
    ax.set(xticks=x,xticklabels=[str(q["base_jobs"]) for q in qs],
           xlabel="Base jobs",title=f"Lookahead K = {K}")
    ax.grid(axis="y",alpha=.15)
axes[0].set_ylabel("Expected-cost certificate")
axes[1].plot([],[],color=COLORS[0],lw=4,alpha=.25,label="Deterministic interval")
axes[1].plot([],[],color=COLORS[1],lw=4,label="95% simultaneous upper")
axes[1].legend(fontsize=8)
finish(fig,"revision_larger_bounds")

scale=read("scaling")
fig,axes=plt.subplots(1,2,figsize=(7.1,3.3))
for seed,color in zip((20261011,20261012,20261013),COLORS):
    qs=[q for q in scale if q["seed"]==seed]
    axes[0].plot([q["jobs"] for q in qs],[q["median_solve_ms"] for q in qs],"o-",color=color,label=str(seed))
    axes[1].plot([q["jobs"] for q in qs],[q["p95_solve_ms"] for q in qs],"s-",color=color)
axes[0].set(xlabel="Base jobs",ylabel="Median LP solve time (ms)",xticks=(50,200,500));axes[0].legend(fontsize=8)
axes[1].set(xlabel="Base jobs",ylabel="95th percentile LP time (ms)",xticks=(50,200,500))
for ax in axes:ax.grid(alpha=.15)
finish(fig,"revision_scaling")

physical=read("physical");battery=read("battery_runs")
fig,axes=plt.subplots(1,2,figsize=(7.1,3.3))
for pue,color in zip((1.1,1.2,1.4),COLORS):
    qs=[q for q in physical if q["marginal_PUE"]==pue]
    axes[0].plot([q["IT_productivity"] for q in qs],[q["robust_relief_MW"] for q in qs],"o-",color=color,label=f"PUE {pue:g}")
axes[0].set(xlabel="IT productivity (accelerator-hours/MWh)",ylabel="Robust promised relief (MW)",xticks=(800,1200,1600));axes[0].legend(fontsize=8)
for Q,color in zip((0.,.5,1.,2.),COLORS):
    qs=[q for q in battery if q["initial_charge"]==Q and q["efficiency"]==1 and q["discharge_rate"]==.5]
    axes[1].plot([q["origins"] for q in qs],[q["frontier"] for q in qs],"o-",color=color,label=f"Initial energy {Q:g}")
axes[1].axhline(.2,color="#666666",ls="--",lw=1)
axes[1].set(xlabel="Successive called origins",ylabel="Finite-run commitment (work units)",xscale="log",xticks=(1,2,4,8,16,32))
axes[1].set_xticklabels(["1","2","4","8","16","32"]);axes[1].legend(fontsize=7)
for ax in axes:ax.grid(alpha=.15)
finish(fig,"revision_physical_battery")
print({"new_figures":7})
