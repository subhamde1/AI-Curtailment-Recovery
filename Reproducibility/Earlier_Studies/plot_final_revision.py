"""Scientific plots and a precise managerial schematic from recorded inputs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Polygon
from matplotlib.path import Path as PlotPath
import numpy as np
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'final_data';FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def read(name):return json.loads((DATA/(name+'.json')).read_text())
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=250,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)

def joint():
    rows=read('joint_costs');fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),constrained_layout=True)
    for ax,phi in zip(axes,(0.,4.)):
        R=[.25,.5,.75,1.];Z=[0.,.15,.3,.45,.6,.9]
        C=np.full((len(R),len(Z)),np.nan)
        for i,r in enumerate(R):
            for j,z in enumerate(Z):
                q=next(x for x in rows if x['commitment']==r and x['z']==z and x['epsilon']==.1 and x['failure_fee']==phi)
                if q['feasible']:C[i,j]=q['policy_cost']
        im=ax.imshow(C,origin='lower',aspect='auto',cmap='viridis')
        ax.set_xticks(range(len(Z)),[str(x) for x in Z]);ax.set_yticks(range(len(R)),[str(x) for x in R])
        ax.set_xlabel('Protected rate z');ax.set_ylabel('Commitment r')
        ax.set_title('Processing only' if phi==0 else 'Processing + episode fee 4')
        for i in range(len(R)):
            for j in range(len(Z)):
                ax.text(j,i,'ineligible' if np.isnan(C[i,j]) else f'{C[i,j]:.2f}',ha='center',va='center',
                    color='black' if np.isnan(C[i,j]) or C[i,j]>np.nanmedian(C) else 'white',fontsize=8)
        fig.colorbar(im,ax=ax,label='Exact expected cost',shrink=.85)
    save(fig,'final_joint_cost')

def reserve():
    rows=read('reserve_boundary');fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for tail,color in ((0.,'#144d6f'),(2.,'#ca6137')):
        q=[x for x in rows if x['B_rate']==.5 and x['root_extra']==0 and x['normal_tail_extra']==tail]
        axes[0].plot([x['urgent_deadline'] for x in q],[x['reserve'] for x in q],
            'o-' if tail==0 else 'x--',label=f'Extra normal tail {tail:g}',color=color)
    axes[0].set(xlabel='Urgent A deadline (slot)',ylabel='Minimum causal protected rate',xticks=[2,3,4]);axes[0].legend(fontsize=8)
    for root,color in ((0.,'#144d6f'),(.1,'#ca6137')):
        q=[x for x in rows if x['urgent_deadline']==2 and x['root_extra']==root and x['normal_tail_extra']==0]
        axes[1].plot([x['B_rate'] for x in q],[x['reserve'] for x in q],'o-',label=f'Guaranteed root extra {root:g}',color=color)
    axes[1].set(xlabel='B combined job-rate ceiling',ylabel='Minimum causal protected rate');axes[1].legend(fontsize=8)
    save(fig,'final_reserve_condition')

def lookahead():
    rows=read('lookahead_depth');fig,axes=plt.subplots(1,3,figsize=(11.5,3.7),constrained_layout=True)
    for n,color in ((20,'#144d6f'),(50,'#ca6137'),(100,'#3a8861')):
        q=[x for x in rows if x['base_jobs']==n and x['complete']];K=[x['K'] for x in q]
        axes[0].plot(K,[x['deterministic_width'] for x in q],'o-',color=color,label=f'{n} jobs')
        axes[1].plot(K,[x['high_confidence_width'] for x in q],'o-',color=color,label=f'{n} jobs')
        axes[2].plot(K,[x['wall_p95_ms'] for x in q],'o-',color=color,label=f'{n} jobs')
    axes[2].axhline(50,color='.3',ls='--',lw=1,label='Observed p95 budget')
    for ax in axes:ax.set_xlabel('Lookahead K');ax.set_xticks(range(4));ax.legend(fontsize=8)
    axes[0].set_ylabel('Deterministic upper − lower');axes[1].set_ylabel('95% simultaneous upper − lower');axes[2].set_ylabel('End-to-end online p95 (ms)')
    save(fig,'final_lookahead_depth')

def ai():
    rows=read('ai_workload');labels=['Urgent mix','Urgent mix patient training','Urgent mix inference slack']
    q=[next(x for x in rows if x['label']==s) for s in labels];fig,axes=plt.subplots(1,2,figsize=(9.8,3.7),constrained_layout=True)
    tick=['Urgent mix','Training slack','Inference slack']
    axes[0].bar(tick,[x['minimum_reserve'] for x in q],color=['#144d6f','#144d6f','#ca6137'])
    axes[0].set_ylabel('Minimum z for fixed r = 0.6');axes[0].set_ylim(0,.44)
    for i,x in enumerate(q):axes[0].text(i,x['minimum_reserve']+.008,f'{x["minimum_reserve"]:.4f}',ha='center')
    axes[1].bar(tick,[x['total_cost'] for x in q],color=['#144d6f','#144d6f','#ca6137']);axes[1].set_ylabel('Exact expected cost at r = 0.6');axes[1].set_ylim(0,2.6)
    for i,x in enumerate(q):axes[1].text(i,x['total_cost']+.04,f'{x["total_cost"]:.4f}',ha='center')
    save(fig,'final_ai_interaction')

def manager():
    fig,ax=plt.subplots(figsize=(7.2,6.45));ax.set(xlim=(-.85,10),ylim=(0,10.1));ax.axis('off')
    def box(x,y,w,h,title,body,fill='#edf4f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=.12',
            edgecolor='#23495b',facecolor=fill,lw=1.2))
        ax.text(x+w/2,y+h-.23,title,ha='center',va='top',fontsize=10.2,fontweight='bold',color='#163b4e')
        ax.text(x+w/2,y+h/2-.12,body,ha='center',va='center',fontsize=9,linespacing=1.45)
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.2,color='#23495b'))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,label,ha='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=1.3))
    box(1.05,8.5,7.9,1.35,'1  OBSERVE','Debt by deadline; rates and channel permissions\nPredictive grid state; charge; contractual obligation')
    arrow((5,8.42),(5,8.05))
    box(1.05,6.67,7.9,1.30,'2  CERTIFY','Test the obligation through the full clearing horizon\nRetain the original audit’s conditional success accounting')
    arrow((3.0,6.59),(2.65,5.98),'Feasible');arrow((7.0,6.59),(7.35,5.98),'Infeasible')
    box(.2,3.98,4.48,1.9,'3  CHOOSE','Reserve and recovery timing\nLookahead under compute budget\nSame resources and service criterion')
    box(5.3,3.98,4.48,1.9,'4  APPROVED RESPONSE','Guaranteed protection or fallback\nApproved deadline / contract revision\nOtherwise record failure and remedy',fill='#fff2e9')
    arrow((2.44,3.90),(4.0,3.45));arrow((7.54,3.90),(6.0,3.45))
    box(1.05,1.99,7.9,1.35,'5  EXECUTE AND UPDATE','Execute only the current causal allocation\nUpdate useful residual work, deadlines, observations and charge')
    route=PlotPath([(0.98,2.7),(-.22,2.7),(-.22,7.3),(.94,7.3)])
    ax.add_patch(FancyArrowPatch(path=route,arrowstyle='-|>',mutation_scale=13,lw=1.2,color='#23495b'))
    ax.text(-.56,5.0,'Re-certify',rotation=90,ha='center',fontsize=8.5,color='#23495b')
    ax.text(5,.84,'NEW ACCREDITATION: choose an eligible promise.\nEXISTING CONTRACT: keep the obligation fixed unless an approved route changes it.',
        ha='center',va='center',fontsize=9,fontweight='bold',linespacing=1.6)
    save(fig,'final_manager_sequence')

if __name__=='__main__':joint();reserve();lookahead();ai();manager()
