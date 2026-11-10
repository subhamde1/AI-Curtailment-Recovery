"""Render Figure 31 from the supplied full-precision holdout results."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY_FIGURES = HERE / "figures"

def render():
    data=pd.read_csv(HERE/'holdout_results.csv')
    base=data[data.design=='base']
    days=sorted(base.day.unique())
    fig,axes=plt.subplots(len(days),1,figsize=(8.4,1.85*len(days)+.6),sharex=True,sharey=True,squeeze=False)
    for day,ax in zip(days,axes[:,0]):
        for k,color,mark in [(1,'#115f88','o'),(2,'#26785c','s'),(3,'#b85c20','^')]:
            part=base[(base.day==day)&(base.calls==k)].sort_values('hour')
            ax.plot(part.hour,part.frontier,label=f'{k} call'+('s' if k>1 else ''),color=color,marker=mark,
                    markersize=3,linewidth=1.15)
        ax.set_title(f'Public trace day {day}',loc='left',fontsize=10)
        ax.set_ylabel('Accredited relief',fontsize=9)
        ax.grid(axis='y',alpha=.24)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(loc='upper right',ncol=3,frameon=False,fontsize=9)
    axes[-1,0].set_xlabel('First called hour within the held-out day',fontsize=9)
    axes[-1,0].set_xticks(list(range(0,22,3)))
    fig.tight_layout(pad=.8)
    STUDY_FIGURES.mkdir(exist_ok=True)
    fig.savefig(STUDY_FIGURES/'additional_holdout.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)



if __name__ == "__main__":
    render()
