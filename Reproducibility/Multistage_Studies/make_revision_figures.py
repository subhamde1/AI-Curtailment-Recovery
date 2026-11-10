"""Generate the two manuscript figures directly from the released study outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
rows = {r['label']: r for r in json.loads((ROOT/'matched_results.json').read_text())}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.spines.top':False,
                     'axes.spines.right':False, 'savefig.dpi':300})
labels = ['base','rate_scale_0.65','slack_1','z_0.05','urgency_0.7']
names = ['Base','Rate × 0.65','Slack + 1','Reserve 0.05','Urgent 0.70']
blue, orange, grey = '#245b79', '#b35b28', '#62736d'
fig, axes = plt.subplots(2, 1, figsize=(6.45,6.4), constrained_layout=True)
x=np.arange(len(labels)); width=.24
for ax in axes:
    ax.set_axisbelow(True); ax.grid(axis='y', color='#dedede', linewidth=.6)
for off,key,color,name in [(-width,'causal',blue,'Future-cost policy'),(0,'shield',orange,'Current-cost shield'),(width,'foresight',grey,'Perfect foresight')]:
    axes[0].bar(x+off,[rows[s][key]['value'] for s in labels], width, label=name, color=color)
axes[0].set_ylabel('Expected audit cost')
axes[0].set_xticks(x,names)
axes[0].set_ylim(0,3.65)
axes[0].legend(loc='upper left',ncols=1,frameon=False, fontsize=8.5)
axes[0].set_title('(a) Identical commitment and success target',loc='left',fontsize=10,pad=8)
for off,key,color,name in [(-.17,'frontier',blue,'Inherited debt retained'),(.17,'static_frontier',grey,'Inherited debt omitted')]:
    axes[1].bar(x+off,[rows[s][key]['commitment'] for s in labels], .34, label=name, color=color)
axes[1].set_ylabel('Accredited commitment')
axes[1].set_xticks(x,names)
axes[1].set_ylim(0,1.32)
axes[1].legend(loc='upper left',ncols=1,frameon=False, fontsize=8.5)
axes[1].set_title('(b) Same reserve and reliability criterion',loc='left',fontsize=10,pad=8)
fig.text(.04,-.038,'Cost is infeasible at event share 0.40 and at the hard target with local-only A.\nBoth commitment frontiers are zero; infeasibility is not plotted as zero cost.', fontsize=8.3)
fig.savefig(OUT/'multistage_decisions.png', bbox_inches='tight')
fig.savefig(OUT/'multistage_decisions.pdf', bbox_inches='tight')
plt.close(fig)

large=json.loads((ROOT/'overlap_100_results.json').read_text())
hist=large['cuts']['history']; it=np.arange(len(hist))
lower=[h['lower'] for h in hist]
opt=large['extensive']['value']
calendar=large['existing_calendar']
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.45,3.45),constrained_layout=True,gridspec_kw={'width_ratios':[1,1.25]})
ax1.plot(it,lower,'o-',color=blue,label='Valid lower value')
ax1.axhline(calendar,color=orange,linestyle='--',label='Calendar upper value')
ax1.axhline(opt,color=grey,linestyle=':',label='Full-tree optimum')
ax1.set_ylim(16.0,18.55);ax1.set_ylabel('Expected cost');ax1.set_xlabel('Restoration solve')
ax1.set_xticks(it,it+1);ax1.set_title('(a) Full cost interval',loc='left',fontsize=10)
ax1.legend(frameon=False,fontsize=7.7,loc='upper center')
ax2.plot(it,np.maximum(0,np.array(opt)-np.array(lower)),'o-',color=blue)
for i,h in enumerate(hist):
    ax2.annotate(str(h['retained']), (i,max(0,opt-h['lower'])), xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
ax2.set_ylim(-.002,.038);ax2.set_ylabel('Optimum minus lower value');ax2.set_xlabel('Restoration solve')
ax2.set_xticks(it,it+1);ax2.set_title('(b) Labels show retained rows',loc='left',fontsize=10)
for ax in (ax1,ax2): ax.set_axisbelow(True); ax.grid(axis='y',color='#dedede',linewidth=.6)
fig.text(.04,-.062,'Connected original 100-job catalogue; declared two-observation law.\nFour solves retain 13 of 274 channel rows. Direct full-tree optimization is faster.',fontsize=8.3)
fig.savefig(OUT/'conflict_certificate.png',bbox_inches='tight')
fig.savefig(OUT/'conflict_certificate.pdf',bbox_inches='tight')
plt.close(fig)
print(json.dumps({'figures':[str(x) for x in OUT.glob('*.png')]}))
