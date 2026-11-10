from pathlib import Path
from fractions import Fraction as F
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'figures');out=p.parse_args().output_dir
out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(9,4.1));cut=8/9
for s,c,label in [(F(8,5),'#17638b',r'$s=8/5$'),(F(7,4),'#bb660e',r'$s=7/4$'),(F(2),'#26785c',r'$s=2$')]:
 z=float((2*s-3)/3);axs[0].plot([cut,1],[z,z],c=c,label=label,lw=2)
 axs[0].plot([cut],[z],marker='o',mfc='white',mec=c,ms=6,zorder=4)
 k=np.arange(1,10);axs[1].plot(k,np.where(k<4,0,z),marker='o',ms=4,color=c,lw=1.7)
axs[0].plot([.7,cut],[0,0],c='#444444',lw=2);axs[0].plot([cut],[0],marker='o',c='#444444',ms=6,zorder=5)
axs[0].axvline(cut,c='#999999',lw=.8,ls=':');axs[0].set(xlim=(.7,1.005),ylim=(-.012,.36),xlabel='Inherited success requirement $b$',ylabel='Minimum protected reserve',title='(a) Reliability changes the reserve decision')
axs[0].set_xticks([.7,.8,cut,1]);axs[0].set_xticklabels(['0.7','0.8',r'$8/9$','1']);axs[0].legend(frameon=False,loc='upper left')
axs[1].axvline(4,c='#999999',lw=.8,ls=':');axs[1].set(xlim=(.7,9.3),ylim=(-.012,.36),xticks=range(1,10),xlabel='Maximum retained leaves $k$',ylabel='Largest subset minimum reserve',title='(b) Four leaves are necessary and sufficient')
for ax in axs:ax.grid(axis='y',alpha=.22)
fig.tight_layout(w_pad=2);fig.savefig(out/'Figure30_Witness_Reliability.pdf',bbox_inches='tight');fig.savefig(out/'Figure30_Witness_Reliability.png',dpi=600,bbox_inches='tight')
