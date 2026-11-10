from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;D=ROOT/'hourly_trace';F=D/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240,'axes.titleweight':'bold'})
BLUE='#235789';ORANGE='#D17A22';GREEN='#477A4C'

def finish(fig,n):
 fig.tight_layout(pad=1.7);fig.savefig(F/f'figure_{n}.png',bbox_inches='tight');fig.savefig(F/f'figure_{n}.pdf',bbox_inches='tight');plt.close(fig)

def run():
 h=pd.read_csv(D/'hourly_metrics.csv');p=pd.read_csv(D/'panel.csv');sp=pd.read_csv(D/'observed_pods.csv');rr=json.loads((D/'results.json').read_text());m=json.loads((D/'calibration.json').read_text())
 fig,ax=plt.subplots(2,2,figsize=(9,6.7))
 for a in ax[0]:a.axvspan(0,23.5,color='#e5edf5');a.axvline(23.5,color='black',lw=.8);a.set_xlabel('Hour from day 109 hour 00')
 ax[0,0].plot(h.t.iloc[1:],h.first_appearances.iloc[1:],color=BLUE,lw=1.5);ax[0,0].set_ylabel('First appearances per hour');ax[0,0].set_title('A  Observed demand arrivals',loc='left')
 ax[0,1].plot(h.t,h.requested_GPUs,color=BLUE,label='Requested GPUs');ax[0,1].plot(h.t,h.attributed_GPU_hours,color=ORANGE,label='Occupation per hour');ax[0,1].plot(h.t,h.eligible_GPU_hours,color=GREEN,label='Eligible occupation');ax[0,1].set_ylabel('GPU requests / GPU-hours per hour');ax[0,1].set_title('B  Concurrency and occupation',loc='left');ax[0,1].legend(fontsize=8.8,frameon=False)
 for name,mask,color in [('Training',p.t<24,BLUE),('Holdout',p.t>=24,ORANGE)]:
  x=np.sort(p.loc[mask,'gpu_request']);ax[1,0].step(x,np.arange(1,len(x)+1)/len(x),color=color,label=name)
 ax[1,0].set_xscale('log');ax[1,0].set_xlabel('GPU request per pod-hour');ax[1,0].set_ylabel('Empirical cumulative fraction');ax[1,0].set_title('C  Resource-request distribution',loc='left');ax[1,0].legend(frameon=False)
 complete=sp[(~sp.left_censored)&(~sp.right_censored)];x=np.sort(complete.observed_span_hours);ax[1,1].step(x,np.arange(1,len(x)+1)/len(x),color=BLUE);ax[1,1].set_xlabel('Observed span in hours');ax[1,1].set_ylabel('Empirical cumulative fraction');ax[1,1].set_title('D  Spans inside the window',loc='left')
 finish(fig,25)
 boundary=json.loads((D/'debt_boundary_results.json').read_text());base=[q for q in rr if q['label']=='base'];q3=[q for q in base if q['K']==3];starts=sorted({q['start'] for q in base})
 fig,ax=plt.subplots(2,2,figsize=(9,6.7))
 for K,color,mark in [(1,BLUE,'o'),(2,ORANGE,'s'),(3,GREEN,'^')]:
  rows=sorted([q for q in base if q['K']==K],key=lambda q:q['start'])
  ax[0,0].plot(starts,[q['frontier']['r'] for q in rows],color=color,marker=mark,label=f'{K} call'+('s' if K!=1 else ''))
  ax[0,1].plot(starts,[q['reserve_at_quarter']['z'] for q in rows],color=color,marker=mark)
 ax[0,0].set_ylabel('Accredited relief r');ax[0,0].set_title('A  Single versus repeated calls',loc='left');ax[0,0].legend(frameon=False)
 ax[0,1].set_ylabel('Required protected rate');ax[0,1].set_title('B  Reserve at relief 0.25',loc='left')
 ax[1,0].plot(starts,[q['debt_free_frontier']['r'] for q in q3],color=ORANGE,marker='s',label='Debt-free');ax[1,0].plot(starts,[q['current_frontier']['r'] for q in q3],color=BLUE,marker='o',label='After existing relief 0.20');ax[1,0].plot(starts,[q['current_frontier']['r'] for q in boundary],color=GREEN,marker='^',label='After existing relief 0.30');ax[1,0].set_ylabel('Third-call relief');ax[1,0].set_title('C  Accreditation after execution',loc='left');ax[1,0].legend(fontsize=8.8,frameon=False)
 ax[1,1].bar(starts,[q['calendar']['value']-q['dynamic']['actual_penalized_cost'] for q in q3],width=1.9,color=BLUE);ax[1,1].set_ylabel('Saving in declared cost units');ax[1,1].set_title('D  Dynamic versus fixed calendar',loc='left')
 for a in ax.ravel():a.set_xlabel('Held-out episode origin');a.set_xticks(starts);a.grid(axis='y',alpha=.18)
 finish(fig,26)
 labels=[('base','Base'),('slack_zero','No extra slack'),('slack_two','Two extra hours'),('reserve_low','Protected 0.10'),('reserve_high','Protected 0.40'),('overhead_20','Training overhead 20%'),('half_rate','Half lineage ceilings'),('training_local','Training local only'),('reliability_95','Reliability 0.95'),('persistent','Persistence 0.75'),('frequent','Stress share 0.40'),('all_training_batch','Allow HP training / batch')]
 fig,ax=plt.subplots(1,2,figsize=(9,5.6),sharey=True);y=np.arange(len(labels));front=[];reserve=[]
 for l,name in labels:
  qs=[q for q in rr if q['label']==l and q['K']==3];front.append(np.mean([q['frontier']['r'] for q in qs]));zs=[q['reserve_at_quarter'].get('z') for q in qs];reserve.append(np.mean([v for v in zs if v is not None]) if any(v is not None for v in zs) else np.nan)
 ax[0].barh(y,front,color=BLUE);ax[1].barh(y,reserve,color=ORANGE);ax[0].set_yticks(y,[l[1] for l in labels]);ax[0].invert_yaxis();ax[0].set_xlabel('Accredited relief');ax[1].set_xlabel('Reserve at relief 0.25');ax[0].set_title('A  Relief at declared resources',loc='left');ax[1].set_title('B  Required reserve',loc='left')
 for k,v in enumerate(reserve):
  if np.isnan(v):ax[1].text(.01,k,'Infeasible',va='center',fontsize=9)
 for a in ax:a.grid(axis='x',alpha=.18)
 finish(fig,27)
 print('Generated figures 25–27')
if __name__=='__main__':run()
