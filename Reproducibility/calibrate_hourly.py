"""Calibrate temporal marks from a complete consecutive hourly source panel.

Build panel.csv from locally retrieved source partitions, or read an existing panel.
The fixed panel uses source pod hashes starting with '00' and gpu_request>0.
First appearances are hourly observation arrivals, not submission timestamps.
"""
from pathlib import Path
import json,hashlib,sys
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'hourly_trace';OUT.mkdir(exist_ok=True)

def quantify(x):
    a=np.asarray(x,dtype=float)
    return dict(n=int(len(a)),mean=float(a.mean()) if len(a) else None,
                q10=float(np.quantile(a,.1)) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                q90=float(np.quantile(a,.9)) if len(a) else None,maximum=float(a.max()) if len(a) else None)

def run(source=None):
    if source is not None:
        import pyarrow.parquet as pq
        frames=[]
        for day in range(109,111):
            for hour in range(24):
                path=Path(source)/f'day{day}_hour{hour:02d}.parquet'
                if not path.exists():raise FileNotFoundError(path)
                f=pq.read_table(path).to_pandas();f['t']=(day-109)*24+hour;frames.append(f)
        df=pd.concat(frames,ignore_index=True).sort_values(['t','pod_id'])
        df.to_csv(OUT/'panel.csv',index=False,float_format='%.17g')
    else:df=pd.read_csv(OUT/'panel.csv')
    assert df.t.nunique()==48 and not df.duplicated(['t','pod_id']).any()
    assert df.pod_id.str.startswith('00').all()
    assert np.isfinite(df.gpu_request).all() and (df.gpu_request>0).all()
    finite=np.isfinite(df.used_gpu_hours)&(df.used_gpu_hours>=0)
    bad=int((~finite).sum());df=df[finite].copy()
    # Keep positive occupation even if a terminal state was observed later in the hour.
    eligible=(df.priority_class=='LP')&df.job_type_public.isin(['training','offline_inference'])
    df['eligible']=eligible
    first=df.groupby('pod_id').t.min();last=df.groupby('pod_id').t.max()
    arrival=first[first>0].value_counts().reindex(range(48),fill_value=0).sort_index()
    hour=df.groupby('t').agg(observed_pods=('pod_id','nunique'),requested_GPUs=('gpu_request','sum'),attributed_GPU_hours=('used_gpu_hours','sum'))
    hour['first_appearances']=arrival
    e=df[eligible].groupby('t').agg(eligible_pods=('pod_id','nunique'),eligible_requested_GPUs=('gpu_request','sum'),eligible_GPU_hours=('used_gpu_hours','sum'))
    hour=hour.join(e).fillna(0)
    hour['split']=np.where(hour.index<24,'training','holdout')
    hour.to_csv(OUT/'hourly_metrics.csv',float_format='%.17g')
    pods=df.groupby('pod_id').agg(first=('t','min'),last=('t','max'),observed_hours=('t','nunique'),attributed_GPU_hours=('used_gpu_hours','sum'),requested_GPUs=('gpu_request','median'),job_type=('job_type_public','first'),priority=('priority_class','first'))
    pods['left_censored']=pods['first']==0;pods['right_censored']=pods['last']==47
    pods['observed_span_hours']=pods['last']-pods['first']+1
    pods.to_csv(OUT/'observed_pods.csv',float_format='%.17g')
    summaries={}
    for split,times in [('training',range(24)),('holdout',range(24,48))]:
        h=hour.loc[list(times)];part=df[df.t.isin(times)]
        summaries[split]={c:quantify(h.loc[h.index>0,c] if c=='first_appearances' else h[c]) for c in hour.columns if c!='split'}
        summaries[split]['pod_hours']=len(part);summaries[split]['unique_pods']=int(part.pod_id.nunique())
        summaries[split]['request_distribution']=quantify(part.gpu_request)
        summaries[split]['positive_occupation_distribution']=quantify(part[part.used_gpu_hours>0].used_gpu_hours)
        births=h.loc[h.index>0,'first_appearances']
        summaries[split]['arrival_variance_to_mean']=float(births.var(ddof=1)/births.mean())
        summaries[split]['arrival_lag1_correlation']=float(births.autocorr())
        summaries[split]['eligible_demand_lag1_correlation']=float(h.eligible_GPU_hours.autocorr())
    complete=pods[(~pods.left_censored)&(~pods.right_censored)]
    meta=dict(source='Alibaba Cluster Trace GPU v2026 asi_opensource_pod_hourly',days=[109,110],hours=48,
              panel_rule="source pod_id starts with '00'; gpu_request>0; all consecutive hourly partitions",
              training_days=[109],holdout_days=[110],first_partition_is_left_censored=True,
              rows=len(df),unique_pods=int(df.pod_id.nunique()),invalid_occupation_rows=bad,
              left_censored_pods=int(pods.left_censored.sum()),right_censored_pods=int(pods.right_censored.sum()),
              complete_observed_spans=int(len(complete)),observed_span_distribution=quantify(complete.observed_span_hours),
              observed_occupation_distribution=quantify(complete.attributed_GPU_hours),
              normalizer_GPU_hours=float(hour.loc[list(range(24)),'eligible_GPU_hours'].mean()),
              categories=df.groupby('job_type_public').size().astype(int).to_dict(),
              category_occupation=df.groupby('job_type_public').used_gpu_hours.sum().to_dict(),
              hour_request_bound_violations=int((df.used_gpu_hours>df.gpu_request+1e-8).sum()),
              split=summaries,sha256_panel=hashlib.sha256((OUT/'panel.csv').read_bytes()).hexdigest(),
              interpretation='Hourly first observed GPU-demand arrivals; observed request concurrency; GPU-occupation demand proxies; no exact submissions or useful-compute/power measurements.')
    (OUT/'calibration.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!='split'},indent=2))
    return meta

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else None)
