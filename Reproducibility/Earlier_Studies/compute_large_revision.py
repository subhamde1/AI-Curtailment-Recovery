"""Larger-instance bounds and frozen multiseed scaling study."""
from pathlib import Path
import argparse,json,math,resource
from time import perf_counter
import numpy as np
from revision_algorithms import calendar_lookahead,operate_lookahead
from lagrangian_bounds import optimize_bound
from compute_extensions import transition,mean_lower,calendar,make_jobs,operate,class_jobs
from compute_revision import save

ROOT=Path(__file__).resolve().parent


def decode_catalog(catalog):
    return [dict(j,id=tuple(j["id"])) for j in catalog]


def paths(H,P,rng,count):
    for _ in range(count):
        e=np.zeros(H,dtype=int)
        for t in range(1,H):e[t]=int(rng.random()<P[e[t-1],1])
        yield e


def originals(resume=False):
    old=json.loads((ROOT/"extension_results.json").read_text())
    P=transition(.2,.6)
    checkpoint=ROOT/'revision_data/larger_bounds.json'
    rows=json.loads(checkpoint.read_text()) if resume and checkpoint.exists() else []
    for previous in old["large"]:
        completed=[q for q in rows if q['H']==previous['H']]
        if len(completed)==2 and all(q['paths']==128 for q in completed):
            continue
        start=perf_counter()
        base=decode_catalog(previous["base_catalog"])
        T,H=previous["T"],previous["H"];r=.4;h=1.2;reserve=1.6
        bound=optimize_bound(base,T,r,H,P,h,reserve,iterations=60)
        Lmean=mean_lower(base,T,r,H,P,h,reserve)
        lower=max(Lmean,bound["lower"])
        upper={K:calendar_lookahead(base,0,H,0,P,h,reserve,T,r,K=K) for K in (0,1,2)}
        assert upper[2]["value"]<=upper[1]["value"]+1e-7<=upper[0]["value"]+1e-7
        assert abs(upper[0]["value"]-previous["U"])<1e-7
        records=[]
        rng=np.random.default_rng(20261020+H);m=128
        for i,e in enumerate(paths(H,P,rng,m)):
            result={}
            for K in (0,1):
                q=operate_lookahead(base,e,T,r,P,h,reserve,K=K)
                result[str(K)]=dict(cost=q["cost"],failures=q["failures"],
                                   max_variables=q["max_variables"])
            records.append(dict(path=i,events=e.tolist(),policies=result))
        total_upper_work=sum(j["work"] for j in base)+(T-1)*1.1*r
        B=2.7*total_upper_work;delta=.05/6
        for K in (0,1):
            costs=[q["policies"][str(K)]["cost"] for q in records]
            mean=float(np.mean(costs));se=float(np.std(costs,ddof=1)/np.sqrt(m))
            margin=B*np.sqrt(math.log(1/delta)/(2*m))
            statistical_upper=min(upper[K]["value"],mean+margin)
            rows.append(dict(base_jobs=len(base),H=H,T=T,K=K,
                U=upper[K]["value"],U2=upper[2]["value"],L_mean=Lmean,L_causal_jobs=bound["lower"],
                lower=lower,deterministic_gap=upper[K]["value"]-lower,
                sample_mean=mean,sample_SE=se,paths=m,seed=20261020+H,
                B=B,delta=delta,upper_95_simultaneous=statistical_upper,
                high_confidence_gap=statistical_upper-lower,
                U_variables=upper[K]["variables"],U_ms=1000*upper[K]["seconds"],
                U_matrix_bytes=upper[K]["matrix_bytes"],deadline_failures=0))
        save("larger_bound_history_"+str(H),bound)
        save("larger_policy_paths_"+str(H),records)
        save("larger_bounds",rows)
        print(json.dumps(dict(part="original_catalog",jobs=len(base),H=H,
                              lower=lower,U0=upper[0]["value"],U1=upper[1]["value"],
                              seconds=perf_counter()-start)),flush=True)
    return rows


def scaling():
    P=transition(.2,.6);r=.4;h=1.2;rows=[]
    for n,H in ((50,24),(200,48),(500,72)):
        T=H-6;reserve=1.35*(.18*n/T+1.1*r)
        for seed in (20261011,20261012,20261013):
            start=perf_counter();rng=np.random.default_rng(seed)
            base=[]
            for j in range(n):
                release=int(rng.integers(0,T));D=int(rng.integers(2,7))
                base.append(dict(id=("base",j),release=release,deadline=release+D-1,
                                 work=float(rng.uniform(.12,.24)),rate=float(rng.uniform(.12,.20))))
            save("scale_catalog_"+str(n)+"_"+str(seed),base)
            try:U=calendar_lookahead(base,0,H,0,P,h,reserve,T,r,K=0)
            except RuntimeError as exc:
                row=dict(jobs=n,H=H,T=T,seed=seed,reserve=reserve,status="infeasible envelope",
                         message=str(exc));rows.append(row);save("scaling",rows);continue
            bound=optimize_bound(base,T,r,H,P,h,reserve,iterations=40)
            Lmean=mean_lower(base,T,r,H,P,h,reserve);L=max(Lmean,bound["lower"])
            costs=[];myopic=[];foresight=[];times=[];model_bytes=0;variables=0;records=[]
            # A separate path stream prevents changes to catalog draw counts from
            # silently changing the evaluation paths.
            eval_rng=np.random.default_rng(seed+10000+H)
            for i,e in enumerate(paths(H,P,eval_rng,8)):
                q=operate_lookahead(base,e,T,r,P,h,reserve,K=0)
                short=operate(base,e,T,r,P,h,reserve,mode="myopic")
                pi=calendar(make_jobs(base,e,T,r),0,H,0,P,h,reserve,events=e)["value"]
                assert pi<=q["cost"]+1e-7 and pi<=short["cost"]+1e-7
                costs.append(q["cost"]);myopic.append(short["cost"]);foresight.append(pi)
                times+=q["times"];variables=max(variables,q["max_variables"])
                model_bytes=max(model_bytes,q["matrix_bytes"])
                records.append(dict(path=i,events=e.tolist(),calendar=q["cost"],
                                    myopic=short["cost"],foresight=pi))
            row=dict(jobs=n,H=H,T=T,seed=seed,reserve=reserve,status="all-support certified",
                     U=U["value"],lower=L,gap=U["value"]-L,causal_job_lower=bound["lower"],
                     sample_calendar=float(np.mean(costs)),sample_myopic=float(np.mean(myopic)),
                     sample_foresight=float(np.mean(foresight)),paths=8,failures=0,
                     median_solve_ms=1000*float(np.median(times)),
                     p95_solve_ms=1000*float(np.quantile(times,.95)),
                     variables=variables,matrix_MiB=model_bytes/2**20,
                     process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                     total_seconds=perf_counter()-start)
            rows.append(row);save("scaling",rows)
            save("scale_paths_"+str(n)+"_"+str(seed),records)
            save("scale_bound_history_"+str(n)+"_"+str(seed),bound)
            print(json.dumps(dict(part="scale",**row)),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument("--part",choices=["originals","scaling","all"],default="all")
    p.add_argument("--resume",action="store_true",help="Reuse completed original-catalog evaluations")
    args=p.parse_args()
    if args.part in ("originals","all"):originals(resume=args.resume)
    if args.part in ("scaling","all"):scaling()


if __name__=="__main__":main()
