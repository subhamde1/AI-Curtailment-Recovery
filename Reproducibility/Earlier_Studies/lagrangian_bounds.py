"""Causal job decomposition and valid finite-resource cost lower certificates."""
from collections import defaultdict
import numpy as np
from compute_extensions import prices, class_jobs, tree_jobs


def job_tables(job, P, multipliers, h, reserve):
    """Exact piecewise-linear DP under full-rate individual channel availability."""
    rate=job["rate"]
    assert reserve >= rate-1e-12 and h >= rate-1e-12
    tables={}
    for t in range(job["deadline"],job["release"]-1,-1):
        for e in (0,1):
            quoted=np.array(prices(t,e))+multipliers[t,e]
            channel=int(np.argmin(quoted)) if e==0 else 1
            now=float(quoted[channel])
            future=P[e,0]*tables[t+1,0]["slopes"]+P[e,1]*tables[t+1,1]["slopes"] if t<job["deadline"] else np.zeros(0)
            slopes=np.sort(np.r_[now,future])
            tables[t,e]=dict(slopes=slopes,future=future,now=now,channel=channel)
    return tables


def curve_value(slopes,rate,work):
    if work<1e-12:return 0.
    assert work <= len(slopes)*rate+1e-9
    full=min(len(slopes),int(np.floor((work+1e-12)/rate)))
    leftover=max(0.,work-full*rate)
    return float(rate*np.sum(slopes[:full])+(slopes[full]*leftover if full<len(slopes) else 0.))


def job_value_and_service(job,distribution,weight,P,multipliers,h,reserve):
    tables=job_tables(job,P,multipliers,h,reserve)
    rate=job["rate"]; work=job["work"]; H=len(multipliers)
    value=sum(distribution[e]*curve_value(tables[job["release"],e]["slopes"],rate,work) for e in (0,1))*weight
    service=np.zeros((H,2,2))
    mass={(e,round(work,12)):float(distribution[e]*weight) for e in (0,1) if distribution[e]*weight>0}
    for t in range(job["release"],job["deadline"]+1):
        following=defaultdict(float)
        for (e,w),prob in mass.items():
            table=tables[t,e]
            before=int(np.sum(table["future"]<table["now"]-1e-12))
            x=min(rate,max(0.,w-rate*before))
            x=min(w,x);left=max(0.,w-x)
            service[t,e,table["channel"]]+=prob*x
            if t<job["deadline"]:
                for en in (0,1):
                    following[en,round(left,12)]+=prob*P[e,en]
            else:assert left<1e-9
        mass=following
    return value,service


def lagrangian_value(base,T,r,H,P,h,reserve,multipliers,current=0,small=False):
    belief=np.eye(2)[current];distributions=[]
    for t in range(H):
        distributions.append(belief.copy());belief=belief@P
    total=0.;service=np.zeros((H,2,2))
    for j in base:
        v,a=job_value_and_service(j,distributions[j["release"]],1.,P,multipliers,h,reserve)
        total+=v;service+=a
    factory=tree_jobs if small else class_jobs
    for origin in range(T):
        event_probability=distributions[origin][1]
        if not event_probability:continue
        for job in factory(origin,r):
            v,a=job_value_and_service(job,P[1],event_probability,P,multipliers,h,reserve)
            total+=v;service+=a
    capacity=np.zeros((H,2,2))
    for t,dist in enumerate(distributions):
        capacity[t,0,0]=dist[0]*h
        capacity[t,0,1]=dist[0]*reserve
        capacity[t,1,1]=dist[1]*reserve
    penalty=float(np.sum(multipliers*capacity))
    return float(total-penalty),service-capacity


def optimize_bound(base,T,r,H,P,h,reserve,current=0,small=False,iterations=60):
    # Each evaluated nonnegative multiplier is independently a valid lower bound.
    # Taking the incumbent maximum yields a nondecreasing certificate sequence.
    mu=np.zeros((H,2,2));best=-np.inf;best_mu=mu.copy();history=[]
    scale=max(1.,sum(j["work"] for j in base)/H)
    for i in range(iterations+1):
        value,gradient=lagrangian_value(base,T,r,H,P,h,reserve,mu,current,small)
        if value>best:best=value;best_mu=mu.copy()
        history.append(dict(iteration=i,evaluated=value,incumbent=best))
        step=.8/(scale*np.sqrt(i+1))
        mu=np.maximum(0.,mu+step*gradient)
        # Local service is already forbidden during stress individually.
        mu[:,1,0]=0.
    return dict(lower=best,multipliers=best_mu.tolist(),history=history,
                iterations=iterations)
