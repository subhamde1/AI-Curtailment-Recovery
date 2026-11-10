"""Reproduce heterogeneous certification and finite-reserve calendar experiments.

All data are synthetic. Small trees are exhaustive; large experiments use a
declared seed. No observed facility data or fitted workload permissions are used.
"""
from pathlib import Path
from itertools import product
from time import perf_counter
import json, math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':250})
SEED=20261008

def sparse_rows(rows,n):
    rr=[];cc=[];vv=[]
    for i,row in enumerate(rows):
        for j,v in row.items():
            if v: rr.append(i);cc.append(j);vv.append(v)
    return coo_matrix((vv,(rr,cc)),shape=(len(rows),n)).tocsr()

def lp(c,ub,bu,eq,be,bounds):
    ans=linprog(c,A_ub=sparse_rows(ub,len(c)) if ub else None,
      b_ub=bu if ub else None,A_eq=sparse_rows(eq,len(c)) if eq else None,
      b_eq=be if eq else None,bounds=bounds,method='highs')
    return ans

def reserve_tree(r=1.,extra_deadline=0):
    # Root followed by two completely revealed three-slot calendars.
    capacities=[[.5,0,2.5,2.5],[.5,2.5,0,0]]
    works=[.5*r,r,r]; rates=[1,.5,1.5]; deadlines=[2,4,4]
    if extra_deadline:
        capacities=[x+[2.5]*extra_deadline for x in capacities]
        deadlines=[d+extra_deadline for d in deadlines]
    H=len(capacities[0]); edges=[(k,j,t) for k in range(2)
           for j in range(3) for t in range(deadlines[j])]
    ix={e:i for i,e in enumerate(edges)}; z=len(edges)
    c=np.zeros(z+1);c[z]=1
    ub=[];bu=[];eq=[];be=[]
    for k in range(2):
        for t in range(H):
            row={ix[k,j,t]:1 for j in range(3) if (k,j,t) in ix}
            row[z]=-1;ub.append(row);bu.append(capacities[k][t])
        for j in range(3):
            eq.append({ix[k,j,t]:1 for t in range(deadlines[j])});be.append(works[j])
    for j in range(3):eq.append({ix[0,j,0]:1,ix[1,j,0]:-1});be.append(0)
    bounds=[(0,rates[j]) for k,j,t in edges]+[(0,None)]
    ans=lp(c,ub,bu,eq,be,bounds);assert ans.success
    # Independently construct the dual of the min-z LP.
    A=sparse_rows(ub,len(c)); E=sparse_rows(eq,len(c))
    bound_rows=[];bound_rhs=[]
    for i,b in enumerate(bounds[:-1]):bound_rows.append({i:1});bound_rhs.append(b[1])
    R=sparse_rows(bound_rows,len(c))
    from scipy.sparse import hstack,vstack
    # Primal min with <= constraints has nonpositive multipliers; equality free.
    Dc=np.r_[bu,be,bound_rhs]
    DA=hstack([A.T,E.T,R.T],format='csr')
    dual=linprog(-Dc,A_ub=DA,b_ub=c,
       bounds=[(None,0)]*len(ub)+[(None,None)]*len(eq)+[(None,0)]*len(bound_rows),method='highs')
    assert dual.success and abs(ans.fun+dual.fun)<1e-8
    return float(ans.fun),float(-dual.fun)

def transition(p,rho):
    q=p*(1-rho);a=p+(1-p)*rho
    return np.array([[1-q,q],[1-a,a]])

def prices(t,z):
    # Positive channel prices. Curtailment removes local recovery service.
    return .7+.22*math.cos(2*math.pi*t/6)+.2*z, 1.65+.35*math.sin(2*math.pi*t/8)+.7*z

def class_jobs(origin,r):
    # Work weights sum to chi=1.1. Durations and rate ceilings differ.
    return [dict(id=('call',origin,k),release=origin+1,deadline=origin+D,
                 work=r*b,rate=rate) for k,(D,b,rate) in
            enumerate([(2,.35,.22),(4,.45,.25),(6,.30,.20)])]

def tree_jobs(origin,r):
    return [dict(id=('call',origin,k),release=origin+1,deadline=origin+D,
                 work=r*b,rate=rate) for k,(D,b,rate) in
            enumerate([(2,.45,.30),(3,.65,.40)])]

def make_jobs(base,events,T,r,small=False):
    out=[dict(j) for j in base]
    for t in range(T):
        if events[t]:out.extend((tree_jobs if small else class_jobs)(t,r))
    return out

def calendar(jobs,t,H,z,P,h,reserve,events=None,mode='forecast'):
    # Existing work is passed in as residual work. Future ghost jobs are upper bounds.
    jobs=[j for j in jobs if j['work']>1e-10 and j['deadline']>=t]
    edges=[(i,s,k) for i,j in enumerate(jobs)
           for s in range(max(t,j['release']),j['deadline']+1) for k in range(2)]
    if not edges:return dict(value=0.,actions={},variables=0,runtime=0.)
    ix={e:i for i,e in enumerate(edges)}
    beliefs={t:np.eye(2)[z]}
    for s in range(t+1,H):beliefs[s]=beliefs[s-1]@P
    c=[];ub=[];bu=[];eq=[];be=[]
    for i,s,k in edges:
        if events is not None:co=prices(s,int(events[s]))[k]
        elif s==t:co=prices(s,z)[k]
        else:co=sum(beliefs[s][e]*prices(s,e)[k] for e in (0,1))
        c.append(co if mode=='forecast' or s==t else 1e-8)
    for s in range(t,H):
        local=h*(1-(int(events[s]) if events is not None else z)) if s==t or events is not None else 0.
        for k,cap in [(0,local),(1,reserve)]:
            ub.append({ix[i,s,k]:1 for i in range(len(jobs)) if (i,s,k) in ix});bu.append(cap)
        for i,j in enumerate(jobs):
            if (i,s,0) in ix:
                ub.append({ix[i,s,0]:1,ix[i,s,1]:1});bu.append(j['rate'])
    for i,j in enumerate(jobs):
        eq.append({ix[i,s,k]:1 for s in range(max(t,j['release']),j['deadline']+1) for k in (0,1)})
        be.append(j['work'])
    start=perf_counter();ans=lp(c,ub,bu,eq,be,[(0,None)]*len(c)); elapsed=perf_counter()-start
    if not ans.success:raise RuntimeError(('Infeasible calendar',t,z,ans.message,jobs))
    actions={}
    for (i,s,k),q in zip(edges,ans.x):
        if s==t and q>1e-10:
            v=actions.setdefault(jobs[i]['id'],np.zeros(2));v[k]+=q
    dualval=float(np.dot(ans.ineqlin.marginals,bu)+np.dot(ans.eqlin.marginals,be))
    assert abs(dualval-ans.fun)<1e-7
    return dict(value=float(ans.fun),actions=actions,variables=len(c),runtime=elapsed)

def operate(base,events,T,r,P,h,reserve,mode='forecast',small=False,cache=None):
    H=len(events); active=[dict(j) for j in base]; cost=0.; misses=0; maxvars=0; times=[]
    for t in range(H):
        # Future calls include this observed call; later calls use upper demand.
        ghost=[]
        for origin in range(t,T):
            if origin>t or events[t]:ghost.extend((tree_jobs if small else class_jobs)(origin,r))
        key=(tuple(events[:t+1]),mode)
        plan=cache.get(key) if cache is not None else None
        if plan is None:
            plan=calendar(active+ghost,t,H,int(events[t]),P,h,reserve,mode=mode)
            if cache is not None:cache[key]=plan
        maxvars=max(maxvars,plan['variables']);times.append(plan['runtime'])
        channel=np.zeros(2)
        for j in active:
            action=plan['actions'].get(j['id'],np.zeros(2))
            assert action.sum()<=j['rate']+1e-7 and action.sum()<=j['work']+1e-7
            j['work']-=action.sum();channel+=action
            if j['deadline']==t and j['work']>1e-7:misses+=1
        assert channel[0]<=h*(1-events[t])+1e-7 and channel[1]<=reserve+1e-7
        cost+=float(channel@prices(t,int(events[t])))
        active=[j for j in active if j['deadline']>t and j['work']>1e-9]
        if t<T and events[t]:active.extend((tree_jobs if small else class_jobs)(t,r))
    assert not active and misses==0
    return dict(cost=cost,misses=misses,max_variables=maxvars,
                mean_ms=1000*float(np.mean(times)),max_ms=1000*max(times))

def causal_cost(base,T,r,P,h,reserve):
    H=T+3; nodes=[(1,)];prob={(1,):1.}
    for t in range(1,H):
        for pre in [n for n in nodes if len(n)==t]:
            for e in (0,1):
                n=pre+(e,);nodes.append(n);prob[n]=prob[pre]*P[pre[-1],e]
    # Distinct calls are indexed by their observed origin prefix.
    jobs={('base',j['id']):dict(j,birth=(),work=j['work']) for j in base}
    for n in nodes:
        origin=len(n)-1
        if origin<T and n[-1]:
            for j in tree_jobs(origin,r):jobs[('event',n,j['id'][-1])]=dict(j,birth=n)
    edges=[]
    for n in nodes:
        t=len(n)-1
        for jid,j in jobs.items():
            if j['release']<=t<=j['deadline'] and (not j['birth'] or n[:len(j['birth'])]==j['birth']):
                edges.extend([(n,jid,0),(n,jid,1)])
    ix={e:i for i,e in enumerate(edges)};c=[];ub=[];bu=[];eq=[];be=[]
    for n,jid,k in edges:c.append(prob[n]*prices(len(n)-1,n[-1])[k])
    for n in nodes:
        for k,cap in [(0,h*(1-n[-1])),(1,reserve)]:
            ub.append({ix[n,jid,k]:1 for jid in jobs if (n,jid,k) in ix});bu.append(cap)
        for jid,j in jobs.items():
            if (n,jid,0) in ix:
                ub.append({ix[n,jid,0]:1,ix[n,jid,1]:1});bu.append(j['rate'])
            if len(n)-1==j['deadline'] and (not j['birth'] or n[:len(j['birth'])]==j['birth']):
                row={}
                for s in range(j['release'],j['deadline']+1):
                    for k in (0,1):row[ix[n[:s+1],jid,k]]=1
                eq.append(row);be.append(j['work'])
    ans=lp(c,ub,bu,eq,be,[(0,None)]*len(c));assert ans.success
    dual=float(np.dot(ans.ineqlin.marginals,bu)+np.dot(ans.eqlin.marginals,be))
    assert abs(ans.fun-dual)<1e-7
    return dict(value=float(ans.fun),dual=dual,variables=len(c),nodes=len(nodes))

def small_experiment():
    T=3;H=6;r=.6;h=1.2;reserve=1.2;p=.2
    base=[dict(id=('base',0),release=0,deadline=1,work=.6,rate=.4),
          dict(id=('base',1),release=0,deadline=3,work=.8,rate=.3)]
    rows=[]
    for rho in (0.,.6,.85):
        P=transition(p,rho);causal=causal_cost(base,T,r,P,h,reserve)
        frontier=0.;myopic=0.;perfect=0.;cache={};cachem={}
        for word in product((0,1),repeat=H-1):
            e=(1,)+word;pw=math.prod(P[e[t-1],e[t]] for t in range(1,H))
            frontier+=pw*operate(base,e,T,r,P,h,reserve,small=True,cache=cache)['cost']
            myopic+=pw*operate(base,e,T,r,P,h,reserve,mode='myopic',small=True,cache=cachem)['cost']
            perfect+=pw*calendar(make_jobs(base,e,T,r,True),0,H,1,P,h,reserve,events=e)['value']
        ghost=base+[j for t in range(T) for j in tree_jobs(t,r)]
        U=calendar(ghost,0,H,1,P,h,reserve)['value']
        assert perfect<=causal['value']+1e-7<=frontier+1e-7<=U+1e-7
        rows.append(dict(rho=rho,perfect=perfect,causal=causal['value'],
                         calendar=frontier,myopic=myopic,U=U,
                         gap_bound=U-perfect,actual_gap=frontier-causal['value'],
                         causal_variables=causal['variables'],tree_nodes=causal['nodes']))
    return rows

def initial_certificate(b,E,T,D,h,z,chi=1.,a=2.):
    H=len(E);best=a
    for u in range(H):
        for v in range(u,H):
            inherited=sum(q for k,q in enumerate(b,1) if u==0 and k-1<=v)
            n=sum(E[t] for t in range(T) if u<=t+1 and t+D<=v)
            cap=sum(h*(1-E[s])+z for s in range(u,v+1))
            if inherited>cap+1e-10:return -1.
            if n:best=min(best,(cap-inherited)/(chi*n))
    return best

def state_experiment():
    T=4;D=3;H=T+D;P=transition(.2,.6);rows=[];checks=0
    for b in ([0,0,0],[0,1.5,0],[1.5,0,0],[0,0,1.5]):
        pairs=[]
        for word in product((0,1),repeat=H-1):
            e=(0,)+word;pw=math.prod(P[e[t-1],e[t]] for t in range(1,H))
            r=initial_certificate(b,e,T,D,1.,.2)
            pairs.append((r,pw))
            for test in (0.,.2,.4,.8,1.2):
                jobs=[dict(id=('i',k),release=0,deadline=k-1,work=q,rate=10.)
                      for k,q in enumerate(b,1) if q]
                for t in range(T):
                    if e[t]:jobs.append(dict(id=('c',t),release=t+1,deadline=t+D,work=test,rate=10.))
                # Independent homogeneous slot allocation, one aggregate channel.
                edges=[(i,s) for i,j in enumerate(jobs) for s in range(j['release'],j['deadline']+1)]
                if not edges: ok=True
                else:
                    ix={q:i for i,q in enumerate(edges)}
                    ub=[{ix[i,s]:1 for i in range(len(jobs)) if (i,s) in ix} for s in range(H)]
                    eq=[{ix[i,s]:1 for s in range(j['release'],j['deadline']+1)} for i,j in enumerate(jobs)]
                    ans=lp(np.zeros(len(edges)),ub,[1-e[s]+.2 for s in range(H)],eq,[j['work'] for j in jobs],[(0,None)]*len(edges));ok=ans.success
                assert ok==(r>=test-1e-8);checks+=1
        eps=.1; candidates=sorted(set([0.]+[max(0,x) for x,p in pairs]))
        eligible=[x for x in candidates if sum(p for cap,p in pairs if cap<x-1e-10)<=eps+1e-10]
        rows.append(dict(debt=b,zero_failure=sum(p for cap,p in pairs if cap<0),
                         frontier=max(eligible) if eligible else None))
    return rows,checks

def mean_lower(base,T,r,H,P,h,reserve,small=False,z0=0):
    # Expected-flow relaxation: no forecast or realized sample is used.
    belief=np.eye(2)[z0]; probs=[]
    for t in range(H):
        probs.append(belief.copy());belief=belief@P
    jobs=[dict(j) for j in base]
    for origin in range(T):
        pe=probs[origin][1]
        if pe:
            for j in (tree_jobs if small else class_jobs)(origin,r):
                jobs.append(dict(j,work=j['work']*pe,rate=j['rate']*pe))
    edges=[(i,s,k) for i,j in enumerate(jobs) for s in range(j['release'],j['deadline']+1) for k in (0,1)]
    ix={e:i for i,e in enumerate(edges)}
    c=[min(prices(s,z)[k] for z in (0,1)) for i,s,k in edges]
    ub=[];bu=[];eq=[];be=[]
    for s in range(H):
        for k,cap in [(0,h*probs[s][0]),(1,reserve)]:
            ub.append({ix[i,s,k]:1 for i in range(len(jobs)) if (i,s,k) in ix});bu.append(cap)
        for i,j in enumerate(jobs):
            if (i,s,0) in ix:ub.append({ix[i,s,0]:1,ix[i,s,1]:1});bu.append(j['rate'])
    for i,j in enumerate(jobs):
        eq.append({ix[i,s,k]:1 for s in range(j['release'],j['deadline']+1) for k in (0,1)});be.append(j['work'])
    ans=lp(c,ub,bu,eq,be,[(0,None)]*len(c));assert ans.success
    dual=float(np.dot(ans.ineqlin.marginals,bu)+np.dot(ans.eqlin.marginals,be))
    assert abs(ans.fun-dual)<1e-7
    return float(ans.fun)

def large_experiment():
    rng=np.random.default_rng(SEED);rows=[];p=.2;rho=.6;P=transition(p,rho)
    paths=24
    for n,H in [(20,12),(50,24),(100,48)]:
        T=H-6;r=.4;h=1.2;reserve=1.6
        base=[]
        for j in range(n):
            release=int(rng.integers(0,max(1,T)))
            D=int(rng.integers(2,7));work=float(rng.uniform(.12,.24))
            base.append(dict(id=('base',j),release=release,deadline=release+D-1,
                             work=work,rate=float(rng.uniform(.12,.20))))
        ghost=base+[j for t in range(1,T) for j in class_jobs(t,r)]
        U=calendar(ghost,0,H,0,P,h,reserve)['value']
        val=[];short=[];clair=[];times=[];maxvar=0
        for k in range(paths):
            e=np.zeros(H,dtype=int)
            for t in range(1,H):e[t]=int(rng.random()<P[e[t-1],1])
            a=operate(base,e,T,r,P,h,reserve);b=operate(base,e,T,r,P,h,reserve,mode='myopic')
            v=calendar(make_jobs(base,e,T,r),0,H,0,P,h,reserve,events=e)['value']
            assert v<=a['cost']+1e-7 and v<=b['cost']+1e-7
            val.append(a['cost']);short.append(b['cost']);clair.append(v)
            times.append(a['mean_ms']);maxvar=max(maxvar,a['max_variables'])
        # Independent audit sample for a bound on E[perfect-information cost].
        audit_rng=np.random.default_rng(SEED+1000+H);audit=[];audit_paths=1024
        for k in range(audit_paths):
            e=np.zeros(H,dtype=int)
            for t in range(1,H):e[t]=int(audit_rng.random()<P[e[t-1],1])
            audit.append(calendar(make_jobs(base,e,T,r),0,H,0,P,h,reserve,events=e)['value'])
        # This is a lower-confidence bound on E[perfect-information cost], not
        # a confidence interval for the empirical policy cost.
        B=2.7*sum(j['work'] for j in ghost)
        delta=.05/3
        L=max(0.,float(np.mean(audit))-B*math.sqrt(math.log(1/delta)/(2*audit_paths)))
        Lmean=mean_lower(base,T,r,H,P,h,reserve)
        Lcert=max(L,Lmean)
        rows.append(dict(base_jobs=n,H=H,T=T,paths=paths,seed=SEED,
                         base_catalog=base,
                         calendar_mean=float(np.mean(val)),myopic_mean=float(np.mean(short)),
                         perfect_mean=float(np.mean(clair)),calendar_se=float(np.std(val,ddof=1)/math.sqrt(paths)),
                         U=U,L_sample=L,L_mean=Lmean,L95=Lcert,gap95=U-Lcert,
                         audit_mean=float(np.mean(audit)),audit_paths=audit_paths,
                         delta_per_configuration=delta,B=B,max_variables=maxvar,
                         mean_lp_ms=float(np.mean(times)),deadline_failures=0))
        print('large',{k:v for k,v in rows[-1].items() if k!='base_catalog'},flush=True)
    return rows

def battery_example(q0):
    H=3; jobs=[dict(release=0,deadline=1,work=1.,rate=.7),
               dict(release=0,deadline=2,work=1.,rate=.7)]
    edges=[(j,t,k) for j,o in enumerate(jobs) for t in range(o['deadline']+1) for k in (0,1)]
    ix={e:i for i,e in enumerate(edges)};nv=len(edges)
    grid=list(range(nv,nv+H));g=list(range(nv+H,nv+2*H));u=list(range(nv+2*H,nv+3*H));q=list(range(nv+3*H,nv+4*H))
    c=np.zeros(nv+4*H)
    for e,i in ix.items():c[i]=3.*e[2]
    for t in range(H):c[grid[t]]=1.;c[g[t]]=c[u[t]]=.02
    ub=[];bu=[];eq=[];be=[]
    for j,o in enumerate(jobs):
        eq.append({ix[j,t,k]:1 for t in range(o['deadline']+1) for k in (0,1)});be.append(o['work'])
        for t in range(o['deadline']+1):ub.append({ix[j,t,0]:1,ix[j,t,1]:1});bu.append(o['rate'])
    for t in range(H):
        ub.append({ix[j,t,1]:1 for j in range(2) if (j,t,1) in ix});bu.append(.5)
        ub.append({g[t]:1/.8,u[t]:1/.8});bu.append(1.)
        row={grid[t]:1,u[t]:1,g[t]:-1}
        for j in range(2):
            if (j,t,0) in ix:row[ix[j,t,0]]=-1
        eq.append(row);be.append(0.)
        row={q[t]:1,g[t]:-1,u[t]:1}
        if t:row[q[t-1]]=-1
        eq.append(row);be.append(q0 if t==0 else 0.)
    bounds=[(0,None)]*nv+[(0,1.),(0,0.),(0,0.)]+[(0,.8)]*(3*H)
    ans=lp(c,ub,bu,eq,be,bounds);assert ans.success
    dual=float(np.dot(ans.ineqlin.marginals,bu)+np.dot(ans.eqlin.marginals,be)+
               np.dot(ans.upper.marginals,[0 if b[1] is None else b[1] for b in bounds]))
    assert abs(ans.fun-dual)<1e-8
    return dict(initial_charge=q0,cost=float(ans.fun),dual=dual,
                grid_energy=float(sum(ans.x[grid])),discharge=float(sum(ans.x[u])),
                remote_work=float(sum(ans.x[i] for e,i in ix.items() if e[2]==1)))

def main():
    start=perf_counter();out={'inputs':'synthetic, normalized units','seed':SEED}
    reserve=[]
    for r in np.linspace(0,1,41):
        z,d=reserve_tree(float(r));formula=max(0,(3*r-2)/8)
        assert abs(z-formula)<1e-8
        reserve.append(dict(r=float(r),z=z,dual=d))
    out['reserve_curve']=reserve;out['reserve_duality_checks']=len(reserve)
    out['extra_slack']=[dict(extra_slots=d,reserve=reserve_tree(1.,d)[0]) for d in (0,1,2)]
    out['states'],out['state_flow_checks']=state_experiment()
    out['small']=small_experiment();print('small',out['small'],flush=True)
    out['large']=large_experiment()
    out['battery']=[battery_example(q) for q in (0.,.4,.8)]
    out['runtime_seconds']=perf_counter()-start
    (ROOT/'extension_results.json').write_text(json.dumps(out,indent=2))
    fig,ax=plt.subplots(figsize=(6.4,3.5))
    ax.plot([x['r'] for x in reserve],[x['z'] for x in reserve],color='#173F5F',lw=2,label='Exact causal reserve')
    ax.plot([0,1],[0,0],color='#A34A28',ls='--',label='Pathwise reserve')
    ax.set(xlabel='Cohort scale r',ylabel='Protected capacity z per slot');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'causal_reserve.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7,3.3))
    x=np.arange(len(out['small']));w=.22
    for k,(key,label,col) in enumerate([('perfect','Foresight','#367C68'),('causal','Optimal causal','#173F5F'),('calendar','Calendar policy','#A34A28')]):
        axs[0].bar(x+(k-1)*w,[q[key] for q in out['small']],w,label=label,color=col)
    axs[0].set(xticks=x,xticklabels=[str(q['rho']) for q in out['small']],xlabel='Persistence',ylabel='Expected cost');axs[0].legend(fontsize=8)
    for key,label,col in [('calendar_mean','Calendar','#173F5F'),('myopic_mean','Myopic shield','#A34A28'),('perfect_mean','Foresight sample mean','#367C68')]:
        axs[1].plot([q['H'] for q in out['large']],[q[key] for q in out['large']],'o-',label=label,color=col)
    axs[1].set(xlabel='Horizon slots',ylabel='Sample mean total cost');axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'heterogeneous_policy.png');plt.close(fig)
    print('extensions complete',out['runtime_seconds'],flush=True)

if __name__=='__main__':main()
