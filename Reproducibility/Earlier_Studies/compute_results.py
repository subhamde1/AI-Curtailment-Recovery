"""Deterministic numerical experiments for the repeatable-flexibility manuscript.

All input data are synthetic. Enumeration is exhaustive, not Monte Carlo.
"""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
from itertools import product
import json, math, time
import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,
 'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,
 'savefig.dpi':300,'axes.labelsize':10,'legend.fontsize':9})
COLORS=['#173F5F','#A34A28','#367C68','#775B8B']

def transition(p, rho):
    q=p*(1-rho); alpha=p+(1-p)*rho
    return np.array([[1-q,q],[1-alpha,alpha]])

def word_probability(e,p,rho):
    P=transition(p,rho)
    w=p if e[0] else 1-p
    for a,b in zip(e,e[1:]):w*=P[a,b]
    return w

def edf(e,T,D,M,h,backup=False,return_trace=False,z=0.):
    # Start-of-slot cohorts have their absolute deadlines attached.
    jobs=[];out=0.;miss=False;trace=[]
    for s,ev in enumerate(e,1):
        H=h*(1-ev)+z; free=0.;q=0.
        jobs.sort(key=lambda j:j[0])
        for j in jobs:
            x=min(j[1],H);j[1]-=x;H-=x;free+=x
        for j in jobs:
            if j[0]==s and j[1]>1e-10:
                if backup:q+=j[1];out+=j[1];j[1]=0.
                else:miss=True
        jobs=[j for j in jobs if j[0]>s and j[1]>1e-10]
        if s<=T and ev:jobs.append([s+D,float(M)])
        trace.append({'slot':s,'event':int(ev),'free':free,'backup':q,
                      'end_debt':sum(j[1] for j in jobs)})
    assert not jobs
    return trace if return_trace else (not miss,out)

def interval_counts(e,T,D):
    N=len(e);prefix=np.r_[0,np.cumsum(e)]
    out=[]
    for u in range(2,N+1):
        for v in range(u,N+1):
            lo=max(1,u-1);hi=min(T,v-D)
            n=int(prefix[hi]-prefix[lo-1]) if hi>=lo else 0
            o=int(v-u+1-prefix[v]+prefix[u-1])
            out.append((u,v,n,o))
    return out

def path_gamma(e,T,D):
    ratios=[Fraction(o,n) for u,v,n,o in interval_counts(e,T,D) if n]
    return min(ratios) if ratios else math.inf

def cut_backup(e,T,D,M,h,z=0.):
    N=len(e); F=np.zeros(N+1)
    cuts=defaultdict(list)
    for u,v,n,o in interval_counts(e,T,D):cuts[v].append((u,n,o))
    for v in range(1,N+1):
        F[v]=max([F[v-1]]+[F[u-1]+M*n-h*o-z*(v-u+1) for u,n,o in cuts[v]])
    return F[N]

def flow_backup(e,T,D,M,h,z=0.):
    # Independent LP: ordinary allocation to cohort-slot edges.
    calls=[t for t in range(1,T+1) if e[t-1]]
    edges=[(t,s) for t in calls for s in range(t+1,t+D+1)]
    if not edges:return 0.
    A=[];B=[]
    for t in calls:A.append([int(t==i) for i,s in edges]);B.append(M)
    for s in range(2,T+D+1):A.append([int(s==j) for i,j in edges]);B.append(h*(1-e[s-1])+z)
    res=linprog(-np.ones(len(edges)),A_ub=np.array(A),b_ub=np.array(B),
                bounds=(0,None),method='highs')
    assert res.success
    return M*len(calls)+res.fun

def run_floor(T,D,p,rho):
    q=p*(1-rho);alpha=p+(1-p)*rho
    Q=np.zeros((D+1,D+1));Q[0,0]=1-q;Q[0,1]=q
    for i in range(1,D+1):
        Q[i,0]=1-alpha
        if i<D:Q[i,i+1]=alpha
    s=np.zeros(D+1);s[0]=1-p;s[1]=p
    return 1-float(s@np.linalg.matrix_power(Q,T+D-1)@np.ones(D+1))

def gamma_all(T,D):
    N=T+D; ix=np.arange(2**N,dtype=np.uint32)
    E=((ix[:,None] >> np.arange(N-1,-1,-1)) & 1).astype(np.int16)
    pref=np.c_[np.zeros(len(ix),dtype=np.int16),np.cumsum(E,axis=1,dtype=np.int16)]
    gam=np.full(len(ix),np.inf)
    for u in range(2,N+1):
        for v in range(u+D-1,N+1):
            hi=min(T,v-D);lo=max(1,u-1)
            if hi<lo:continue
            n=pref[:,hi]-pref[:,lo-1]
            o=v-u+1-pref[:,v]+pref[:,u-1]
            g=np.divide(o,n,out=np.full(len(ix),np.inf),where=n>0)
            np.minimum(gam,g,out=gam)
    return E,gam

def enum_distribution(E,gamma,p,rho):
    P=transition(p,rho)
    w=np.where(E[:,0]==1,p,1-p).astype(float)
    for s in range(1,E.shape[1]):w*=P[E[:,s-1],E[:,s]]
    assert abs(w.sum()-1)<1e-11
    vals,idx=np.unique(gamma,return_inverse=True)
    masses=np.bincount(idx,weights=w,minlength=len(vals))
    return vals,masses,w

def frontier(vals,masses,epsilon,a=2,h=1,chi=1):
    cum=0.;best=0.
    for g,m in zip(vals,masses):
        if cum<=epsilon+1e-12:best=min(a,h*g/chi)
        else:break
        cum+=m
    return float(best)

def risk_dp(T,D,M,h,p,rho,z=0):
    # Integer work sizes give exact finite debt states, with floating probabilities.
    P=transition(p,rho);states={(0,(0,)*D):1-p,(1,(0,)*D):p}
    for s in range(1,T+D+1):
        nxt=defaultdict(float)
        for (e,b),w in states.items():
            for j in (0,1):
                H=h*(1-j)+z;rem=[]
                for bk in b:
                    x=min(H,bk);H-=x;rem.append(bk-x)
                if rem[0]>0:continue
                bn=tuple(rem[1:]+[M*j if s<=T else 0])
                nxt[j,bn]+=w*P[e,j]
        states=nxt
    return 1-sum(states.values()),len(states)

def make_tree(T,M,h,p,rho):
    N=T+2;P=transition(p,rho);rows=[];byword={};variables=[]
    for s in range(1,N+1):
        for word in product((0,1),repeat=s):
            parent=byword.get(word[:-1],-1)
            weight=(p if word[0] else 1-p) if s==1 else rows[parent]['w']*P[word[-2],word[-1]]
            n=M*word[-2] if 2<=s<=T+1 else 0.
            var=len(variables) if n else -1
            if n:variables.append(len(rows))
            row={'word':word,'parent':parent,'w':weight,'n':n,
                 'H':h*(1-word[-1]),'var':var}
            byword[word]=len(rows);rows.append(row)
    return rows,variables

def tree_cost(y,rows,variables,M,gradient=False):
    val=0.;g=np.zeros(len(variables));qvals=[]
    for i,row in enumerate(rows):
        vi=row['var'];pi=rows[row['parent']]['var'] if row['parent']>=0 else -1
        b=y[pi] if pi>=0 else 0.;z=y[vi] if vi>=0 else 0.
        q=max(0.,b+row['n']-z-row['H']);qvals.append(q)
        val+=row['w']*.5*q*q
        if vi>=0:g[vi]-=row['w']*q
        if pi>=0:g[pi]+=row['w']*q
    return (val,g) if gradient else val

def quadratic_causal(T,M,h,p,rho):
    rows,variables=make_tree(T,M,h,p,rho)
    yedf=np.zeros(len(variables))
    for row in rows:
        vi=row['var'];parent=row['parent'];pi=rows[parent]['var'] if parent>=0 else -1
        b=yedf[pi] if pi>=0 else 0.
        if vi>=0:yedf[vi]=max(0.,row['n']-max(0.,row['H']-b))
    edfcost=tree_cost(yedf,rows,variables,M)
    if not variables:return {'causal':0.,'lower':0.,'gap':0.,'edf':0.,'variables':0}
    result=minimize(lambda y:tree_cost(y,rows,variables,M,True), yedf,
                    method='L-BFGS-B',jac=True,bounds=[(0,M)]*len(variables),
                    options={'ftol':1e-15,'gtol':1e-12,'maxiter':30000,'maxls':50,'maxcor':35})
    val,g=tree_cost(result.x,rows,variables,M,True)
    lower=val+np.sum(np.where(g<0,g*(M-result.x),-g*result.x))
    return {'causal':val,'lower':lower,'gap':val-lower,'edf':edfcost,
            'variables':len(variables),'converged':bool(result.success),'iterations':result.nit}

def quadratic_perfect(T,M,h,p,rho):
    N=T+2;upper=0.;lower=0.
    for word in product((0,1),repeat=N):
        w=word_probability(word,p,rho)
        n=np.array([M*word[s-2] if 2<=s<=T+1 else 0. for s in range(1,N+1)])
        H=h*(1-np.array(word));inds=np.where(n>0)[0];a=np.zeros((N,len(inds)))
        for j,s in enumerate(inds):a[s,j]=-1;a[s+1,j]=1
        def fun(y):
            q=np.maximum(0,n-H+a@y)
            return .5*q@q,a.T@q
        if len(inds):
            r=minimize(fun,np.zeros(len(inds)),method='L-BFGS-B',jac=True,
              bounds=[(0,M)]*len(inds),options={'ftol':1e-15,'gtol':1e-12,'maxiter':3000})
            v,g=fun(r.x);lo=v+np.sum(np.where(g<0,g*(M-r.x),-g*r.x))
        else:v=lo=0.
        upper+=w*v;lower+=w*lo
    return {'perfect':upper,'perfect_lower':lower,'perfect_gap':upper-lower}

def z_threshold(A,pi,h):
    A=np.asarray(A)
    return np.where(A<=0,0,np.where(A<=(1+pi)*h,A/(1+pi),(A+(1-pi)*h)/2))

def robust_work(L,D,h,z):
    """Sharp work generated per call, without the physical commitment cap."""
    rate=(h+(L+1)*z)/L
    boundary=min(((k+D-1)*z+h*((D+k)//(L+1)))/k
                 for k in range(1,L+1))
    return min(rate,boundary)

def verify_robust():
    """Enumerate finite interval words independently of the frontier formula."""
    cases=0
    for ell in range(1,14):
        ix=np.arange(2**(ell+1),dtype=np.uint32)
        E=((ix[:,None]>>np.arange(ell,-1,-1))&1).astype(np.int16)
        runs=np.zeros(len(ix),int);maxruns=runs.copy()
        for col in E.T:
            runs=(runs+1)*col;maxruns=np.maximum(maxruns,runs)
        for L in range(1,7):
            for D in range(1,min(ell,8)+1):
                n=E[:,:ell-D+1].sum(axis=1)
                o=ell-E[:,1:].sum(axis=1)
                yes=(maxruns<=L)&(n>0)
                if not yes.any():continue
                for h in [0,.25,1,4,20]:
                    for z in [0,.25,1,3]:
                        M=robust_work(L,D,h,z)
                        assert np.max(M*n[yes]-z*ell-h*o[yes])<1e-9
                        cases+=1
    return cases

def verify():
    count=0;maxerr=0.
    # Exhaustive EDF versus the independent cut representation.
    for T,D in [(4,1),(4,2),(3,3),(3,4)]:
        for e in product((0,1),repeat=T+D):
            gam=path_gamma(e,T,D)
            for M in [.25,.5,1.,1.5,2.]:
                ok,total=edf(e,T,D,M,1,True)
                cut=cut_backup(e,T,D,M,1)
                assert abs(total-cut)<1e-9,(e,T,D,M,total,cut)
                ok,_=edf(e,T,D,M,1,False)
                assert ok==(gam>=M)
                count+=1
    rng=np.random.default_rng(713)
    for _ in range(180):
        T=int(rng.integers(1,9));D=int(rng.integers(1,6));e=tuple(rng.integers(0,2,T+D))
        M=float(rng.uniform(.05,3));h=float(rng.uniform(.1,2));z=float(rng.uniform(0,1))
        a=edf(e,T,D,M,h,True,z=z)[1];b=flow_backup(e,T,D,M,h,z)
        cut=cut_backup(e,T,D,M,h,z)
        assert abs(a-cut)<1e-8
        maxerr=max(maxerr,abs(a-b));assert abs(a-b)<1e-8
    for T,D,M,h in [(4,1,1,2),(4,2,3,2),(3,3,2,3)]:
        for p,rho in [(.2,0),(.2,.6),(.5,.9)]:
            enum=sum(word_probability(e,p,rho) for e in product((0,1),repeat=T+D)
                     if not edf(e,T,D,M,h)[0])
            dpr,_=risk_dp(T,D,M,h,p,rho)
            assert abs(enum-dpr)<1e-11
            enumzero=sum(word_probability(e,p,rho) for e in product((0,1),repeat=T+D)
                         if path_gamma(e,T,D)==0)
            assert abs(enumzero-run_floor(T,D,p,rho))<1e-11
    # EDF rate-limit counterexample checked against a separate capacitated LP.
    # Variables A1,A2,B1,B2,B3; bounds 1; free C=(1,2,0).
    Aeq=np.array([[1,1,0,0,0],[0,0,1,1,1]])
    Aub=np.array([[1,0,1,0,0],[0,1,0,1,0],[0,0,0,0,1]])
    flow=linprog(np.zeros(5),A_eq=Aeq,b_eq=[1,2],A_ub=Aub,b_ub=[1,2,0],bounds=(0,1),method='highs')
    assert flow.success
    return {'exhaustive_word_commitment_cases':count,'independent_flow_cases':180,
            'maximum_flow_discrepancy':maxerr,'risk_dp_cases':9,
            'rate_limited_feasible_schedule':flow.x.tolist(),
            'bounded_run_interval_parameter_checks':verify_robust()}

def main():
    started=time.time();out={'inputs':'synthetic, normalized work units','verification':verify()}
    print('verification',out['verification'],flush=True)
    # Exact finite-horizon frontiers, including deadline extension clearing slots.
    table=[];cache={}
    for D in range(1,9):
        E,gamma=gamma_all(10,D);cache[D]=(E,gamma)
        for rho in [0.,.3,.6,.85]:
            vals,masses,w=enum_distribution(E,gamma,.2,rho)
            row={'T':10,'D':D,'p':.2,'rho':rho,'blackout_floor':run_floor(10,D,.2,rho),
                 'r_star_10pct':frontier(vals,masses,.1),'r_star_5pct':frontier(vals,masses,.05),
                 'risk_r_half':float(w[gamma<.5].sum())}
            table.append(row)
        print('frontier deadline',D,flush=True)
    out['frontiers']=table
    out['joint_robust']=[{'L':4,'D':D,'h':1.,'z':z,'chi':1.,'a':2.,
                         'r_star':min(2.,robust_work(4,D,1.,z))}
                        for D in range(1,6) for z in [0,.25,.5,1.]]
    fig,ax=plt.subplots(figsize=(6.4,3.8));zgrid=np.linspace(0,1.5,301)
    for k,D in enumerate([1,2,3,4]):
        ax.plot(zgrid,[min(2.,robust_work(4,D,1.,z)) for z in zgrid],
                color=COLORS[k],label=f'D = {D}',lw=2,
                linestyle='--' if D==4 else '-')
    ax.set(xlabel='Protected recovery capacity z',
           ylabel='Robust commitment r',ylim=(0,2.05))
    ax.legend();ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIG/'joint_robust.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    for k,rho in enumerate([0.,.3,.6,.85]):
        z=[r for r in table if r['rho']==rho]
        ax.plot([r['D'] for r in z],[r['r_star_10pct'] for r in z],'-o',color=COLORS[k],label=f'Persistence {rho:g}')
    ax.set(xlabel='Recovery deadline D (slots)',ylabel='Maximum commitment r / h',xticks=range(1,9),ylim=(-.05,2.05))
    ax.legend(loc='upper left');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIG/'frontier.png');plt.close(fig)
    # Persistence creates an interior maximum in a finite-horizon all-deadlines risk.
    rhogrid=np.linspace(0,.995,201);fig,ax=plt.subplots(figsize=(6.4,3.8))
    for k,D in enumerate([1,2,4,8]):
        ax.plot(rhogrid,[run_floor(24,D,.2,r) for r in rhogrid],color=COLORS[k],label=f'D = {D}')
    ax.axhline(.1,color='#666666',ls='--',lw=1,label='10% tolerance')
    ax.set(xlabel='Persistence ρ (event share p = 0.2)',ylabel='Zero-commitment probability',ylim=(0,1))
    ax.legend(ncol=2);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIG/'risk_floor.png');plt.close(fig)
    out['persistence_counterexample']=[{'T':4,'D':1,'p':.2,'rho':r,'failure':run_floor(4,1,.2,r)} for r in [0,.5,.99]]
    out['horizon_floors']=[{'T':T,'D':D,'rho':rho,'floor':run_floor(T,D,.2,rho)}
          for T in [10,24,96,365] for D in [2,4,8] for rho in [0,.6,.85]]
    # Same number of calls, different repeatability.
    words={'Separated':(1,0,0,1,0,0,1,0,0,0,0,0,0),
           'Clustered':(1,1,1,0,0,0,0,0,0,0,0,0,0)}
    traces={k:edf(e,10,3,1.2,1,True,True) for k,e in words.items()}
    out['paths']={k:{'events':list(e),'gamma':float(path_gamma(e,10,3)),
                      'linear_backup':edf(e,10,3,1.2,1,True)[1]} for k,e in words.items()}
    fig,axs=plt.subplots(2,1,figsize=(6.4,4.8),sharex=True)
    for k,(name,e) in enumerate(words.items()):
        tr=traces[name];slots=np.arange(1,len(e)+1)
        axs[k].bar(slots,e,color='#E8D4CB',label='Curtailment indicator')
        axs[k].step(slots,[x['end_debt'] for x in tr],where='mid',color=COLORS[k],lw=2,label='End-of-slot recovery debt')
        axs[k].plot(slots,[x['backup'] for x in tr],'s',color='#333333',ms=4,label='Deadline backup')
        axs[k].set_ylabel(name+'\nwork units');axs[k].legend(loc='upper right',fontsize=8,ncol=1);axs[k].set_ylim(0,4)
    axs[-1].set(xlabel='Slot',xticks=range(1,14));fig.tight_layout();fig.savefig(FIG/'paths.png');plt.close(fig)
    # Closed-form D=1 cost and commitment economics.
    rgrid=np.linspace(0,2,251);fig,axs=plt.subplots(1,2,figsize=(6.8,3.4))
    out['d1_economics']=[]
    for k,rho in enumerate([0,.3,.6,.85]):
        alpha=.2+.8*rho;J=.2/2*(alpha*rgrid*rgrid+(1-alpha)*np.maximum(rgrid-1,0)**2)
        axs[0].plot(rgrid,J,color=COLORS[k],label=f'ρ = {rho:g}')
        lam=.3;ru=lam/alpha if lam<=alpha else lam+(1-alpha)
        out['d1_economics'].append({'rho':rho,'alpha':alpha,'lambda':lam,'r_unconstrained':min(2,ru),
                                  'cost_at_r_1':.1*alpha,'cost_at_r_1_5':.1*(alpha*2.25+(1-alpha)*.25)})
        lams=np.linspace(0,1,201);rstars=np.where(lams<=alpha,lams/alpha,lams+1-alpha)
        axs[1].plot(lams,np.minimum(2,rstars),color=COLORS[k])
    axs[0].set(xlabel='Commitment r (h = 1)',ylabel='Expected backup cost per slot');axs[0].legend()
    axs[1].set(xlabel='Payment per call and unit λ',ylabel='Economic commitment r')
    for ax in axs:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'cost_economics.png');plt.close(fig)
    # Analytical two-stage marginal-price balance.
    fig,axs=plt.subplots(1,2,figsize=(6.8,3.4));A=np.linspace(0,4,401)
    for k,pi in enumerate([.1,.3,.6,.9]):
        y=z_threshold(A,pi,1);axs[0].plot(A,y,color=COLORS[k],label=f'π = {pi:g}')
        # isolated cohort n=A, H_current=0, b=0, h_next=1
        opt=.5*((A-y)**2+(1-pi)*np.maximum(y-1,0)**2+pi*y*y)
        late=.5*((1-pi)*np.maximum(A-1,0)**2+pi*A*A)
        axs[1].plot(A,np.divide(late-opt,late,out=np.full(len(A),np.nan),where=late>1e-10)*100,color=COLORS[k])
    axs[0].set(xlabel='Net work A',ylabel='Optimal residual work y*');axs[0].legend()
    axs[1].set(xlabel='Net work A',ylabel='Saving from early backup (%)')
    for ax in axs:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'threshold.png');plt.close(fig)
    # Continuous convex stochastic optimization and independent perfect-information bounds.
    costs=[]
    for rho in [0,.3,.6,.85]:
        row={'T':6,'D':2,'M':1.5,'h':1.,'p':.2,'rho':rho}
        row.update(quadratic_causal(6,1.5,1,.2,rho));row.update(quadratic_perfect(6,1.5,1,.2,rho))
        assert row['gap']<1e-6 and row['perfect_gap']<1e-7,row
        assert row['perfect']<=row['causal']+1e-7<=row['edf']+1e-7
        row['savings_percent']=100*(row['edf']-row['causal'])/row['edf']
        row['information_gap_percent']=100*(row['causal']-row['perfect'])/row['causal']
        costs.append(row);print('convex cost',row,flush=True)
    out['quadratic_costs']=costs
    fig,ax=plt.subplots(figsize=(6.4,3.8));x=np.arange(len(costs));width=.24
    for k,(field,label) in enumerate([('perfect','Perfect information'),('causal','Optimal causal'),('edf','Backup only at deadlines')]):
        ax.bar(x+(k-1)*width,[row[field] for row in costs],width,color=COLORS[k],label=label)
    ax.set(xticks=x,xticklabels=[str(row['rho']) for row in costs],xlabel='Persistence ρ',ylabel='Expected total backup cost');ax.legend();ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'policy_cost.png');plt.close(fig)
    out['deadline_design']=[]
    for T in [10,24,96,365]:
        for rho in [0,.6,.85]:
            alpha=.2+.8*rho
            d=next(D for D in range(1,150) if run_floor(T,D,.2,rho)<=.1)
            sufficient=max(1,math.ceil(math.log(T*.2/.1)/(-math.log(alpha))))
            out['deadline_design'].append({'T':T,'rho':rho,'exact_min_D_for_floor':d,
                                          'union_bound_sufficient_D':sufficient})
    out['runtime_seconds']=time.time()-started
    (ROOT/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print('complete seconds',out['runtime_seconds'],flush=True)

if __name__=='__main__':main()
