import itertools, json, time, math
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint, linprog
from scipy.sparse import lil_matrix, csr_matrix

OUT=Path(__file__).parent

def chain(p,rho):
    q=p*(1-rho); alpha=p+(1-p)*rho
    return np.array([[1-q,q],[1-alpha,alpha]])

def tree(N,p,rho,e0=1):
    P=chain(p,rho); nodes={}
    for t in range(1,N+1):
        for tail in itertools.product((0,1),repeat=t-1):
            hist=(e0,)+tail
            prob=np.prod([P[hist[k-1],hist[k]] for k in range(1,t)])
            nodes[hist]=float(prob)
    return nodes

def recovery(T=3,D=2,p=.25,rho=.4,h=1,z=.4,u=.1,v=.5,b=.9,r=.4,
             mode='cost',pi=False,rate=10,fee=2,e0=1,selected=None):
    N=max(T+D,2); nodes=tree(N,p,rho,e0); leaves=[n for n in nodes if len(n)==N]
    prices=[1.6,1.1,.8,1.8,1.,1.3,1.4,1.]
    js=[('u',1,1,u,0),('v',1,2,v,0)]+[(str(t),t+1,t+D,0,1) for t in range(1,T+1)]
    vr={}; costs=[]; lower=[]; upper=[]; ints=[]
    def add(key,c=0,lo=0,hi=np.inf,integer=0):
        vr[key]=len(costs);costs.append(c);lower.append(lo);upper.append(hi);ints.append(integer)
    add(('r',),-1 if mode=='frontier' else 0,r if mode!='frontier' else 0,r if mode!='frontier' else 1)
    add(('z',),1 if mode=='reserve' else 0,0 if mode=='reserve' else z,np.inf if mode=='reserve' else z)
    contexts=[(leaf,nodes[leaf]) for leaf in leaves] if pi else [(None,1)]
    def exists(j,hist):
        return j in ('u','v') or hist[int(j)-1]==1
    for leaf,lp in contexts:
        nn=[leaf[:t] for t in range(1,N+1)] if pi else list(nodes)
        for hist in nn:
            t=len(hist); prob=lp if pi else nodes[hist]
            for j,a,d,w,beta in js:
                if a<=t<=d and exists(j,hist):
                    for k in (0,1):
                        price=prices[t-1] if k==0 else 4
                        add(('x',leaf,hist,j,k),prob*price if mode=='cost' else 0)
    for leaf in leaves:
        yl=1 if selected is not None and leaf in selected else 0
        if b==1:yl=1
        add(('y',leaf),-fee*nodes[leaf] if mode=='cost' else 0,yl,1,1)
    rows=[];lbs=[];ubs=[]
    def row(co,lo=-np.inf,hi=np.inf):
        rows.append({vr[k]:val for k,val in co.items() if val});lbs.append(lo);ubs.append(hi)
    for leaf,lp in contexts:
        nn=[leaf[:t] for t in range(1,N+1)] if pi else list(nodes)
        for hist in nn:
            t=len(hist)
            for k in (0,1):
                co={('x',leaf,hist,j,k):1 for j,a,d,w,be in js if ('x',leaf,hist,j,k) in vr}
                if k==1:co[('z',)]=-1
                row(co,hi=h*(1-hist[-1]) if k==0 else 0)
            for j,a,d,w,be in js:
                if ('x',leaf,hist,j,0) not in vr:continue
                row({('x',leaf,hist,j,k):1 for k in (0,1)},hi=rate)
                co={('x',leaf,hist[:tt],j,k):1 for tt in range(a,t+1) for k in (0,1)}
                co[('r',)]=-be;row(co,hi=w)
    for leaf in leaves:
        ctx=leaf if pi else None
        for j,a,d,w,be in js:
            if not exists(j,leaf):continue
            M=w+be
            co={('x',ctx,leaf[:tt],j,k):1 for tt in range(a,d+1) for k in (0,1)}
            co[('r',)]=-be;co[('y',leaf)]=-M
            row(co,lo=w-M)
    row({('y',leaf):nodes[leaf] for leaf in leaves},lo=b)
    A=lil_matrix((len(rows),len(vr)))
    for ii,row0 in enumerate(rows):
        for jj,val in row0.items():A[ii,jj]=val
    t0=time.perf_counter()
    res=milp(costs,integrality=ints,bounds=Bounds(lower,upper),constraints=LinearConstraint(csr_matrix(A),lbs,ubs),options={'time_limit':60,'mip_rel_gap':1e-9,'presolve':False})
    elapsed=time.perf_counter()-t0
    info={'status':int(res.status),'seconds':elapsed,'variables':len(vr),'rows':len(rows)}
    if res.x is None:return info
    xx=res.x;rr=xx[vr[('r',)]];zz=xx[vr[('z',)]]
    success=[];successmass=0
    for leaf in leaves:
        ctx=leaf if pi else None;ok=True
        for j,a,d,w,be in js:
            if not exists(j,leaf):continue
            got=sum(xx[vr[('x',ctx,leaf[:tt],j,k)]] for tt in range(a,d+1) for k in (0,1))
            ok &= got>=w+be*rr-1e-7
        if ok:success.append(leaf);successmass+=nodes[leaf]
    root=(e0,);early=sum(xx[vr[('x',None,root,'v',k)]] for k in (0,1)) if not pi else None
    info.update(value=float(res.fun+fee if mode=='cost' else (-res.fun if mode=='frontier' else res.fun)),
                dual=float(res.mip_dual_bound+fee if mode=='cost' else res.mip_dual_bound),r=rr,z=zz,
                success=successmass,early=early,mip_gap=float(res.mip_gap))
    if not pi:
        budgets=[]
        for hist,pr in nodes.items():
            mass=sum(nodes[ell] for ell in success if ell[:len(hist)]==hist)/pr
            budgets.append({'history':''.join(map(str,hist)),'budget':mass})
        info['budgets']=budgets
    info['success_leaves']=[''.join(map(str,k)) for k in success]
    return info

def debt_floor(T=3,D=2,p=.25,rho=.4,h=1,z=.4,u=.1,v=.5,b=.9,r=.4,e0=1,**kw):
    N=max(T+D,2); nodes=tree(N,p,rho,e0); vals=[]
    # Tail slots 2..N: the residual patient debt is already present, due at 2.
    for ell,prob in nodes.items():
        if len(ell)!=N:continue
        jobs=[(t+1,t+D,r) for t in range(1,T+1) if ell[t-1]]
        Q=np.inf;bad=False
        for a in range(2,N+1):
            for d in range(a,N+1):
                cap=sum(h*(1-ell[tt-1])+z for tt in range(a,d+1))
                need=sum(w for aa,dd,w in jobs if a<=aa and dd<=d)
                slack=cap-need
                if slack < -1e-8:bad=True
                if a==2:Q=min(Q,slack)
        vals.append((-np.inf if bad else Q,prob))
    cand=sorted(set(max(0,q) for q,pp in vals if q>=-1e-8))
    admiss=[q for q in cand if sum(pp for qq,pp in vals if qq>=q-1e-8)>=b-1e-10]
    if not admiss:return {'Q':None,'floor':None,'maxsuccess_zero':sum(pp for q,pp in vals if q>=-1e-8)}
    Q=max(admiss);return {'Q':Q,'floor':max(v-Q,0),'maximum_early':min(v,h*(1-e0)+z-u)}

def maxrun(hist):
    best=cur=0
    for x in hist:cur=cur+1 if x else 0;best=max(best,cur)
    return best

def selected_runs(args):
    nd=tree(args.get('T',3)+args.get('D',2),args.get('p',.25),args.get('rho',.4),args.get('e0',1))
    N=args.get('T',3)+args.get('D',2);leaves=[n for n in nd if len(n)==N]
    for k in range(1,N+1):
        ss={l for l in leaves if maxrun(l)<=k}
        if sum(nd[l] for l in ss)>=args.get('b',.9)-1e-10:return ss

def run_dynamic():
    cases=[('base',{}),('normal_now',{'e0':0}),('low_frequency',{'p':.15}),('high_frequency',{'p':.4}),('independent',{'rho':0}),('persistent',{'rho':.75}),('low_reserve',{'z':.35}),('high_reserve',{'z':.5}),('high_reliability',{'b':.97}),('low_reliability',{'b':.8}),('low_debt',{'v':.2}),('high_debt',{'v':.7}),('urgent_mix',{'u':.3,'v':.3}),('rate_limited',{'rate':.3}),('slack_one',{'D':1}),('slack_three',{'D':3})]
    out=[]
    for name,kw in cases:
        a={'T':3,'D':2,'p':.25,'rho':.4,'h':1.,'z':.4,'u':.1,'v':.5,'b':.9,'r':.4,'rate':10.,'fee':2.,'e0':1};a.update(kw)
        exact=recovery(**a)
        cal=recovery(**{**a,'b':1})
        sel=recovery(**a,selected=selected_runs(a))
        foresight=recovery(**a,pi=True)
        frontier=recovery(**a,mode='frontier')
        static=recovery(**{**a,'u':0,'v':0},mode='frontier')
        pathfront=recovery(**a,mode='frontier',pi=True)
        reserve=recovery(**a,mode='reserve')
        pireserve=recovery(**a,mode='reserve',pi=True)
        floor=debt_floor(**a) if a['rate']==10 else None
        rec={'name':name,'inputs':a,'dynamic':exact,'hard_optimum':cal,'selected':sel,'foresight':foresight,'frontier':frontier,'static':static,'pathfront':pathfront,'reserve':reserve,'pireserve':pireserve,'floor':floor}
        out.append(rec)
        print(name,{k:rec[k].get('value') for k in ['dynamic','hard_optimum','selected','foresight','frontier','static','pathfront','reserve','pireserve']},floor,flush=True)
        (OUT/'dynamic_results.json').write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':run_dynamic()
