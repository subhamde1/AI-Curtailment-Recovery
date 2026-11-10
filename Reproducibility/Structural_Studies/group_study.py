import itertools, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix,csr_matrix
from studies import chain

OUT=Path(__file__).parent

def instance(p=.25,rho=.4,rate=.4,urgent=False,h=1.2,z=.8,parts=None,foresight=False):
    P=chain(p,rho); pi=np.array([1-p,p]);nodes={}
    for t in range(1,4):
        for hist in itertools.product((0,1),repeat=t):
            nodes[hist]=float(pi[hist[0]]*np.prod([P[hist[i-1],hist[i]] for i in range(1,t)]))
    jobs=[{'w':.6,'a':1,'d':3,'rate':rate} for _ in range(4)]
    if urgent: jobs[0].update(w=.4,d=2);jobs[3]['w']=.8
    if parts is None:parts=[[0,1,2,3]]
    leaves=[n for n in nodes if len(n)==3]
    contexts=leaves if foresight else [None]
    vs={};cost=[]
    prices=[2.5,1.,.5]
    for ell in contexts:
        ns=[ell[:t] for t in range(1,4)] if ell else list(nodes)
        for hist in ns:
            for j,job in enumerate(jobs):
                if len(hist)>job['d']:continue
                for k in (0,1):
                    vs[(ell,hist,j,k)]=len(cost)
                    cost.append((nodes[ell] if ell else nodes[hist])*(prices[len(hist)-1] if k==0 else 4))
    rows=[];rhs=[];eq=[];beq=[]
    def add(mapping,b,equality=False):
        (eq if equality else rows).append({vs[k]:v for k,v in mapping.items()});(beq if equality else rhs).append(b)
    for ell in contexts:
        ns=[ell[:t] for t in range(1,4)] if ell else list(nodes)
        for hist in ns:
            for j,job in enumerate(jobs):
                if len(hist)>job['d']:continue
                add({(ell,hist,j,k):1 for k in (0,1)},job['rate'])
                add({(ell,hist[:t],j,k):1 for t in range(1,len(hist)+1) for k in (0,1)},job['w'])
            for group in parts:
                for k in (0,1):
                    add({(ell,hist,j,k):1 for j in group if (ell,hist,j,k) in vs},h*(1-hist[-1]) if k==0 else z)
        for leaf in ([ell] if ell else leaves):
            for j,job in enumerate(jobs):
                add({(ell,leaf[:t],j,k):1 for t in range(1,job['d']+1) for k in (0,1)},job['w'],True)
    # Dualizing these conditional-mean constraints gives Markov-state resource prices.
    if not foresight:
        for t in range(1,4):
            for e in (0,1):
                for k in (0,1):
                    histset=[n for n in nodes if len(n)==t and n[-1]==e]
                    add({(None,n,j,k):nodes[n] for n in histset for j in range(4) if (None,n,j,k) in vs},
                        sum(nodes[n] for n in histset)*(h*(1-e) if k==0 else z))
    def sparse(rr):
        A=lil_matrix((len(rr),len(vs)))
        for i,row in enumerate(rr):
            for j,val in row.items():A[i,j]=val
        return csr_matrix(A)
    t0=time.perf_counter();res=linprog(cost,A_ub=sparse(rows),b_ub=rhs,A_eq=sparse(eq),b_eq=beq,bounds=(0,None),method='highs')
    info={'status':int(res.status),'seconds':time.perf_counter()-t0,'variables':len(vs),'rows':len(rows)+len(eq)}
    if res.success:info.update(value=res.fun,dual=float(np.dot(res.ineqlin.marginals,rhs)+np.dot(res.eqlin.marginals,beq)),max_violation=float(max(0,-res.ineqlin.residual.min(),np.abs(res.eqlin.residual).max())))
    return info

def calendar(p=.25,rho=.4,rate=.4,urgent=False,h=1.2,z=.8):
    P=chain(p,rho);pi=[1-p,p];prices=[2.5,1,.5];jobs=[(.6,3,rate)]*4
    if urgent:jobs=[(.4,2,rate),(.6,3,rate),(.6,3,rate),(.8,3,rate)]
    vals=[]
    for hist in itertools.product((0,1),repeat=3):
        rem=np.array([j[0] for j in jobs]);total=0
        for t in range(1,4):
            vs=[(j,s,k) for j,(_,d,a) in enumerate(jobs) for s in range(t,d+1) for k in (0,1)]
            A=[];b=[];E=[];f=[];c=[];bounds=[]
            for j,s,k in vs:
                c.append((prices[t-1] if k==0 else 4)+1e-8*(4-s))
                bounds.append((0,None))
            for j,(_,d,a) in enumerate(jobs):
                E.append([int(jj==j) for jj,s,k in vs]);f.append(rem[j])
                for s in range(t,d+1):A.append([int(jj==j and ss==s) for jj,ss,k in vs]);b.append(a)
            for s in range(t,4):
                for k in (0,1):
                    A.append([int(ss==s and kk==k) for jj,ss,kk in vs])
                    b.append(h*(1-hist[t-1]) if k==0 and s==t else (0 if k==0 else z))
            rr=linprog(c,A_ub=A,b_ub=b,A_eq=E,b_eq=f,bounds=bounds,method='highs')
            if not rr.success:return {'status':rr.status}
            for val,(j,s,k) in zip(rr.x,vs):
                if s==t:rem[j]-=val;total+=val*(prices[t-1] if k==0 else 4)
        prob=pi[hist[0]]*P[hist[0],hist[1]]*P[hist[1],hist[2]]
        vals.append((total,float(prob)))
    return {'status':0,'value':sum(c*p for c,p in vals)}

def run():
    rows=[]
    for p,rho,rate,urgent,h,z in [(p,rho,.4,False,1.2,.8) for p in [.15,.25,.4] for rho in [0,.4,.75]]+[(.25,.4,.35,False,1.2,.8),(.25,.4,.5,False,1.2,.8),(.25,.4,.4,True,1.2,.8),(.25,.4,.4,False,1.2,1.0),(.25,.4,.4,False,1.0,.8)]:
        args=dict(p=p,rho=rho,rate=rate,urgent=urgent,h=h,z=z)
        vals={nm:instance(**args,parts=part) for nm,part in [('single',[[0],[1],[2],[3]]),('pairs',[[0,1],[2,3]]),('triple',[[0,1,2],[3]]),('full',[[0,1,2,3]])]}
        vals['foresight']=instance(**args,foresight=True);vals['calendar']=calendar(**args)
        rows.append({'inputs':args,**vals});print(args,{k:v.get('value') for k,v in vals.items()},flush=True)
    (OUT/'group_results.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':run()
