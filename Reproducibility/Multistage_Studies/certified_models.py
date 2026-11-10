"""Shared-history optimization and violated-resource constraint generation.

The separate extensive-form solve is the numerical reference. Missing physical
capacity rows are separated over the entire declared tree, never just samples.
"""
from itertools import product
from time import perf_counter
import numpy as np
from scipy.optimize import milp, linprog, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, vstack

def mat(rows,n):
    ri=[];ci=[];da=[]
    for i,row in enumerate(rows):
        for j,v in row.items():
            if v:ri.append(i);ci.append(j);da.append(v)
    return coo_matrix((da,(ri,ci)),shape=(len(rows),n)).tocsr()

def descendants(n,birth):return n[:len(birth)]==birth

def model(nodes,jobs,b=1.,z=.2,mode='cost',foresight=False,fee=5.,mean_rows=False):
    leaves=[n for n,v in nodes.items() if not v['children']]
    edges=[];contexts=leaves if foresight else [None]
    for ctx in contexts:
        ns=[n for n in nodes if ctx is None or descendants(ctx,n)]
        for n in ns:
            t=nodes[n]['t']
            for j,w in enumerate(jobs):
                if w['release']<=t<=w['deadline'] and descendants(n,w.get('birth',())):
                    for k in w.get('elig',(0,1)):
                        edges.append((ctx,n,j,k))
    ix={e:i for i,e in enumerate(edges)};nv=len(edges)
    design=nv if mode in ('reserve','frontier') else None
    yi={l:nv+(design is not None)+i for i,l in enumerate(leaves)}
    N=nv+(design is not None)+len(leaves)
    objective=np.zeros(N);lb=np.zeros(N);ub=np.full(N,np.inf);integrality=np.zeros(N,int)
    for e,i in ix.items():
        ctx,n,j,k=e
        if mode=='cost':objective[i]=(nodes[ctx]['prob'] if ctx else nodes[n]['prob'])*nodes[n]['prices'][k]
        if mode=='current':objective[i]=nodes[n]['prices'][k] if len(n)==1 else 0.
    if design is not None:objective[design]=1 if mode=='reserve' else -1;ub[design]=100 if mode=='reserve' else 1
    for l,i in yi.items():
        ub[i]=1;lb[i]=1 if b>=1-1e-12 else 0;integrality[i]=0 if b>=1-1e-12 else 1
        if mode=='cost':objective[i]=-fee*nodes[l]['prob']
    rows=[];lower=[];upper=[];resource=[];meta=[]
    def add(row,L=-np.inf,U=np.inf):rows.append(row);lower.append(L);upper.append(U)
    add({yi[l]:nodes[l]['prob'] for l in leaves},b,np.inf)
    for ctx in contexts:
        ns=[n for n in nodes if ctx is None or descendants(ctx,n)]
        for n in ns:
            t=nodes[n]['t']
            lineages={}
            for j,w in enumerate(jobs):
                if not (w['release']<=t<=w['deadline'] and descendants(n,w.get('birth',()))):continue
                row={ix[ctx,n,j,k]:1 for k in w.get('elig',(0,1))}
                add(row,U=w['rate'])
                lin=w.get('lineage',('job',j));lineages.setdefault(lin,{}).update(row)
                # Useful service at every shared node, including unselected paths.
                cum={ix[ctx,a,j,k]:1 for a in ns if descendants(n,a) for k in w.get('elig',(0,1)) if (ctx,a,j,k) in ix}
                wc=w.get('coef',0);w0=w['work']
                if mode=='frontier':cum[design]=-wc;bound=w0
                else:bound=w0+wc*w.get('r',0)
                add(cum,U=bound)
            for lin,row in lineages.items():
                group=[w for j,w in enumerate(jobs) if w.get('lineage',('job',j))==lin]
                add(row,U=min(w['rate'] for w in group))
            for k in (0,1):
                row={ix[ctx,n,j,k]:1 for j in range(len(jobs)) if (ctx,n,j,k) in ix}
                if mode=='reserve' and k==1:row[design]=-1;cap=0
                else:cap=nodes[n]['local'] if k==0 else z
                resource.append(row);meta.append((ctx,n,k,cap))
        for leaf in ([ctx] if ctx else leaves):
            for j,w in enumerate(jobs):
                if not descendants(leaf,w.get('birth',())):continue
                cum={ix[ctx,n,j,k]:1 for n in ns if descendants(leaf,n) and nodes[n]['t']<=w['deadline'] for k in w.get('elig',(0,1)) if (ctx,n,j,k) in ix}
                wc=w.get('coef',0);w0=w['work'];big=w0+wc if mode=='frontier' else w0+wc*w.get('r',0)
                if mode=='frontier':cum[design]=-wc;right=w0-big
                else:right=big-big
                cum[yi[leaf]]=-big;add(cum,L=right)
    if mean_rows and not foresight:
        for t in sorted({v['t'] for v in nodes.values()}):
            for e in sorted({v['state'] for v in nodes.values() if v['t']==t}):
                ns=[n for n,v in nodes.items() if v['t']==t and v['state']==e]
                for k in (0,1):
                    row={ix[None,n,j,k]:nodes[n]['prob'] for n in ns for j in range(len(jobs)) if (None,n,j,k) in ix}
                    cap=sum(nodes[n]['prob']*(nodes[n]['local'] if k==0 else z) for n in ns)
                    add(row,U=cap)
    return dict(nodes=nodes,jobs=jobs,edges=edges,ix=ix,yi=yi,design=design,N=N,c=objective,lb=lb,ub=ub,integer=integrality,A=mat(rows,N),L=np.array(lower),U=np.array(upper),R=mat(resource,N),meta=meta,resource_upper=np.array([q[3] for q in meta]),constant=fee if mode=='cost' else 0,mode=mode)

def solve(M,kept=None,root_fixed=None,limit=60):
    kept=range(len(M['meta'])) if kept is None else kept
    A=vstack([M['A'],M['R'][list(kept)]],format='csr')
    L=np.r_[M['L'],np.full(len(kept),-np.inf)];U=np.r_[M['U'],M['resource_upper'][list(kept)]]
    if root_fixed:
        fixed=[];rhs=[]
        for key,val in root_fixed.items():
            n,j,k=key
            fixed.append({M['ix'][None,n,j,k]:1});rhs.append(val)
        A=vstack([A,mat(fixed,M['N'])],format='csr');L=np.r_[L,rhs];U=np.r_[U,rhs]
    t0=perf_counter()
    ans=milp(M['c'],integrality=M['integer'],bounds=Bounds(M['lb'],M['ub']),constraints=LinearConstraint(A,L,U),options={'time_limit':limit,'mip_rel_gap':1e-9})
    elapsed=perf_counter()-t0
    if ans.success:
        lower=float(getattr(ans,'mip_dual_bound',ans.fun)+M['constant']) if getattr(ans,'mip_dual_bound',None) is not None else float(ans.fun+M['constant'])
        info=dict(status=int(ans.status),value=float(ans.fun+M['constant']),lower=lower,seconds=elapsed,variables=M['N'],rows=A.shape[0],resource_rows=len(M['meta']),max_capacity_violation=float(max(0,np.max(M['R']@ans.x-M['resource_upper']))),mip_gap=float(getattr(ans,'mip_gap',0) or 0))
    else:info=dict(status=int(ans.status),seconds=elapsed,message=ans.message)
    return info,ans

def constraint_generation(M,batch=8,max_rounds=100,limit=60):
    kept=[];history=[];total=0
    for it in range(max_rounds):
        info,ans=solve(M,kept,limit=limit);total+=info['seconds']
        if not ans.success:return dict(status=info['status'],history=history,seconds=total),ans
        violation=M['R']@ans.x-M['resource_upper']
        violated=[i for i,v in enumerate(violation) if v>1e-8 and i not in kept]
        history.append(dict(round=it,lower=info['lower'],value=info['value'],retained=len(kept),max_violation=float(max(0,violation.max()))))
        if not violated:
            info.update(history=history,rounds=it+1,retained_rows=len(kept),total_seconds=total,certified_gap=max(0,info['value']-info['lower']));return info,ans
        # Largest normalized physical shortage, with stable deterministic ties.
        violated.sort(key=lambda i:(-violation[i]/max(.1,M['resource_upper'][i]),i))
        kept.extend(violated[:batch])
    return dict(status=1,history=history,total_seconds=total),ans

def two_class_instance(r=.4,p=.25,rho=.4,rate_scale=1.,slack=0,z=.15,urgency=.35,eligibility='mixed',debt=(.25,.3),h=.9):
    q=p*(1-rho);alpha=p+(1-p)*rho;P=np.array([[1-q,q],[1-alpha,alpha]])
    H=4+slack;ds=[3+slack,H];nodes={}
    for t in range(1,H+1):
        if t==1:words=[(0,)]
        elif t<=3:words=[(0,)+w for w in product((0,1),repeat=t-1)]
        else:
            words=[(0,a,b)+(b,)*(t-3) for a,b in product((0,1),repeat=2)]
        for n in words:
            prob=1 if t==1 else P[0,n[1]]*(P[n[1],n[2]] if t>=3 else 1)
            children=[n+(e,) for e in ((0,1) if t<3 else (n[-1],))] if t<H else []
            rates=np.array([.55,.45])*rate_scale;rates=np.array([rates[j] if t<=ds[j] else 0 for j in (0,1)])
            elig=[[0],[0,1]] if eligibility=='mixed' else [[0,1],[0,1]]
            arr=(np.array([urgency,1-urgency])*r if t==3 and n[1]==1 else np.zeros(2))
            nodes[n]=dict(t=t,state=n[-1],prob=float(prob),children=children,local=h*(1-n[-1]),prices=[(1.2,3,.7,3.5,1.1,2.2)[(t-1)%6],4.],arrival=arr.tolist(),rates=rates.tolist(),elig=elig,deadlines=ds)
    jobs=[]
    for j in (0,1):
        jobs.append(dict(id=f'initial{j}',release=1,deadline=ds[j],work=debt[j],coef=urgency if j==0 else 1-urgency,r=r,rate=(.55 if j==0 else .45)*rate_scale,elig=nodes[(0,)]['elig'][j],lineage=j,birth=()))
        jobs.append(dict(id=f'call{j}',release=3,deadline=ds[j],work=0,coef=urgency if j==0 else 1-urgency,r=r,rate=(.55 if j==0 else .45)*rate_scale,elig=nodes[(0,)]['elig'][j],lineage=j,birth=(0,1)))
    return nodes,jobs
