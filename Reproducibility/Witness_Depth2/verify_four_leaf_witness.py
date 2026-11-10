"""Independent cumulative-service LP and exact policy audit for Theorem 13c."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import json
import numpy as np
from scipy.optimize import linprog

N={"r":dict(parent=None,a=(4,4),h=1,g=(0,0))}
for branch in "IJK":
    N[branch]=dict(parent="r",a={"I":(0,2),"J":(2,0),"K":(4,4)}[branch],h=2 if branch!="K" else 6,g=(0,0))
    N[branch+"d"]=dict(parent=branch,a=(3,3),h=1 if branch!="K" else 6,g={"I":(0,2),"J":(2,0),"K":(0,0)}[branch])
    for j in "abc":
        a={"a":(1,3),"b":(3,1),"c":(3,3)}[j] if branch!="K" else (3,3)
        N[branch+j]=dict(parent=branch+"d",a=a,h=sum(a),g=(0,0))
LEAVES=tuple(b+j for b in "IJK" for j in "abc")
CRITICAL=("Ia","Ib","Ja","Jb")

def path(n):
    a=[]
    while n is not None:a.append(n);n=N[n]['parent']
    return tuple(reversed(a))

def lp(leaves,s=F(8,5),z=None):
    nodes=sorted({n for l in leaves for n in path(l)},key=lambda n:len(path(n)))
    pos={(n,j):2*i+j for i,n in enumerate(nodes) for j in range(2)}
    dim=2*len(nodes)+(z is None);c=np.zeros(dim)
    if z is None:c[-1]=1
    A=[];b=[];E=[];d=[];bounds=[]
    for n in nodes:
        bounds.extend((0,a) for a in N[n]['a'])
        row=np.zeros(dim);row[pos[n,0]]=row[pos[n,1]]=1
        if z is None:row[-1]=-1
        A.append(row);b.append(N[n]['h']+(0 if z is None else float(z)))
        for j in range(2):
            row=np.zeros(dim)
            for k in path(n):row[pos[k,j]]=1
            total=s+sum(N[k]['g'][j] for k in path(n))
            A.append(row);b.append(float(total))
            if n in leaves:E.append(row);d.append(float(total))
    if z is None:bounds.append((0,None))
    a=np.array(A);bb=np.array(b);e=np.array(E);dd=np.array(d)
    r=linprog(c,A_ub=a,b_ub=bb,A_eq=e,b_eq=dd,bounds=bounds,method='highs')
    if not r.success:return dict(feasible=False,status=int(r.status))
    return dict(feasible=True,reserve=float(r.fun) if z is None else float(z),
                max_inequality_violation=float(max(0,np.max(a@r.x-bb))),
                max_equality_error=float(np.max(np.abs(e@r.x-dd))),
                actions={n:[float(r.x[pos[n,j]]) for j in range(2)] for n in nodes})

def ranks(leaves,z):
    z=F(z);nodes={n for l in leaves for n in path(l)};f={}
    for n in sorted(nodes,key=lambda n:len(path(n)),reverse=True):
        a,b=N[n]['a'];h=F(N[n]['h'])+z
        q=(min(F(a),h),min(F(b),h),min(F(a+b),h))
        children=[k for k in nodes if N[k]['parent']==n]
        if not children:R=(F(0),)*3
        else:
            raw=tuple(min(f[k][j]-F(N[k]['g'][j] if j<2 else sum(N[k]['g'])) for k in children) for j in range(3))
            if min(raw)<0:return None
            aa=min(raw[0],raw[2]);bb=min(raw[1],raw[2]);R=(aa,bb,min(raw[2],aa+bb))
        f[n]=tuple(q[j]+R[j] for j in range(3))
    return f

def exact_policy(s):
    t=2*s/3;z=t-1
    x={'r':(s/3,s/3),'I':(F(0),t),'J':(t,F(0)),'K':(F(0),F(0)),
       'Id':(t-1,F(1)),'Jd':(F(1),t-1),'Kd':(F(0),F(0))}
    for l in LEAVES:x[l]=(t,t) if l.startswith('K') else (F(1),F(1))
    for n,service in x.items():
        assert all(0<=service[j]<=N[n]['a'][j] for j in range(2))
        assert sum(service)<=N[n]['h']+z
        residual=tuple(s+sum(N[k]['g'][j] for k in path(n))-sum(x[k][j] for k in path(n)) for j in range(2))
        assert min(residual)>=0
        if n in LEAVES:assert residual==(0,0)
    return dict(all_16_node_balances_exact=True,actions={n:[str(v) for v in a] for n,a in x.items()})

def run():
    cases=[]
    for size in range(1,4):
        for subset in combinations(LEAVES,size):
            v=lp(subset,z=0);assert v['feasible'],subset
            f=ranks(subset,0)['r'];assert F(8,5)<=min(f[:2]) and F(16,5)<=f[2]
            cases.append(dict(leaves=subset,violation=v['max_inequality_violation']))
    full=lp(LEAVES);support=lp(CRITICAL)
    assert abs(full['reserve']-1/15)<1e-9 and abs(support['reserve']-1/15)<1e-9
    assert not lp(LEAVES,z=0)['feasible']
    deletions={l:lp(tuple(k for k in LEAVES if k!=l)) for l in CRITICAL}
    assert all(v['reserve']<1e-9 for v in deletions.values())
    family=[]
    for s in [F(3,2),F(8,5),F(7,4),F(19,10),F(2)]:
        expected=max(F(0),(2*s-3)/3);v=lp(LEAVES,s=s)
        assert abs(v['reserve']-float(expected))<1e-9
        for l in CRITICAL:assert lp(tuple(k for k in LEAVES if k!=l),s=s,z=0)['feasible']
        family.append(dict(class_work=str(s),exact_reserve=str(expected),LP_reserve=v['reserve'],exact_policy=exact_policy(s)))
    for z in [F(0),F(1,30),F(1,15),F(1,5),F(1)]:
        f=ranks(LEAVES,z)
        assert f['I']==(1+z,2+z,3+z) and f['J']==(2+z,1+z,3+z)
        assert f['r']==(2+2*z,2+2*z,3+3*z)
    result=dict(nodes=16,leaves=9,branching_depth=2,exact_reserve='1/15',critical_leaves=CRITICAL,
                proper_subtrees_checked=len(cases),max_proper_subtree_violation=max(c['violation'] for c in cases),
                full_tree=full,four_leaf_support=support,critical_leaf_deletions=deletions,
                family_checks=family,rank_checks=5,subtree_checks=cases,node_inputs=N)
    Path(__file__).with_name('four_leaf_witness_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['nodes','leaves','branching_depth','exact_reserve','proper_subtrees_checked','max_proper_subtree_violation']}))
if __name__=='__main__':run()
