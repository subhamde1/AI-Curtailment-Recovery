"""Common finite-tree service, reserve and cost models for the final revision.

Indices are zero based. All physical constraints remain on failed leaves.
The full solver policy is executable from the observed history. Failure is
recorded at the first missed audited deadline; optional recovery then stops.
"""
from collections import defaultdict
from itertools import product
from time import perf_counter
import math
import warnings
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from revision_algorithms import matrix, solve_lp, make_tree
from compute_extensions import prices


def class_tree(base, classes, T, H, current, P, rmax=1.2):
    leaves=[(current,)+w for w in product((0,1),repeat=H-1)]
    nodes=sorted({ell[:t+1] for ell in leaves for t in range(H)},key=lambda n:(len(n),n))
    probability={(current,):1.}
    for n in nodes:
        if len(n)>1:probability[n]=probability[n[:-1]]*P[n[-2],n[-1]]
    jobs={('base',i):dict(j,birth=(),w0=j['work'],beta=0.) for i,j in enumerate(base)}
    for birth in nodes:
        origin=len(birth)-1
        if origin>=T or not birth[-1]:continue
        for k,c in enumerate(classes):
            jobs[('call',birth,k)]=dict(id=('call',birth,k),birth=birth,
                release=origin+1,deadline=origin+c['D'],rate=c['rate'],
                channels=tuple(c.get('channels',(0,1))),w0=0.,beta=c['beta'])
    assert max(j['deadline'] for j in jobs.values())<H
    return dict(leaves=leaves,nodes=nodes,probability=probability,jobs=jobs,
                H=H,T=T,rmax=rmax)


def run_bound_set(tree,epsilon):
    """Predeclared success set: smallest maximum stress run attaining target mass."""
    def run(ell):
        longest=now=0
        for e in ell:
            now=now+1 if e else 0;longest=max(longest,now)
        return longest
    for L in range(tree['H']+1):
        selected=[i for i,ell in enumerate(tree['leaves']) if run(ell)<=L]
        mass=sum(tree['probability'][tree['leaves'][i]] for i in selected)
        if mass>=1-epsilon-1e-12:return selected,L,mass
    raise AssertionError('complete support must be feasible as a probability set')


def service_program(tree,h,reserve=None,r=None,epsilon=0.,objective='cost',
                    penalty=0.,selected=None,time_limit=30.,keep_policy=False):
    """Exact deterministic-causal accreditation/minimum reserve/expected cost.

Cost penalizes one failed episode, not each job or missing work unit. At a
positive penalty, an optimum marks every actually successful leaf as success.
At zero penalty flags may understate success, so actual success is recomputed.
"""
    if objective=='frontier':assert r is None and reserve is not None
    if objective=='reserve':assert reserve is None and r is not None
    if objective=='cost':assert reserve is not None and r is not None
    if selected is not None:assert penalty==0., 'Evaluate actual failure fees after fixing a success set'
    leaves=tree['leaves'];nodes=tree['nodes'];jobs=tree['jobs'];H=tree['H']
    edges=[(n,jid,k) for n in nodes for jid,j in jobs.items()
           if j['release']<=len(n)-1<=j['deadline']
           and (not j['birth'] or n[:len(j['birth'])]==j['birth'])
           for k in j.get('channels',(0,1))]
    ix={e:i for i,e in enumerate(edges)};nv=len(edges)
    rv=nv if r is None else None;nv+=int(r is None)
    zv=nv if reserve is None else None;nv+=int(reserve is None)
    y={ell:nv+i for i,ell in enumerate(leaves)};nv+=len(y)
    rows=[];rhs=[];bycap=defaultdict(dict);byrate=defaultdict(dict)
    for (n,jid,k),i in ix.items():bycap[n,k][i]=1.;byrate[n,jid][i]=1.
    for n in nodes:
        for k in (0,1):
            row=dict(bycap[n,k]);cap=h*(1-n[-1]) if k==0 else reserve
            if k==1 and zv is not None:row[zv]=-1.;cap=0.
            rows.append(row);rhs.append(cap)
    for (n,jid),row in byrate.items():
        j=jobs[jid];rows.append(row);rhs.append(j['rate'])
        prefix={ix[n[:t+1],jid,k]:1. for t in range(j['release'],len(n))
                for k in j.get('channels',(0,1))}
        if rv is not None:prefix[rv]=-j['beta']
        rows.append(prefix);rhs.append(j['w0']+(0. if rv is not None else j['beta']*r))
    for ell in leaves:
        for jid,j in jobs.items():
            if j['birth'] and ell[:len(j['birth'])]!=j['birth']:continue
            row={ix[ell[:t+1],jid,k]:-1. for t in range(j['release'],j['deadline']+1)
                 for k in j.get('channels',(0,1))}
            M=j['w0']+j['beta']*tree['rmax'];row[y[ell]]=M
            if rv is not None:row[rv]=j['beta']
            rows.append(row);rhs.append(M-j['w0']-(0. if rv is not None else j['beta']*r))
    rows.append({y[ell]:-tree['probability'][ell] for ell in leaves});rhs.append(-(1-epsilon))
    c=np.zeros(nv);constant=0.
    if objective=='cost':
        for (n,jid,k),i in ix.items():c[i]=tree['probability'][n]*prices(len(n)-1,n[-1])[k]
        for ell,i in y.items():c[i]=-penalty*tree['probability'][ell]
        constant=penalty
    elif objective=='frontier':c[rv]=-1.
    else:c[zv]=1.
    lower=np.zeros(nv);upper=np.full(nv,np.inf)
    if rv is not None:upper[rv]=tree['rmax']
    if zv is not None:upper[zv]=10.
    for i in y.values():upper[i]=1.
    if selected is not None:
        chosen={leaves[i] for i in selected}
        for ell,i in y.items():lower[i]=upper[i]=float(ell in chosen)
    elif epsilon==0.:
        for i in y.values():lower[i]=upper[i]=1.
    A=matrix(rows,nv);start=perf_counter()
    if selected is not None or epsilon==0.:
        ans=solve_lp(c,rows,rhs,bounds=[(lo,None if np.isinf(hi) else hi) for lo,hi in zip(lower,upper)])
        bound=float(ans.fun)+constant if ans.success else None
    else:
        integrality=np.zeros(nv)
        for i in y.values():integrality[i]=1.
        # SciPy forwards solver-specific feasibility options to HiGHS.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            ans=milp(c,integrality=integrality,bounds=Bounds(lower,upper),
                constraints=LinearConstraint(A,-np.inf,rhs),
                options={'mip_rel_gap':1e-9,'time_limit':time_limit,
                         'presolve':False,
                         'mip_feasibility_tolerance':1e-9,
                         'primal_feasibility_tolerance':1e-9,
                         'dual_feasibility_tolerance':1e-9})
        bound=getattr(ans,'mip_dual_bound',None)
        if bound is not None:bound=float(bound)+constant
    elapsed=perf_counter()-start
    record=dict(status=int(ans.status),seconds=elapsed,variables=nv,constraints=len(rows),
        matrix_bytes=int(A.data.nbytes+A.indices.nbytes+A.indptr.nbytes),
        objective=objective,epsilon=epsilon,penalty=penalty)
    if ans.x is None:return dict(record,feasible=False,solver_message=ans.message)
    R=float(ans.x[rv]) if rv is not None else r
    Z=float(ans.x[zv]) if zv is not None else reserve
    violation=float(np.max(A@ans.x-np.asarray(rhs)))
    assert violation<1e-7,(objective,violation)
    selected_mass=sum(tree['probability'][ell] for ell,i in y.items() if ans.x[i]>.5)
    assert selected_mass>=1-epsilon-1e-7
    leaf_records=[];actual_success=0.;processing=0.
    for ell in leaves:
        first_failure=None
        for jid,j in jobs.items():
            if j['birth'] and ell[:len(j['birth'])]!=j['birth']:continue
            got=sum(ans.x[ix[ell[:t+1],jid,k]] for t in range(j['release'],j['deadline']+1)
                    for k in j.get('channels',(0,1)))
            if got<j['w0']+j['beta']*R-1e-7:
                first_failure=j['deadline'] if first_failure is None else min(first_failure,j['deadline'])
        cost=0.
        for t in range(H):
            if objective=='cost' and first_failure is not None and t>first_failure:continue
            for jid,j in jobs.items():
                for k in j.get('channels',(0,1)):
                    edge=(ell[:t+1],jid,k)
                    if edge in ix:cost+=ans.x[ix[edge]]*prices(t,ell[t])[k]
        p=tree['probability'][ell];processing+=p*cost
        if first_failure is None:actual_success+=p
        leaf_records.append(dict(events=list(ell),probability=p,success=first_failure is None,
                                 selected_success=bool(ans.x[y[ell]]>.5),
                                 first_missed_deadline=first_failure,processing_cost=float(cost)))
    total=processing+penalty*(1-actual_success)
    value=float(ans.fun)+constant
    if objective=='cost':
        if abs(value-total)>=2e-6:
            for ell,i in y.items():
                rec=next(q for q in leaf_records if q['events']==list(ell))
                if rec['success']!=(ans.x[i]>.5):
                    shortages=[j['w0']+j['beta']*R-sum(ans.x[ix[ell[:t+1],jid,k]]
                        for t in range(j['release'],j['deadline']+1) for k in j.get('channels',(0,1)))
                        for jid,j in jobs.items() if not j['birth'] or ell[:len(j['birth'])]==j['birth']]
                    print('Mismatched success:',rec,'max shortage',max(shortages),flush=True)
        assert abs(value-total)<2e-6,(value,total)
        assert bound is not None and bound<=total+2e-6
    output=dict(record,feasible=True,completed=bool(ans.success),r=float(R),reserve=float(Z),
        maximum_physical_residual=violation,
        value=value,solver_lower_bound=bound,selected_success=float(selected_mass),
        actual_success=float(actual_success),processing_cost=float(processing),
        failure_cost=float(penalty*(1-actual_success)),policy_cost=float(total),
        policy_gap=max(0.,float(total-bound)) if objective=='cost' else None,
        leaf_records=leaf_records)
    if keep_policy:
        output['node_actions']=[dict(history=list(n),job=repr(jid),channel=k,service=float(ans.x[i]))
                                for (n,jid,k),i in ix.items() if ans.x[i]>1e-10]
        output['success_flags']=[list(ell) for ell,i in y.items() if ans.x[i]>.5]
    return output


def two_job_cuts(calendars,jobs,z=0.):
    """Closed-form projected first-slot cuts for the two-branch family."""
    assert len(jobs)==2 and len(calendars)==2
    u=np.array([min(j['work'],j['rate']) for j in jobs]);lower=[]
    for capacities in calendars:
        row=[]
        for subset in ((0,),(1,),(0,1)):
            tail=sum(min(capacities[t]+z,sum(jobs[j]['rate'] for j in subset
                         if t<=jobs[j]['deadline'])) for t in range(1,len(capacities)))
            row.append(max(0.,sum(jobs[j]['work'] for j in subset)-tail))
        lower.append(row)
    ell=np.max(lower,axis=0);root=calendars[0][0]+z
    feasible=(ell[0]<=u[0]+1e-10 and ell[1]<=u[1]+1e-10 and
              max(ell[0]+ell[1],ell[2])<=min(root,sum(u))+1e-10)
    leaf_feasible=[q[0]<=u[0]+1e-10 and q[1]<=u[1]+1e-10 and
        max(q[0]+q[1],q[2])<=min(root,sum(u))+1e-10 for q in lower]
    return dict(feasible=bool(feasible),leaf_feasible=list(map(bool,leaf_feasible)),
                lower=ell.tolist(),branch_lower=lower,root_capacity=root,
                shared_conflict=bool(all(leaf_feasible) and ell[0]+ell[1]>root+1e-10))
