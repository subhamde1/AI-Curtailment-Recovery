"""Compare the exchangeable-class corridor with independent shared-tree LPs."""
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def retain(leaves):
    return {leaf[:t] for leaf in leaves for t in range(len(leaf)+1)}


def ranks(leaves, resources, classes):
    """Before-service ranks and residual ranks for a retained tree."""
    nodes = retain(leaves)
    children = defaultdict(list)
    for node in nodes:
        if node:
            children[node[:-1]].append(node)
    before, after = {}, {}
    indices = np.arange(classes+1)
    for node in sorted(nodes, key=lambda n:(-len(n),n)):
        rate, capacity, _ = resources[node]
        branches = sorted(children[node])
        if not branches:
            residual = np.zeros(classes+1)
        elif any(before[child] is None for child in branches):
            before[node] = None
            continue
        else:
            raw = np.min([before[child]-indices*resources[child][2] for child in branches],axis=0)
            if np.min(raw)<-1e-10:
                before[node] = None
                continue
            residual = np.minimum.accumulate(np.maximum(0,raw)[::-1])[::-1]
        after[node] = residual
        before[node] = np.minimum(indices*rate,capacity)+residual
    return before[()],after.get(())


def prefixes(work):
    return np.r_[0.,np.cumsum(sorted(work,reverse=True))]


def balanced_residual(work, rate, capacity):
    """Allocate as much as possible while minimizing sorted residual prefixes."""
    floor = np.maximum(0.,work-rate)
    target = max(floor.sum(),work.sum()-capacity)
    if target <= floor.sum()+1e-12:
        return floor
    events = defaultdict(int)
    for lo,hi in zip(floor,work):
        events[lo] += 1
        events[hi] -= 1
    last, value, slope = 0.,floor.sum(),0
    for point,change in sorted(events.items()):
        next_value = value+slope*(point-last)
        if slope and target<=next_value+1e-12:
            return np.clip(last+(target-value)/slope,floor,work)
        last,value,slope = point,next_value,slope+change
    return work.copy()


def independent_lp(leaves, resources, classes, work=None, rank_size=0):
    """Node variables, prefix useful-service constraints and leaf balances."""
    nodes=sorted(retain(leaves),key=lambda n:(len(n),n))
    ix={(n,j):i*classes+j for i,n in enumerate(nodes) for j in range(classes)}
    service=len(ix)
    size=service+(classes if work is None else 0)
    bounds=[(0.,resources[n][0]) for n in nodes for _ in range(classes)]
    if work is None:
        bounds += [(0.,None)]*classes
    a,b,e,d=[],[],[],[]
    for node in nodes:
        a.append({ix[node,j]:1. for j in range(classes)})
        b.append(resources[node][1])
        for j in range(classes):
            row={ix[node[:t],j]:1. for t in range(len(node)+1)}
            arrivals=sum(resources[node[:t]][2] for t in range(1,len(node)+1))
            if work is None:
                row[service+j]=-1.
                rhs=arrivals
            else:
                rhs=arrivals+work[j]
            if node in leaves:
                e.append(row);d.append(rhs)
            else:
                a.append(row);b.append(rhs)

    def sparse(rows):
        i,j,v=[],[],[]
        for k,row in enumerate(rows):
            for col,x in row.items():
                i.append(k);j.append(col);v.append(x)
        return coo_matrix((v,(i,j)),shape=(len(rows),size)).tocsr()

    objective=np.zeros(size)
    if work is None:
        objective[service:service+rank_size]=-1.
    else:
        objective[:classes]=1.
    return linprog(objective,A_ub=sparse(a),b_ub=b,A_eq=sparse(e),b_eq=d,
                   bounds=bounds,method='highs')


def follows_rule(leaf,resources,work):
    residual=np.array(work,dtype=float)
    for t in range(len(leaf)+1):
        rate,capacity,arrival=resources[leaf[:t]]
        residual=balanced_residual(residual+arrival,rate,capacity)
    return residual.sum()<1e-8


def check_pivot(rng):
    """Independent slot variables for the asymmetric pivot family."""
    feasible=0
    for case in range(64):
        K=(3,4,6,8)[case%4]
        w=rng.uniform(0.,(1.25,1.6,2.)[case%3],K)
        w[0]=rng.uniform(0.,1.15)
        rates=rng.uniform(.3,1.2,K-1)
        capacity=float(rng.uniform(.2,K-1))
        floors=np.maximum(0,w[0]+w[1:]-1)
        formula=w[0]<=1 and np.all(floors<=rates) and np.all(np.arange(1,K)*w[0]+prefixes(w[1:])[1:]<=capacity+np.arange(1,K))
        count=K*K
        equalities=[];targets=[];inequalities=[];caps=[]
        row=np.zeros(count);row[1:K]=1.
        inequalities.append(row);caps.append(capacity)
        for branch in range(1,K):
            row=np.zeros(count);row[branch*K]=row[branch*K+branch]=1.
            inequalities.append(row);caps.append(1.)
            for j in range(K):
                row=np.zeros(count);row[j]=row[branch*K+j]=1.
                equalities.append(row);targets.append(w[j])
        result=linprog(np.zeros(count),A_ub=inequalities,b_ub=caps,A_eq=equalities,b_eq=targets,
                       bounds=[(0,0)]+[(0,u) for u in rates]+[(0,1)]*(K*(K-1)),method='highs')
        assert result.status in (0,2) and result.success==formula
        feasible+=int(result.success)
    return dict(pivot_lp_checks=64,feasible_pivot_cases=feasible)


def main():
    rng=np.random.default_rng(20261103)
    checked=rank_lp=individual_leaves=empty=feasible=0
    max_rank_error=max_service_error=0.
    for case in range(48):
        classes=(3,4,6,8)[case%4]
        depth=2+case%2
        width=2+case%3
        full=list(product(range(width),repeat=depth))
        leaves=[leaf for leaf in full if rng.random()<.65] or full[:1]
        nodes=retain(leaves)
        resources={n:(rng.integers(1,9)/5.,rng.integers(1,5*classes+1)/5.,
                      rng.integers(0,3)/5. if n and case%3 else 0.)
                   for n in sorted(nodes)}
        before,after=ranks(leaves,resources,classes)
        if before is None:
            empty+=1
        for k in range(1,classes+1):
            solve=independent_lp(leaves,resources,classes,rank_size=k)
            assert solve.status in (0,2)
            assert solve.success==(before is not None)
            if solve.success:
                max_rank_error=max(max_rank_error,abs(-solve.fun-before[k]))
            rank_lp+=1
        for scale in (.0,.4,.8,1.2,1.6):
            work=rng.uniform(0.,scale,size=classes)
            criterion=before is not None and np.all(prefixes(work)<=before+1e-9)
            solve=independent_lp(leaves,resources,classes,work=work)
            assert solve.status in (0,2) and solve.success==criterion
            assert all(follows_rule(leaf,resources,work) for leaf in leaves)==criterion
            if criterion:
                feasible+=1
                predicted=max(0.,float(np.max(prefixes(work)-after)))
                max_service_error=max(max_service_error,abs(predicted-solve.fun))
            if case<12:
                for leaf in leaves:
                    path=independent_lp([leaf],resources,classes,work=work)
                    assert path.status in (0,2)
                    assert path.success==follows_rule(leaf,resources,work)
                    individual_leaves+=1
            checked+=1
    assert max_rank_error<1e-8 and max_service_error<1e-8
    resources={(): (.4,.9,0.),(0,):(1.,3.,0.),(1,):(1.5,1.8,0.)}
    before,after=ranks([(0,),(1,)],resources,3)
    example=independent_lp([(0,),(1,)],resources,3,work=np.array([1.35,.65,.6]))
    assert np.allclose(before,[0,1.4,2.6,2.7]) and abs(example.fun-.8)<1e-10
    report=dict(trees=48,rank_lp_checks=rank_lp,root_workload_checks=checked,feasible_root_workloads=feasible,
                empty_trees=empty,individual_leaf_lp_checks=individual_leaves,
                max_rank_error=max_rank_error,max_minimum_service_error=max_service_error,
                crossing_example=dict(root_ranks=before.tolist(),residual_ranks=after.tolist(),minimum_service=example.fun))
    report.update(check_pivot(rng))
    (Path(__file__).resolve().parent/'symmetric_corridor_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
