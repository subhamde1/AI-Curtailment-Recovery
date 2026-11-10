"""Joint service/cost, exact reserve boundary, and coordinated AI studies."""
from pathlib import Path
from itertools import product
import argparse,csv,json,math,copy
from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from compute_extensions import transition
from revision_algorithms import make_tree,reserve_identity,operate_lookahead,calendar_lookahead
from final_algorithms import service_program,run_bound_set,class_tree,two_job_cuts

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'final_data';DATA.mkdir(exist_ok=True)

def save(name,obj):
    (DATA/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def flat_csv(name,rows):
    keys=list(dict.fromkeys(k for q in rows for k,v in q.items() if not isinstance(v,(list,dict))))
    with (DATA/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader()
        w.writerows({k:v for k,v in q.items() if k in keys} for q in rows)

def joint():
    P=transition(.2,.6);base=[dict(id=('base',0),release=0,deadline=1,work=.6,rate=.4),
                             dict(id=('base',1),release=0,deadline=2,work=.8,rate=.3)]
    T,H=3,6;h=1.2;tree=make_tree(base,T,H,0,P,rmax=1.2)
    R=(.25,.5,.75,1.);Z=(0.,.15,.3,.45,.6,.9);E=(0.,.05,.1,.2)
    frontiers=[];reserve=[];costs=[];start=perf_counter()
    for z,eps in product(Z,E):
        q=service_program(tree,h,z,epsilon=eps,objective='frontier')
        frontiers.append({**q,'z':z,'epsilon':eps})
    save('joint_frontiers',frontiers);flat_csv('joint_frontiers',frontiers)
    for r,eps in product(R,E):
        q=service_program(tree,h,r=r,epsilon=eps,objective='reserve')
        reserve.append({**q,'commitment':r,'epsilon':eps})
    save('joint_reserves',reserve);flat_csv('joint_reserves',reserve)
    for r,z,eps in product(R,Z,E):
        selected,L,mass=run_bound_set(tree,eps)
        fixed=service_program(tree,h,z,r,eps,selected=selected,keep_policy=True)
        calendar=None
        try:
            U=calendar_lookahead(base,0,H,0,P,h,z,T,r,K=0,small=True)
            cache={};realized=[]
            for ell in tree['leaves']:
                pi=operate_lookahead(base,ell,T,r,P,h,z,K=0,small=True,cache=cache)
                realized.append(pi['cost']*tree['probability'][ell])
            calendar=dict(U=U['value'],executed_expected_cost=sum(realized),success=1.)
        except RuntimeError:pass
        for phi in (0.,4.):
            q=service_program(tree,h,z,r,eps,penalty=phi,keep_policy=True)
            row={**q,'commitment':r,'z':z,'epsilon':eps,'failure_fee':phi,
                 'run_cap':L,'run_set_mass':mass,'selected_tree_feasible':fixed['feasible'],
                 'calendar_feasible':calendar is not None}
            if fixed['feasible']:
                row['selected_tree_processing']=fixed['processing_cost']
                row['selected_tree_success']=fixed['actual_success']
                row['selected_tree_policy_upper']=fixed['processing_cost']+phi*(1-fixed['actual_success'])
                row['selected_tree_policy']=fixed['node_actions']
            if calendar is not None:row.update(calendar_upper=calendar['U'],calendar_cost=calendar['executed_expected_cost'])
            if q['feasible'] and q['completed']:
                if fixed['feasible']:assert row['selected_tree_policy_upper']>=q['solver_lower_bound']-2e-6
                if calendar is not None:assert calendar['executed_expected_cost']>=q['solver_lower_bound']-2e-6
            costs.append(row)
        if len(costs)%48==0:print('Joint cost cases',len(costs),'elapsed',round(perf_counter()-start,2),flush=True)
    save('joint_costs',costs);flat_csv('joint_costs',costs)
    save('joint_inputs',dict(base=base,T=T,H=H,current=0,h=h,P=P.tolist(),R=R,Z=Z,E=E,rmax=1.2))
    print('Joint complete',len(costs),round(perf_counter()-start,2),flush=True)

def reserve_boundary():
    calendars=[[1,0,2,2],[1,2,0,0]]
    jobs=[dict(release=0,deadline=1,work=.8,rate=1.),dict(release=0,deadline=3,work=.8,rate=.5)]
    cases=[]
    for D,b,root,tail in product((1,2,3),(.35,.5,.75),(0.,.1),(0.,2.)):
        c=copy.deepcopy(calendars);c[0][0]+=root;c[1][0]+=root
        for k in range(2):
            for t in range(1,4):
                if c[k][t]>0:c[k][t]+=tail
        js=copy.deepcopy(jobs);js[0]['deadline']=D;js[1]['rate']=b
        q=reserve_identity(c,js,reveal=2);leaf=[reserve_identity(c,js,reveal=2,selected=[k]) for k in (0,1)]
        cuts=two_job_cuts(c,js)
        if q['status']=='optimal':
            lo,hi=0.,10.
            for _ in range(48):
                mid=(lo+hi)/2
                if two_job_cuts(c,js,mid)['feasible']:hi=mid
                else:lo=mid
            assert abs(q['reserve']-hi)<1e-8
            assert cuts['shared_conflict']==(max(x['reserve'] for x in leaf)<1e-8 and q['reserve']>1e-8)
        cases.append(dict(urgent_deadline=D+1,B_rate=b,root_extra=root,normal_tail_extra=tail,
            reserve=q.get('reserve'),leaf_max=max((x.get('reserve',0) for x in leaf)),**cuts))
    rng=np.random.default_rng(20261028);errors=[];conflicts=0
    for i in range(200):
        H=int(rng.integers(4,8));c0=float(rng.uniform(.1,1.2))
        cs=[[c0]+rng.choice([0.,.25,.5,1.5],H-1).tolist() for _ in range(2)]
        js=[]
        for j in range(2):
            D=int(rng.integers(1,H));a=float(rng.uniform(.15,1.2))
            js.append(dict(release=0,deadline=D,rate=a,work=float(rng.uniform(.05,(D+1)*a))))
        q=reserve_identity(cs,js,reveal=2);assert q['status']=='optimal'
        lo,hi=0.,10.
        for _ in range(48):
            mid=(lo+hi)/2
            if two_job_cuts(cs,js,mid)['feasible']:hi=mid
            else:lo=mid
        errors.append(abs(hi-q['reserve']));conflicts+=two_job_cuts(cs,js)['shared_conflict']
    save('reserve_boundary',cases);flat_csv('reserve_boundary',cases)
    save('reserve_boundary_checks',dict(seed=20261028,cases=200,max_reserve_error=max(errors),
        zero_leaf_shared_conflicts=conflicts,example_reserve=reserve_identity(calendars,jobs,reveal=2)['reserve']))
    print('Reserve boundary complete',max(errors),flush=True)

def ai_study():
    P=transition(.2,.6);base=[dict(id=('base','training'),release=0,deadline=3,work=.25,rate=.2),
                             dict(id=('base','batch'),release=0,deadline=5,work=.2,rate=.25)]
    standard=[dict(name='short-window inference',D=1,beta=.15,rate=.2,channels=(0,1)),
              dict(name='checkpointable training',D=2,beta=.6,rate=.35,channels=(0,1)),
              dict(name='asynchronous batch',D=4,beta=.35,rate=.22,channels=(0,1))]
    specs=[]
    def add(label,classes,z=.4):specs.append(dict(label=label,classes=classes,reserve=z))
    add('Balanced baseline',copy.deepcopy(standard))
    c=copy.deepcopy(standard);c[1]['D']=4;add('More patient training',c)
    c=copy.deepcopy(standard);c[0]['D']=2;add('Inference deadline two',c)
    c=copy.deepcopy(standard);c[0]['rate']=.08;add('Inference rate limit',c)
    c=copy.deepcopy(standard);c[1]['beta']=.9;add('Training overhead 0.80',c)
    c=copy.deepcopy(standard);c[0]['channels']=(0,);add('Inference local only',c)
    c=copy.deepcopy(standard);c[1]['channels']=(0,);add('Training local only',c)
    add('Protected rate 0.25',copy.deepcopy(standard),.25)
    add('Protected rate 0.60',copy.deepcopy(standard),.6)
    urgent=copy.deepcopy(standard);urgent[0].update(beta=.6,rate=.75);urgent[1].update(beta=.3,D=3);urgent[2]['beta']=.15
    add('Urgent mix',copy.deepcopy(urgent))
    c=copy.deepcopy(urgent);c[1]['D']=4;add('Urgent mix patient training',c)
    c=copy.deepcopy(urgent);c[0]['D']=2;add('Urgent mix inference slack',c)
    c=copy.deepcopy(urgent);c[1]['D']=4;c[0]['channels']=(0,);add('Urgent mix local inference',c)
    rows=[]
    for spec in specs:
        tree=class_tree(base,spec['classes'],2,6,1,P,rmax=1.2)
        frontier=service_program(tree,1.2,spec['reserve'],epsilon=.1,objective='frontier')
        assert frontier['feasible'] and frontier['completed'],(spec['label'],frontier)
        R=max(0.,min(.6,frontier['r']-1e-10))
        reserve=service_program(tree,1.2,r=.6,epsilon=.1,objective='reserve')
        cost=service_program(tree,1.2,spec['reserve'],R,.1,penalty=2.,keep_policy=True)
        assert cost['feasible'] and cost['completed'],(spec['label'],cost)
        rows.append(dict(label=spec['label'],z=spec['reserve'],frontier=frontier['r'],r_eval=R,
            minimum_reserve=reserve.get('reserve') if reserve['feasible'] else None,
            reference_obligation_feasible=reserve['feasible'],processing_cost=cost['processing_cost'],
            failure_fee=cost['failure_cost'],total_cost=cost['policy_cost'],success=cost['actual_success'],
            policy_lower=cost['solver_lower_bound'],policy_upper=cost['policy_cost'],width=cost['policy_gap'],
            classes=spec['classes'],frontier_record=frontier,reserve_record=reserve,cost_record=cost))
        print('AI',spec['label'],frontier['r'],reserve.get('reserve'),cost['policy_cost'],flush=True)
    save('ai_workload',rows);flat_csv('ai_workload',rows)
    save('ai_inputs',dict(base=base,specifications=specs,current=1,h=1.2,T=2,H=6,
                         epsilon=.1,rmax=1.2,r_ref=.6,penalty=2.,P=P.tolist()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['joint','reserve','ai','all'],default='all');a=p.parse_args()
    if a.part in ('joint','all'):joint()
    if a.part in ('reserve','all'):reserve_boundary()
    if a.part in ('ai','all'):ai_study()

if __name__=='__main__':main()
