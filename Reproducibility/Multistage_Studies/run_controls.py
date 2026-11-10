from pathlib import Path
from itertools import product
import json
from certified_models import model,solve,constraint_generation,two_class_instance
from run_studies import actual_success
from multistage_structure import rank_backward

ROOT=Path(__file__).resolve().parent

def conflict(z):
    nodes={}
    for t in range(1,5):
        words=[(0,)] if t==1 else ([(0,a) for a in (0,1)] if t==2 else [(0,a,b)+(b,)*(t-3) for a,b in product((0,1),repeat=2)])
        for n in words:
            prob=1. if t==1 else (.5 if t==2 else .5*(.95 if n[2]!=n[1] else .05))
            state=0 if t==1 else n[-1];local=1. if t==1 else 2.*(1-state)
            nodes[n]=dict(t=t,state=state,prob=prob,children=[n+(e,) for e in ((0,1) if t<3 else (n[-1],))] if t<4 else [],local=local,prices=[1.,4.],arrival=[0.,0.],rates=[1. if t<=2 else 0.,.5],elig=[[0,1],[0,1]],deadlines=[2,4])
    jobs=[dict(id='A',release=1,deadline=2,work=.8,rate=1.,birth=()),dict(id='B',release=1,deadline=4,work=.8,rate=.5,birth=())]
    return nodes,jobs

def run():
    out=[]
    for label,z,b in [('uniform_unit_cost',.4,1.),('binding_conflict',.025,.9),('conflict_with_more_reserve',.1,.9)]:
        if label=='uniform_unit_cost':
            nodes,jobs=two_class_instance(z=z,eligibility='both')
            for v in nodes.values():v['prices']=[1.,1.]
        else:nodes,jobs=conflict(z)
        M=model(nodes,jobs,b=b,z=z);row=dict(label=label,z=z,b=b)
        row['causal'],a=solve(M);row['foresight'],f=solve(model(nodes,jobs,b=b,z=z,foresight=True))
        row['reserve'],rz=solve(model(nodes,jobs,b=b,z=z,mode='reserve'));row['foresight_reserve'],pz=solve(model(nodes,jobs,b=b,z=z,mode='reserve',foresight=True))
        S=model(nodes,jobs,b=b,z=z,mode='current');first,s=solve(S)
        fixed={(n,j,k):float(s.x[i]) for (ctx,n,j,k),i in S['ix'].items() if len(n)==1}
        row['shield'],ss=solve(M,root_fixed=fixed);row['cuts'],c=constraint_generation(M,batch=2)
        assert a.success and ss.success and c.success
        row['success'],good=actual_success(M,a);row['actions']=[dict(history=list(n),job=j,channel=k,quantity=float(a.x[i])) for (ctx,n,j,k),i in M['ix'].items() if a.x[i]>1e-9]
        initial=[sum(w['work']+w.get('coef',0)*w.get('r',0) for w in jobs if w.get('lineage',jobs.index(w))==j and w['release']==1) for j in (0,1)] if label=='uniform_unit_cost' else [.8,.8]
        F,R=rank_backward(nodes,z,good);row['rank_root']=F[(0,)].tolist();row['tail_rank_root']=R[(0,)].tolist()
        row['reserve_premium']=row['reserve']['value']-row['foresight_reserve']['value']
        row['policy_saving']=row['shield']['value']-row['causal']['value']
        assert abs(row['cuts']['value']-row['causal']['value'])<1e-8
        if label=='uniform_unit_cost':assert abs(row['policy_saving'])<1e-8
        if label=='binding_conflict':assert abs(row['reserve_premium']-.025)<1e-8 and abs(row['policy_saving'])<1e-8
        out.append(row);print(label,{k:row[k].get('value') for k in ('causal','shield','foresight','reserve','foresight_reserve')})
    (ROOT/'control_results.json').write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':run()
