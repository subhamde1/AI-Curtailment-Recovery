from pathlib import Path
from itertools import product
from time import perf_counter
import sys,json,hashlib,resource
import numpy as np
from certified_models import model,solve,constraint_generation,two_class_instance
from multistage_structure import rank_backward,reserve_rank,robust_lp

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent

def actual_success(M,ans):
    if not ans.success:return None
    good=[]
    for leaf,i in M['yi'].items():
        passed=True
        for j,w in enumerate(M['jobs']):
            if leaf[:len(w.get('birth',()))]!=w.get('birth',()):continue
            total=sum(ans.x[ix] for (ctx,n,jj,k),ix in M['ix'].items() if jj==j and (ctx is None or ctx==leaf) and leaf[:len(n)]==n and M['nodes'][n]['t']<=w['deadline'])
            r=ans.x[M['design']] if M['mode']=='frontier' else w.get('r',0)
            if total<w['work']+w.get('coef',0)*r-1e-8:passed=False
        if passed:good.append(leaf)
    return sum(M['nodes'][l]['prob'] for l in good),good

def small():
    spec=json.loads((ROOT/'study_specification.json').read_text());base=spec['base'];configs=[('base',base.copy())]
    for key,vals in spec['one_factor_ranges'].items():
        for v in vals:
            if v==base[key]:continue
            cfg={**base,key:v};configs.append((f'{key}_{v}',cfg))
    keys=list(spec['crossed_ranges'])
    for vals in product(*(spec['crossed_ranges'][k] for k in keys)):
        cfg={**base,**dict(zip(keys,vals))};configs.append(('cross_'+'_'.join(map(str,vals)),cfg))
    out=[];maxerror=0.;rankchecks=0
    for label,cfg in configs:
        args={k:v for k,v in cfg.items() if k!='b'};nodes,jobs=two_class_instance(**args);b=cfg['b'];z=cfg['z']
        row=dict(label=label,inputs=cfg)
        costM=model(nodes,jobs,b=b,z=z,mode='cost');row['causal'],sol=solve(costM)
        row['foresight'],fsol=solve(model(nodes,jobs,b=b,z=z,mode='cost',foresight=True))
        row['reserve'],rsol=solve(model(nodes,jobs,b=b,z=z,mode='reserve'))
        row['foresight_reserve'],rr=solve(model(nodes,jobs,b=b,z=z,mode='reserve',foresight=True))
        row['frontier'],fs=solve(model(nodes,jobs,b=b,z=z,mode='frontier'))
        ns,js=two_class_instance(**{**args,'debt':(0,0)})
        row['static_frontier'],ss=solve(model(ns,js,b=b,z=z,mode='frontier'))
        shieldM=model(nodes,jobs,b=b,z=z,mode='current');row['shield_first'],sh=solve(shieldM)
        if sh.success:
            fixed={(n,j,k):float(sh.x[i]) for (ctx,n,j,k),i in shieldM['ix'].items() if len(n)==1}
            row['shield'],s=solve(costM,root_fixed=fixed)
            if s.success:row['shield_success']=actual_success(costM,s)[0]
        if sol.success:
            row['cuts'],cs=constraint_generation(costM,batch=2)
            assert cs.success and abs(row['cuts']['value']-row['causal']['value'])<1e-7
            maxerror=max(maxerror,abs(row['cuts']['value']-row['causal']['value']))
            row['success'],good=actual_success(costM,sol);assert row['success']>=b-1e-8
            row['actions']=[dict(history=list(n),job=j,channel=k,quantity=float(sol.x[i])) for (ctx,n,j,k),i in costM['ix'].items() if sol.x[i]>1e-9]
            # Independent rank verification for the actual selected-success tree.
            initial=np.array(cfg['debt'])+np.array([cfg['urgency'],1-cfg['urgency']])*cfg['r']
            F,R=rank_backward(nodes,z,good);assert F[(0,)] is not None and np.all(np.r_[initial,initial.sum()]<=F[(0,)]+1e-7);rankchecks+=1
            row['rank_root']=F[(0,)].tolist();row['tail_rank_root']=R[(0,)].tolist();row['selected_leaves']=[list(l) for l in good]
            row['initial_debt']=initial.tolist()
            if row['foresight']['status']==0:assert row['foresight']['value']<=row['causal']['value']+1e-7
            if row.get('shield',{}).get('status')==0:assert row['shield']['value']>=row['causal']['value']-1e-7
        for key in ('frontier','static_frontier'):
            if row[key]['status']==0:row[key]['commitment']=-row[key]['value']
        if row['reserve']['status']==0 and row['foresight_reserve']['status']==0:
            row['reserve_premium']=row['reserve']['value']-row['foresight_reserve']['value'];assert row['reserve_premium']>=-1e-7
        out.append(row);print(label,{k:row[k].get('value') for k in ('causal','foresight','reserve','frontier')},flush=True)
    (ROOT/'matched_results.json').write_text(json.dumps(out,indent=2))
    validation=dict(configurations=len(out),rank_selected_policy_checks=rankchecks,max_cut_vs_extensive_error=maxerror,feasible_cost_cases=sum(r['causal']['status']==0 for r in out),infeasible_cost_cases=[r['label'] for r in out if r['causal']['status']!=0])
    (ROOT/'matched_verification.json').write_text(json.dumps(validation,indent=2))
    return out

def large():
    sys.path.insert(0,str(PRIOR/'earlier_studies'))
    from compute_extensions import transition,prices,class_jobs,operate
    d=json.loads((PRIOR/'earlier_studies/extension_results.json').read_text())
    original=next(q for q in d['large'] if q['base_jobs']==100);base=original['base_catalog']
    catalog_hash=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest()
    base=[{**w,'id':tuple(w['id'])} for w in base]
    spec=json.loads((ROOT/'study_specification.json').read_text())['large_catalog'];H=48;T=42;P=transition(spec['p'],spec['rho']);nodes={}
    for t in range(1,H+1):
        if t<10:words=[(0,)*t]
        elif t<24:words=[(0,)*9+(a,)*(t-9) for a in (0,1)]
        else:words=[(0,)*9+(a,)*14+(b,)*(t-23) for a,b in product((0,1),repeat=2)]
        for n in words:
            prob=1 if t<10 else P[0,n[9]]*(P[n[9],n[23]] if t>=24 else 1)
            nexts=(0,1) if t in (9,23) else (n[-1],)
            nodes[n]=dict(t=t,state=n[-1],prob=float(prob),children=[n+(e,) for e in nexts] if t<H else [],local=spec['h']*(1-n[-1]),prices=list(prices(t-1,n[-1])))
    jobs=[dict(w,release=w['release']+1,deadline=w['deadline']+1,birth=()) for w in base]
    for n,v in nodes.items():
        origin=v['t']-1
        if origin<T and v['state']:
            for w in class_jobs(origin,spec['r']):jobs.append(dict(w,release=w['release']+1,deadline=w['deadline']+1,birth=n))
    M=model(nodes,jobs,b=1,z=spec['z'],fee=0,mean_rows=True)
    t0=perf_counter();relax,a=solve(M,kept=[]);cg,c=constraint_generation(M,batch=12);full,b=solve(M)
    assert c.success and b.success and abs(cg['value']-full['value'])<1e-7
    perfect,pf=solve(model(nodes,jobs,b=1,z=spec['z'],fee=0,foresight=True))
    leaves=[n for n,v in nodes.items() if not v['children']];cal=0.;short=0.
    from compute_extensions import lp
    def calendar_replay(leaf,myopic=False):
        active=[dict(w) for w in base];value=0.
        for t in range(H):
            history=leaf[:t+1];v=nodes[history]
            ghost=[]
            for origin in range(t,T):
                if origin>t or leaf[t]:ghost.extend(class_jobs(origin,spec['r']))
            queue=[w for w in active+ghost if w['work']>1e-10]
            edges=[(j,s,k) for j,w in enumerate(queue) for s in range(max(t,w['release']),w['deadline']+1) for k in (0,1)]
            if not edges:
                if t<T and leaf[t]:active.extend(class_jobs(t,spec['r']))
                continue
            ix={e:i for i,e in enumerate(edges)};cost=[];rows=[];rhs=[];eq=[];be=[]
            future={}
            for s in range(t,H):
                descendants=[n for n,x in nodes.items() if x['t']==s+1 and n[:len(history)]==history]
                future[s]=sum(nodes[n]['prob']*np.array(nodes[n]['prices']) for n in descendants)/v['prob']
            for j,s,k in edges:cost.append(float(future[s][k]) if not myopic or s==t else 1e-8)
            for j,w in enumerate(queue):
                eq.append({ix[j,s,k]:1 for s in range(max(t,w['release']),w['deadline']+1) for k in (0,1)});be.append(w['work'])
                for s in range(max(t,w['release']),w['deadline']+1):rows.append({ix[j,s,k]:1 for k in (0,1)});rhs.append(w['rate'])
            for s in range(t,H):
                for k in (0,1):
                    rows.append({ix[j,s,k]:1 for j in range(len(queue)) if (j,s,k) in ix});rhs.append((spec['h']*(1-leaf[t]) if s==t else 0) if k==0 else spec['z'])
            ans=lp(cost,rows,rhs,eq,be,[(0,None)]*len(edges));assert ans.success
            current={w['id']:np.zeros(2) for w in queue}
            for (j,s,k),i in ix.items():
                if s==t:current[queue[j]['id']][k]+=ans.x[i]
            for w in active:
                a=current.get(w['id'],np.zeros(2));w['work']-=a.sum();value+=float(a@v['prices'])
                assert w['deadline']!=t or w['work']<1e-7
            active=[w for w in active if w['deadline']>t and w['work']>1e-9]
            if t<T and leaf[t]:active.extend(class_jobs(t,spec['r']))
        assert not active
        return value
    for leaf in leaves:
        probability=nodes[leaf]['prob']
        cal+=probability*calendar_replay(leaf)
        short+=probability*calendar_replay(leaf,True)
    out=dict(inputs=spec,catalog_sha256=catalog_hash,base_jobs=len(base),branch_jobs=len(jobs)-len(base),nodes=len(nodes),leaves=len(leaves),conditional_mean=relax,cuts=cg,extensive=full,perfect=perfect,existing_calendar=cal,current_oriented_calendar=short,peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,total_seconds=perf_counter()-t0)
    assert perfect['value']<=full['value']+1e-7<=cal+1e-7
    (ROOT/'overlap_100_results.json').write_text(json.dumps(out,indent=2));print('large',json.dumps({k:v for k,v in out.items() if k not in ('cuts',)},indent=2),flush=True)
    return out

if __name__=='__main__':
    small();large()
