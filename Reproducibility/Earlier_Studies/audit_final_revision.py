"""Check the published common-service and matched-lookahead numerical records."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'final_data'
def read(name):return json.loads((DATA/(name+'.json')).read_text())
def main():
    joint=read('joint_costs');ai=read('ai_workload');large=read('lookahead_depth')
    assert len(joint)==192 and len(ai)==13 and len(large)==12
    for q in joint:
        if not q['feasible']:assert q['status']==2;continue
        assert q['completed'] and q['status']==0
        assert q['actual_success']>=1-q['epsilon']-1e-7
        assert q['selected_success']>=1-q['epsilon']-1e-7
        assert q['maximum_physical_residual']<1e-7
        assert abs(q['policy_cost']-q['processing_cost']-q['failure_cost'])<1e-8
        assert q['solver_lower_bound']<=q['policy_cost']+1e-7
        if q['selected_tree_feasible']:assert q['selected_tree_policy_upper']>=q['solver_lower_bound']-1e-7
        if q['calendar_feasible']:assert q['calendar_cost']>=q['solver_lower_bound']-1e-7
    for q in ai:
        assert q['success']>=.9-1e-7 and q['width']<1e-7
        assert q['r_eval']<=min(.6,q['frontier'])+1e-9
        assert q['cost_record']['maximum_physical_residual']<1e-7
    for q in large:
        assert q['complete'] and q['paths_completed']==32 and q['deadline_failures']==0
        assert q['solver_budget_failures']==0
        assert len(q['records'])==32 and q['online_calls']==1+32*q['H']
        assert abs(q['deterministic_width']-(q['U']-q['lower']))<1e-8
        assert q['upper_95_simultaneous']<=q['U']+1e-8
    for n in (20,50,100):
        q=[x for x in large if x['base_jobs']==n]
        assert all(a['U']>=b['U']-1e-8 for a,b in zip(q,q[1:]))
        assert all(a['records'][i]['events']==b['records'][i]['events'] for a,b in zip(q,q[1:]) for i in range(32))
    assert abs(read('pair_group_diagnostic')['max_pair_rate']-.5)<1e-12
    assert abs(read('reserve_boundary_checks')['example_reserve']-.025)<1e-8
    assert read('reserve_boundary_checks')['max_reserve_error']<1e-8
    urgent=next(q for q in ai if q['label']=='Urgent mix');patient=next(q for q in ai if q['label']=='Urgent mix patient training')
    assert urgent['minimum_reserve']==patient['minimum_reserve']==.36
    assert abs(urgent['frontier']-patient['frontier'])<1e-8 and patient['total_cost']<urgent['total_cost']
    result=dict(
        joint_cases=192,joint_maximum_width=max(x['policy_gap'] for x in joint if x['feasible']),
        workload_cases=13,workload_maximum_width=max(x['width'] for x in ai),
        matched_large_cases=12,common_paths_per_case=32,solver_limit_failures=0,
        reserve_random_checks=200,reserve_random_maximum_error=read('reserve_boundary_checks')['max_reserve_error'])
    (DATA/'integrated_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
