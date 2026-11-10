"""Replay exported actions without the optimization constraint matrix."""
from pathlib import Path
import json
from collections import defaultdict
from certified_models import two_class_instance

ROOT = Path(__file__).resolve().parent

def verify():
    rows = json.loads((ROOT / 'matched_results.json').read_text())
    checked = 0
    worst = 0.0
    cost_error = 0.0
    probability_error = 0.0
    for row in rows:
        if row['causal']['status'] != 0:
            continue
        cfg = row['inputs']
        nodes, jobs = two_class_instance(**{k: v for k, v in cfg.items() if k != 'b'})
        actions = {}
        for a in row['actions']:
            n, j, k = tuple(a['history']), a['job'], a['channel']
            assert (n, j, k) not in actions
            assert n in nodes and 0 <= j < len(jobs)
            w = jobs[j]
            assert k in w.get('elig', (0, 1))
            assert w['release'] <= nodes[n]['t'] <= w['deadline']
            assert n[:len(w.get('birth', ()))] == w.get('birth', ())
            assert a['quantity'] >= 0
            actions[n, j, k] = a['quantity']
        def cumulative(n, j):
            return sum(v for (h, jj, k), v in actions.items()
                       if jj == j and n[:len(h)] == h)
        for n, node in nodes.items():
            channels = defaultdict(float)
            lineage = defaultdict(float)
            for j, w in enumerate(jobs):
                amount = sum(actions.get((n, j, k), 0.0) for k in (0, 1))
                worst = max(worst, amount - w['rate'])
                lineage[w.get('lineage', ('job', j))] += amount
                worst = max(worst, cumulative(n, j) - w['work'] - w.get('coef', 0) * w.get('r', 0))
                for k in (0, 1):
                    channels[k] += actions.get((n, j, k), 0.0)
            worst = max(worst, channels[0] - node['local'], channels[1] - cfg['z'])
            for lin, amount in lineage.items():
                limit = min(w['rate'] for j, w in enumerate(jobs)
                            if w.get('lineage', ('job', j)) == lin)
                worst = max(worst, amount - limit)
        success = 0.0
        for leaf, node in nodes.items():
            if node['children']:
                continue
            passed = True
            for j, w in enumerate(jobs):
                if leaf[:len(w.get('birth', ()))] != w.get('birth', ()):
                    continue
                demand = w['work'] + w.get('coef', 0) * w.get('r', 0)
                if cumulative(leaf, j) < demand - 1e-8:
                    passed = False
            if passed:
                success += node['prob']
        assert success >= cfg['b'] - 1e-8
        probability_error = max(probability_error, abs(success - row['success']))
        cost = sum(nodes[n]['prob'] * nodes[n]['prices'][k] * v
                   for (n, j, k), v in actions.items()) + 5 * (1 - success)
        cost_error = max(cost_error, abs(cost - row['causal']['value']))
        checked += 1
    assert worst < 1e-7 and cost_error < 1e-7 and probability_error < 1e-7
    report = dict(exported_policies_checked=checked, max_physical_violation=max(0, worst),
                  max_expected_cost_error=cost_error, max_success_probability_error=probability_error,
                  representation='Direct exported-history actions; no optimizer matrix used')
    (ROOT / 'saved_policy_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    verify()
