"""Independent primal/dual checks of equation (S7a), using synthetic inputs."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
SEED = 20261009

def main():
    rng = np.random.default_rng(SEED)
    maximum_gap = 0.0
    for case in range(100):
        jobs = int(rng.integers(2, 9))
        channels = int(rng.integers(1, 5))
        eligible = rng.random((jobs, channels)) < 0.6
        capacity = rng.uniform(0, 2, channels)
        rates = rng.uniform(0, 1.5, jobs)
        gamma = rng.uniform(0, 4, jobs)
        if case % 10 == 0:
            capacity[0] = 0.0
        if case % 11 == 0:
            rates[0] = 0.0
        edges = [(j, k) for j in range(jobs) for k in range(channels)
                 if eligible[j, k]]
        if edges:
            a = np.zeros((jobs + channels, len(edges)))
            for e, (j, k) in enumerate(edges):
                a[j, e] = 1.0
                a[jobs + k, e] = 1.0
            primal = linprog(-np.array([gamma[j] for j, _ in edges]),
                             A_ub=a, b_ub=np.r_[rates, capacity],
                             bounds=(0, None), method='highs')
            assert primal.success
            value = -float(primal.fun)
        else:
            value = 0.0
        a_dual = np.zeros((len(edges), channels + jobs))
        for e, (j, k) in enumerate(edges):
            a_dual[e, k] = -1.0
            a_dual[e, channels + j] = -1.0
        dual = linprog(np.r_[capacity, rates],
                       A_ub=a_dual if edges else None,
                       b_ub=-np.array([gamma[j] for j, _ in edges]) if edges else None,
                       bounds=(0, None), method='highs')
        assert dual.success
        alpha = dual.x[:channels]
        phi = float(capacity @ alpha)
        for j in range(jobs):
            if eligible[j].any():
                phi += rates[j] * max(0.0, gamma[j] - float(alpha[eligible[j]].min()))
        gap = max(abs(value - float(dual.fun)), abs(value - phi))
        maximum_gap = max(maximum_gap, gap)
        assert gap < 1e-8, (case, value, dual.fun, phi)
    result = {'seed': SEED, 'synthetic_cases': 100,
              'maximum_primal_dual_support_discrepancy': maximum_gap}
    (ROOT / 'channel_support_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
