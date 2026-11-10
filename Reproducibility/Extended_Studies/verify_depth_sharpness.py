"""Check the arbitrary-depth reserve witnesses using cumulative-service LPs."""
from dataclasses import dataclass
from fractions import Fraction
import json

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Node:
    parent: int | None
    ordinary: float
    ceiling: tuple[float, float]
    arrival: tuple[int, int]


def make_tree(depth):
    nodes = []
    leaves = []
    critical = []

    def add(parent, ordinary, ceiling, arrival=(0, 0)):
        nodes.append(Node(parent, ordinary, ceiling, arrival))
        return len(nodes) - 1

    def gadget(m, parent=None, arrival=(0, 0), active=True):
        if active:
            ordinary, ceiling = (0.5, (0.5, 0.5)) if m else (1.5, (1, 1))
        else:
            ordinary, ceiling = 0, (0, 0)
        root = add(parent, ordinary, ceiling, arrival)
        if m == 0:
            leaves.append(root)
            if active:
                critical.append(root)
            return
        if active:
            i = add(root, 1, (0, 1))
            j = add(root, 1, (1, 0))
            k = add(root, 2, (1, 1))
            gadget(m - 1, i, (0, 1))
            gadget(m - 1, j, (1, 0))
            gadget(m - 1, k, active=False)
        else:
            for _ in range(3):
                child = add(root, 0, (0, 0))
                gadget(m - 1, child, active=False)

    gadget(depth)
    return nodes, leaves, critical


def solve(nodes, leaves, reserve=None, root_class=0, root_direction=0):
    paths = []
    for i, node in enumerate(nodes):
        parent = () if node.parent is None else paths[node.parent]
        paths.append(parent + (i,))
    retained = sorted({n for leaf in leaves for n in paths[leaf]})
    position = {n: 2 * i for i, n in enumerate(retained)}
    variable_reserve = reserve is None
    dimension = 2 * len(retained) + variable_reserve
    objective = np.zeros(dimension)
    if variable_reserve:
        objective[-1] = 1
    else:
        objective[root_class] = root_direction

    limits = lil_matrix((3 * len(retained), dimension))
    rhs = np.empty(3 * len(retained))
    bounds = []
    for i, n in enumerate(retained):
        node = nodes[n]
        bounds.extend((0, a) for a in node.ceiling)
        row = 3 * i
        limits[row, position[n]:position[n] + 2] = 1
        if variable_reserve:
            limits[row, -1] = -1
        rhs[row] = node.ordinary + (0 if variable_reserve else reserve)
        for j in range(2):
            limits[row + j + 1, [position[k] + j for k in paths[n]]] = 1
            rhs[row + j + 1] = 1 + sum(nodes[k].arrival[j] for k in paths[n])
    if variable_reserve:
        bounds.append((0, None))
    limits = limits.tocsr()
    end_rows = [3 * (position[leaf] // 2) + j + 1 for leaf in leaves for j in range(2)]
    return linprog(objective, A_ub=limits, b_ub=rhs, A_eq=limits[end_rows],
                   b_eq=rhs[end_rows], bounds=bounds, method="highs")


def main():
    reports = []
    for m in range(1, 6):
        nodes, leaves, critical = make_tree(m)
        assert len(leaves) == 3**m and len(critical) == 2**m
        exact = Fraction(1, 2 * (2**(m + 1) - 1))
        current = (Fraction(1, 2) + exact) / 2
        for retained in (leaves, critical):
            result = solve(nodes, retained)
            assert result.success and abs(result.fun - float(exact)) < 1e-9
            assert solve(nodes, retained, reserve=0).status == 2
        for leaf in critical:
            reduced = [k for k in leaves if k != leaf]
            assert solve(nodes, reduced, reserve=0).success
        for job_class in range(2):
            for sign in (-1, 1):
                result = solve(nodes, leaves, reserve=float(exact), root_class=job_class,
                               root_direction=sign)
                assert result.success and abs(result.x[job_class] - float(current)) < 1e-9
        reports.append({
            "branching_depth": m,
            "leaves": len(leaves),
            "necessary_leaves": len(critical),
            "minimum_reserve": str(exact),
            "root_service_per_class": str(current),
            "zero_reserve_after_critical_deletion": len(critical),
        })
    print(json.dumps({"arbitrary_depth_witness_checks": reports}))


if __name__ == "__main__":
    main()
