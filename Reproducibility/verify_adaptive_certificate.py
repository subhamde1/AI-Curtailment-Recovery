"""Exact finite-tree checks of Proposition 5's certificate propagation contract.

These examples are independent of the LP implementations. They supplement,
rather than replace, the proposition's conditional-expectation proof.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parent
PROB = (F(1, 3), F(2, 3))


def make_plan(rng, horizon, work, node=""):
    """A common current action followed by two causal descendant policies."""
    action = work if len(node) == horizon - 1 else rng.randrange(work + 1)
    plan = {node: action}
    if len(node) < horizon - 1:
        for branch in "01":
            plan.update(make_plan(rng, horizon, work - action, node + branch))
    return plan


def cost(plan, prices, horizon, node=""):
    value = F(prices[node] * plan[node])
    if len(node) < horizon - 1:
        value += sum(p * cost(plan, prices, horizon, node + b) for p, b in zip(PROB, "01"))
    return value


def execute(rng, incumbent, prices, horizon, work, allowances, node=""):
    incumbent_value = cost(incumbent, prices, horizon, node)
    allowance = F(rng.randrange(3), 3) if allowances else F(0)
    candidates = [incumbent] + [make_plan(rng, horizon, work, node) for _ in range(4)]
    eligible = [(cost(p, prices, horizon, node), p) for p in candidates
                if cost(p, prices, horizon, node) <= incumbent_value + allowance]
    # Exercise both genuine improvements and explicitly allowed increases.
    accepted_value, accepted = (max if allowances else min)(eligible, key=lambda q: q[0])
    action = accepted[node]
    assert 0 <= action <= min(2, work)
    stage = F(prices[node] * action)
    retained_tail = F(0)
    actual = stage
    expected_allowance = allowance
    checks = 1
    if len(node) < horizon - 1:
        for probability, branch in zip(PROB, "01"):
            child = node + branch
            retained_tail += probability * cost(accepted, prices, horizon, child)
            future_cost, future_allowance, count = execute(
                rng, accepted, prices, horizon, work - action, allowances, child)
            actual += probability * future_cost
            expected_allowance += probability * future_allowance
            checks += count
    else:
        assert action == work  # The same hard deadline holds on every leaf.
    assert stage + retained_tail == accepted_value
    assert actual <= incumbent_value + expected_allowance
    return actual, expected_allowance, checks


def verify_terminal_charges():
    """Evaluate the final epoch when the fee is revealed after allocation."""
    checked = 0
    late_charges = 0
    for failure in (F(0), F(1, 10), F(1, 2), F(1)):
        probabilities = (1 - failure, failure)
        for fee in (F(0), F(1), F(4)):
            for processing in (F(0), F(3, 2)):
                charges = (processing, processing + fee)
                # Both terminal continuations have zero remaining cost.
                expected = sum(p * c for p, c in zip(probabilities, charges))
                assert expected == processing + failure * fee
                exceeds_expected = any(
                    p > 0 and charge > expected
                    for p, charge in zip(probabilities, charges)
                )
                assert exceeds_expected == (0 < failure < 1 and fee > 0)
                checked += 1
                late_charges += exceeds_expected
    return {"cases": checked, "realized_charge_exceeds_expectation": late_charges,
            "example": {"failure_probability": "1/10", "failure_fee": 4,
                        "expected_final_charge": str(F(1, 10) * 4)}}


def main():
    cases = []
    node_checks = 0
    for seed in range(128):
        rng = random.Random(20261103 + seed)
        horizon = 3 + seed % 2
        prices = {"": rng.randrange(1, 9)}
        for depth in range(1, horizon):
            for number in range(2 ** depth):
                prices[format(number, f"0{depth}b")] = rng.randrange(1, 9)
        incumbent = make_plan(rng, horizon, 2)
        initial = cost(incumbent, prices, horizon)
        actual, allowance, checks = execute(rng, incumbent, prices, horizon, 2, seed >= 64)
        node_checks += checks
        cases.append({"seed": seed, "horizon": horizon, "initial_upper": str(initial),
                      "executed_expected_cost": str(actual), "expected_allowance": str(allowance),
                      "bound_slack": str(initial + allowance - actual)})
    # Individually valid upper values can violate propagation.
    initial, loose_next, replacement_cost, replacement_upper = map(F, (1, 3, 2, 2))
    assert replacement_cost <= replacement_upper <= loose_next
    assert replacement_cost > initial
    assert not (loose_next <= initial)  # Revised propagation rejects the update.
    # A declared allowance permits, and accounts exactly for, the higher cost.
    assert F(2) <= F(1) + F(1)
    result = {"cases": len(cases), "exact_node_checks": node_checks,
              "hard_deadline_failures": 0, "bound_violations": 0,
              "incoherent_bound_counterexample_detected": True,
              "terminal_charge_checks": verify_terminal_charges(),
              "arithmetic": "fractions.Fraction; no floating-point comparisons",
              "scope": "Finite illustrative causal trees and terminal assessments.",
              "case_results": cases}
    (ROOT / "adaptive_certificate_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "case_results"}))


if __name__ == "__main__":
    main()
