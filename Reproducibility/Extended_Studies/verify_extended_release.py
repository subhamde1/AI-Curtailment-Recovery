"""Compare the replication records with the frozen published values."""
import argparse
import csv
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def main(recomputed_window=False):
    published = load('published_reference.json')
    scaling = HERE.parent / 'publication_snapshot' / 'scaling.json'
    assert hashlib.sha256(scaling.read_bytes()).hexdigest() == published['scaling_table_23_data_sha256']
    corridors = load('symmetric_corridor_results.json')
    assert corridors['trees'] == 48 and corridors['rank_lp_checks'] == 252
    assert corridors['root_workload_checks'] == 240 and corridors['individual_leaf_lp_checks'] == 815
    assert corridors['pivot_lp_checks'] == 64
    assert corridors['max_rank_error'] < 1e-12
    assert corridors['max_minimum_service_error'] < 1e-12
    comparison = load('window_bounds_verification.json')
    assert comparison['independent_full_tree_cases'] == 12
    assert comparison['window_comparisons'] == 36 and comparison['max_full_window_error'] < 1e-12
    certificates = [r for r in load('window_certificate_results.json')
                    if r['conditioning_window'] >= 3]
    assert [r['conditioning_window'] for r in certificates] == [3, 4, 5]
    printed_by_window = {row['window']: row for row in published['table_47']}
    assert list(printed_by_window) == [3, 4, 5]
    lower, upper = Fraction(0), Fraction(100)
    printed_lower, printed_upper = lower, upper
    for result in certificates:
        window = result['conditioning_window']
        assert result['verified_exactly']
        lo, hi = [Fraction(str(q)) for q in result['outward_decimal_interval']]
        assert lower <= lo <= Fraction(result['exact_lower_fraction'])
        assert Fraction(result['exact_lower_fraction']) <= Fraction(result['exact_upper_fraction']) <= hi <= upper
        lower, upper = lo, hi
        printed = printed_by_window[window]
        assert printed['service_variables'] == result['decisions']
        assert printed['pathwise_capacity_rows'] == result['pathwise_capacity_rows']
        printed_lo, printed_hi = Fraction(printed['lower']), Fraction(printed['upper'])
        assert printed_lower <= printed_lo <= printed_hi <= printed_upper
        printed_lower, printed_upper = printed_lo, printed_hi
        if recomputed_window:
            assert printed_lo <= lo <= hi <= printed_hi
        else:
            assert (lo, hi) == (printed_lo, printed_hi)
    assert (printed_upper - printed_lower) / printed_lower < Fraction(172, 100000)
    assert printed_upper - printed_lower == Fraction('0.03218087')

    manifest = load('additional_trace_manifest.json')
    selection = [(q['day'], q['hour']) for q in manifest['partitions']]
    assert selection == [(d, h) for d in manifest['days'] for h in range(24)]
    for q in manifest['partitions']:
        assert q['source_rows'] > q['selected_rows'] > 0
        assert q['source_bytes'] > 0 and 0 < q['archive_offset'] < manifest['archive_bytes']

    report = load('holdout_summary.json')
    days = [110] + manifest['days']
    assert report['holdout_days'] == days and report['observed_hours'] == 24 * (1 + len(days))
    assert report['observed_rows'] == sum(r['rows'] for r in report['daily_panel'])
    assert sum(q['selected_rows'] for q in manifest['partitions']) == sum(
        r['rows'] for r in report['daily_panel'] if r['day'] in manifest['days'])
    with (HERE / 'holdout_results.csv').open() as records:
        rows = list(csv.DictReader(records))
    assert len(rows) == report['all_models'] == 5 * 22 * len(days)
    base = {(int(q['day']), int(q['hour']), int(q['calls'])): q for q in rows if q['design'] == 'base'}
    assert set(base) == {(day, hour, k) for day in days for hour in range(22) for k in (1,2,3)}
    assert len(base) == report['base_models']
    assert all(float(q['all_stress_mass']) > .1 for q in rows)
    assert all(float(q['max_residual']) < 1e-8 for q in rows)
    declines = [float(base[d,h,1]['frontier']) - float(base[d,h,3]['frontier']) for d in days for h in range(22)]
    assert len(declines) == report['matched_origins']
    assert sum(q > 1e-9 for q in declines) == report['strict_single_to_three_declines']
    assert abs(mean(declines) - report['mean_single_to_three_decline']) < 1e-12
    printed_holdout = []
    for stats in report['daily']:
        group = [base[stats['day'], h, stats['calls']] for h in range(22)]
        f = [Decimal(q['frontier']) for q in group]
        z = [Decimal(q['reserve']) for q in group if q['reserve']]
        printed = [f"{stats['day']}",f"{stats['calls']}",f'{mean(f):.4f}',
                   f'{min(f):.4f}--{max(f):.4f}',f'{len(z)}/22',f'{mean(z):.4f}']
        printed_holdout.append(printed)
    assert printed_holdout == published['table_48']
    print(f"Extended release guard passed: {len(base)} base cases, {len(rows)} total cases, "
          f"{len(selection)} added partition records, published interval "
          f"[{float(printed_lower):.8f}, {float(printed_upper):.8f}].")
    if recomputed_window:
        print(f"Recomputed interval [{float(lower):.8f}, {float(upper):.8f}].")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recomputed-window', action='store_true',
                        help='Accept an exact recomputed interval contained in the printed interval.')
    main(parser.parse_args().recomputed_window)
