"""Verify the saved studies and optionally recompute selected results."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXTENDED = ROOT / 'extended_studies'
HOURLY = ROOT / 'hourly_trace'
STRUCTURAL = ROOT / 'structural_studies'


def verify_trace_inputs():
    """Check the locally reconstructed Alibaba extracts against the release."""
    sources = [
        (HOURLY / 'source_manifest.json', HOURLY / 'source_partitions'),
        (EXTENDED / 'additional_trace_manifest.json', EXTENDED / 'additional_trace_partitions'),
    ]
    for manifest, directory in sources:
        for row in json.loads(manifest.read_text())['partitions']:
            path = directory / f"day{row['day']}_hour{row['hour']:02d}.parquet"
            if not path.is_file():
                raise SystemExit(f'Missing {path.name}. Follow Alibaba data and retrieval in README.md.')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise SystemExit(f'Extract checksum mismatch for {path.name}. Check the source and PyArrow version.')
    if not (STRUCTURAL / 'trace_sample.parquet').is_file():
        raise SystemExit('Missing the execution-summary sample. Follow Alibaba data and retrieval in README.md.')


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-studies', action='store_true',
                        help='Recompute the conflict benchmark and, with --trace-data, the hourly and debt studies.')
    parser.add_argument('--legacy-quick', action='store_true',
                        help='Run the partial replay of the earlier experiments.')
    parser.add_argument('--multistage-studies', action='store_true',
                        help='Recompute the matched multistage cases and controls.')
    parser.add_argument('--extended-studies', action='store_true',
                        help='Recompute the extended structural and Markov studies and, with --trace-data, the holdout.')
    parser.add_argument('--trace-data', action='store_true',
                        help='Use the locally retrieved Alibaba extracts to replay the saved trace policies.')
    args = parser.parse_args()
    commands = [(ROOT, ['verify_adaptive_certificate.py'])]
    if args.trace_data:
        verify_trace_inputs()
        commands.append((ROOT, ['calibrate_hourly.py', 'hourly_trace/source_partitions']))
    if args.new_studies:
        commands.append((ROOT, ['conflict_benchmark.py']))
        if args.trace_data:
            commands += [(ROOT, [f]) for f in ('hourly_trace_study.py', 'debt_boundary.py', 'figures_hourly.py')]
    if args.multistage_studies:
        commands += [(ROOT / 'multistage_studies', [f]) for f in
                     ('run_studies.py', 'run_controls.py', 'make_revision_figures.py')]
    if args.legacy_quick:
        commands.append((ROOT / 'earlier_studies', ['run_all.py', '--quick']))

    commands += [(ROOT / 'earlier_studies', [f]) for f in
                 ('audit_revision.py', 'audit_final_revision.py', 'verify_decomposition.py')]
    commands += [(STRUCTURAL, [f]) for f in ('audit_mathematics.py', 'verify_new.py')]
    if args.trace_data:
        commands += [(ROOT, [f]) for f in ('verify_hourly.py', 'verify_debt_boundary.py')]
    commands += [(ROOT / 'multistage_studies', [f]) for f in
                 ('multistage_structure.py', 'verify_witnesses.py', 'verify_saved_matched.py',
                  'exact_overlap_certificate.py', 'verify_service_ranges.py')]
    commands.append((ROOT / 'witness_depth2', ['verify_four_leaf_witness.py']))

    release_guard = ['verify_extended_release.py']
    if args.extended_studies:
        commands += [(EXTENDED, [f]) for f in
                     ('verify_symmetric_corridor.py', 'verify_depth_sharpness.py',
                      'verify_window_bounds.py', 'window_certificate.py')]
        if args.trace_data:
            commands += [(EXTENDED, [f]) for f in ('holdout_validation.py', 'plot_holdout.py')]
        release_guard.append('--recomputed-window')
    commands.append((EXTENDED, release_guard))
    for directory, command in commands:
        print(f"Running {directory.name}/{' '.join(command)}", flush=True)
        subprocess.run([sys.executable, *command], cwd=directory, check=True)
    print('Requested replay completed. See README.md for its scope.', flush=True)


if __name__ == '__main__':
    run()
