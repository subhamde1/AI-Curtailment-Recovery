"""One-command offline replay; defaults to all computations, without checkpoints."""
from pathlib import Path
import argparse, subprocess, sys, time

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true',help='Partial audit that omits the larger-catalog, scaling, and matched-depth runs')
    args=parser.parse_args()
    commands=[['compute_results.py'],['compute_extensions.py'],['verify_channel_support.py'],
              ['compute_revision.py','--part','all'],['audit_revision.py'],['verify_decomposition.py']]
    if not args.quick:commands.append(['compute_large_revision.py','--part','all'])
    commands += [['compute_final_revision.py','--part','all']]
    if not args.quick:commands.append(['profile_final_lookahead.py'])
    commands += [['plot_revision.py'],['plot_final_revision.py'],['audit_final_revision.py']]
    start=time.perf_counter()
    for command in commands:
        print('Running '+' '.join(command),flush=True)
        subprocess.run([sys.executable]+command,cwd=ROOT,check=True)
    print({'scope':'partial quick audit' if args.quick else 'complete numerical replay',
           'elapsed_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
