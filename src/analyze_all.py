import sys
import argparse
from pathlib import Path

from plot_perfstats import run as run_perfstats
from plot_rdb import run as run_rdb
from plot_violin import run as run_violin

"""
.
├── carp_P3584M_intvl250000.log
├── exp-info
│   └── vpic-perfstats.log.<rank>
├── outsize.txt
├── plfs
│   └── particle
│       └── manifest.e<epoch>.csv (overlap stats)
│       └── vpic-manifest.<rank> (parsed manifest)
├── plots
└── vpic
"""


def analyze_input(path_in: str, path_out: str):
    run_violin(path_in, path_out)


def analyze_output(path_in: str):
    path_exp = Path(path_in) / 'exp-info'
    path_plfs = Path(path_in) / 'plfs' / 'particle'
    path_plots = Path(path_in) / 'plots'

    if not path_plots.exists():
        path_plots.mkdir()

    path_exp = str(path_exp)
    path_plfs = str(path_plfs)
    path_plots = str(path_plots)

    run_perfstats(path_exp, path_plots)
    run_rdb(path_plots, path_plots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='CARP Analysis Suite',
        description='''
        Run the entire set of analyses on a CARP run
        ''',
        epilog='''
        TBD
        '''
    )

    parser.add_argument('--input-path', '-i', type=str,
                        help='Path to analyze', required=True)
    parser.add_argument('--violin', '-v', type=str,
                        help='Run a Violin analysis and write to this path ('
                             'pdf/png), path is VPIC data',
                        required=False)
    parser.add_argument('--suite', '-s', action='store_true',
                        help='Run the post-analysis suite, path is output dir',
                        default=False, required=False)

    options = parser.parse_args()

    if not options.input_path:
        parser.print_help()
        sys.exit(0)

    if not options.violin and not options.suite:
        parser.print_help()
        sys.exit(0)

    if options.violin:
        analyze_input(options.input_path, options.violin)
    elif options.suite:
        analyze_output(options.input_path)
