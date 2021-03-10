from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import argparse
import sys
import re

OVERRIDES = {
    '/Users/schwifty/Repos/workloads/rundata/Mar8/vpic-carp8m-0-carp.38446/I'
    '-0/carp_P3584M_intvl250000':
        {
            'rintvl': 'once_global',
            'rundata': {
                'mdb': {
                    'max':
                        [0.26, 0.27, 1.63, 3.98, 5.53]
                }
            }
        }
}

COLORS = {}
COL_IDX = 0


def get_color(v: str) -> Tuple:
    global COLORS
    global COL_IDX

    if v not in COLORS:
        COLORS[v] = plt.cm.tab10(COL_IDX % 10)
        COL_IDX += 1

    return COLORS[v]


def override_exists_impl(obj, props):
    if len(props) == 0:
        return obj
    elif props[0] in obj:
        return override_exists_impl(obj[props[0]], props[1:])
    else:
        return None


def override_exists(*args):
    props = list(args)
    if len(props) == 0:
        return None
    else:
        return override_exists_impl(OVERRIDES, props)


def get_file_params(fpath: str):
    rintvl = 250000
    pvtcnt = 256

    params = re.findall('intvl(\d+)', fpath)
    if (params):
        rintvl = params[0]

    params = re.findall('pvtcnt(\d+)', fpath)
    if (params):
        pvtcnt = params[0]

    override = override_exists(fpath, 'rintvl')
    if override:
        rintvl = override

    override = override_exists(fpath, 'pvtcnt')
    if override:
        pvtcnt = override

    return (rintvl, pvtcnt)


def plot_runtime(jobdir: str, param: str, ax: object) -> None:
    runtime_csv = jobdir + '/plots/runtime.csv'
    runtime = pd.read_csv(runtime_csv, header=None)
    ax.plot(range(len(runtime[0])), runtime[0], label=param,
            color=get_color(param))


def run_runtime(all_jobdirs: str, plot_dir: str):
    PKEY = 0
    fig, ax = plt.subplots(1, 1)
    for dir in all_jobdirs:
        print(dir)
        params = get_file_params(dir)
        param = params[PKEY]
        plot_runtime(dir, param, ax)

    ax.legend()
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Epoch I/O time (seconds)')
    ax.set_title('Runtime vs RTP Interval')
    fig.savefig(plot_dir + '/runtime.pdf', dpi=300)


def plot_olap(jobdir: str, param: str, ax: object, key: str, op: str) -> None:
    override = override_exists(jobdir, 'rundata', key, op)
    if override:
        data = override
        data_x = list(range(len(data)))
        data_y = data
        ax.plot(data_x, data_y, label=param, color=get_color(param))
        return

    glob_template = '/plots/{0}.olap.*csv'.format(key)
    olap_globstr = jobdir + glob_template
    epoch_olaps = glob.glob(olap_globstr)
    olap_data = []
    for epoch_csv in epoch_olaps:
        epoch_pd = pd.read_csv(epoch_csv)
        epoch_idx = epoch_csv.split('.')[-2].strip('e')
        if op == 'avg':
            mass_max = epoch_pd['MatchMass'].mean()
        else:
            mass_max = epoch_pd['MatchMass'].max()
        mass_total = epoch_pd['TotalMass'][0]

        if mass_total == 0:
            continue

        mass_max = mass_max * 100.0 / mass_total;
        olap_data.append((epoch_idx, mass_max, mass_total))
    olap_data.sort()

    if len(olap_data) == 0:
        return

    data_x = list(zip(*olap_data))[0]
    data_y = list(zip(*olap_data))[1]
    ax.plot(data_x, data_y, label=param, color=get_color(param))


def run_olap_impl(all_jobdirs: str, plot_dir: str, key: str, op: str):
    PKEY = 0
    fig, ax = plt.subplots(1, 1)
    for dir in all_jobdirs:
        params = get_file_params(dir)
        param = params[PKEY]
        plot_olap(dir, param, ax, key, op)

    ax.legend()
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Epoch Overlap (Percent)')
    ax.set_title(
        '{1} {0} Overlap vs RTP Interval'.format(key.upper(), op.title()))
    plot_path = '{0}/{1}.{2}.olap.pdf'.format(plot_dir, key, op)
    print(plot_path)
    fig.savefig(plot_path, dpi=300)


def run_olap(all_dirs: str, dir_out: str):
    run_olap_impl(all_dirs, dir_out, 'rtp', op='max')
    run_olap_impl(all_dirs, dir_out, 'rtp', op='avg')
    run_olap_impl(all_dirs, dir_out, 'mdb', op='max')
    run_olap_impl(all_dirs, dir_out, 'mdb', op='avg')
    run_olap_impl(all_dirs, dir_out, 'rdb', op='max')


def run(dir_in: str, dir_out: str):
    print(dir_in)
    all_runs_glob = dir_in + '/*carp8*/**/carp*'
    all_dirs = glob.glob(all_runs_glob)
    all_dirs.sort()
    run_runtime(all_dirs, dir_out)
    run_olap(all_dirs, dir_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Aggregate Run Plotter',
        description='''Plot aggregate runs''',
        epilog='''
        TBD
        '''
    )

    parser.add_argument('--input-path', '-i', type=str,
                        help='Path to batch dir', required=True)
    parser.add_argument('--output-path', '-o', type=str,
                        help='Destination for plotted graphs', required=True)

    options = parser.parse_args()
    if not options.input_path or not options.output_path:
        parser.print_help()
        sys.exit(0)

    run(options.input_path, options.output_path)
