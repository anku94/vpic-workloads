import argparse
import glob
import multiprocessing
import sys
from typing import Iterable, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cache

import IPython

PERFLOGFMT = '/vpic-perfstats.log.{0}'


def read_bincnt(perf_path: str) -> np.ndarray:
    data = pd.read_csv(perf_path)
    col_type = data.columns[1]
    col_val = data.columns[2]
    # all_pivots = data[data[col_type] == 'RENEG_AGGR_PIVOTS']
    all_bincnts = data[data[col_type] == 'RENEG_BINCNT']
    col = all_bincnts[col_val]
    np_arr = np.stack(
        col.map(lambda x: np.array(x.strip().split(' '), dtype=int)))
    return np_arr


def read_pivots(perf_path: str):
    perf_path += PERFLOGFMT.format(0)
    data = pd.read_csv(perf_path)
    col_type = data.columns[1]
    col_val = data.columns[2]
    all_pivots = data[data[col_type] == 'RENEG_AGGR_PIVOTS']
    col = all_pivots[col_val]
    np_arr = np.stack(
        col.map(lambda x: np.array(x.strip().split(' '), dtype=float)))

    print(np_arr.shape)
    return np_arr


def read_all(perf_path: str) -> Tuple[Iterable, Iterable]:
    aggr_bincnts = None

    cache_obj = cache.Cache()
    if cache_obj.exists(perf_path):
        aggr_bincnts = cache_obj.get(perf_path)
    else:
        all_fpaths = sorted(glob.glob(perf_path + PERFLOGFMT.format('*')))
        # all_fpaths = all_fpaths[:8]

        with multiprocessing.Pool(8) as pool:
            parsed_fpaths = pool.map(read_bincnt, all_fpaths)

        aggr_bincnts = sum(parsed_fpaths)
        cache_obj.put(perf_path, aggr_bincnts)

    row_sum = aggr_bincnts.sum(1)
    epoch_idx = np.argwhere(row_sum < 100).flatten()
    assert (epoch_idx[0] == 0)
    epoch_idx = np.delete(epoch_idx, 0)

    epoch_bincnts = np.split(aggr_bincnts, epoch_idx)

    aggr_pivots = read_pivots(perf_path)
    epoch_pivots = np.split(aggr_pivots, epoch_idx)

    print('Total Epochs: ', len(epoch_bincnts))
    print('Total Mass: ', sum(aggr_bincnts[-1]))

    return epoch_pivots, epoch_bincnts


def analyze_overlap_epoch(all_pivots: np.ndarray,
                          all_counts: np.ndarray, npts: int) -> Tuple[
    np.ndarray, List, np.int64]:
    piv_min = all_pivots.min()
    piv_max = all_pivots.max()
    piv_step = (piv_max - piv_min) / npts

    probe_points = np.arange(piv_min, piv_max, piv_step)
    overlaps = []

    all_deltas = np.diff(all_counts, axis=0)

    for point in probe_points:
        point_olap = 0
        for pvt, cnt in zip(all_pivots, all_deltas):
            for i in range(len(cnt)):
                if pvt[i] <= point <= pvt[i + 1]:
                    point_olap += cnt[i]

        overlaps.append(point_olap)

    sum_shuffled = all_counts[-1].sum()
    return probe_points, overlaps, sum_shuffled


def dump_csv(epoch: int, points: np.ndarray, overlaps: List, total: np.int64,
             fpath: str):
    csv_header = "Epoch,Point,MatchMass,TotalMass,MatchCount"
    cols = csv_header.split(',')

    df = pd.DataFrame()
    df_len = len(points)

    df[cols[0]] = [epoch] * df_len
    df[cols[1]] = points
    df[cols[2]] = overlaps
    df[cols[3]] = [total] * df_len

    df.to_csv(fpath, index=False)

    return


def analyze_overlap(all_pivots: List[np.ndarray],
                    all_counts: List[np.ndarray],
                    out_path: str):
    path_fmt = '{0}/rtp.olap.e{1}.csv'
    npts = 100

    for epoch in range(len(all_pivots)):
        points, overlaps, total = analyze_overlap_epoch(all_pivots[epoch],
                                                        all_counts[epoch], npts)
        print('Epoch {:d}: Max Overlap: {:.2f}%'.format(
            epoch, max(overlaps) * 100.0 / total))

        csv_path = path_fmt.format(out_path, epoch)
        dump_csv(epoch, points, overlaps, total, csv_path)
        print('Epoch {0} saved to ...{1}'.format(epoch, csv_path[-20:]))


def plot_reneg_std(bincnts: Iterable[np.ndarray], fig_path: str) -> None:
    fig, ax = plt.subplots(1, 1)

    linestyle = '-'

    for epoch, epoch_stats in enumerate(bincnts):
        epoch_loads = np.diff(epoch_stats, n=1, axis=0)

        if (epoch_loads.shape[0] == 0): continue

        epoch_cum_loads = np.delete(epoch_stats.sum(1), 0)
        epoch_stds = epoch_loads.std(1) / epoch_loads.mean(1)

        ax.plot(epoch_cum_loads, epoch_stds, linestyle, mec='purple',
                label='Epoch {0}'.format(epoch))

    ax.set_xlabel('Total Data Volume')
    ax.set_ylabel('Normalized Stddev (CoV)')
    ax.set_title('Renegotiation Events vs Interval Stddev')
    ax.legend()
    fig.show()

    plot_out = fig_path + '/reneg_vs_std.pdf'
    # fig.savefig(plot_out, dpi=300)
    print('Plot saved: ', plot_out)


def query_pivots(epoch: int, rank: int, all_pivots: List, all_counts: List):
    pivots_epoch = all_pivots[epoch]
    counts_epoch = all_counts[epoch]
    deltas_epoch = np.diff(counts_epoch, n=1, axis=0)

    print('Rank {:d}, Epoch {:d}'.format(rank, epoch))
    rank_total = 0

    for pivots, counts in zip(pivots_epoch, deltas_epoch):
        print(pivots[rank], pivots[rank + 1], counts[rank])
        rank_total += counts[rank]

    print('Rank Total: ', rank_total)


def run_query(perf_path: str, epoch: int, rank: int):
    pivots, counts = read_all(perf_path)
    query_pivots(epoch, rank, pivots, counts)


def run(perf_path: str, path_out: str) -> None:
    pivots, counts = read_all(perf_path)
    plot_reneg_std(counts, path_out)
    analyze_overlap(pivots, counts, path_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Manifest Plotter',
        description='''
        Plot manifest CSVs generated by the RDB reader''',
        epilog='''
        TBD
        '''
    )

    parser.add_argument('--input-path', '-i', type=str,
                        help='Path to perflog files', required=True)
    parser.add_argument('--output-path', '-o', type=str,
                        help='Destination for plotted graphs', required=False)
    parser.add_argument('--query', '-q', action='store_true',
                        help='Query counts for a rank/epoch pair',
                        default=False)
    parser.add_argument('--rank', '-r', type=int,
                        help='Query counts for this rank', required=False)
    parser.add_argument('--epoch', '-e', type=int,
                        help='Query counts for this rank-epoch', required=False)

    options = parser.parse_args()
    if not options.input_path:
        parser.print_help()
        sys.exit(0)

    if not options.output_path:
        options.output_path = options.input_path

    if options.query:
        run_query(options.input_path, options.epoch, options.rank)
    else:
        run(options.input_path, options.output_path)
