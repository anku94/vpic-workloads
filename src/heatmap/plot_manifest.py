import glob
from itertools import zip_longest
import re
from typing import List, Tuple, Dict, NewType

import numpy as np
import pandas as pd
import sys

import IPython

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

RankManifestItem = NewType('RankManifestItem', Tuple[float, float, int, int])
RdbManifestItem = NewType('RdbManifestItem',
                          Tuple[float, float, int, int, int])
RankManifest = List[RankManifestItem]
RdbManifest = List[RdbManifestItem]
Manifest = Dict[int, RdbManifest]


def read_mdb_manifest(data_path: str, rank: int = 0) -> RankManifest:
    f = open(data_path).read().splitlines()
    all_items = list()

    for item in f:
        item = re.split('\ |-', item)
        item = list(filter(lambda x: x != '', item))
        item[0] = float(item[0])
        item[1] = float(item[1])
        item[2] = int(item[2])
        item.append(rank)
        item = RankManifestItem(
            (float(item[0]), float(item[1]), int(item[2]), rank))
        all_items.append(item)

    return all_items


def read_rdb_manifest(data_path: str, rank: int = 0,
                      epoch: int = 0) -> RdbManifest:
    f = open(data_path).read().splitlines()
    all_items = list()

    rnum_max = 0

    for item in f:
        item = item.strip().split(',')

        sst_epoch = item[0]
        range_begin = float(item[4])
        range_end = float(item[5])
        sst_count = int(item[6])
        rnum = int(item[8])
        rnum = max(rnum_max, rnum)
        rnum_max = rnum

        if int(sst_epoch) != epoch: continue

        item = RdbManifestItem((range_begin, range_end, sst_count, rank, rnum))
        all_items.append(item)

    return all_items


def read_entire_manifest(data_path: str, epoch: int) -> Manifest:
    all_data = {}
    all_files = glob.glob(data_path + '/*manifest*')
    all_data_len = 0

    for file in all_files:
        print(file)
        file_rank = int(file.split('.')[-1])
        data = read_rdb_manifest(file, file_rank, epoch)
        all_data[file_rank] = data
        all_data_len += len(data)

    print('Reading manifest for epoch: {0}, items read: {1}'.format(epoch,
                                                                    all_data_len))
    return all_data


def _norm_fwd(x):
    cutoff = 2
    if x > cutoff:
        return cutoff + (np.log(x + 1 - cutoff) / 10)
    else:
        return x


def norm_fwd(x):
    return np.vectorize(_norm_fwd)(x)


def _norm_inv(x):
    cutoff = 2

    if x > cutoff:
        return np.exp((x - cutoff) * 10) + cutoff - 1
    else:
        return x


def norm_inv(x):
    return np.vectorize(_norm_inv)(x)


class SkewNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def split_log(self, value):
        cutoff = 3
        print(value)
        if value > 1e9:
            return 1
        elif value > cutoff:
            return np.log(value + 1 - cutoff)
        else:
            return value

    def norm(self, value):
        return self.split_log(value) / self.split_log(self.vmax)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)


def flatten_cell_line(line: List[Tuple[float, float]]) -> List:
    # avg = lambda x: (x[0] + x[1]) / 2
    avg = lambda x: x[1] - x[0]
    avg = lambda x: x[1]
    avg_none = lambda x: max(0, avg(x)) if x else 0
    line_out = np.array(list(map(avg_none, line)), dtype=np.float64)
    # normalize
    # norm = lambda x: np.log(x + 1)
    # line_out = np.array(map(norm, line_out))
    return line_out


def compute_heatmap_cells(manifest: Manifest) -> None:
    num_ranks = len(manifest)
    # num_ranks = 20
    range_min, range_max, item_count, rank_counts = get_stats(manifest)
    rank_count_max = max(rank_counts.values())

    cell_vol = 1e3
    num_cells = int(rank_count_max / cell_vol)
    print(num_cells)

    all_cells = []
    all_reneg_points = []
    for rank in range(num_ranks):
        rank_manifest = manifest[rank]
        cell_carried_over = 0
        cells = []

        rank_volume = 0
        rank_prev_round = rank_manifest[0][4]
        rank_prev_epoch = 0
        rank_reneg_points = []

        for item in rank_manifest:
            rank_round = item[4]
            if rank_round != rank_prev_round:
                rank_reneg_points.append(rank_volume / cell_vol)
                rank_prev_round = rank_round

            rank_volume += item[2]

            cell_carried_over += item[2]
            # XXX: we're essentially skipping small SSTs for plot
            cell_min, cell_max = item[0], item[1]
            while cell_carried_over > cell_vol:
                cells.append([cell_min, cell_max])
                cell_carried_over -= cell_vol

        all_cells.append(cells)
        all_reneg_points.append(rank_reneg_points)

    all_lines_in = list(zip_longest(*all_cells, fillvalue=[2e9, 2e9]))
    all_lines_out = list(map(flatten_cell_line, all_lines_in))
    # for line in all_lines_out:
    #     print(line)
    print(all_lines_out[:3])
    all_reneg_points = list(zip(*all_reneg_points))
    print(all_reneg_points)
    print('Rounds detected: ', len(all_reneg_points))
    return (all_lines_out, all_reneg_points)


def plot_manifest_heatmap(path_in: str, path_out: str,
                          plot_label: str = 'range.density',
                          annotate: bool = False) -> None:
    for epoch in range(5):
        print(path_in)
        manifest = read_entire_manifest(path_in, epoch=epoch)
        data_heatmap, data_reneg = compute_heatmap_cells(manifest)
        data_heatmap = np.array(data_heatmap)
        fig, ax = plt.subplots(1, 1)
        imret = ax.imshow(data_heatmap, aspect='auto',
                          origin='lower',
                          cmap='nipy_spectral',
                          norm=colors.FuncNorm((norm_fwd, norm_inv), vmin=0,
                                               vmax=500, clip=True)
                          )

        if annotate:
            for reneg_line in data_reneg:
                num_ranks = len(reneg_line)
                ax.plot(range(num_ranks), reneg_line)

        cbar = fig.colorbar(imret, ax=ax)
        cbar.set_ticks([0, 0.5, 1.5, 2, 5, 10, 100, 250])
        ax.set_xlabel('Rank (0 - 511)')
        ax.set_ylabel('Items Written (In Million) - Grows With Time')
        ax.set_title('Partition vs Rank Over Time (Epoch {0})'.format(epoch))
        ax.set_ylim([0, max([ax.get_ylim()[1], 20000])])
        ax.set_facecolor('#CCC')

        def numfmt(x, pos):
            return '{}'.format(x / 1000)

        yfmt = tkr.FuncFormatter(numfmt)
        plt.gca().yaxis.set_major_formatter(yfmt)
        # fig.show()
        fig.savefig('{0}/{1}.{2}.pdf'.format(path_out, plot_label, epoch),
                    dpi=300)


def plot_manifest_heatmap2x2(path_in: str, path_out: str,
                             plot_label: str = 'range.density') -> None:
    fig, axes = plt.subplots(2, 2)
    print(fig)
    print(axes)

    def numfmt(x, pos):
        return '{}'.format(x / 1000)

    yfmt = tkr.FuncFormatter(numfmt)
    plt.gca().yaxis.set_major_formatter(yfmt)

    ax1d = [axes[i][j] for i in range(2) for j in range(2)]
    for idx, epoch in enumerate([0, 1, 3, 4]):
        print(path_in)
        ax = ax1d[idx]
        manifest = read_entire_manifest(path_in, epoch=epoch)
        data_heatmap, data_reneg = compute_heatmap_cells(manifest)
        data_heatmap = np.array(data_heatmap)
        imret = ax.imshow(data_heatmap, aspect='auto',
                          origin='lower',
                          cmap='nipy_spectral',
                          norm=colors.FuncNorm((norm_fwd, norm_inv), vmin=0,
                                               vmax=500, clip=True)
                          )

        # cbar = fig.colorbar(imret, ax=ax)
        # cbar.set_ticks([0, 0.5, 1.5, 2, 5, 10, 250])
        # ax.set_xlabel('Rank (0 - 511)')
        # ax.set_ylabel('Items Written (In Million) - Grows With Time')
        ax.set_title('Epoch {0}'.format(epoch))
        ax.set_ylim([0, max([ax.get_ylim()[1], 20000])])
        ax.set_facecolor('#CCC')
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(imret, cax=cbar_ax)
    cbar.set_ticks([0, 0.5, 1.5, 2, 10, 250])

    fig.suptitle(path_in.split('/')[-3])
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    # fig.show()
    fig.savefig('{0}/{1}.2x2.pdf'.format(path_out, plot_label, epoch),
                dpi=300)


def plot_heatmap_internal(mf_path, epoch, ax):
    manifest = read_entire_manifest(mf_path, epoch=epoch)
    data_heatmap, data_reneg = compute_heatmap_cells(manifest)
    data_heatmap = np.array(data_heatmap)
    imret = ax.imshow(data_heatmap, aspect='auto',
                      origin='lower',
                      cmap='nipy_spectral',
                      norm=colors.FuncNorm((norm_fwd, norm_inv), vmin=0,
                                           vmax=500, clip=True)
                      )

    return imret


def plot_manifest_heatmap2x2_intvl(path_in: str, path_out: str) -> None:
    print(path_in)
    runs = glob.glob(path_in + '/batch*')
    print(runs)
    all_runs = []
    for run in runs:
        print(run)
        intvl, iter = run.split('.')[-2:]
        intvl = int(intvl)
        iter = int(iter)
        print(intvl, iter)
        all_runs.append((intvl, iter, run))
    plot_iter = 3
    plot_runs = sorted(list(filter(lambda x: x[1] == plot_iter, all_runs)))
    plot_epoch = 1
    print(plot_runs)

    fig, axes = plt.subplots(2, 2)
    print(fig)
    print(axes)

    def numfmt(x, pos):
        return '{}'.format(x / 1000)

    yfmt = tkr.FuncFormatter(numfmt)
    plt.gca().yaxis.set_major_formatter(yfmt)

    ax1d = [axes[i][j] for i in range(2) for j in range(2)]
    for idx, run in enumerate(plot_runs):
        path_in = run[2] + '/plfs/particle'
        print(path_in)
        ax = ax1d[idx]
        print(run[2])
        imret = plot_heatmap_internal(path_in, plot_epoch, ax)
        ax.set_title('Interval {0}'.format(run[0]))
        ax.set_ylim([0, max([ax.get_ylim()[1], 20000])])
        ax.set_facecolor('#CCC')
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(imret, cax=cbar_ax)
    cbar.set_ticks([0, 0.5, 1.5, 2, 10, 250])

    fig.suptitle(path_in.split('/')[-3])
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    # fig.show()
    fig.savefig('{0}/epoch{1}.iter{2}.intvl.2x2.pdf'.format(path_out, plot_epoch, plot_iter),
                dpi=300)


def get_stats(manifest) -> Tuple[float, float, int, Dict[int, int]]:
    range_min = manifest[0][0][0]
    range_max = manifest[0][0][1]
    item_count = 0
    rank_counts = {}

    for rank in manifest:
        item_count_rank = 0
        for item in manifest[rank]:
            range_min = min(item[0], range_min)
            range_max = max(item[1], range_max)
            item_count_rank += item[2]
        item_count += item_count_rank
        rank_counts[rank] = item_count_rank

    print(range_min, range_max, item_count)
    return (range_min, range_max, item_count, rank_counts)


if __name__ == '__main__':
    run_path = sys.argv[1]
    plfs_path = run_path + '/plfs/particle'
    plot_path = run_path + '/../plots'
    run_label = run_path.split('/')[-1]
    print(plfs_path, run_label, plot_path)
    # data_path = sys.argv[1]
    # plot_path = sys.argv[2]
    # plot_manifest_heatmap(plfs_path, plot_path, run_label, False)
    # plot_manifest_heatmap2x2(plfs_path, plot_path, run_label)
    plot_manifest_heatmap2x2_intvl(run_path + '/..', plot_path)
    # plot_manifest_heatmap(plfs_path, plot_path, run_label + '.annotated', True)
