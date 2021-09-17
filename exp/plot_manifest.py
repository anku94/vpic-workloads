import glob
from itertools import zip_longest
import re
from typing import List, Tuple, Dict, NewType

import numpy as np
import pandas as pd
import sys

import IPython

import matplotlib.pyplot as plt
import matplotlib.colors as colors

RankManifestItem = NewType('RankManifestItem', Tuple[float, float, int, int])
RankManifest = List[RankManifestItem]
Manifest = Dict[int, RankManifest]


def read_manifest_file(data_path: str, rank: int = 0) -> RankManifest:
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


def read_entire_manifest(data_path: str) -> Manifest:
    all_data = {}
    all_files = glob.glob(data_path + '/*manifest*')

    for file in all_files:
        file_rank = int(file.split('.')[-1])
        data = read_manifest_file(file, file_rank)
        all_data[file_rank] = data

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
        if value > cutoff:
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
    range_min, range_max, item_count, rank_counts = get_stats(manifest)
    rank_count_max = max(rank_counts.values())

    cell_vol = 1e3
    num_cells = int(rank_count_max / cell_vol)
    print(num_cells)

    all_cells = []
    for rank in range(num_ranks):
        rank_manifest = manifest[rank]
        cell_carried_over = 0
        cells = []
        for item in rank_manifest:
            cell_carried_over += item[2]
            # XXX: we're essentially skipping small SSTs for plot
            cell_min, cell_max = item[0], item[1]
            while cell_carried_over > cell_vol:
                cells.append([cell_min, cell_max])
                cell_carried_over -= cell_vol

        all_cells.append(cells)

    all_lines_in = list(zip_longest(*all_cells))
    all_lines_out = list(map(flatten_cell_line, all_lines_in))
    # for line in all_lines_out:
    #     print(line)
    print(all_lines_out[:3])
    return all_lines_out


def plot_manifest_heatmap(path_in: str, path_out: str) -> None:
    print(path_in)
    manifest = read_entire_manifest(path_in)
    data = compute_heatmap_cells(manifest)
    data = np.array(data)
    fig, ax = plt.subplots(1, 1)
    # imret = ax.imshow(data, aspect='auto',
    #                   origin='lower',
    #                   norm=colors.FuncNorm((norm_fwd, norm_inv)))
    imret = ax.imshow(data, aspect='auto',
                      origin='lower',
                      cmap='nipy_spectral',
                      norm=colors.FuncNorm((norm_fwd, norm_inv), vmin=0, vmax=500, clip=True))
    cbar = fig.colorbar(imret, ax=ax)
    cbar.set_ticks([0, 0.5, 1.5, 2, 5, 10, 100, 250])
    ax.set_xlabel('Rank (0 - 511)')
    ax.set_ylabel('Data (MB) - Grows With Time')
    ax.set_title('Partition vs Rank Over Time')
    # fig.show()
    fig.savefig('range-density.pdf', dpi=300)


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


def get_overlapping_count(manifest, point):
    overlapping_ssts = 0
    overlapping_mass = 0

    overlapping_items = []

    ranks = set()

    for item in manifest:
        if point >= item[0] and point <= item[1]:
            overlapping_ssts += 1
            overlapping_mass += item[2]
            overlapping_items.append(item)
            ranks.add(item[3])

    return overlapping_ssts, overlapping_mass, ranks


def do_ranges_overlap(a1, b1, a2, b2):
    if a1 >= a2 and a1 <= b2:
        return True

    if b1 >= a2 and b1 <= b2:
        return True

    if a2 >= a1 and a2 <= b2:
        return True

    if b2 >= a1 and b2 <= a2:
        return True

    return False


def get_overlapping_range_count(manifest, p, q):
    overlapping_ssts = 0
    overlapping_mass = 0

    for item in manifest:
        if do_ranges_overlap(item[0], item[1], p, q):
            overlapping_ssts += 1
            overlapping_mass += item[2]

    return overlapping_ssts, overlapping_mass


def run_manifest_analysis(data_path: str):
    mf_items = read_entire_manifest(data_path)
    # print(mf_items)
    range_min, range_max, item_sum = get_stats(mf_items)
    item_count = len(mf_items)

    count = get_overlapping_count(mf_items, 0.75)
    x_values_p1 = np.linspace(range_min, 2, num=100)
    # x_values_p2 = np.linspace(2, range_max, num=100)
    # x_values = np.concatenate([x_values_p1, x_values_p2])
    x_values = x_values_p1
    overlap_stats = map(lambda x: get_overlapping_count(mf_items, x), x_values)
    overlap_stats = list(zip(*overlap_stats))
    y_count_values = np.array(overlap_stats[0])
    y_mass_values = np.array(overlap_stats[1])

    y_mass_percent = y_mass_values * 100.0 / item_sum
    # print(x_values)
    # print(y_count_values)
    # print(y_mass_values)
    # print(y_mass_percent)
    range_max = 2

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_values, y_count_values)
    ax.plot([range_min, range_max], [item_count / 512.0, item_count / 512.0],
            '--')
    # ax.set_ylim(0, 100)
    ax.set_ylabel('Partitioning (% of total size)')
    ax.set_ylabel('Number of overlapping SSTs')
    ax.set_xlabel('Indexed Attribute (Energy)')
    ax.set_title(
        'Selectivity (%age) of range-partitioning across keyspace (T1900)')
    ax.set_title('Overlapping SST Count vs Keyspace (Ranks 512, SSTs = 30,000)')
    timestep = data_path.split('.')[-1]
    print(timestep)
    fig.show()
    # fig.savefig('../vis/manifest/vpic.512.pdf')
    print(len(mf_items))


def run_manifest_cdf_analysis(data_path: str):
    mf_items = read_entire_manifest(data_path)
    # print(mf_items)
    range_min, range_max, item_sum = get_stats(mf_items)
    item_count = len(mf_items)

    get_overlapping_count(mf_items, 0.5)
    x_values_p1 = np.linspace(range_min, 2, num=100)
    # x_values_p2 = np.linspace(2, range_max, num=100)
    # x_values = np.concatenate([x_values_p1, x_values_p2])
    x_values = x_values_p1
    overlap_stats = map(lambda x: get_overlapping_range_count(mf_items, 0, x),
                        x_values)
    overlap_stats = list(zip(*overlap_stats))
    y_count_values = np.array(overlap_stats[0])
    y_mass_values = np.array(overlap_stats[1])

    y_mass_percent = y_mass_values * 100.0 / item_sum
    # print(x_values)
    # print(y_count_values)
    # print(y_mass_values)
    # print(y_mass_percent)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_values, y_count_values)
    ax.plot([range_min, range_max], [item_count / 32.0, item_count / 32.0],
            '--')
    # ax.set_ylim(0, 100)
    ax.set_ylabel('Partitioning (% of total size)')
    ax.set_ylabel('Number of overlapping SSTs')
    ax.set_xlabel('Indexed Attribute (Energy)')
    ax.set_title(
        'Selectivity (%age) of range-partitioning across keyspace (T1900)')
    ax.set_title('Overlapping SST Count for query in (0, X)')
    timestep = data_path.split('.')[-1]
    fig.show()
    # fig.savefig('../vis/manifest/cdffromzero.T{0}.1.5M.32.32.pdf'.format(timestep))


def plot_misc_2():
    df = pd.read_csv('../rundata/rtp-params/pivot-std.csv')
    print(df)

    df = df[9:]

    df_1900_data = df[df['ts'] == 1900]
    df_2850_data = df[df['ts'] == 2850]

    df_1900_y = df_1900_data['stddev'] * 100
    df_1900_x = df_1900_data['renegcnt']

    df_2850_y = df_2850_data['stddev'] * 100
    df_2850_x = df_2850_data['renegcnt']

    fig, ax = plt.subplots(1, 1)
    ax.plot(df_1900_x, df_1900_y, label='Timestep 1900')
    ax.plot(df_2850_x, df_2850_y, label='Timestep 2850')
    ax.set_ylabel('Standard Deviation of Load (%)')
    ax.set_xlabel('Num. of Fixed RTP Rounds During I/O')
    ax.set_title('Renegotiation Freq vs Load Balance')
    ax.legend()

    # fig.show()
    fig.savefig('../vis/manifest/renegfreq.pdf')


def plot_misc():
    df = pd.read_csv('../rundata/rtp-params/pivot-std.csv')
    print(df)

    df1 = df[0:8]
    df_x = df1['pvtcnt']
    df_data = df1['stddev']
    df_xnum = range(len(df_data))
    print(df_data)
    fig, ax = plt.subplots(1, 1)
    ax.bar(df_xnum, df_data * 100)
    ax.set_ylabel('Standard deviation in load (%)')
    ax.set_xlabel('Number of pivots')
    ax.set_title('Pivots Sampled vs Load stddev')
    plt.xticks(df_xnum, df_x)
    # fig.show()
    fig.savefig('../vis/manifest/pvtcnt.pdf')
    return


def run_manifest_adhoc_analysis(data_path: str):
    mf_items = read_entire_manifest(data_path)
    # print(mf_items)
    range_min, range_max, item_sum = get_stats(mf_items)
    item_count = len(mf_items)

    count = get_overlapping_count(mf_items, 0.75)
    x_values_p1 = np.linspace(range_min, 2, num=100)
    # x_values_p2 = np.linspace(2, range_max, num=100)
    # x_values = np.concatenate([x_values_p1, x_values_p2])
    x_values = x_values_p1
    overlap_stats = map(lambda x: get_overlapping_count(mf_items, x), x_values)
    overlap_stats = list(zip(*overlap_stats))
    y_count_values = np.array(overlap_stats[0])
    y_mass_values = np.array(overlap_stats[1])

    y_max = max(y_count_values)

    y_max_idx = np.where(y_count_values == y_max)[0][0]
    x_max = x_values[y_max_idx]
    print(x_max, y_max)

    #  IPython.embed()


def get_bins(fpath):
    f = open(fpath).read().split('\n')
    f = [i.strip() for i in f if len(i.strip()) > 0]
    f = [i for i in f if 'RENEG_AGGR_PIVOTS' in i]
    f = list(map(lambda x: list(map(float, x.split(',')[-1].split(' '))), f))
    return f


def assert_float_arr_equal(all_arrs):
    all_elems = list(zip(*all_arrs))
    for elem_set in all_elems:
        max_elem = max(elem_set)
        min_elem = min(elem_set)
        print(min_elem, max_elem)
        assert (abs(max_elem - min_elem) < 1e-4)


def assert_all_bins_equal(path):
    all_files = glob.glob(data_path + '/*perfstats*')
    #  all_files = all_files[:10]
    print(all_files)
    all_bins = list(map(get_bins, all_files))
    all_bins = list(zip(*all_bins))
    for cur_bin_set in all_bins:
        print(cur_bin_set)
        assert_float_arr_equal(cur_bin_set)


def get_bins_for(bins, rank):
    rank_bins = []
    for bin_set in bins:
        bs = bin_set[rank]
        be = bin_set[rank + 1]
        rank_bins.append((bs, be))
    return rank_bins


def make_sense_of(path, rank):
    bins = get_bins(path + '/vpic-perfstats.log.{0}'.format(rank))
    rank_bins = get_bins_for(bins, rank)
    rank_manifest = read_manifest_file(path + '/vpic-manifest.{0}'.format(rank),
                                       rank)
    rank_manifest = [(i[0], i[1]) for i in rank_manifest]
    print(rank_bins)
    print(rank_manifest)

    bounds = rank_bins[0]
    last_bounds = bounds
    rbidx = 1

    fits = lambda item, bounds: item[0] >= bounds[0] and item[1] <= bounds[1]

    for item in rank_manifest:
        while not fits(item, bounds):
            print("Not fit", item, bounds)
            last_bounds = rank_bins[rbidx]
            mbs = min(bounds[0], last_bounds[0])
            mbe = max(bounds[1], last_bounds[1])
            bounds = (mbs, mbe)
            rbidx += 1

        print("Fit found", item, bounds)
        bounds = last_bounds


def make_sense_of_manifest(data_path):
    for rank in range(200, 512):
        print(rank)
        make_sense_of(data_path, rank)
    pass


if __name__ == '__main__':
    # plot_misc_2()
    data_path = sys.argv[1]
    #  assert_all_bins_equal(data_path)
    # make_sense_of_manifest(data_path)
    #  run_manifest_adhoc_analysis(data_path)
    plot_manifest_heatmap(data_path, data_path)
    sys.exit(0)

    data_path_base = "../rundata/manifest-data/T."
    timesteps = ['100', '950', '1900', '2850']
    for timestep in timesteps:
        data_path = data_path_base + timestep
        print(data_path)
        run_manifest_cdf_analysis(data_path)
