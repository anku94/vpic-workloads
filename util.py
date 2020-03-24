import functools
import numpy as np
import operator
import re
import struct

from itertools import zip_longest
from math import fabs
from pathlib import Path
from typing import List

EPSILON = 1e-4

def approx_altb(a, b):
    if a < b - EPSILON:
        return True
    return False

def approx_alteqb(a, b):
    if a < b + EPSILON:
        return True
    return False

def approx_aeqb(a, b):
    if fabs(a - b) < EPSILON:
        return True
    return False

def approx_agtb(a, b):
    if a > b + EPSILON:
        return True
    return False

def approx_agteqb(a, b):
    if a > b - EPSILON:
        return True
    return False


class VPICReader:
    data_root = None
    timesteps = None

    def __init__(self, path: str) -> None:
        self.data_root = Path(path)
        self.timesteps = self._find_all_ts()
        return

    def _find_all_ts(self):
        candidates = list(self.data_root.glob('T*'))
        timesteps = list(filter(lambda x: re.match('T\.\d+', x.name) and x.is_dir(), candidates))
        return timesteps

    def get_num_ranks(self) -> int:
        num_ranks = len(list(self.timesteps[0].glob('e*')))
        return num_ranks

    def read_a_rank(self, timestep: int, rank: int, ftype: str = 'eparticle') -> List[str]:
        ts_str = re.findall('\d+', self.timesteps[timestep].name)[0]
        rank_fname = self.timesteps[timestep] / ("%s.%s.%s" % (ftype, ts_str, rank))
        print(rank_fname)

        values = []
        with open(rank_fname, 'rb') as f:
            while True:
                raw = f.read(4)
                if not raw:
                    break

                values.append(struct.unpack('f', raw)[0])

        print(len(values))
        return values

    def read_global(self, timestep: int, ftype: str = 'eparticle') -> List[str]:
        num_ranks = self.get_num_ranks()

        all_data = []

        for rank in range(num_ranks):
            print(rank)
            all_data.append(self.read_a_rank(timestep, rank, ftype))

        print(list(zip_longest(*all_data))[0])
        all_data = functools.reduce(operator.iconcat, zip_longest(*all_data), [])
        print(all_data[0])

        return all_data

class Histogram:
    hist = None
    bin_edges = None

    def __init__(self, data : List[float], bins) -> None:
        if type(bins) == type([]):
            self.hist, self.bin_edges = data, bins
        else:
            self.hist, self.bin_edges = np.histogram(data, bins)

    def rebalance(self):
        nbins = len(self.hist)
        new_edges, mass_per_bin = self._rebalance(nbins)
        print(new_edges, mass_per_bin)

        self.bin_edges = new_edges
        self.hist = [mass_per_bin] * nbins

    def _rebalance(self, nsamples):
        assert(nsamples > 2)

        old_hist = self.hist
        old_edges = self.bin_edges

        start = old_edges[0]
        end = old_edges[-1]

        nbins = len(old_hist)

        mass_per_bin = sum(old_hist) * 1.0 / nsamples

        new_hist = [start]

        bin_idx = 0
        mass_cur = 0

        while True:
            if bin_idx == nbins:
                break

            if approx_altb(mass_cur + old_hist[bin_idx], mass_per_bin):
                mass_cur += old_hist[bin_idx]
                bin_idx += 1
                continue

            # mass_cuur + cur_bin > mass_per_bin; so divide cur into multiple

            cur_bin_total = old_hist[bin_idx]
            cur_bin_left = old_hist[bin_idx]
            cur_bin_start = old_edges[bin_idx]
            cur_bin_end = old_edges[bin_idx + 1]

            while approx_agteqb(mass_cur + cur_bin_left, mass_per_bin):
                take_from_bin = mass_per_bin - mass_cur
                mass_cur = 0

                len_diff = cur_bin_end - cur_bin_start
                len_take = len_diff * take_from_bin * 1.0 / cur_bin_left

                new_hist.append(cur_bin_start + len_take)

                cur_bin_start += len_take
                cur_bin_left -= take_from_bin

            mass_cur = cur_bin_left
            bin_idx += 1

        if len(new_hist) == nsamples + 1:
            new_hist[-1] = end
        elif len(new_hist) == nsamples:
            new_hist.append(end)
        else:
            assert(False)

        return new_hist, mass_per_bin

def load_bins(rank_bins):
    num_ranks = len(rank_bins)
    bins_per_rank = len(rank_bins[0])

    rbvec = []

    for rank in range(num_ranks):
        for bidx in range(bins_per_rank - 1):
            bin_start = rank_bins[rank][bidx]
            bin_end = rank_bins[rank][bidx + 1]

            if (bin_start == bin_end): continue

            rbvec.append((rank, bin_start, bin_end, True))
            rbvec.append((rank, bin_end, bin_start, False))

    rbvec = sorted(rbvec, key = lambda x: (x[1], x[3]))
    return rbvec

def pivot_union(rb_items, rank_bin_widths, num_ranks):
    assert(len(rb_items) > 2)

    BIN_EMPTY = rb_items[0][1] - 10

    rank_bin_start = [BIN_EMPTY] * num_ranks
    rank_bin_end = [BIN_EMPTY] * num_ranks

    unified_bins = []
    unified_bin_counts = []

    prev_bin_val = rb_items[0][1]
    prev_bp_bin_val = rb_items[0][1]

    active_ranks = []

    cur_bin_count = 0
    rb_item_sz = len(rb_items)

    for idx, item in enumerate(rb_items):
        bp_rank = item[0]
        bp_bin_val = item[1]
        bp_bin_other = item[2]
        bp_is_start = item[3]

        remove_item = None

        if bp_bin_val != prev_bin_val:
            cur_bin = bp_bin_val
            cur_bin_count = 0

            for rank in active_ranks:
                assert(rank_bin_start[rank] != BIN_EMPTY)

                rank_total_range = rank_bin_end[rank] - rank_bin_start[rank]
                rank_left_range = cur_bin - prev_bp_bin_val
                rank_contrib = rank_bin_widths[rank] * rank_left_range * 1.0 / rank_total_range

                cur_bin_count += rank_contrib

                if rank == bp_rank:
                    assert(not bp_is_start)
                    remove_item = rank

            if cur_bin_count > EPSILON:
                unified_bin_counts.append(cur_bin_count)
                unified_bins.append(cur_bin)

            if remove_item != None:
                active_ranks.remove(remove_item)

            prev_bp_bin_val = bp_bin_val

        if bp_is_start:
            assert(rank_bin_start[bp_rank] == BIN_EMPTY)
            rank_bin_start[bp_rank] = bp_bin_val
            rank_bin_end[bp_rank] = bp_bin_other
            active_ranks.append(bp_rank)
            if (idx == 0):
                unified_bins.append(bp_bin_val)
        else:
            assert(rank_bin_start[bp_rank] != BIN_EMPTY)
            rank_bin_start[bp_rank] = BIN_EMPTY
            rank_bin_end[bp_rank] = BIN_EMPTY

            if remove_item == None:
                old_len = len(active_ranks)
                active_ranks.remove(bp_rank)
                new_len = len(active_ranks)

                assert(old_len == new_len + 1)

    return Histogram(unified_bin_counts, unified_bins)


