import functools
from collections.abc import Iterable

import numpy as np
import operator
import re
import struct
import os

from itertools import zip_longest, product
from math import fabs
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union, Tuple

EPSILON = 1e-4


def chunk_and_sum(ls, nchunks):
    chunk_size = int(len(ls) / nchunks)
    # print('chunk size: ', chunk_size)
    chunks = []
    for chunk_start in range(0, len(ls), chunk_size):
        ls_chunk = ls[chunk_start:chunk_start + chunk_size]
        ls_chunk = sum(ls_chunk)
        chunks.append(ls_chunk)
    return chunks


class ApproxComp:
    @staticmethod
    def approx_altb(a, b):
        if a < b - EPSILON:
            return True
        return False

    @staticmethod
    def approx_alteqb(a, b):
        if a < b + EPSILON:
            return True
        return False

    @staticmethod
    def approx_aeqb(a, b):
        if fabs(a - b) < EPSILON:
            return True
        return False

    @staticmethod
    def approx_agtb(a, b):
        if a > b + EPSILON:
            return True
        return False

    @staticmethod
    def approx_agteqb(a, b):
        if a > b - EPSILON:
            return True
        return False


class VPICReader:
    data_root = None
    timesteps = None
    num_ranks = None

    def __init__(self, path: str, num_ranks=None) -> None:
        self.data_root = Path(path)
        self.timesteps = sorted(self._find_all_ts(),
                                key=lambda x: int(x.name[2:]))
        if num_ranks is None:
            self.num_ranks = self._compute_num_ranks()
        else:
            self.num_ranks = num_ranks
        return

    def _find_all_ts(self):
        candidates = list(self.data_root.glob('T*'))
        timesteps = list(filter(lambda x: re.match(
            'T\.\d+', x.name) and x.is_dir(), candidates))
        return timesteps

    def _compute_num_ranks(self) -> int:
        num_ranks = len(list(self.timesteps[0].glob('e*')))
        return num_ranks

    def get_num_ranks(self) -> int:
        return self.num_ranks

    def get_num_ts(self) -> int:
        return len(self.timesteps)

    def get_ts(self, ts_idx: int) -> int:
        ts_int = re.findall('\d+', self.timesteps[ts_idx].name)[0]
        return ts_int

    def read_a_rank(self, timestep: int, rank: int, ftype: str = 'eparticle') -> \
            List[str]:
        ts_str = re.findall('\d+', self.timesteps[timestep].name)[0]
        rank_fname = self.timesteps[timestep] / \
                     ("%s.%s.%s" % (ftype, ts_str, rank))
        # print(rank_fname)

        values = []
        with open(rank_fname, 'rb') as f:
            while True:
                raw = f.read(4)
                if not raw:
                    break

                values.append(struct.unpack('f', raw)[0])

        # print(len(values))
        return values

    def read_global(self, timestep: int, ftype: str = 'eparticle') -> List[str]:
        num_ranks = self.get_num_ranks()

        all_data = []

        for rank in range(num_ranks):
            print(rank)
            all_data.append(self.read_a_rank(timestep, rank, ftype))

        # print(list(zip_longest(*all_data))[0])
        all_data = functools.reduce(
            operator.iconcat, zip_longest(*all_data), [])
        # print(all_data[0])

        return list(filter(lambda x: x is not None, all_data))

    @staticmethod
    def sample_a_rank(fpath: str, intvl_cnt: int = 10,
                      intvl_samples: int = 2) -> List[float]:
        print(fpath)
        fsize = os.path.getsize(fpath)
        offsets = np.arange(intvl_cnt) * (fsize / intvl_cnt)
        offsets = offsets.astype(int)
        offsets = offsets & (~3)
        vals = []
        with open(fpath, 'rb') as f:
            for offset in offsets:
                f.seek(offset, 0)
                for sample in range(intvl_samples):
                    data = f.read(4)
                    val = struct.unpack('f', data)[0]
                    if val: vals.append(val)

        return vals

    @staticmethod
    def sample_a_rank_unpack(args: Tuple) -> List[float]:
        return VPICReader.sample_a_rank(args[0], intvl_cnt=args[1],
                                        intvl_samples=args[2])

    def rank_fpath(self, timestep: int, rank: int,
                   ftype: str = 'eparticle') -> str:
        ts_str = re.findall('\d+', self.timesteps[timestep].name)[0]
        rank_fname = self.timesteps[timestep] / \
                     ("%s.%s.%s" % (ftype, ts_str, rank))
        return rank_fname

    def sample_global(self, timestep: int, ftype: str = 'eparticle') -> List[
        str]:
        num_ranks = self.get_num_ranks()
        rank_paths = [self.rank_fpath(timestep, rank, ftype) for rank in
                      range(num_ranks)]
        INTVL_COUNT = 50
        INTVL_SAMPLES =  50
        rpath_args = [(x, INTVL_COUNT, INTVL_SAMPLES) for x in rank_paths]
        print(rank_paths)

        data = None
        with Pool(processes=64) as pool:
            data = pool.map(self.sample_a_rank_unpack, rpath_args)

        data = functools.reduce(operator.iconcat, data, [])
        return data


class Histogram:
    hist = None
    bin_edges = None

    def __init__(self, data: List[float] = None,
                 bin_weights: Union[float, List[float]] = None,
                 bin_edges: List[float] = None,
                 nbins: int = None
                 ) -> None:
        if bin_weights is not None and bin_edges is not None:
            if isinstance(bin_weights, Iterable):
                self.hist, self.bin_edges = bin_weights, bin_edges
            elif isinstance(bin_weights, float):
                self.bin_edges = bin_edges
                self.hist = [bin_weights] * (len(bin_edges) - 1)
            else:
                raise Exception('Invalid constructor parameters: %s' %
                                (type(bin_weights),))
        elif bin_edges is not None:
            self.hist, self.bin_edges = np.histogram(data, bin_edges)
        elif nbins is not None:
            self.hist, self.bin_edges = np.histogram(data, nbins)
        else:
            raise Exception('Invalid constructor parameters')

    def get_mass(self) -> float:
        return sum(self.hist)

    def rebalance(self, nsamples):
        nbins = len(self.hist)
        new_edges, mass_per_bin = self._rebalance(nsamples)
        assert (ApproxComp.approx_aeqb(mass_per_bin * nsamples, sum(self.hist)))

        self.bin_edges = new_edges
        self.hist = [mass_per_bin] * nsamples

    def _rebalance(self, nsamples):
        assert (nsamples > 2)

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

            if ApproxComp.approx_altb(mass_cur + old_hist[bin_idx],
                                      mass_per_bin):
                mass_cur += old_hist[bin_idx]
                bin_idx += 1
                continue

            # mass_cur + cur_bin > mass_per_bin; so divide cur into multiple

            cur_bin_total = old_hist[bin_idx]
            cur_bin_left = old_hist[bin_idx]
            cur_bin_start = old_edges[bin_idx]
            cur_bin_end = old_edges[bin_idx + 1]

            while ApproxComp.approx_agteqb(mass_cur + cur_bin_left,
                                           mass_per_bin):
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
            raise Exception("Not sure what happened here")

        return new_hist, mass_per_bin

    def __str__(self):
        return 'Histogram (num_bins: {0}, edges: {1}, weights: {2})'.format(
            len(self.hist),
            ', '.join(map(lambda x: "%0.4f" % x, self.bin_edges)),
            ', '.join(map(lambda x: str(x), self.hist))
        )


class RenegUtils:
    @staticmethod
    def load_bins(rank_bins):
        num_ranks = len(rank_bins)
        bins_per_rank = len(rank_bins[0])

        rbvec = []

        for rank in range(num_ranks):
            for bidx in range(bins_per_rank - 1):
                bin_start = rank_bins[rank][bidx]
                bin_end = rank_bins[rank][bidx + 1]

                if bin_start == bin_end:
                    continue

                rbvec.append((rank, bin_start, bin_end, True))
                rbvec.append((rank, bin_end, bin_start, False))

        rbvec = sorted(rbvec, key=lambda x: (x[1], x[3]))
        return rbvec

    @staticmethod
    def pivot_union(rb_items: List, rank_bin_widths: List[float],
                    num_ranks: int):
        assert (len(rb_items) > 2)

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
                    assert (rank_bin_start[rank] != BIN_EMPTY)

                    rank_total_range = rank_bin_end[rank] - rank_bin_start[rank]
                    rank_left_range = cur_bin - prev_bp_bin_val
                    rank_contrib = rank_bin_widths[rank] \
                                   * rank_left_range * 1.0 / rank_total_range
                    cur_bin_count += rank_contrib

                    # print('Rank ', rank, rank_total_range, rank_left_range)

                    if rank == bp_rank:
                        assert (not bp_is_start)
                        remove_item = rank

                if cur_bin_count > EPSILON:
                    unified_bin_counts.append(cur_bin_count)
                    unified_bins.append(cur_bin)

                if remove_item is not None:
                    active_ranks.remove(remove_item)

                prev_bp_bin_val = bp_bin_val

            if bp_is_start:
                assert (rank_bin_start[bp_rank] == BIN_EMPTY)
                rank_bin_start[bp_rank] = bp_bin_val
                rank_bin_end[bp_rank] = bp_bin_other
                active_ranks.append(bp_rank)
                if idx == 0:
                    unified_bins.append(bp_bin_val)
            else:
                assert (rank_bin_start[bp_rank] != BIN_EMPTY)
                rank_bin_start[bp_rank] = BIN_EMPTY
                rank_bin_end[bp_rank] = BIN_EMPTY

                if remove_item is None:
                    old_len = len(active_ranks)
                    active_ranks.remove(bp_rank)
                    new_len = len(active_ranks)

                    assert (old_len == new_len + 1)

        return Histogram(bin_weights=unified_bin_counts, bin_edges=unified_bins)
