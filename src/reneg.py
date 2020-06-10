from util import VPICReader, RenegUtils, Histogram, ApproxComp
from rank import Rank
import itertools, numpy
from copy import deepcopy

import matplotlib.pyplot as plt
from functools import reduce
from typing import List


def flatten(ls: list):
    return list(itertools.chain.from_iterable(ls))


class Renegotiation:
    num_bins_final: int
    num_pivots_stored: int
    num_pivots_sent: int

    def __init__(self, num_ranks: int, timestep: int,
                 vpic_reader: VPICReader) -> None:
        self.vpic_reader = vpic_reader
        self.num_ranks = num_ranks
        self.time_step = timestep

        self.ranks = [Rank(vpic_reader, ridx) for ridx in range(num_ranks)]
        # all data, including what's not yet produced
        self.ranks_data = []
        # data that has been produced. reset after reneg
        self.ranks_produced = [[] for ridx in range(num_ranks)]
        # data that has been produced, flattened. reset after reneg.
        self.ranks_produced_flattened = []
        self.ranks_cursors = []

        """
        number of bins finally returned by the renegotiation.
        usually equal to the number of ranks, but configurable in case we want to
        find out 'how much information' the global distribution contains
        """
        self.num_bins_final = num_ranks
        """
        number of pivots gathered from each rank. ideally, this wouldn't be a function
        of the scale. we hope that in practice, this is some logarithmic function of
        scale or something
        """
        self.num_pivots_sent = num_ranks * 4
        """
        number of counters maintained by each rank to construct the pivots to be sent
        in the previous step. 1x-2x should be sufficient if we expect the distribution
        to change slowly. we don't think (yet) that this needs to be a function of scale
        """
        self.num_pivots_stored = self.num_pivots_sent * 2

        return

    def set_num_bins_final(self, new_num_bins_final: int):
        self.num_bins_final = new_num_bins_final

    def set_num_pivots_sent(self, new_num_pivots_sent: int):
        self.num_pivots_sent = new_num_pivots_sent

    def set_num_pivots_stored(self, new_num_pivots_stored: int):
        self.num_pivots_stored = new_num_pivots_stored

    def read_a_particle(self) -> None:
        return

    def read_all(self) -> None:
        assert(len(self.ranks_data) == 0)

        for rank in self.ranks:
            rank_data = self.vpic_reader.read_a_rank(self.time_step, rank.rank_id)
            self.ranks_data.append(rank_data)
            self.ranks_cursors.append(0)

        return

    def insert(self, percent: float) -> int:
        total_produced = 0
        for ridx in range(self.num_ranks):
            data = self._produce_at_rank(ridx, percent, ro=False)
            # print('Inserting %s at Rank %s' % (len(data), ridx))
            total_produced += len(data)
        return total_produced

    def peek_ahead(self, percent: float) -> List[float]:
        all_data = []

        for ridx in range(self.num_ranks):
            data = self._produce_at_rank(ridx, percent, ro=True)
            all_data.extend(data)

        return all_data

    def _produce_at_rank(self, rank_id: int, percent: float, ro: bool = False):
        assert(0 <= percent <= 1)

        cur_pos = self.ranks_cursors[rank_id]
        data_len = len(self.ranks_data[rank_id])

        if cur_pos >= data_len:
            raise Exception("EOF")

        pos_increment = int(data_len * percent)
        new_pos = cur_pos + pos_increment

        if (abs(new_pos - data_len) * 1.0 / data_len) < 0.001:
            new_pos = data_len
        elif new_pos >= data_len:
            raise Exception("EOF")

        data_to_produce = self.ranks_data[rank_id][cur_pos:new_pos]

        # print("Producing {0} @ Rank {1}".format(len(data_to_produce), rank_id))

        if not ro:
            self.ranks_produced[rank_id].extend(data_to_produce)
            self.ranks_produced_flattened.extend(data_to_produce)

            self.ranks[rank_id].produce(data_to_produce)
            self.ranks_cursors[rank_id] = new_pos

        # print("Inserting {0} elems for Rank {1}"
        #       .format(new_pos - cur_pos, rank_id))

        return data_to_produce

    def renegotiate(self) -> Histogram:
        all_pivots = []
        all_widths = []

        all_masses = []

        for rank in self.ranks:
            pivots, pivot_width = rank.compute_pivots(self.num_pivots_sent)
            assert(len(pivots) == self.num_pivots_sent)

            all_pivots.append(pivots)
            all_widths.append(pivot_width)

            all_masses.append(rank.get_total_produced())

        rbvec = RenegUtils.load_bins(all_pivots)
        merged_pivots = RenegUtils.pivot_union(rbvec, all_widths,
                                               self.num_ranks)
        mass_init = (self.num_pivots_sent - 1) * sum(all_widths)
        mass_fin = merged_pivots.get_mass()

        assert(abs(mass_fin - mass_init) < 1e-3)

        # print(all_pivots, all_widths)
        # print(merged_pivots.bin_edges, merged_pivots.hist)

        merged_pivots.rebalance(self.num_bins_final)

        mass_reb = merged_pivots.get_mass()

        assert(ApproxComp.approx_aeqb(mass_init, sum(all_masses)))
        assert(ApproxComp.approx_aeqb(mass_init, mass_fin))
        assert(ApproxComp.approx_aeqb(mass_init, mass_reb))

        # print(merged_pivots.bin_edges, merged_pivots.hist)

        return merged_pivots

    def update_pivots(self, new_pivots: Histogram):
        rank_pivots = deepcopy(new_pivots)
        if len(rank_pivots.hist) != self.num_pivots_stored:
            rank_pivots.rebalance(self.num_pivots_stored)

        for rank in self.ranks:
            rank.update_and_flush(rank_pivots.bin_edges)

        self.reset_between_reneg()
        return

    def reset_between_reneg(self):
        self.ranks_produced = [[] for ridx in range(self.num_ranks)]
        self.ranks_produced_flattened = []
        return

    def get_aggr_pivot_counts(self):
        aggr_sum = sum([numpy.array(rank.pivot_counts) for rank in self.ranks])
        return aggr_sum

    def get_pivot_count_sum(self):
        all_sums = [sum(rank.pivot_counts) for rank in self.ranks]
        return sum(all_sums)

    @staticmethod
    def _renegotiate_tree_stage(pivots, widths, fanout: int, num_merged: int):
        rbvec = RenegUtils.load_bins(pivots)
        merged_pivots = RenegUtils.pivot_union(rbvec, widths, fanout)

        pivots_per_rank = len(pivots[0])
        assert(len(widths) == fanout)

        mass_init = (pivots_per_rank - 1) * sum(widths)
        mass_fin = merged_pivots.get_mass()

        assert(ApproxComp.approx_aeqb(mass_init, mass_fin))

        merged_pivots.rebalance(num_merged)

        return merged_pivots

    def renegotiate_tree(self, fanout: List[int], num_merged: List[int]):
        num_leaves = reduce(lambda x, y: x * y, fanout)
        assert(num_leaves == self.num_ranks)

        all_pivots = []
        all_widths = []

        all_masses = []

        for rank in self.ranks:
            pivots, pivot_width = rank.compute_pivots(self.num_pivots_sent)
            assert(len(pivots) == self.num_pivots_sent)

            all_pivots.append(pivots)
            all_widths.append(pivot_width)

            all_masses.append(rank.get_total_produced())

        prev_pivots = all_pivots
        prev_widths = all_widths
        prev_num_per_rank = self.num_pivots_sent

        next_pivots = []
        next_widths = []

        cur_hist = None

        for stage_fanout, stage_merged in zip(fanout, num_merged):
            for chunk_start in range(0, len(prev_pivots), stage_fanout):
                chunk_pivots = prev_pivots[chunk_start:chunk_start + stage_fanout]
                chunk_widths = prev_widths[chunk_start:chunk_start + stage_fanout]

                cur_hist = self._renegotiate_tree_stage(chunk_pivots, chunk_widths,
                                                        stage_fanout, stage_merged)
                next_pivots.append(cur_hist.bin_edges)
                next_widths.append(cur_hist.hist[0])

            prev_pivots = next_pivots
            prev_widths = next_widths

            next_pivots = []
            next_widths = []

        assert(len(prev_pivots) == 1)
        return cur_hist

    def plot(self):
        reneg_bins = self.renegotiate()
        self.ranks_data = flatten(self.ranks_data)

        ref_hist = Histogram(data=self.ranks_produced_flattened, nbins=self.num_pivots_sent)
        ref_hist.rebalance(self.num_bins_final)

        cur_hist = Histogram(data=self.ranks_data, bin_edges=reneg_bins.bin_edges)

        fig, ax = plt.subplots()
        plot1 = ax.bar(range(32), cur_hist.hist)

        mean_load = len(self.ranks_data) / 32
        ax.plot([-1, 32], [mean_load, mean_load], color='orange', linewidth=1)
        ax.text(21, mean_load * 1.05, 'Ideal (balanced) load', color='#c04e01')

        ax.set_xlabel("Rank ID")
        ax.set_ylabel("Load")

        plt.tight_layout()
        plt.savefig("../vis/ASCR/naive_lb_2.pdf")
