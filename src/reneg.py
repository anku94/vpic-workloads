from util import VPICReader, RenegUtils, Histogram, ApproxComp
from rank import Rank
import itertools, numpy

import matplotlib.pyplot as plt

NUM_PIVOTS = 4
NUM_FINAL = 32


def flatten(ls: list):
    return list(itertools.chain.from_iterable(ls))


class Renegotiation:
    vpic_reader = None
    num_ranks = None
    time_step = None

    ranks = []
    ranks_data = []
    ranks_produced = []
    ranks_cursors = []

    def __init__(self, num_ranks: int, timestep: int,
                 vpic_reader: VPICReader) -> None:
        self.vpic_reader = vpic_reader
        self.num_ranks = num_ranks
        self.time_step = timestep

        self.ranks = [Rank(vpic_reader, ridx) for ridx in range(num_ranks)]

        return

    @staticmethod
    def set_NUM_PIVOTS(new_num_pivots: int):
        global NUM_PIVOTS
        NUM_PIVOTS = new_num_pivots

    def read_a_particle(self) -> None:
        return

    def read_all(self) -> None:
        assert(len(self.ranks_data) == 0)

        for rank in self.ranks:
            rank_data = self.vpic_reader.read_a_rank(self.time_step, rank.rank_id)
            self.ranks_data.append(rank_data)
            self.ranks_cursors.append(0)

        return

    def insert(self, percent: float) -> None:
        for ridx in range(self.num_ranks):
            self._produce_at_rank(ridx, percent)

        return

    def _produce_at_rank(self, rank_id: int, percent: float):
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

        self.ranks_produced.extend(data_to_produce)

        # print("Inserting {0} elems for Rank {1}"
        #       .format(new_pos - cur_pos, rank_id))

        self.ranks[rank_id].produce(data_to_produce)
        self.ranks_cursors[rank_id] = new_pos
        return

    def renegotiate(self) -> Histogram:
        all_pivots = []
        all_widths = []

        all_masses = []

        for rank in self.ranks:
            pivots, pivot_width = rank.compute_pivots(NUM_PIVOTS)
            assert(len(pivots) == NUM_PIVOTS)

            all_pivots.append(pivots)
            all_widths.append(pivot_width)

            all_masses.append(rank.get_total_produced())

        rbvec = RenegUtils.load_bins(all_pivots)
        merged_pivots = RenegUtils.pivot_union(rbvec, all_widths,
                                               self.num_ranks)
        mass_init = (NUM_PIVOTS - 1) * sum(all_widths)
        mass_fin = merged_pivots.get_mass()

        assert(abs(mass_fin - mass_init) < 1e-3)

        # print(all_pivots, all_widths)
        # print(merged_pivots.bin_edges, merged_pivots.hist)

        merged_pivots.rebalance(NUM_FINAL)
        mass_reb = merged_pivots.get_mass()

        assert(ApproxComp.approx_aeqb(mass_init, sum(all_masses)))
        assert(ApproxComp.approx_aeqb(mass_init, mass_fin))
        assert(ApproxComp.approx_aeqb(mass_init, mass_reb))

        # print(merged_pivots.bin_edges, merged_pivots.hist)

        return merged_pivots

    def plot(self):
        reneg_bins = self.renegotiate()
        self.ranks_data = flatten(self.ranks_data)

        ref_hist = Histogram(data=self.ranks_produced, nbins=NUM_PIVOTS)
        ref_hist.rebalance(NUM_FINAL)

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
