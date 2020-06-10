import sys

from util import VPICReader
from typing import List, Tuple
import bisect


class Rank:
    """
    A renegotiating rank

    Mostly to be operated by Renegotiation

    :param vpic_reader:
    :param rank_id:
    :param pivot_precision:
    """
    def __init__(self, vpic_reader: VPICReader, rank_id: int, pivot_precision: int = None) -> None:
        self.rank_id = rank_id
        self.vpic_reader = vpic_reader

        self.pivots = None
        self.pivot_counts = None

        if pivot_precision:
            self.pivot_precision = pivot_precision

        self.oob_left = []
        self.oob_right = []
        return

    def get_id(self) -> int:
        return self.rank_id

    def read(self, timestep: int) -> None:
        data = self.vpic_reader.read_a_rank(timestep, self.rank_id)
        return

    def produce(self, data: List[float]) -> None:
        for elem in data:
            if self.pivots is None or elem < self.pivots[0]:
                self.oob_left.append(elem)
            elif elem >= self.pivots[-1]:
                self.oob_right.append(elem)
            else:
                assert(self.pivots is not None)
                assert(self.pivot_counts is not None)
                bidx = bisect.bisect_left(self.pivots, elem) - 1
                self.pivot_counts[bidx] += 1
        pivot_sum = 0 if self.pivot_counts is None else sum(self.pivot_counts)
        # print('Producing at %s - %s - sum - %s' % (self.rank_id, len(data), pivot_sum))

    def get_total_produced(self) -> float:
        oob_left_sz = len(self.oob_left)
        shuffled_sz = sum(self.pivot_counts) if self.pivot_counts else 0
        oob_right_sz = len(self.oob_right)

        return oob_left_sz + shuffled_sz + oob_right_sz

    # def _update_pivots(self, new_pivots: List[float]) -> None:
    #     self.pivots = new_pivots
    #     self.pivot_counts = [0] * (len(new_pivots) - 1)
    #
    # def _flush_oobs(self) -> None:
    #     oobl = self.oob_left
    #     self.oob_left = []
    #     oobr = self.oob_right
    #     self.oob_right = []
    #     self.produce(oobl)
    #     self.produce(oobr)

    def update_and_flush(self, new_pivots: List[float]) -> None:
        self.pivots = new_pivots
        self.pivot_counts = [0] * (len(new_pivots) - 1)

        oobl = self.oob_left
        self.oob_left = []
        oobr = self.oob_right
        self.oob_right = []

        self.produce(oobl)
        self.produce(oobr)

        # Reset anyway, since we don't use counts from older rounds anymore
        self.pivot_counts = [0] * (len(new_pivots) - 1)

    def compute_pivots(self, num_pivots: int) -> Tuple[List[float], float]:
        assert(num_pivots > 2)

        self.oob_left.sort()
        self.oob_right.sort()

        new_pivots = [0.0] * num_pivots

        oobl = self.oob_left
        oobr = self.oob_right

        oobl_sz = len(oobl)
        #pvt_sz = 0 if self.pivots is None else sum(self.pivots)
        pvt_sz = 0 if self.pivots is None else sum(self.pivot_counts)
        oobr_sz = len(oobr)

        # print('OOB: ', oobl_sz, pvt_sz, oobr_sz)
        range_start = 0
        range_end = 0

        if self.pivots is None:
            range_start = oobl[0] if oobl_sz > 0 else 0
            range_end = oobl[-1] if oobl_sz > 0 else 0
        elif pvt_sz > 1e-3:
            range_start = self.pivots[0]
            range_end = self.pivots[-1]

        if oobl_sz > 0:
            range_start = oobl[0]
            if pvt_sz < 1e-3 and oobr_sz == 0:
                range_end = oobl[-1]

        if oobr_sz > 0:
            range_end = oobr[-1]
            if pvt_sz < 1e-3 and oobl_sz == 0:
                range_start = oobr[0]

        assert(range_end >= range_start)

        new_pivots[0] = range_start
        new_pivots[-1] = range_end

        total_mass = oobl_sz + pvt_sz + oobr_sz
        mass_per_pivot = total_mass * 1.0 / (num_pivots - 1)

        # print("Total mass: ", oobl_sz, pvt_sz, oobr_sz, "MPP: ", mass_per_pivot)
        # print("Pivot start, end: ", range_start, range_end)

        cur_pivot = 1
        if mass_per_pivot < 1e-3:
            raise Exception("Probably fill pivots with zero?")

        pvt_ss = self.pivots
        pvcnt_ss = self.pivot_counts
        accumulated_ppp = 0.0
        particles_carried_over = 0.0

        oob_index = 0

        while True:
            part_left = oobl_sz - oob_index
            if mass_per_pivot < 1e-3 or part_left < mass_per_pivot:
                particles_carried_over += part_left
                break
            accumulated_ppp += mass_per_pivot
            # XXX: round semantics?
            cur_part_idx = round(accumulated_ppp)
            new_pivots[cur_pivot] = oobl[cur_part_idx]
            cur_pivot += 1

            oob_index = cur_part_idx + 1

        bin_idx = 0
        # assert state is RENEGO - why? makes no sense

        if pvt_ss is not None:
            for bidx in range(len(pvt_ss) - 1):
                cur_bin_left = pvcnt_ss[bidx]
                bin_start = pvt_ss[bidx]
                bin_end = pvt_ss[bidx + 1]

                while particles_carried_over + cur_bin_left >= mass_per_pivot - 1e-05:
                    # print('Loop condition: A: {0}, B: {1}'.format(particles_carried_over + cur_bin_left,
                    #       mass_per_pivot - 1e05))

                    take_from_bin = mass_per_pivot - particles_carried_over

                    # advance bin_start s.t. take_from_bin is removed
                    bin_width = bin_end - bin_start
                    width_to_remove = take_from_bin / cur_bin_left * bin_width
                    # print('TFB: {0}, CBL: {1}, BW: {2}'.format(take_from_bin,
                    #                                            cur_bin_left, bin_width))
                    bin_start += width_to_remove
                    new_pivots[cur_pivot] = bin_start

                    cur_pivot += 1

                    cur_bin_left -= take_from_bin
                    particles_carried_over = 0

                assert(cur_bin_left >= -1e-3)

                particles_carried_over += cur_bin_left

        oob_index = 0

        while True:
            part_left = oobr_sz - oob_index
            if (mass_per_pivot < 1e-3 or
                    part_left + particles_carried_over < mass_per_pivot - 1e-3):
                particles_carried_over += part_left
                break

            next_idx = oob_index + mass_per_pivot - particles_carried_over

            particles_carried_over = 0

            cur_part_idx = round(next_idx)
            if cur_part_idx >= oobr_sz:
                cur_part_idx = oobr_sz - 1

            new_pivots[cur_pivot] = oobr[cur_part_idx]
            cur_pivot += 1
            oob_index = cur_part_idx + 1

        while cur_pivot < num_pivots - 1:
            new_pivots[cur_pivot] = pvt_ss[-1]
            cur_pivot += 1

        pivot_width = mass_per_pivot

        return new_pivots, pivot_width

    @staticmethod
    def repartition_bin_counts(pivots_old: List[float],
                               counts_old: List[int],
                               pivots_new: List[float],
                               counts_new: List[int]) -> None:
        assert(counts_new is not None and len(counts_new) == 0)

        op_sz = len(pivots_old) - 1
        np_sz = len(pivots_new) - 1

        counts_new.extend([0] * np_sz)

        oidx = 0
        nidx = 0

        npvt_sum = 0
        opvt_left = counts_old[0]

        while True:
            if oidx == op_sz:
                break

            if nidx == np_sz:
                break

            ops = pivots_old[oidx]  # old pivot start
            ope = pivots_old[oidx + 1]  # old pivot end
            opw = ope - ops  # old pivot width

            opc = counts_old[oidx]  # old pivot count

            nps = pivots_new[nidx]
            npe = pivots_new[nidx + 1]
            npw = npe - nps

            if ope <= nps:
                oidx += 1
                continue

            if npe <= ops:
                counts_new[nidx] = npvt_sum
                npvt_sum = 0
                nidx += 1
                continue

            # Condition on types of overlap
            if ops <= nps and ope <= npe:
                # Left overlap
                npvt_sum += (ope - nps) * opc / opw
                oidx += 1
            elif ops <= nps and ope > npe:
                # OP engulfs NP
                npvt_sum += (npw * opc / opw)
                counts_new[nidx] = npvt_sum
                npvt_sum = 0
                nidx += 1
            elif ops > nps and ope <= npe:
                # NP engulfs OP
                npvt_sum += opc
                oidx += 1
            else:
                # Right overlap
                npvt_sum += (npe - ops) * opc / opw
                counts_new[nidx] = npvt_sum
                npvt_sum = 0
                nidx += 1

        if nidx < np_sz:
            counts_new[nidx] = npvt_sum

        osum_temp = sum(counts_old)
        nsum_temp = sum(counts_new)

        assert(abs(osum_temp - nsum_temp) < 1e-3)

        return
