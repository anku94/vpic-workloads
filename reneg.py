from util import VPICReader
from typing import List
import bisect


class Rank:
    rankId = None
    vpicReader = None

    pivots = None
    pivotCounts = None
    oobLeft = []
    oobRight = []

    def __init__(self, vpic_reader: VPICReader, rank_id: int) -> None:
        self.rankId = rank_id
        self.vpicReader = vpic_reader
        return

    def get_id(self) -> int:
        return self.rankId

    def read(self, timestep: int) -> None:
        data = self.vpicReader.read_a_rank(timestep, self.rankId)
        print(len(data))
        print(data[:10])
        return

    def insert(self, data: List[float]) -> None:
        for elem in data:
            if self.pivots is None or elem < self.pivots[0]:
                self.oobLeft.append(elem)
            elif elem >= self.pivots[-1]:
                self.oobRight.append(elem)
            else:
                assert(self.pivots is not None)
                assert(self.pivotCounts is not None)
                bidx = bisect.bisect_left(self.pivots, elem) - 1
                self.pivotCounts[bidx] += 1

    def update_pivots(self, new_pivots: List[float]) -> None:
        self.pivots = new_pivots
        # XXX: Not exactly - TODO: repartition old counts
        self.pivotCounts = [0] * (len(new_pivots) - 1)

    def flush_oobs(self) -> None:
        oobl = self.oobLeft
        self.oobLeft = []
        oobr = self.oobRight
        self.oobRight = []
        self.insert(oobl)
        self.insert(oobr)

    def compute_pivots(self, num_pivots: int) -> list:
        assert(self.pivots is not None)
        assert(num_pivots > 2)

        new_pivots = [None] * num_pivots

        oobl_sz = len(self.oobLeft)
        pvt_sz = 0 if self.pivots is None else sum(self.pivots)
        oobr_sz = len(self.oobRight)

        oobl = self.oobLeft
        oobr = self.oobRight

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

        print("----------")

        print("mass: ", total_mass, mass_per_pivot)

        cur_pivot = 1
        if mass_per_pivot < 1e-3:
            raise Exception("Probably fill pivots with zero?")

        pvt_ss = self.pivots
        pvcnt_ss = self.pivotCounts
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
                cur_bin_left = pvt_ss[bidx]
                bin_start = pvt_ss[bidx]
                bin_end = pvt_ss[bidx]

                while particles_carried_over + cur_bin_left >= mass_per_pivot - 1e05:
                    take_from_bin = mass_per_pivot - particles_carried_over

                    # advance bin_start s.t. take_from_bin is removed
                    bin_width = bin_end - bin_start
                    width_to_remove = take_from_bin / cur_bin_left * bin_width

                    bin_start += width_to_remove
                    new_pivots[cur_pivot] = bin_start

                    cur_pivot += 1

                    cur_bin_left -= take_from_bin
                    particles_carried_over = 0

                assert(cur_bin_left >= -1e-3)

                particles_carried_over += cur_bin_left

        oob_index = 0

        while True:
            part_left = oobr_sz = oob_index
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

        return new_pivots

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


class Renegotiation:
    vpicReader = None
    numRanks = None

    def __init__(self, numRanks, vpicReader: VPICReader) -> None:
        self.vpicReader = vpicReader
        self.numRanks = numRanks

    def read_a_particle(self) -> None:
        return
