from util import VPICReader, Histogram, ApproxComp
from reneg import Renegotiation
import numpy


def compare_balanced_hists(ref_hist: Histogram, our_hist: Histogram):
    sum_ref = sum(ref_hist.hist)
    sum_our = sum(ref_hist.hist)
    assert(ApproxComp.approx_aeqb(sum_ref, sum_our))

    our_weights = our_hist.hist
    our_max = max(our_weights)
    our_min = min(our_weights)
    our_avg = numpy.mean(our_weights)
    our_var = numpy.var(our_weights)

    # print("Max: %0.2f%%, Min: %.2f%%, Var: %.2f%%" % (our_max * 100 / our_avg,
    #       our_min * 100 / our_avg, our_var))

    ref_num_pivots = len(ref_hist.hist)
    our_num_pivots = len(our_hist.hist)

    # print("Num Pivots: ", ref_num_pivots, our_num_pivots)
    assert(ref_num_pivots == our_num_pivots)
    num_pivots = our_num_pivots

    print("%s,%.2f,%.2f,%.2f" % (num_pivots, our_max * 100 / our_avg,
                           our_min * 100 / our_avg, our_var / our_avg))


def bench_pivot_accuracy(num_ranks, time_step, data_path):
    """
    Pivot accuracy for a range of pivots - simple aggregation
    :param num_ranks:
    :param time_step:
    :param data_path:
    :return:
    """

    vpic_reader = VPICReader(data_path, num_ranks=4)
    reneg = Renegotiation(num_ranks, time_step, vpic_reader)
    reneg.set_NUM_PIVOTS(256)
    reneg.read_all()
    reneg.insert(0.05)

    reneg_bins = reneg.renegotiate()
    ref_hist = Histogram(data=reneg.ranks_produced, nbins=num_ranks)
    ref_hist.rebalance(num_ranks)

    cur_hist = Histogram(data=reneg.ranks_produced, nbins=reneg_bins.bin_edges)
    compare_balanced_hists(ref_hist, cur_hist)
    return


if __name__ == "__main__":
    bench_pivot_accuracy(32, 0, "../data")
