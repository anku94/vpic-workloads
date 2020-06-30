import random
import sys
from collections.abc import Iterable

from util import VPICReader, Histogram, ApproxComp
from reneg import Renegotiation
from typing import List, Tuple
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pprint


def log_tailed(x, cutoff):
    if x < cutoff:
        return x

    return cutoff + math.log(1 + (x - cutoff))


def compare_balanced_hists(ref_hist: Histogram, our_hist: Histogram, num_pivots):
    sum_ref = sum(ref_hist.hist)
    sum_our = sum(ref_hist.hist)
    assert(ApproxComp.approx_aeqb(sum_ref, sum_our))

    our_weights = numpy.array(our_hist.hist)
    our_avg = numpy.mean(our_weights)

    our_weights = our_weights * 100.0 / our_avg

    print(["%.2f" % (i,) for i in our_weights])

    our_max = max(our_weights)
    our_min = min(our_weights)

    our_var = numpy.var(our_weights)
    our_std = our_var ** 0.5

    # print("Max: %0.2f%%, Min: %.2f%%, Var: %.2f%%" % (our_max * 100 / our_avg,
    #       our_min * 100 / our_avg, our_var))

    ref_num_pivots = len(ref_hist.hist)
    our_num_pivots = len(our_hist.hist)

    # print("Num Pivots: ", ref_num_pivots, our_num_pivots)
    assert(ref_num_pivots == our_num_pivots)

    print("%s,%.2f,%.2f,%.2f" % (num_pivots, our_max, our_min, our_std))
    return our_max, our_min, our_std

def _bench_pivot_accuracy(num_ranks, time_step, data_path, num_pivots):
    """
    Pivot accuracy for a range of pivots - simple aggregation
    :param num_ranks:
    :param time_step:
    :param data_path:
    :return:
    """

    vpic_reader = VPICReader(data_path, num_ranks=32)
    reneg = Renegotiation(num_ranks, time_step, vpic_reader)
    reneg.set_NUM_PIVOTS(num_pivots)
    reneg.read_all()
    reneg.insert(0.05)

    reneg_bins = reneg.renegotiate()
    ref_hist = Histogram(data=reneg.ranks_produced_flattened, nbins=num_ranks)
    ref_hist.rebalance(num_ranks)

    cur_hist = Histogram(data=reneg.ranks_produced_flattened, nbins=reneg_bins.bin_edges)
    print(["%.2f" % (i,) for i in cur_hist.bin_edges])
    comp_data = compare_balanced_hists(ref_hist, cur_hist, num_pivots)

    return comp_data


def _bench_pivot_subdivide_accuracy(num_ranks, time_step, data_path,
                                    num_pivots, num_subdivisions):
    """
    Pivot accuracy for a range of pivots - more repartitioning
    The distribution generated from 32 ranks is divided num_subdivisions ways
    :param num_ranks:
    :param time_step:
    :param data_path:
    :param num_pivots:
    :param num_subdivisions:
    :return:
    """

    vpic_reader = VPICReader(data_path, num_ranks=32)
    reneg = Renegotiation(num_ranks, time_step, vpic_reader)
    reneg.set_NUM_PIVOTS(num_pivots)
    reneg.set_NUM_FINAL(num_subdivisions)
    reneg.read_all()
    reneg.insert(0.05)

    reneg_bins = reneg.renegotiate()
    ref_hist = Histogram(data=reneg.ranks_produced_flattened, nbins=num_subdivisions)
    ref_hist.rebalance(num_subdivisions)

    cur_hist = Histogram(data=reneg.ranks_produced_flattened, nbins=reneg_bins.bin_edges)
    print(["%.2f" % (i,) for i in cur_hist.bin_edges])
    comp_data = compare_balanced_hists(ref_hist, cur_hist, num_pivots)

    return comp_data


def benchmark_range_accuracy(num_ranks, time_step, data_path):
    all_pivots = [4, 8, 16, 32, 64, 128, 256]
    all_max = []
    all_min = []
    all_std = []

    for num_pivots in all_pivots:
        our_max, our_min, our_std = \
            _bench_pivot_accuracy(num_ranks, time_step, data_path, num_pivots)
        all_max.append(our_max)
        all_min.append(our_min)
        all_std.append(our_std)

    # all_max = [200] * 7
    # all_min = [90] * 7
    # all_std = [5] * 7

    fig, ax = plt.subplots(1, 1)
    print(ax)

    ax.plot(all_pivots, all_min, 'x-', label='Min')
    ax.plot(all_pivots, all_max, 'x-', label='Max')
    ax.plot(all_pivots, all_std, 'x-', label='Stdev')

    ax.plot(all_pivots, [100] * len(all_pivots), ',--')
    ax.plot(all_pivots, [0] * len(all_pivots), ',--')

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(all_pivots)
    ax.set_xlabel("Number of samples from each rank")

    ax.legend(loc='center left')

    # plt.show()
    # plt.savefig(
    #     '../vis/renegbench/samplesvsacc.1024from32.ts%s.png'
    #     % (time_step,)
    # )
    return


def benchmark_range_subdivide_accuracy(num_ranks, time_step, data_path):
    num_pivots = 256
    all_subdivisions = [32, 64, 128, 256, 512, 1024, 2048]
    all_max = []
    all_min = []
    all_std = []

    for num_subdivisions in all_subdivisions:
        our_max, our_min, our_std = \
            _bench_pivot_subdivide_accuracy(num_ranks, time_step,
                                            data_path, 256, num_subdivisions)
        all_max.append(our_max)
        all_min.append(our_min)
        all_std.append(our_std)

    # all_max = [200] * 7
    # all_min = [90] * 7
    # all_std = [5] * 7

    fig, ax = plt.subplots(1, 1)
    print(ax)

    ax.plot(all_subdivisions, all_min, 'x-', label='Min')
    ax.plot(all_subdivisions, all_max, 'x-', label='Max')
    ax.plot(all_subdivisions, all_std, 'x-', label='Stdev')

    ax.plot(all_subdivisions, [100] * len(all_subdivisions), ',--')
    ax.plot(all_subdivisions, [0] * len(all_subdivisions), ',--')

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(all_subdivisions)
    ax.set_xlabel("Number of subdivisions from %d*%d samples" % (num_pivots,32,))

    ax.legend(loc='center left')

    # plt.show()
    plt.savefig(
        '../vis/renegbench/subdivvsacc.simple.ts%s.png'
        % (time_step,)
    )
    return

def _get_hist_stats(hist: Histogram):
    our_weights = numpy.array(hist.hist)
    our_avg = numpy.mean(our_weights)

    our_weights = our_weights * 100.0 / our_avg

    our_max = max(our_weights)
    our_min = min(our_weights)

    our_var = numpy.var(our_weights)
    our_std = our_var ** 0.5

    return our_max, our_min, our_std


def benchmark_rtp_vs_reg(num_ranks: int, timesteps: List[int], data_path: str,
                         rtp_pivots: List[int]):
    stddev_reneg = []
    stddev_rtp = []

    for ts in timesteps:
        vpic_reader = VPICReader(data_path, num_ranks=32)
        reneg = Renegotiation(num_ranks, ts, vpic_reader)
        reneg_pivots = 256
        reneg.set_NUM_PIVOTS(256)
        reneg.set_NUM_FINAL(32)
        reneg.read_all()
        reneg.insert(0.05)

        reneg_bins = reneg.renegotiate()
        rtp_bins = reneg.renegotiate_tree([4, 4, 2], rtp_pivots)
        print(reneg_bins)
        print(rtp_bins)

        reneg_hist = Histogram(data=reneg.ranks_produced_flattened, nbins=reneg_bins.bin_edges)
        rtp_hist = Histogram(data=reneg.ranks_produced_flattened, nbins=rtp_bins.bin_edges)

        _, _, reneg_stddev = _get_hist_stats(reneg_hist)
        _, _, rtp_stddev = _get_hist_stats(rtp_hist)

        stddev_reneg.append(reneg_stddev)
        stddev_rtp.append(rtp_stddev)

    fig, ax = plt.subplots(1, 1)

    ax.plot(timesteps, stddev_reneg, 'x-', label='Naive Stddev')
    ax.plot(timesteps, stddev_rtp, 'x-', label='RTP Stddev')
    ax.plot(timesteps, [0] * len(timesteps), '--')
    ax.set_xticks(timesteps)
    ax.set_xlabel('Timestep Index')
    ax.legend(loc='upper left')

    # plt.show()
    plt.savefig(
        '../vis/renegbench/renegvsrtp.%s.%s.png' %
        (reneg_pivots, '.'.join([str(i) for i in rtp_pivots]))
    )


def _eval_pivots(data, hist) -> Tuple[float, float, float]:
    nphist, bin_edges = numpy.histogram(data, hist.bin_edges)
    norm_hist = nphist / numpy.mean(nphist)
    norm_hist_std = numpy.var(norm_hist) ** 0.5
    print("HistSize: %s, Var: %0.3f" %(len(nphist), norm_hist_std))

    # print(norm_hist)
    # bins = norm_hist
    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # fig, ax = plt.subplots(1, 1)
    # ax.bar(center, norm_hist, align='center', width=width)
    # ax.set_title('Sample Load Imbalance for Std: %.2f' % (norm_hist_std,))
    # ax.savefig('../vis/renegbench/sampleimbalance.eps', dpi=600)
    # sys.exit(0)

    perfect_hist = Histogram(bin_weights=nphist, bin_edges=bin_edges)
    perfect_hist.rebalance(len(nphist))
    perf_hist, perf_edges = numpy.histogram(data, perfect_hist.bin_edges)
    norm_perf_hist = perf_hist / numpy.mean(perf_hist)

    # print('Perfect hist: ', _pprint_float(norm_perf_hist))
    # print("PerfHistSize: %s, Var: %0.3f" %(len(nphist), numpy.var(norm_perf_hist) ** 0.5,))
    return min(norm_hist), max(norm_hist), norm_hist_std


def benchmark_predictive_power(num_ranks: int, timestep: int, data_path: str, ax,
                               num_chunks: int):
    """
    Compute how well each x% predicts the next x%
    :param num_ranks:
    :param timestep:
    :param data_paath:
    :return:
    """
    ts = timestep
    num_ranks=32
    vpic_reader = VPICReader(data_path, num_ranks=num_ranks)
    reneg = Renegotiation(num_ranks, ts, vpic_reader)
    reneg_pivots = 256
    reneg.set_num_pivots_sent(256)
    reneg.set_num_pivots_stored(256)
    reneg.set_num_bins_final(32)
    reneg.read_all()

    old_data = None
    old_pivots = None

    all_min = []
    all_max = []
    all_std = []

    for chunk_idx in range(num_chunks):
        print('Iteration %s' % (chunk_idx,))
        read_per_iter = 1.0 / num_chunks
        cur_data = reneg.peek_ahead(read_per_iter)
        reneg.insert(read_per_iter)

        if old_pivots:
            min_val, max_val, std_val = \
                _eval_pivots(cur_data, old_pivots)
            all_min.append(min_val * 100)
            all_max.append(max_val * 100)
            all_std.append(std_val * 100)

        new_pivots = reneg.renegotiate()
        reneg.update_pivots(new_pivots)

        old_pivots = new_pivots
        old_data = cur_data

    x_values = [(i + 1) * 1.0 / num_chunks for i in range(num_chunks - 1)]

    ax.plot(x_values, all_min, 'x-', label='Min')
    ax.plot(x_values, all_max, 'x-', label='Max')
    ax.plot(x_values, all_std, 'x-', label='Std')

    ax.plot([0, 1], [100, 100], '--')
    ax.plot([0, 1], [0, 0])
    ax.set_ylim([0, 200])

    ax.set_xlabel('Experiment Progress')
    ax.set_ylabel('Percent')
    ax.legend(loc='upper right')

    ax.set_title('Timestep Index: {0}, '
                 'Renegotiation Frequency: {1:.1f}%'
                 .format(ts, 100.0/num_chunks))


def benchmark_predictive_power_adaptive(num_ranks: int, timestep: int,
                                        data_path: str, ax, num_chunks: int):
    """
    Compute how well each x% predicts the next x%
    :param num_ranks:
    :param timestep:
    :param data_paath:
    :return:
    """
    ts = timestep
    vpic_reader = VPICReader(data_path, num_ranks=32)
    reneg = Renegotiation(num_ranks, ts, vpic_reader)
    reneg_pivots = 256
    reneg.set_num_pivots_sent(256)
    reneg.set_num_pivots_stored(256)
    reneg.set_num_bins_final(32)
    reneg.read_all()

    old_data = None
    old_pivots = None

    all_min = []
    all_max = []
    all_std = []
    all_skew = []

    reneg_points = []

    reneg_threshold = 0.10

    for chunk_idx in range(num_chunks):
        print('Iteration %s' % (chunk_idx,))
        read_per_iter = 1.0 / num_chunks
        cur_data = reneg.peek_ahead(read_per_iter)
        reneg.insert(read_per_iter)

        if chunk_idx == 0:
            new_pivots = reneg.renegotiate()
            reneg.update_pivots(new_pivots)

            old_pivots = new_pivots
            old_data = cur_data

            continue

        if old_pivots:
            min_val, max_val, std_val = \
                _eval_pivots(cur_data, old_pivots)
            all_min.append(min_val * 100)
            all_max.append(max_val * 100)
            all_std.append(std_val * 100)

            cur_skew = reneg.get_skew(window=True)
            all_skew.append(cur_skew * 100)

            if cur_skew > reneg_threshold:
                new_pivots = reneg.renegotiate()
                reneg.update_pivots(new_pivots)

                reneg_points.append(chunk_idx)

                old_pivots = new_pivots
                old_data = cur_data

    x_values = [(i + 1) * 1.0 / num_chunks for i in range(num_chunks - 1)]

    ax.plot(x_values, all_min, '-', label='Min')
    ax.plot(x_values, all_max, '-', label='Max')
    ax.plot(x_values, all_std, '-', label='Std')
    ax.plot(x_values, all_skew, '-', label='LBI')

    filter_indices = lambda arr, idx_arr: [arr[idx-1] for idx in idx_arr]
    x_filtered = filter_indices(x_values, reneg_points)
    all_min_filtered = filter_indices(all_min, reneg_points)
    all_max_filtered = filter_indices(all_max, reneg_points)
    all_std_filtered = filter_indices(all_std, reneg_points)
    all_skew_filtered = filter_indices(all_skew, reneg_points)

    # ax.plot(x_filtered, all_min_filtered, 'ob')
    # ax.plot(x_filtered, all_max_filtered, 'og')
    # ax.plot(x_filtered, all_std_filtered, 'or')
    ax.plot(x_filtered, all_skew_filtered, 'o', color='orange')

    print(x_values, all_std)
    print(reneg_points)

    rt100 = reneg_threshold * 100

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 200])
    ax.plot([0, 1], [100, 100], '--')
    ax.plot([0, 1], [rt100, rt100], '--', color='crimson')
    ax.plot(0, rt100, 'go')
    plt.annotate("{0}%".format(rt100), xy=(0, rt100),
                 xytext=(-35, -4), textcoords='offset points', color='crimson')
    ax.plot([0, 1], [0, 0])

    ax.set_xlabel('Experiment Progress')
    ax.set_ylabel('Percent')
    ax.legend(loc='upper right')

    ax.set_title('tsidx: {0}, '
                 'Dynamic Reneg Freq: {1:.1f}%, ({2} rounds)'
                 .format(ts, 100.0/num_chunks, len(reneg_points)))


def plot_rand_ax(ax):
    x = range(20)
    y = [100 + random.randint(-20, 20) for i in range(20)]
    ax.plot(x, y)
    num_chunks = 20

    ts = 0

    reneg_points = [2, 5, 6, 7, 9]

    rt100 = 10.0

    ax.plot([0, 1], [100, 100], '--')
    ax.plot([0, 1], [rt100, rt100], '--', color='crimson')
    ax.plot(0, rt100, 'go')
    plt.annotate("{0}%".format(rt100), xy=(0, rt100),
                 xytext=(-35, -4), textcoords='offset points', color='crimson')
    ax.plot([0, 1], [0, 0])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 200])

    ax.set_xlabel('Experiment Progress')
    ax.set_ylabel('Percent')
    ax.legend(loc='upper right')

    ax.set_title('tsidx: {0}, '
                 'Dynamic Reneg Freq: {1:.1f}%, ({2} rounds)'
                 .format(ts, 100.0/num_chunks, len(reneg_points)))
    return


def benchmark_predictive_power_suite():
    # for num_chunks in [20, 100, 1000]:
    for num_chunks in [100]:
        for ts in range(4):
            fig, ax = plt.subplots(1, 1)
            # plot_rand_ax(ax)
            # benchmark_predictive_power(32, ts, '../data', ax, num_chunks)
            benchmark_predictive_power_adaptive(32, ts, '../data', ax, num_chunks)
            plt.savefig(
                '../vis/renegbench/renegsim.dynamic.2k.times%s.ts%s.eps'
                % (num_chunks, ts), dpi=600
            )
            # plt.show()
            # break
            # sys.exit(0)

    # for num_chunks in [100]:
    #     for ts in range(4):
    #         fig, ax = plt.subplots(1, 1)
    #         # plot_rand_ax(ax)
    #         benchmark_predictive_power(32, ts, '../data', ax, num_chunks)
    #         # plt.savefig(
    #         #     '../vis/renegbench/renegsim.dynamic.nowindow.times%s.ts%s.eps'
    #         #     % (num_chunks, ts), dpi=600
    #         # )
    #         plt.show()
    #         sys.exit(0)
    pass
    pass


def _subarray_len(array: List):
    return len(array), [len(l) for l in array]


def _pprint_float(item):
    if isinstance(item, Iterable):
        return ', '.join(['%0.3f' % elem for elem in item])
    elif isinstance(item, float):
        return '%0.3f' % item
    else:
        raise Exception('Unknown type: %s' % type(item))


def rankwise_pivot_fit_check(num_ranks, timestep, data_path):
    """
    For each rank, check how well its pivots fit its own data

    This is more of a sanity check evaluation than anything else.
    Each rank's pivtos must fit that rank's data, since that's what
    the pivots are derived from. This test simply verifies the fact
    for all ranks

    :param num_ranks:
    :param timestep:
    :param data_path:
    :return:
    """
    ts = timestep
    vpic_reader = VPICReader(data_path, num_ranks=32)
    reneg = Renegotiation(num_ranks, ts, vpic_reader)
    reneg_pivots = 256
    reneg.set_num_bins_final(32)
    reneg.set_num_pivots_sent(256)
    reneg.set_num_pivots_stored(256)
    reneg.read_all()

    pp = pprint.PrettyPrinter(width=40, compact=True)

    reneg.insert(0.05)
    print("Len produced: %s %s" % (_subarray_len(reneg.ranks_produced)))
    print("Len produced across ranks: %s", len(reneg.ranks_produced_flattened))
    pivots = reneg.renegotiate()

    reneg.update_pivots(pivots)

    reneg.insert(0.05)
    produced_0, produced_1 = _subarray_len(reneg.ranks_produced)
    print("Len produced: {0} {1}".format(produced_0, produced_1))
    # pivots = reneg.renegotiate()

    print('---------------')

    all_stds = []
    num_pivots = 32

    for ridx in range(num_ranks):
        # ridx = 16
        print('Rank {0}'.format(ridx))
        rank = reneg.ranks[ridx]
        pivots, pivot_width = rank.compute_pivots(num_pivots)
        print(_pprint_float(pivots))
        print(pivot_width, len(reneg.ranks_produced[ridx]))
        pivot_hist = Histogram(bin_edges=pivots, bin_weights=pivot_width)
        pivot_data = reneg.ranks_produced[ridx]
        rank_std = _eval_pivots(pivot_data, pivot_hist)
        all_stds.append(rank_std)
        # break

    print('---------------')
    print(_pprint_float(all_stds))

if __name__ == "__main__":
    # for ts in [0, 1, 2, 3]:
    #     benchmark_range_accuracy(32, ts, "../data")

    # for ts in [0, 1, 2, 3]:
    #     benchmark_range_subdivide_accuracy(32, ts, "../data")

    # all_rtp_pivots = [
    #     [256, 256, 32],
    #     [256, 512, 32],
    #     [512, 256, 32],
    #     [512, 512, 32]
    # ]
    # for rtp_pivots in all_rtp_pivots:
    #     benchmark_rtp_vs_reg(32, [0, 1, 2, 3], "../data", rtp_pivots=rtp_pivots)

    # rankwise_pivot_fit_check(32, 0, "../data")
    benchmark_predictive_power_suite()
