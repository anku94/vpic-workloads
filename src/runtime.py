import glob
import matplotlib.pyplot as plt
import numpy

from util import chunk_it


class RuntimeLog:
    def __init__(self, data_path: str):
        self.data_path = data_path
        rank_files = glob.glob(data_path + '/*perfstats.log*')
        self.num_ranks = len(rank_files)

        self.load_data()

    def load_data(self):
        data = [self._load_rank(ridx) for ridx in range(self.num_ranks)]
        data_len = map(lambda x: len(x), data)
        data_len = list(data_len)
        print(data_len)

        common_len = min(data_len)
        data = [numpy.array(data[ridx][:common_len]) for ridx in range(self.num_ranks)]
        self.data = data

    def plot_data(self, plot_path, skew_pct):
        data = sum(self.data)

        fig, ax = plt.subplots(1, 1)
        skew_str = 'none' if int(skew_pct) == 0 else skew_pct
        ax.set_title('Aggregate Disk B/W for %d ranks, skew %s' % (self.num_ranks,skew_str))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('B/W (MB/s)')
        ax.plot(data)

        # fig.savefig('../vis/runtime/diskbw.temp.16x16.0s0.aggr.eps', dpi=600)

        rankwise_data = chunk_it(self.data, 16)
        rankwise_len = max(map(lambda x: len(x), rankwise_data))
        rankwise_band_min = numpy.array([99999] * rankwise_len)
        rankwise_band_max = numpy.array([0] * rankwise_len)
        num_ranks = len(rankwise_data)

        for ridx in range(num_ranks):
            cur_data = rankwise_data[ridx]
            print(cur_data)
            rankwise_band_max = numpy.maximum(rankwise_band_max, cur_data)
            rankwise_band_min = numpy.minimum(rankwise_band_min, cur_data)

        print(rankwise_band_min, rankwise_band_max)

        ax2 = ax.twinx()
        ax2_color=(0.8, 0.2, 0.2, 0.8)
        ax2.fill_between(range(rankwise_len),
                        rankwise_band_min,
                        rankwise_band_max,
                        facecolor=ax2_color)
        ax2.tick_params(axis='y', labelcolor=ax2_color)

        # fig, axes = plt.subplots(2, 2)
        # num_ranks = 4
        # for ridx in range(num_ranks):
        #     y = int(ridx/2)
        #     x = int(ridx%2)
        #     ax = axes[y][x]
        #     ax.plot(rankwise_data[ridx])
        #     ax.set_title('Node %d' % (ridx,))
        #     ax.set_xlabel('Time (seconds)')
        #     ax.set_ylabel('B/W (MB/s)')

        fig.tight_layout()
        fig.savefig(plot_path, dpi=600)

    @staticmethod
    def _adjacent_diff(data):
        data = map(lambda x: float(x), data)
        data = list(data)

        new_data = []
        prev = None
        for elem in data[::10]:
            if prev is None:
                prev = elem
            else:
                new_data.append(elem - prev)
                prev = elem
        return new_data

    def _load_rank(self, rank_id):
        fpath = self.data_path + '/vpic-perfstats.log.%d' % (rank_id)
        fdata = open(fpath, 'r').read().splitlines()

        fdata = fdata[1:]
        fdata = map(lambda x: x.split(','), fdata)
        fdata = list(zip(*fdata))

        intervals = self._adjacent_diff(fdata[0])
        MBps = numpy.array(self._adjacent_diff(fdata[1])) / (1024 * 1024)

        return MBps
