import glob
import matplotlib.pyplot as plt
import numpy


def chunk_it(ls, nchunks):
    chunk_size = int(len(ls) / nchunks)
    print('chunk size: ', chunk_size)
    chunks = []
    for chunk_start in range(0, len(ls), nchunks):
        ls_chunk = ls[chunk_start:chunk_start + chunk_size]
        ls_chunk = sum(ls_chunk)
        chunks.append(ls_chunk)
    return chunks


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

    def plot_data(self):
        data = sum(self.data)

        fig, ax = plt.subplots(1, 1)
        ax.set_title('Aggregate Disk B/W for %d ranks, skew none' % (self.num_ranks,))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('B/W (MB/s)')
        ax.plot(data)

        fig.savefig('../vis/runtime/diskbw.16x16.0s0.aggr.eps', dpi=600)

        # rankwise_data = chunk_it(self.data, 4)
        #
        # fig, axes = plt.subplots(2, 2)
        # num_ranks = 4
        # for ridx in range(num_ranks):
        #     y = int(ridx/2)
        #     x = int(ridx%2)
        #     ax = axes[y][x]
        #     ax.plot(rankwise_data[ridx])
        #     ax.set_title('Rank %d' % (ridx,))
        #     ax.set_xlabel('Time (seconds)')
        #     ax.set_ylabel('B/W (MB/s)')
        #
        # fig.tight_layout()
        # fig.savefig('../vis/runtime/diskbw.64x1.8s60.rankwise.eps', dpi=600)

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
