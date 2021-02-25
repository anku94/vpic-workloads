import glob
import matplotlib.pyplot as plt
import numpy
import pandas as pd

from util import chunk_and_sum


class RuntimeLog:
    def __init__(self, data_path: str):
        self.data_path = data_path
        rank_files = glob.glob(data_path + '/*perfstats.log*')
        self.num_ranks = len(rank_files)

        self.data_log = None
        self.data_phy = None

        self.load_data()

    def load_data(self):
        #data = [self._load_rank(ridx) for ridx in range(self.num_ranks)]
        data = [self._load_rank_pandas(ridx) for ridx in range(self.num_ranks)]
        data = list(zip(*data))
        data_log = data[0]
        data_phy = data[1]

        print('---------------')
        #  print(data_log)
        #  print(data_phy)

        def clip_common(data):
            data_len = map(lambda x: len(x), data)
            data_len = list(data_len)
            print(data_len)

            common_len = min(data_len)
            data = [numpy.array(data[ridx][:common_len]) for ridx in range(self.num_ranks)]
            return data

        self.data_log = clip_common(data_log)
        self.data_phy = clip_common(data_phy)
        # self.data = data

    def plot_data(self, plot_path, skew_pct):
        data = sum(self.data_log)

        fig, ax = plt.subplots(1, 1)
        skew_str = 'none' if int(skew_pct) == 0 else skew_pct
        ax.set_title('Aggregate Disk B/W for %d ranks, skew %s' % (self.num_ranks,skew_str))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('B/W (MB/s)')
        ax.plot(data)

        # fig.savefig('../vis/runtime/diskbw.temp.16x16.0s0.aggr.eps', dpi=600)

        nodewise_data_log = chunk_and_sum(self.data_log, 16)
        nodewise_data_phy = self.data_phy[::16]

        def get_band(data):
            data_len = max(map(lambda x: len(x), data))
            band_min = numpy.array([99999] * data_len)
            band_max = numpy.array([0] * data_len)
            band_sum = numpy.array([0.0] * data_len)
            num_ranks = len(data)

            for ridx in range(num_ranks):
                cur_data = data[ridx]
                band_max = numpy.maximum(band_max, cur_data)
                band_min = numpy.minimum(band_min, cur_data)
                band_sum += cur_data

            band_mean = band_sum * 1.0 / len(data)

            return data_len, band_min, band_mean, band_max

        log_len, log_band_min, _, log_band_max = get_band(nodewise_data_log)
        phy_len, phy_band_min, phy_band_mean, phy_band_max =\
            get_band(nodewise_data_phy)

        ax2 = ax.twinx()
        ax2_color=(0.8, 0.2, 0.2, 0.8)
        ax2.fill_between(range(log_len),
                        log_band_min,
                        log_band_max,
                        facecolor=ax2_color)

        ax2.fill_between(range(phy_len),
                         phy_band_min,
                         phy_band_max,
                         facecolor=(0.2, 0.8, 0.2, 0.6))

        ax2.plot(phy_band_mean, color=(0.05, 0.4, 0.1, 0.9))
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
        # fig.savefig('../vis/runtime/diskbw.temp.64x1.8s60.nodewise.pdf', dpi=600)
        fig.savefig(plot_path, dpi=600)

    # Resamples 10:1, 100ms -> 1s
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
        # 512B sectors, so MBps = delta * 512 / (1024 * 1024)
        MBps_disk = numpy.array(self._adjacent_diff(fdata[2])) / (1024 * 2)

        return MBps, MBps_disk

    def _load_rank_pandas(self, rank_id):
        fpath = self.data_path + '/vpic-perfstats.log.%d' % (rank_id)
        fdata = pd.read_csv(fpath)

        logical_key_str = 'Logical Bytes Written'
        physical_key_str = 'Disk Sectors Written'

        if logical_key_str in fdata:
            logical_bytes = fdata['Logical Bytes Written']
            # convert ms level data to sec level
            logical_bytes = logical_bytes.iloc[::10]
            logical_mbps = logical_bytes.diff().dropna() / (1024.0 ** 2)
        else:
            logical_mbps = pd.Series()

        if physical_key_str in fdata:
            physical_secs = fdata['Disk Sectors Written']

            # convert ms level data to sec level
            physical_secs = physical_secs.iloc[::10]

            # 512B sectors, so MBps = delta * 512 / (1024 * 1024)
            physical_mbps = physical_secs.diff().dropna() / (1024.0 * 2)
        else:
            physical_mbps = pd.Series()

        return (logical_mbps, physical_mbps)
