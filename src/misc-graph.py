import pickle

import IPython

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import StrMethodFormatter


def gen():
    data_file = '../rundata/vpic-512/stddev.csv'
    df = pd.read_csv(data_file)
    print(df)

    fig, ax = plt.subplots(1, 1)

    for pvtcnt in df['pivotcount'].unique():
        print(pvtcnt)
        df_cur = df[df['pivotcount'] == pvtcnt]
        print(df_cur)
        x = df_cur['timestep']
        y = df_cur['stddev'] * 100

        ax.plot(x, y, label='{0} pivots'.format(pvtcnt))

    ax.legend(loc='lower right')
    ax.set_ylim(0, 30)
    ax.set_title('Pivot Count vs Load Stddev (512 ranks)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Stddev (%)')
    fig.savefig('../vis/rtp_traces/vpic.512.pvtcntvsstd.pdf')
    return


def gen_hist():
    data_file = open('../rundata/vpic-512/loadcount.txt').read().split('\n')
    data_file = [i.strip() for i in data_file if len(i.strip()) > 0]
    data_left = data_file[0].split(',')[-1].split(' ')
    data_right = data_file[1].split(',')[-1].split(' ')
    data_left = list(map(int, data_left))
    data_right = list(map(int, data_right))

    fig, axes = plt.subplots(1, 2)
    print(len(data_left))
    print(len(data_right))

    axes[0].bar(range(len(data_left)), data_left)
    axes[1].bar(range(len(data_right)), data_right)

    axes[0].set_xlabel('Stddev = 13%')
    axes[1].set_xlabel('Stddev = 5%')
    axes[0].set_ylabel('Aggregate Load (#particles)')
    fig.suptitle('Load Distribution For Two Stddev Values')
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    fig.savefig('../vis/rtp_traces/load_hist.pdf')


def gen_rt_sst():
    path = '/Users/schwifty/Repos/workloads/rundata/range.read.rel.midway.dropcache'
    df = pd.read_csv(path)
    print(df)
    fig, ax = plt.subplots(1, 1)
    ax.plot(df['index'], df['total'], label='Read + Sort')
    ax.plot(df['index'], df['read'], label='Read')

    ax.set_xlabel('Cumulative SSTs Read')
    ax.set_ylabel('Time Taken (us)')
    ax.legend()

    ax.set_title('Query Performance vs SSTs Read')

    # ax.show_legend()
    fig.show()
    fig.savefig('../vis/query-vs-numsst.pdf', dpi=600)


def gen_sort_size():
    dir = '/Users/schwifty/Repos/workloads/rundata/sort'
    dir += '/sort.{0}B.csv'
    f4 = dir.format(4)
    f8 = dir.format(8)
    f40 = dir.format(40)

    f4 = pd.read_csv(f4)
    f8 = pd.read_csv(f8)
    f40 = pd.read_csv(f40)

    data_x = f4['Index']
    data_y4 = f4['Total']
    data_y8 = f8['Total']
    data_y40 = f40['Total']

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_y4, label='4B items')
    ax.plot(data_x, data_y8, label='8B items')
    ax.plot(data_x, data_y40, label='40B items')

    ax.set_xlabel('Number of SSTs Read')
    ax.set_ylabel('Time Taken (ms)')

    ax.set_title('Sorting Time vs Item Size (i7 CFL, MBP)')
    ax.legend()

    # plt.show()
    plt.savefig('../vis/sort_vs_size.pdf', dpi=600)
    pass


def gen_sort_plat():
    dir = '/Users/schwifty/Repos/workloads/rundata/sort'
    dir += '/sort.{0}.csv'
    f1 = dir.format('mbp')
    f2 = dir.format('sus')

    f1 = pd.read_csv(f1)
    f2 = pd.read_csv(f2)

    data_x = f1['Index']
    data_mbp_tot = f1['Total']
    data_mbp_read = f1['Read']
    data_sus_tot = f2['Total']
    data_sus_read = f2['Read']

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_sus_tot, label='sus sort + read')
    ax.plot(data_x, data_mbp_tot, label='mbp sort + read')
    ax.plot(data_x, data_sus_read, label='sus read')
    ax.plot(data_x, data_mbp_read, label='mbp read')

    ax.set_xlabel('Number of SSTs Read')
    ax.set_ylabel('Time Taken (ms)')

    ax.set_title('Sorting Time vs CPU gen')
    ax.legend()

    # plt.show()
    plt.savefig('../vis/sort_vs_plat.pdf', dpi=600)


def gen_sort_scale():
    path = "/Users/schwifty/Repos/workloads/rundata/sort/sort.cpp17.tbb"
    seq_data = pd.read_csv(path + '/sort.sus.seq.csv')
    par_data = pd.read_csv(path + '/sort.sus.par.csv')

    seq_data = seq_data.groupby('cores', as_index=False).mean()
    par_data = par_data.groupby('cores', as_index=False).mean()

    fig, ax = plt.subplots(1, 1)
    ax.plot(seq_data['cores'], seq_data['dur_ms'], label='Sequential')
    ax.plot(par_data['cores'], par_data['dur_ms'], label='Parallel')

    ax.legend()

    par_data['scale1'] = 1 / (par_data['dur_ms'] / par_data['dur_ms'].shift())
    par_data['scale2'] = 1 / (par_data['dur_ms'] / par_data['dur_ms'][0])

    ax2 = ax.twinx()
    ax2.plot(par_data['cores'], par_data['scale2'], 'r')
    ax2.plot(par_data['cores'], par_data['cores'], 'g:')

    ax2.set_ylabel('Speedup (Ratio)')

    ax.set_xlabel('Num Cores')
    ax.set_ylabel('Time (ms)')

    ax.set_title('C++17/TBB seq::sort vs par::sort, Susitna')

    # plt.show()
    plt.savefig('../vis/sort_vs_scale.pdf', dpi=600)


def gen_db_bench():
    path = '/Users/schwifty/Repos/workloads/rundata/rangewriter_vs_pdb/bench.csv'
    data = pd.read_csv(path)

    fig, ax = plt.subplots(1, 1)

    labels = {
        'pdb': 'Bloom/PDB',
        'rdb': 'RangeDB with Sort',
        'rdbbu': 'RangeDB with Sort+MidFlush'
    }

    for prov in data['prov'].unique():
        print(prov)
        plot_data = data[data['prov'] == prov]
        ax.plot(plot_data['disk_cap'] / (1024 ** 2),
                plot_data['disk_speed'] / plot_data['disk_cap'],
                label=labels[prov])

    ax.set_ylim([0, 1])
    ax.set_xlabel('Disk Speed (MB/s)')
    ax.set_ylabel('Utilization Achieved')
    ax.set_title('Backend Disk Utilization vs Disk Speed')
    ax.legend()
    plt.savefig('../vis/db_bench.pdf', dpi=600)
    pass


def gen_epoch_parq():
    path = '/Users/schwifty/Repos/workloads/rundata/multiepoch.csv'
    data = pd.read_csv(path)

    data_sp = data[data['R_250000X'] == 1]

    fig, ax = plt.subplots(1, 1)
    for pvtcnt in data_sp['pvtcnt'].unique():
        data_cur = data_sp[data_sp['pvtcnt'] == pvtcnt]
        ax.plot(range(5), data_cur['overlap_pct'],
                label='{0} pivots'.format(pvtcnt))
        print(pvtcnt)

    ax.plot(range(5), [100.0 / 512] * 5, '--')
    ax.legend()
    ax.set_xlabel('Epoch Index')
    ax.set_ylim([0, 0.5])
    ax.set_ylabel('Peak SST Overlap (% of total SST mass)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Partitioning Quality Over Epochs as f(pvtcnt)')
    # fig.show()
    fig.savefig('../vis/overlap_vs_epoch_vs_pvtcnt.pdf', dpi=600)
    pass


def gen_reneg_intvl_parq():
    path = '/Users/schwifty/Repos/workloads/rundata/multiepoch.csv'
    data = pd.read_csv(path)

    data_sp = data[data['pvtcnt'] == 256]

    fig, ax = plt.subplots(1, 1)
    for freq_x in data_sp['R_250000X'].unique():
        data_cur = data_sp[data_sp['R_250000X'] == freq_x]
        ax.plot(range(5), data_cur['overlap_pct'],
                label='Interval {0}X'.format(freq_x))

    ax.plot(range(5), [100.0 / 512] * 5, '--')
    ax.legend()
    ax.set_xlabel('Epoch Index')
    ax.set_ylim([0, 0.5])
    ax.set_ylabel('Peak SST Overlap (% of total SST mass)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Partitioning Quality Over Epochs as f(IntervalX)')
    # fig.show()
    fig.savefig('../vis/overlap_vs_reneg_int.pdf', dpi=600)
    pass


def misc_overlap():
    basedir = '/Users/schwifty/Repos/workloads/rundata/carp_stat_trigger_apr26'
    fpath = basedir + '/overlaps_apr13.csv'
    f = open(fpath, 'r').read().split('\n')
    y0 = f[0].split(',')[:5]
    y1 = f[1].split(',')[:5]

    y0 = [float(y) for y in y0]
    y1 = [float(y) for y in y1]
    print(y0)
    print(y1)

    fig, ax = plt.subplots(1, 1)

    ax.plot(range(5), y0, label='Dynamic/Fstat/500k/4.0')
    ax.plot(range(5), y1, label='OOBOnly')

    ax.legend()

    ticks = 2400 * np.arange(1, 6)
    ticks = ['T.' + str(t) for t in ticks]
    ax.set_xticks(range(5))
    ax.set_xticklabels(ticks)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Overlap Percent')
    ax.set_title('Dynamic vs OOB-Only Trigger')
    ax.set_ylim([0, ax.get_ylim()[1]])

    fig.savefig(basedir + '/overlaps_apr13.pdf', dpi=600)
    # fig.show()


def err_bars():
    data = '/Users/schwifty/Repos/workloads/exp/heatmap-data/runs_intvl_single/errs.csv'
    fig, ax = plt.subplots(1, 1)

    df = pd.read_csv(data)
    for intvl in sorted(df['intvl'].unique()):
        idf = df[df['intvl'] == intvl]
        idf = idf.groupby('epochs').agg(mean=('overlaps', 'mean'),
                                        min=('overlaps', 'min'),
                                        max=('overlaps', 'max'),
                                        var=('overlaps', 'var'))
        print(idf)
        # errs_lower = idf['var'] ** 0.5
        # errs_upper = idf['mean'] + (idf['var'] ** 0.5)
        # errs = [errs_lower, errs_upper]
        errs = [idf['var'] ** 0.5, idf['var'] ** 0.5]
        ax.errorbar(idf.index.values.astype(int), idf['mean'], yerr=errs,
                    fmt='.-',
                    capsize=2, label=intvl)
    ax.set_title('Variation in Overlaps for Epochs (Intvl 800)')
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Overlap Percent')
    ax.set_ylim([0, 2])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot_path = data + '.pdf'
    ax.legend()
    # plt.show()
    fig.savefig(plot_path, dpi=300)


def mfread_subpart():
    basedir = '/Users/schwifty/Repos/workloads/rundata/subpart_jun15'
    lat_data = [1117, 1197, 1277]
    size_data = [254933, 315184, 451355]
    size_data = np.array(size_data) / 1000
    label_data = ['Subpart x1', 'Subpart x2', 'Subpart x4']

    fig, ax = plt.subplots(1, 1)

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}K'))

    for tuple in zip(lat_data, size_data, label_data):
        print(tuple)
        mx = tuple[1]
        my = tuple[0]
        p = ax.plot(mx, my, 'o', ms=12, label=tuple[2])
        color = p[-1].get_color()
        ax.plot([0, mx], [my, my], '--', color=color, alpha=0.5)
        ax.plot([mx, mx], [0, my], '--', color=color, alpha=0.5)

    ax.set_xlim([200, 500])
    ax.set_ylim([0, ax.get_ylim()[1] * 1.2])
    ax.legend(loc="lower right")

    ax.set_ylabel('Manifest Read Latency (ms)')
    ax.set_xlabel('Manifest Size (# SSTs)')
    ax.set_title('Impact of Subpartitioning on Manifest Size/Latency')

    # fig.show()
    fig.savefig(basedir + '/mflat.pdf', dpi=300)


def plot_triton():
    basedir = '/Users/schwifty/Repos/workloads/rundata/triton'
    runs = [
        [7.64, 108.66, 172.68, 2.06, '34N/10G/10G/C3'],
        [11.7, 146.33, 139.40, 2.04, '20N/10G/10G/C3'],
        [14.68, 237.70, 263.72, 2.05, '20N/17G/17G/M1'],
        [20.24, 283.98, 244.64, 2.04, '20N/17G/500M/M1']
    ]
    runs_t = list(zip(*runs))
    X = runs_t[4]
    Y0 = np.array(runs_t[0])
    Y1 = np.array(runs_t[1])
    Y2 = np.array(runs_t[2])
    Y3 = np.array(runs_t[3])
    YT = Y0 + Y1 + Y2 + Y3
    data_per_node = [10, 10, 17, 17]
    print(X)
    print(Y0)
    print(Y1)

    fig, ax = plt.subplots(1, 1)
    ax.bar(X, Y0, 0.35, label='Phase 0')
    ax.bar(X, Y1, 0.35, bottom=Y0, label='Phase 1')
    ax.bar(X, Y2, 0.35, bottom=Y0 + Y1, label='Phase 2')
    rects = ax.bar(X, Y3, 0.35, bottom=Y0 + Y1 + Y2, label='Phase 3')

    for idx, rect in enumerate(rects):
        height = rect.get_height()
        ypos = rect.get_y() + height / 2

        data = data_per_node[idx] * 1e3
        time = YT[idx]
        print(data, time)
        MBps = data / time
        label = '%.1f MBps' % (MBps)

        ax.text(rect.get_x() + rect.get_width() / 2., ypos,
                '%s' % label, ha='center', va='bottom')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('TritonSort - Config vs Runtime')
    ax.legend()

    fig.show()
    # fig.savefig(basedir + '/tsruns.pdf', dpi=300)


def plot_triton_2():
    data = np.array([
        [11.24, 157.59, 187.42, 2.04],
        [8.17, 158.24, 190.22, 2.04],
        [8.17, 160.74, 183.25, 2.56],
        [10.21, 183.57, 187.48, 2.05],
        [10.21, 185.16, 189.53, 2.04],
        [10.64, 186.07, 191.03, 2.05],
        [7.15, 143.89, 184.69, 2.04],
        [7.15, 141.68, 188.84, 2.04],
        [7.14, 137.69, 187.25, 2.56],
        [8.68, 147.48, 193.26, 2.05],
        [8.62, 153.93, 193.84, 2.05],
        [8.66, 153.41, 194.1, 2.04]
    ])

    data = [data[0:3], data[3:6], data[6:9], data[9:12]]
    means = []
    stds = []

    for d in data:
        dmean = np.mean(d, 0)
        dstd = np.std(np.sum(d, 1))
        print(dmean, dstd)
        means.append(dmean)
        stds.append(dstd)

    means = np.array(means)
    means = np.transpose(means)
    stds = np.array(stds)

    X = ['10B/90B/M1.XL', '4B/56B/M1.XL', '10B/90B/C3.2XL', '4B/56B/C3.2XL']
    Y0 = means[0]
    Y1 = means[1]
    Y2 = means[2]
    Y3 = means[3]
    YT = np.sum(means, 0)

    print(means, stds)

    fig, ax = plt.subplots(1, 1)
    ax.bar(X, Y0, 0.3, label='Phase 0')
    ax.bar(X, Y1, 0.3, bottom=Y0, label='Phase 1')
    ax.bar(X, Y2, 0.3, bottom=Y0 + Y1, label='Phase 2')
    rects = ax.bar(X, Y3, 0.3, bottom=Y0 + Y1 + Y2, label='Phase 3')
    # ax.errorbar(X, YT, yerr=stds*1, fmt='.', color='black')
    # print(stds)

    ax.legend(loc='lower right')
    ax.set_xlabel('Key/Value Sizes')
    ax.set_ylabel('Sorting Time (seconds, lower the better)')
    ax.set_title('TritonSort: Key Size vs Sorting Time for 250GB')

    for idx, rect in enumerate(rects):
        height = rect.get_height()
        ypos = rect.get_y() + height / 2

        data = 250 / 20.0 * 1e3
        time = YT[idx]
        print(data, time)
        MBps = data / time
        label = '%.1f MBps' % (MBps)

        ax.text(rect.get_x() + rect.get_width() / 2., ypos + 2,
                '%s' % label, ha='center', va='bottom')

    basedir = '/Users/schwifty/Repos/workloads/rundata/triton'
    # plt.show()
    plt.savefig(basedir + '/kvsizes.pdf', dpi=300)


def plot_triton_vs_carp():
    basedir = '/Users/schwifty/Repos/workloads/rundata/triton'
    fig, ax = plt.subplots(1, 1)
    X = [
        'TS/200G',
        'TS/250G',
        'TS/250G*4',
        'NOSHUF',
        'NOSHUF/TS',
        'DeltaFS',
        'CARP'
    ]

    Y = [
        299,
        373,
        1492,
        440,
        1932,
        484,
        590
    ]

    ax.bar(X, Y, width=0.35)

    for rect in ax.patches:
        height = rect.get_height()
        ypos = rect.get_y() + height / 2
        ypos = height + 2
        ax.text(rect.get_x() + rect.get_width() / 2., ypos,
                '%d' % int(height), ha='center', va='bottom')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Storage/Indexing Scheme vs Projected Runtime')
    # fig.show()
    fig.savefig(basedir + '/comp.pdf', dpi=300)


def plot_hist():
    basedir = '/Users/schwifty/Repos/workloads/data/toplot'
    timesteps = list(range(200, 19400 + 3200, 3200))
    fig, ax = plt.subplots(2, 2)
    for tsidx in [0, 2, 4, 6]:
        ts = timesteps[tsidx]
        print(tsidx, ts)
        data1 = np.load('%s/T.%s/sample.npy' % (basedir, ts))
        hist1, edges = np.histogram(data1, bins=1000, range=(0, 1000))
        hist2 = pickle.loads(
            open('%s/hist.data.%s' % (basedir, tsidx), 'rb').read())
        mass1 = sum(hist1)
        mass2 = sum(hist2)
        hist1 = hist1 / mass1
        hist2 = hist2 / mass2
        x = range(1000)
        # y1 = np.cumsum(hist1, axis=0)
        # y2 = np.cumsum(hist2, axis=0)
        y1 = hist1
        y2 = hist2

        print(int(sum(y1[4:]) * 100))
        print(int(sum(y2[4:]) * 100))

        print(y1.shape)
        print(y2.shape)

        plt_idx = 30 + 30 * tsidx

        ax_x = int(tsidx / 4)
        ax_y = int((tsidx / 2) % 2)

        print(ax, ax_x, ax_y)

        cax = ax[ax_x][ax_y]

        cax.plot(x[:plt_idx], y1[:plt_idx])
        cax.plot(x[:plt_idx], y2[:plt_idx])

    print(timesteps)
    fig.show()
    pass


if __name__ == '__main__':
    # gen_hist()
    # gen_rt_sst()
    # gen_sort_size()
    # gen_sort_plat()
    # gen_sort_scale()
    # gen_db_bench()
    # gen_epoch_parq()
    # gen_reneg_intvl_parq()
    # misc_overlap()
    # err_bars()
    # mfread_subpart()
    # plot_triton()
    # plot_triton_2()
    # plot_triton_vs_carp()
    plot_hist()
