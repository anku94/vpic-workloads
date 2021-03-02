import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator



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
        ax.plot(range(5), data_cur['overlap_pct'], label='{0} pivots'.format(pvtcnt))
        print(pvtcnt)

    ax.plot(range(5), [100.0/512] * 5, '--')
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
        ax.plot(range(5), data_cur['overlap_pct'], label='Interval {0}X'.format(freq_x))

    ax.plot(range(5), [100.0/512] * 5, '--')
    ax.legend()
    ax.set_xlabel('Epoch Index')
    ax.set_ylim([0, 0.5])
    ax.set_ylabel('Peak SST Overlap (% of total SST mass)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Partitioning Quality Over Epochs as f(IntervalX)')
    # fig.show()
    fig.savefig('../vis/overlap_vs_reneg_int.pdf', dpi=600)
    pass

if __name__ == '__main__':
    # gen_hist()
    # gen_rt_sst()
    # gen_sort_size()
    # gen_sort_plat()
    # gen_sort_scale()
    # gen_db_bench()
    # gen_epoch_parq()
    gen_reneg_intvl_parq()
