""" Created Sep 15, 2022 """

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
import pandas as pd

from common import plot_init


def std(s):
    return np.std(s)


def plot_allrun_df(run_df):
    run_df = run_df.groupby('pvtcnt', as_index=False).agg(
        {'total_io_time_mean': 'mean',
         'total_io_time_std': 'mean',
         'max_fin_dura_mean': 'mean',
         'wr_min_mean': 'mean',
         'wr_max_mean': 'mean'
         })
    # run_df.columns = ["_".join(col).strip("_") for col in run_df.columns]

    labels_x = run_df['pvtcnt']
    data_x = np.arange(len(labels_x))
    data_y1a = run_df['total_io_time_mean']
    data_y1a_err = run_df['total_io_time_std']

    print(data_y1a_err)

    data_y1b = run_df['max_fin_dura_mean']
    data_y2a = run_df['wr_min_mean']
    data_y2b = run_df['wr_max_mean']

    ax1_ylim = 160 * 1e3
    ax2_ylim = 14 * 1e6

    fig, ax = plt.subplots(1, 1)

    # ax.plot(data_x, data_y1a, label='io_time', marker='x')
    ax.errorbar(data_x,
                data_y1a, yerr=data_y1a_err, label='io_time', marker='x')
    ax.plot(data_x, data_y1b, label='max_findur', marker='x')

    ax.set_title('Runtime/Load Balance as f(pivot_count)')
    ax.set_xlabel('#pivots')
    ax.set_ylabel('Runtime (one epoch)')

    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x / 1e3))
    ax.set_xticks(data_x)
    ax.set_xticklabels([str(x) for x in labels_x])
    ax.minorticks_off()

    ax2 = ax.twinx()
    width = 0.35
    ax2.bar(data_x - width / 2, data_y2a, width=width, label='min_load',
            alpha=0.5)
    ax2.bar(data_x + width / 2, data_y2b, width=width, label='max_load',
            alpha=0.5)
    ax2.yaxis.set_major_formatter(lambda x, pos: '{:.0f}M'.format(x / 1e6))
    ax2.set_ylabel('Load Per Rank')

    ax.yaxis.set_minor_locator(MultipleLocator(5000))
    ax.yaxis.grid(b=True, which='major', color='#aaa')
    ax.yaxis.grid(b=True, which='minor', color='#ddd')

    ax.set_ylim([0, ax1_ylim])
    ax2.set_ylim([0, ax2_ylim])

    fig.legend(ncol=2, bbox_to_anchor=(0.25, 0.78), loc='lower left')

    fig.tight_layout()
    return fig, ax


def plot_allrun_intvlwise(run_df, ax, label_fmt):
    run_df = run_df.groupby(['intvl', 'pvtcnt'], as_index=False).agg(
        {'total_io_time_mean': 'mean',
         'total_io_time_std': 'mean',
         'max_fin_dura_mean': 'mean',
         'wr_min_mean': 'mean',
         'wr_max_mean': 'mean'
         })

    ax1_ylim = 3000 * 1e3

    labels_x = None
    data_x = None

    intvls = run_df['intvl'].unique()
    for intvl in intvls:
        intvl_df = run_df[run_df['intvl'] == intvl].sort_values(['pvtcnt'])
        labels_x = intvl_df['pvtcnt']
        data_x = np.arange(len(labels_x))
        data_y = intvl_df['total_io_time_mean']
        data_y_err = intvl_df['total_io_time_std']
        # ax.errorbar(data_x, data_y, yerr=data_y_err, label=label_fmt.format(intvl),
        #             capsize=8)
        ax.plot(data_x, data_y, label=label_fmt.format(intvl))

    ax.set_xticks(data_x)
    ax.set_xticklabels([str(i) for i in labels_x])

    ax.set_xlabel('#pivots')
    ax.set_ylabel('Runtime (one epoch)')
    ax.set_title('Pivot Count vs Runtime')

    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x / 1e3))
    ax.yaxis.set_minor_locator(MultipleLocator(5000))
    ax.yaxis.grid(b=True, which='major', color='#aaa')
    ax.yaxis.grid(b=True, which='minor', color='#ddd')

    ax.set_ylim([0, ax1_ylim])
    ax.legend()

    return ax


def preprocess_allrun_df(run_df):
    params_agg = [
        p
        for p in list(run_df.columns)
        if p not in ["Unnamed: 0", "run", "intvl", "pvtcnt", "drop"]
    ]
    agg_ops = {p: ["mean", std] for p in params_agg}

    run_df = run_df.groupby(["intvl", "pvtcnt", "drop"], as_index=False).agg(
        agg_ops)
    run_df.columns = ["_".join(col).strip("_") for col in run_df.columns]

    all_intvls = run_df['intvl'].unique()
    all_drop = run_df["drop"].unique()

    dropzero_df = run_df[run_df['drop'] == 0]
    return dropzero_df


def plot_datascal(plot_dir, carp_df):
    fig, ax = plt.subplots(1, 1)
    carp_df = carp_df.groupby(['epcnt'], as_index=False).agg({
        'total_io_time': 'mean'
    })
    print(carp_df)

    data_x_carp = carp_df['epcnt'].astype(str)
    data_y_carp = carp_df['total_io_time']

    ax.plot(data_x_carp, data_y_carp, label='CARP (3MB/s FakeIO)')
    data_y_dfs = np.array([150, 450, 900, 1800]) * 1e3
    ax.plot(data_x_carp, data_y_dfs, '--', label='DeltaFS (3MB/s FakeIO)')
    data_y_ior = np.array([ 100, 300, 600, 1200 ]) * 1e3
    ax.plot(data_x_carp, data_y_ior, '--', label='IOR')

    ax.set_xlabel('Number Of Epochs')
    ax.set_ylabel('Runtime')
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x / 1000))
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.yaxis.grid(b=True, which='major', color='#aaa')
    ax.yaxis.grid(b=True, which='minor', color='#ddd')
    ax.set_title('Data Scalability (CARP vs Others)')
    ax.legend()

    data_path = '/Users/schwifty/Repos/workloads/rundata/20220825-pvtcnt-analysis/data'
    carp_path = '{}/carp-suite-repfirst.csv'.format(data_path)
    carp_df = pd.read_csv(carp_path)
    print(carp_df)

    dfs_path = '{}/deltafs-jobdir.csv'.format(data_path)
    dfs_df = pd.read_csv(dfs_path)
    # ax.plot(carp_df['epcnt'], carp_df['total_io_time'] / 1000.0, label='CARP')
    # ax.plot(dfs_df['epcnt'], dfs_df['total_io_time'] / 1000.0, label='DeltaFS')

    # ax.plot(fio_ep, fio_data, label='FIO')
    # ax.plot(carp_df['epcnt'], carp_df['total_io_time'] / 1000.0, label='CARP')
    # ax.plot(dfs_df['epcnt'], dfs_df['total_io_time'] / 1000.0, label='DeltaFS')

    # fig.tight_layout()
    # fig.show()

    fig.savefig('{}/datascal.pdf'.format(plot_dir), dpi=300)

    pass


def run_plot_allrun(plot_dir):
    allrun_dir = '/Users/schwifty/Repos/workloads/rundata/20220915-rtpbench-throttledruns'

    dfname_3m1 = 'carp-suite-repfirst-allpvtcnt-throttled.csv'
    df_3m1 = pd.read_csv('{}/{}'.format(allrun_dir, dfname_3m1),
                         index_col=False)
    # print(df_3m1)

    dfname_3m2 = 'carp-suite-repfirst-allpvtcnt-throttle-3m-fakeio.csv'
    df_3m2 = pd.read_csv('{}/{}'.format(allrun_dir, dfname_3m2),
                         index_col=False)
    df_3m2_pp = preprocess_allrun_df(df_3m2)
    # print(df_3m2)

    dfname_4m = 'carp-suite-repfirst-allpvtcnt-throttle-4m-fakeio.csv'
    df_4m = pd.read_csv('{}/{}'.format(allrun_dir, dfname_4m), index_col=False)
    df_4m_pp = preprocess_allrun_df(df_4m)

    print(df_3m2_pp)
    print(df_4m_pp)

    # fig, ax = plt.subplots(1, 1)
    # plot_allrun_intvlwise(df_3m2_pp, ax, "3MB/s/rank (Intvl: {})")
    # plot_allrun_intvlwise(df_4m_pp, ax, "4MB/s/rank (Intvl: {})")
    # ax.yaxis.set_minor_locator(MultipleLocator(5000 * 25))
    # ax.set_ylim([0, 3000 * 1e3])
    # fig.show()
    # fig.savefig('{}/pvtcnt.vs.runtime.throttled.pdf'.format(plot_dir), dpi=300)

    plot_datascal(plot_dir, pd.concat([df_3m1, df_3m2]))
    pass


if __name__ == "__main__":
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20220915-rtpbench-throttledruns"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot_init()
    run_plot_allrun(plot_dir)
