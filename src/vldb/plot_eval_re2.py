import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import multiprocessing
import numpy as np
import os
import pandas as pd
import re
import sys

from common import plot_init



def df_stat(exp_dir, rank, stat, op="match"):
    pfpath_rank = "{}/vpic-perfstats.log.{}".format(exp_dir, rank)
    df = pd.read_csv(pfpath_rank).dropna()
    df.columns = ["ts", "stat", "val"]
    df["rank"] = rank

    if op == "contains":
        df = df[df["stat"].str.contains(stat)]
    elif op == "match":
        df = df[df["stat"] == stat]

    return df


def df_stat_reneg(exp_dir, rank):
    stat = "RENEG_PIVOTS_E"
    df = df_stat(exp_dir, rank, stat, op="contains")

    return df["ts"].to_numpy().astype(int)


def read_reneg_times(exp_dir, nranks=16):
    all_ranks = range(nranks)
    all_args = map(lambda x: (exp_dir, x), all_ranks)

    all_ts = None
    with multiprocessing.Pool(16) as p:
        all_ts = p.starmap(df_stat_reneg, all_args)

    tsmat = np.array(all_ts)
    tsmat_mean = tsmat.mean(axis=0)
    return tsmat_mean


def df_stat_bw(exp_dir, rank):
    stat = "LOGICAL_BYTES_WRITTEN"
    df = df_stat(exp_dir, rank, stat)
    df = df.astype({"ts": np.int64, "val": np.int64})

    df["ts"] = df["ts"] / 100
    df["ts"] = df["ts"].astype(np.int64) * 100
    df["bw"] = df["val"].diff().fillna(0)

    df = df[df["bw"] > 1e-2]

    return df


def read_bw_usage(exp_dir, nranks=16):
    all_ranks = range(nranks)
    all_args = map(lambda x: (exp_dir, x), all_ranks)

    all_dfs = None
    with multiprocessing.Pool(16) as p:
        all_dfs = p.starmap(df_stat_bw, all_args)

    aggr_df = pd.concat(all_dfs)
    aggr_df = aggr_df.groupby("ts", as_index=None).agg({"bw": ["sum", "count"]})

    aggr_df.columns = ["_".join(i).strip("_") for i in aggr_df.columns]

    aggr_df["bw_sum"] = aggr_df["bw_sum"].rolling(window=20).mean()
    return aggr_df


def plot_bwusage(exp_dir, plot_dir, save=False):
    nranks = 512
    polling_delta = 100  # ms
    pdf = 1000.0 / polling_delta  # bytes_delta to bytes/sec
    pdf /= 2 ** 20  # bytes/sec to mbytes/sec

    bwdf = read_bw_usage(exp_dir, nranks)
    bwdf["mbps"] = bwdf["bw_sum"] * pdf

    fig, ax = plt.subplots(1, 1)
    ax.plot(bwdf["ts"], bwdf["mbps"], label="B/W")

    # plot reneg_times
    rtp_ts = read_reneg_times(exp_dir, nranks)
    rtp_ymax = bwdf["mbps"].dropna().max()
    for ts in rtp_ts:
        ax.plot([ts, ts], [0, rtp_ymax], linestyle="--", alpha=0.5)

    ax2 = ax.twinx()
    #  ax2.plot(bwdf["ts"], bwdf["bw_count"], label="Ranks Active", color="orange")

    ax.xaxis.set_major_formatter(lambda x, pos: "{:.0f}s".format(x / 1000))
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} MB/s".format(x))

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Aggregate Bandwidth (MB/s)")
    ax2.set_ylabel("Num Ranks Writing")

    ax.set_title("")

    fig.legend(bbox_to_anchor=[0.5, 0.96], loc="center", ncol=2)

    fig.tight_layout()
    fig_path = "{}/logical_bw.pdf".format(plot_dir)

    if save:
        fig.savefig(fig_path, dpi=300)
    else:
        fig.show()

    return


def run_bwusage(plot_dir):
    trace_dir = "/mnt/ltio/jobs-big-run/vpic-carp8m/carp_P3584M_intvl500000"
    trace_dir = "/mnt/ltio/jobs-big-run/vpic-carp8m/carp_P3584M_intvl1000000"
    exp_dir = "{}/exp-info".format(trace_dir)
    plot_bwusage(exp_dir, plot_dir, save=True)


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


def plot_allrun_intvlwise(run_df):
    run_df = run_df.groupby(['intvl', 'pvtcnt'], as_index=False).agg(
        {'total_io_time_mean': 'mean',
         'total_io_time_std': 'mean',
         'max_fin_dura_mean': 'mean',
         'wr_min_mean': 'mean',
         'wr_max_mean': 'mean'
         })

    ax1_ylim = 160 * 1e3
    fig, ax = plt.subplots(1, 1)

    labels_x = None
    data_x = None

    intvls = run_df['intvl'].unique()
    for intvl in intvls:
        intvl_df = run_df[run_df['intvl'] == intvl].sort_values(['pvtcnt'])
        labels_x = intvl_df['pvtcnt']
        data_x = np.arange(len(labels_x))
        data_y = intvl_df['total_io_time_mean']
        data_y_err = intvl_df['total_io_time_std']
        ax.errorbar(data_x, data_y, yerr=data_y_err, label='{}'.format(intvl),
                    capsize=8)

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

    fig.tight_layout()
    return fig, ax


def plot_ior(plot_dir, save=False) -> None:
    df_path = '/Users/schwifty/Repos/workloads/rundata/20220912-aggr-data/ior-logs.csv'
    df = pd.read_csv(df_path, on_bad_lines='skip', index_col=False)
    df.rename(columns={"bw(MiB/s)": "bw"}, inplace=True)
    df_aggr = df.groupby(["blksz", "nranks", "epcnt"], as_index=False).agg({
        "bw": "mean"
    })

    all_nranks = df_aggr["nranks"].unique()
    fig, ax = plt.subplots(1, 1)

    for nranks in all_nranks:
        df_nranks = df_aggr[df_aggr["nranks"] == nranks]
        data_y = df_nranks["bw"]
        data_x = df_nranks["epcnt"].astype(str)
        ax.plot(data_x, data_y, label="{} ranks".format(nranks))

    ax.set_title("IOR Bandwidth vs Epochs Written")
    ax.set_xlabel("Epochs Written")
    ax.set_ylabel("Reported Bandwidth")
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f} GB/s".format(x / 1024))

    ax.set_ylim([0, ax.get_ylim()[1]])
    fig.tight_layout()
    ax.legend()

    plot_path = "{}/ior.bw.pdf".format(plot_dir)
    if save:
        fig.savefig(plot_path, dpi=300)
    else:
        fig.show()
    pass


def run_allrun_plots(plot_dir):
    run_df = pd.read_csv('.run_df')

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
    fig, ax = plot_allrun_intvlwise(dropzero_df)
    fig_path = '{}/run.intvlwise.pdf'.format(plot_dir)
    fig.savefig(fig_path)
    sys.exit(0)

    for intvl in all_intvls:
        intvl_df = run_df[run_df['intvl'] == intvl]
        fig, ax = plot_allrun_df(intvl_df)
        fig_path = '{}/run.intvl{}.pdf'.format(plot_dir, intvl)
        fig.savefig(fig_path)
        # fig.show()
        # sys.exit(0)

    for intvl in all_intvls:
        for drop in all_drop:
            param_df = run_df[(run_df['intvl'] == intvl)
                              & (run_df["drop"] == drop)]
            fig, ax = plot_allrun_df(param_df)
            fig_path = '{}/run.intvl{}.drop{}.pdf'.format(plot_dir, intvl, drop)
            fig.savefig(fig_path)


def read_fio_df(data_path):
    fio_dfpath = '{}/fio-stats.csv'.format(data_path)
    fio_df = pd.read_csv(fio_dfpath, names=['rname', 'io_bytes', 'bw',
                                            'runtime']).dropna()
    fio_bs = fio_df['rname'].map(lambda x: x.split('_')[1][1:])
    fio_epcnt = fio_df['rname'].map(lambda x: x.split('_')[2][2:].strip('.fio'))
    print(fio_df)
    print(fio_bs)
    print(fio_epcnt)

    fig, ax = plt.subplots(1, 1)
    fio_ep = [1, 3, 6, 9, 12]
    fio_data = [209, 665, 1242, 1750, 2672]

    carp_path = '{}/carp-suite-repfirst.csv'.format(data_path)
    carp_df = pd.read_csv(carp_path)
    print(carp_df)

    dfs_path = '{}/deltafs-jobdir.csv'.format(data_path)
    dfs_df = pd.read_csv(dfs_path)

    ax.plot(fio_ep, fio_data, label='FIO')
    ax.plot(carp_df['epcnt'], carp_df['total_io_time'] / 1000.0, label='CARP')
    ax.plot(dfs_df['epcnt'], dfs_df['total_io_time'] / 1000.0, label='DeltaFS')

    ax.set_title('Data Scalability')

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time taken')
    # fig.show()
    fig.savefig('{}/datascal.pdf'.format(data_path), dpi=300)

    pass


def run_datascal_vs_dfs_fio():
    data_path = '/Users/schwifty/Repos/workloads/rundata/20220825-pvtcnt-analysis/data'
    read_fio_df(data_path)


def plot_rtp_lat(eval_dir: str, save: False):
    latdata_path = '/Users/schwifty/Repos/workloads/rundata/post-sc-jul28-onwards/rtp-bench-runs.csv'
    df = pd.read_csv(latdata_path)
    print(df)

    fig, ax = plt.subplots(1, 1)
    linestyles = {
        100: '-',
        10: '-.',
        1: ':'
    }

    for rnum in linestyles.keys():
        print(rnum)
        df_plot = df[df['rounds'] == rnum]
        data_x = df_plot['nranks']
        data_y = df_plot['mean']
        ls = linestyles[rnum]
        label = 'Avg ({} rounds)'.format(rnum)
        ax.plot(data_x, data_y, ls, label=label)

    df_std = df[df['rounds'] == 100]
    data_y1 = df_std['mean'] - df_std['std']
    data_y2 = df_std['mean'] + df_std['std']
    data_x = df_std['nranks']

    ax.fill_between(data_x, data_y1, data_y2, facecolor='green', alpha=0.1)

    ax.set_xscale('log')
    xticks = df['nranks'].unique()
    ax.set_xticks(xticks)
    ax.minorticks_off()
    ax.set_xticklabels([str(i) for i in xticks])
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}ms'.format(x / 1000))

    ax.set_title('RTP Round Latency')
    ax.set_xlabel('Number of Ranks')
    ax.set_ylabel('Time')

    ax.legend(loc='upper left')

    if save:
        fig.savefig(eval_dir + '/post-sc/rtp.lat.pdf', dpi=600)
    else:
        fig.show()


def run_plot_rtpbench(plot_dir):
    plot_rtp_lat(plot_dir, False)
    pass


if __name__ == "__main__":
    # plot_dir for narwhal
    run_dir = "/mnt/lt20ad1/carp-jobdir/load-balancing-paramsweep"
    plot_dir = "figures/20220815"
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20220825-pvtcnt-analysis"
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20220912-aggr-data"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot_init()
    #  run_bwusage(plot_dir)
    # gen_allrun_df(run_dir, cached=False)
    # run_allrun_plots(plot_dir)
    # run_datascal_vs_dfs_fio()
    # plot_ior(plot_dir)
    run_plot_rtpbench(plot_dir)
