import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import multiprocessing
import numpy as np
import pandas as pd
import re
import sys


def plot_init():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc(
        "font", size=SMALL_SIZE
    )  # controls default text sizes plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def humrd_to_num(s):
    s = s.strip()
    s_num = re.findall("[0-9.]+", s)[0]
    num = float(s_num)
    num_x = 1

    if s[-1] == "K":
        num_x = 1000
    elif s[-1] == "M":
        num_x = 1000 * 1000
    else:
        assert s[-1].isdigit()

    num = int(num * num_x)
    return num


def humrd_to_time_ms(s):
    s = s.strip()
    s_unit = re.findall("[a-z]+", s)[0]
    s_num = re.findall("\d+\.\d+", s)[0]
    s_num = float(s_num)

    unitx_map = {"us": 1e-3, "ms": 1, "s": 1e3, "secs": 1e3}

    unitx = unitx_map[s_unit]
    return s_num * unitx


def std(s):
    return np.std(s)


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


def get_run_params(run_path):
    params_from_dirname = run_path.split("/")[-1].split(".")

    all_params = {}

    for p in params_from_dirname:
        match_obj = re.search("(\D+)([0-9]+)", p)
        param_name = match_obj.group(1)
        param_val = int(match_obj.group(2))
        all_params[param_name] = param_val

    run_log = "{}/log.txt".format(run_path)
    with open(run_log, errors="replace") as f:
        lines = f.readlines()
        lines = lines[-200:]

        l = [line for line in lines if "per rank" in line][-2]
        mobj = re.search("min: (.*), max: (.*)\)", l)
        wr_min, wr_max = mobj.group(1), mobj.group(2)
        wr_min = humrd_to_num(wr_min)
        wr_max = humrd_to_num(wr_max)

        l = [line for line in lines if "normalized load stddev" in line][0]
        mobj = re.search("\d+\.\d+", l)
        load_std = float(mobj.group(0))

        wr_dropped = 0

        l = [line for line in lines if "particles dropped" in line]
        if len(l) > 0:
            l = l[0]
            mobj = re.search("> (.*) particles", l)
            wr_dropped = humrd_to_num(mobj.group(1))

        all_params["wr_min"] = wr_min
        all_params["wr_max"] = wr_max
        all_params["load_std"] = load_std
        all_params["wr_dropped"] = wr_dropped

        all_l = [line for line in lines if "@ epoch #" in line]
        for l in all_l:
            mobj = re.search("epoch #(\d+)\s+(\d+\.\d+.*?)- (\d+\.\d+.*?)\(", l)
            ep_id = mobj.group(1)
            ep_tmin = mobj.group(2)
            ep_tmax = mobj.group(3)
            all_params["epoch{}_tmin".format(ep_id)] = humrd_to_time_ms(ep_tmin)
            all_params["epoch{}_tmax".format(ep_id)] = humrd_to_time_ms(ep_tmax)

    l = [line for line in lines if "final compaction draining" in line][0]
    mobj = re.search("\d+\.\d+.*", l.strip())
    all_params["max_fin_dura"] = humrd_to_time_ms(mobj.group(0))

    l = [line for line in lines if "total io time" in line][0]
    mobj = re.search("\d+\.\d+.*", l.strip())
    all_params["total_io_time"] = humrd_to_time_ms(mobj.group(0))

    return all_params


def gen_allrun_df():
    all_runs = glob.glob(run_dir + "/run*")

    all_params = []
    for run in all_runs:
        try:
            params = get_run_params(run)
            all_params.append(params)
        except Exception as e:
            print("Failed to parse: {}".format(run))

    run_df = pd.DataFrame.from_dict(all_params)
    run_df.to_csv(".run_df")


def plot_allrun_df(run_df):
    run_df = run_df.groupby('pvtcnt', as_index=False).agg(
        {'total_io_time_mean': 'mean',
         'max_fin_dura_mean': 'mean',
         'wr_min_mean': 'mean',
         'wr_max_mean': 'mean'
         })

    labels_x = run_df['pvtcnt']
    data_x = np.arange(len(labels_x))
    data_y1a = run_df['total_io_time_mean']
    data_y1b = run_df['max_fin_dura_mean']
    data_y2a = run_df['wr_min_mean']
    data_y2b = run_df['wr_max_mean']

    ax1_ylim = 160 * 1e3
    ax2_ylim = 14 * 1e6

    fig, ax = plt.subplots(1, 1)

    ax.plot(data_x, data_y1a, label='io_time', marker='x')
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

    for intvl in all_intvls:
        intvl_df = run_df[run_df['intvl'] == intvl]
        fig, ax = plot_allrun_df(intvl_df)
        fig_path = '{}/run.intvl{}.pdf'.format(plot_dir, intvl)
        fig.savefig(fig_path)

    for intvl in all_intvls:
        for drop in all_drop:
            param_df = run_df[(run_df['intvl'] == intvl)
                              & (run_df["drop"] == drop)]
            fig, ax = plot_allrun_df(param_df)
            fig_path = '{}/run.intvl{}.drop{}.pdf'.format(plot_dir, intvl, drop)
            fig.savefig(fig_path)


if __name__ == "__main__":
    # plot_dir for narwhal
    run_dir = "/mnt/lt20ad1/carp-jobdir/load-balancing-paramsweep"
    plot_dir = "figures/20220815"
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20220825-pvtcnt-analysis"
    plot_init()
    #  run_bwusage(plot_dir)
    # gen_allrun_df(run_dir, cached=False)
    run_allrun_plots(plot_dir)
