import glob
import os

from common import PlotSaver, plot_init_bigfont
from staged import StagedBuildout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator


def plot_intvls_addpt(plot_key, df, intvl, ax, **kwargs):
    print(df[df["intvl"] == intvl][["pvtcnt", "total_io_time"]].to_string())
    df_intvl = (
        df[df["intvl"] == intvl]
        .groupby("pvtcnt", as_index=False)
        .agg({"total_io_time": "mean", "max_fin_dura": "mean", "load_std": "mean"})
    )
    data_pvtcnt = df_intvl["pvtcnt"]
    if plot_key == "runtime":
        data_y = df_intvl["total_io_time"]
    else:
        data_y = df_intvl["load_std"]
    print(f"[Intvl] {intvl}, dx: {data_pvtcnt.tolist()}")
    print(f"[Intvl] {intvl}, dy: {(data_y / 1000).astype(int).tolist()}")

    data_x = range(len(data_y))
    ax.plot(data_x, data_y, "-o", **kwargs)
    pass


def plot_roofline_util_addpt(ax, df, key, label, linecolor, marker="o"):
    df_bw, df_bw_aggr = filter_df_by_run(df, key)

    # ax.plot(df_bw['x'].astype(str), df_bw['y'], 'x', ms=6, color=linecolor)
    # ax.plot(df_bw_aggr['x'].astype(str), df_bw_aggr['y'], label=label,
    #         marker='o', linewidth=2, ms=6,
    #         color=linecolor)

    if label == "CARP":
        ax.errorbar(
            df_bw_aggr["x"].astype(str),
            df_bw_aggr["y"],
            yerr=df_bw_aggr["yerr"],
            label=label,
            marker="o",
            linewidth=2,
            ms=10,
            color="#4E9B8F",
            capsize=4,
            mec="black",
            mew="1",
            mfc=linecolor[0],
        )
    elif label == "DeltaFS":
        ax.errorbar(
            df_bw_aggr["x"].astype(str),
            df_bw_aggr["y"],
            yerr=df_bw_aggr["yerr"],
            label=label,
            marker="s",
            linewidth=2,
            ms=13,
            color="#C1443C",
            capsize=4,
            mec="black",
            mew="1",
            mfc=linecolor[3],
        )
    else:
        ax.errorbar(
            df_bw_aggr["x"].astype(str),
            df_bw_aggr["y"],
            yerr=df_bw_aggr["yerr"],
            label=label,
            marker=marker,
            linewidth=2,
            ms=13,
            color="#4F7697",
            capsize=4,
            mec="black",
            mew="1",
            mfc=linecolor[4],
        )


def get_bw_mbps(run_df):
    run_nranks = run_df["nranks"]
    if "total_io_time_mean" in run_df.columns:
        run_time = run_df["total_io_time_mean"]
    else:
        run_time = run_df["total_io_time"]
    run_epcnt = np.array(run_df["epcnt"], dtype=float)

    for idx, r_eps in enumerate(zip(run_nranks, run_epcnt)):
        r, eps = r_eps
        if r > 512:
            print(f"[get_bw] Truncating epcnt for {r} ranks")
            run_epcnt[idx] = 512.0 / r
        # print(idx, r, eps)

    data_1r1ep_bytes = 6.55 * 1e6 * 60

    time_s = run_time / 1e3
    data_bytes = run_nranks * run_epcnt * data_1r1ep_bytes
    data_mb = data_bytes / (2**20)
    bw_mbps = data_mb / time_s

    data_x = run_nranks
    data_y = bw_mbps

    time_s_str = ", ".join(time_s.astype(int).astype(str).map(lambda x: x + "s"))
    runtype = run_df["runtype"].unique()[0]
    print(f"[df_getrun] {runtype}: {time_s_str}")
    bw_x = ", ".join(data_x.astype(str))
    bw_y = ", ".join(data_y.astype(int).astype(str))
    print(f"[df_getrun] [bw_x] {bw_x}")
    print(f"[df_getrun] [bw_y] {bw_y}")
    return data_x, data_y


def filter_df_by_run(df, runtype):
    df_run = df[df["runtype"] == runtype]
    dx, dy = get_bw_mbps(df_run)
    df_bw = pd.DataFrame({"x": dx, "y": dy})
    df_bw_aggr = df_bw.groupby("x", as_index=False).agg({"y": ["mean", "std"]})
    df_bw_aggr.columns = ["x", "y", "yerr"]
    print(df_bw_aggr)
    return df_bw, df_bw_aggr


def filter_params(df, params):
    df_params = pd.DataFrame(params)
    df_params = df_params.merge(df, on=df_params.columns.to_list(), how="left")
    return df_params


def prep_data_sources(rootdir):
    all_files = glob.glob(rootdir + "/*.csv")
    all_files = [f for f in all_files if "simpl" not in f]
    all_files = [f for f in all_files if "deltafs" in f or "carp" in f]
    if "20221118" in rootdir:
        all_files = [f for f in all_files if "ad3" in f]

    print(f"Scanning Root dir: {rootdir}... {len(all_files)} files found.")
    for idx, f in enumerate(all_files):
        print(f"[{idx}] {os.path.basename(f)}")

    all_dfs = []

    for fname in all_files:
        if rootdir:
            fpath = f"{rootdir}/{os.path.basename(fname)}"
        else:
            fpath = fname
        file_df = pd.read_csv(fpath, index_col=None)
        file_df = file_df.fillna(0)
        # if 'carp' in fname:
        #     file_df['runtype'] = 'carp'
        all_dfs.append(file_df)

    df = pd.concat(all_dfs)
    return df


def aggr_data_sources():
    repo_path = "/Users/schwifty/Repos/vpic-workloads"

    df_dir = f"{repo_path}/rundata/20221128-roofline-ss1024-re"
    df = prep_data_sources(df_dir)

    all_masks = [
        df["run"] >= 40,
        df["runtype"] == "network-suite",
        df["runtype"].str.contains("nowrite"),
    ]

    df_plot = pd.concat(map(lambda x: df[x], all_masks))

    runs_in_suite = "\n".join(df_plot["runtype"].unique())
    print(f"\n[Runs in suite]: \n{runs_in_suite}\n----")

    return df_plot


def filter_strongscale(df):
    print("\n--- Applying Strongscale Filter ---")
    params_strongscale = {
        "nranks": [32, 64, 128, 256, 512, 1024],
        "epcnt": [12, 12, 6, 3, 1, 1],
    }

    df = df.astype({"nranks": int, "epcnt": int})

    df_params = filter_params(df, params_strongscale)

    # df_params[df_params['nranks'] == 1024]['epcnt'] = 0.5

    return df_params


def plot_roofline_internal_addshadedreg(df, ax):
    cm = plt.cm.get_cmap("Pastel1")
    colors = list(cm.colors)

    key = "network-suite"
    _, df_b = filter_df_by_run(df, key)
    ax.fill_between(
        df_b["x"].astype(str),
        df_b["y"],
        0,
        color=colors[6],
        edgecolor="#333",
        linewidth=0,
        alpha=0.6,
        label="Network Bound",
        hatch="\\",
    )

    key = "dfs-ioonly"
    _, df_a = filter_df_by_run(df, key)

    ax.plot(
        df_a["x"].astype(str),
        df_a["y"],
        "--",
        ms=6,
        color="#444",
        label="",
        linewidth=2,
    )
    ax.fill_between(
        df_a["x"].astype(str),
        df_a["y"],
        0,
        color=colors[1],
        edgecolor="#777",
        linewidth=0,
        alpha=0.2,
        label="Storage Bound",
        hatch="/",
    )
    #
    # # dy_min = np.minimum(df_a["y"].tolist(), df_b["y"].tolist())
    # ax.fill_between(df_a["x"].astype(str), dy_min, 0, color='#bbb',
    #                 edgecolor="b",
    #                 linewidth=0, alpha=0.5, label="")


def plot_roofline_internal_vldb(df, ax):
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [c for c in prop_cycle.by_key()["color"]]

    cm = plt.cm.get_cmap("Dark2")
    cm = plt.cm.get_cmap("Set3")
    colors = list(cm.colors)

    all_labels = {
        "dfs-reg-suite": "DeltaFS",
        "carp-suite-intraepoch-nowrite": "CARP - Shuffle Only",
        "carp-suite-intraepoch": "CARP - IntraEpoch",
        "carp-suite-intraepoch-skipsort": "CARP - IntraEpoch/NoSort",
        "carp-suite-intraepoch-nowrite": "CARP - IntraEpoch/NoWrite",
        "carp-suite-interepoch": "CARP",  # renamed from CARP-InterEpoch
        "carp-suite-interepoch-skipsort": "CARP - InterEpoch/NoSort",
        "carp-suite-interepoch-nowrite": "CARP/ShuffleOnly",  # renamed
        "dfs-ioonly": "Storage Bound - Line",
        "network-suite": "Network Bound - Line",
        "network-suite-psm": "Max Shuffle Xput (PSM)",
        "network-suite-bigrpc": "Max Shuffle Xput (IPoIB, RPC32K)",
        "network-suite-1hopsim": "Max Shuffle Xput (IPoIB, 1HOPSIM)",
        "network-suite-1hopsim-node2x": "Max Shuffle Xput (IPoIB, 1HOPSIM, PPN1/2)",
    }

    keys_to_plot = [
        "dfs-reg-suite",
        "carp-suite-intraepoch-skipsort",
        "carp-suite-interepoch",
        "carp-suite-interepoch-nowrite",
    ]

    keys_to_plot = [
        # "carp-suite-interepoch-nowrite",
        "dfs-reg-suite",
        "carp-suite-interepoch",
    ]

    markers = [
        # 'X',
        "s",
        "o",
    ]

    for kidx, key in enumerate(keys_to_plot):
        print(f"Plotting datapoints: {key}")
        plot_roofline_util_addpt(
            ax, df, key, all_labels[key], colors, marker=markers[kidx]
        )


def plot_tritonsort(df, ax):
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [c for c in prop_cycle.by_key()["color"]]

    cm = plt.cm.get_cmap("Set3")
    colors = list(cm.colors)

    ts_sort_per200 = 3360 / 12.0
    size_ep_g = (6.55 * 1e6 * 60 * 512) / (2**30)
    time_ts_sort = ts_sort_per200 * size_ep_g / 200.0
    print(time_ts_sort)  # ---- time needed for one epoch

    time_ioonly = (
        df[(df["runtype"] == "dfs-ioonly") & (df["nranks"] == 512)]["total_io_time"]
        .mean()
        .tolist()
    )
    time_ioonly /= 1000
    # time_ioonly = time_ioonly[4:]
    print(time_ioonly)
    data_mb = size_ep_g * (2**10)
    time_total = time_ioonly + time_ts_sort

    bw_ts_mbps = data_mb / time_total
    _, df = filter_df_by_run(df, "dfs-ioonly")
    data_x = df["x"]
    ax.plot(
        data_x.astype(str),
        [bw_ts_mbps] * len(data_x),
        "-D",
        label="TritonSort",
        color="#C97F3C",
        ms=11,
        mec="black",
        mfc=colors[5],
    )
    print(f"[TritonSort] {bw_ts_mbps} MB/s")
    pass


def plot_fq(df, ax):
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [c for c in prop_cycle.by_key()["color"]]

    cm = plt.cm.get_cmap("Set3")
    colors = list(cm.colors)

    # fq, 320 ranks, time = 121s, 128s, 124s
    times_fq = [121, 128, 124]
    time_fq_mean = np.mean(times_fq)
    time_fq_std = np.std(times_fq)

    print(time_fq_mean)  # ---- time needed for one epoch

    time_ioonly = (
        df[(df["runtype"] == "dfs-ioonly") & (df["nranks"] == 512)]["total_io_time"]
        .mean()
        .tolist()
    )
    time_ioonly /= 1000
    # time_ioonly = time_ioonly[4:]
    print(time_ioonly)

    size_ep_g = (6.55 * 1e6 * 60 * 512) / (2**30)
    data_mb = size_ep_g * (2**10)
    time_total = time_ioonly + time_fq_mean

    bw_ts_mbps = data_mb / time_total
    _, df = filter_df_by_run(df, "dfs-ioonly")
    data_x = df["x"]
    ax.plot(
        data_x.astype(str),
        [bw_ts_mbps] * len(data_x),
        "-^",
        label="FastQuery",
        color="#7D7999",
        ms=12,
        mec="black",
        mfc=colors[2],
    )
    print(f"[FastQuery] {bw_ts_mbps} MB/s")
    pass


def plot_roofline_inner(ax: plt.Axes):
    plot_init_bigfont()
    df_plot = aggr_data_sources()
    df_ss = filter_strongscale(df_plot)
    df = df_ss

    plot_roofline_internal_addshadedreg(df, ax)
    plot_roofline_internal_vldb(df, ax)
    plot_tritonsort(df, ax)
    plot_fq(df, ax)

    handles, labels = ax.get_legend_handles_labels()

    handles[0].set_linestyle("dashed")
    handles[0].set_linewidth(2)
    new_handles = []
    #
    # for h in handles:
    #     if isinstance(h, ErrorbarContainer):
    #         new_handles.append(h[0])
    #     else:
    #         new_handles.append(h)
    #
    # handle_order = [0, 1, 3, 2, 4, 5, 6]
    handle_order = [0, 1, 2, 3, 4, 5]
    new_handles = [handles[i] for i in handle_order]
    labels = [labels[i] for i in handle_order]

    leg = ax.legend(
        new_handles,
        labels,
        ncol=2,
        fontsize=17,
        framealpha=0.5,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.03),
    )

    for h in leg.legendHandles:
        if isinstance(h, Rectangle):
            h.set(ec="black", linewidth=1)
    #
    """
    Formatting
    """
    x_label = "Number of Ranks"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Ingestion Throughput")
    ax.yaxis.set_minor_locator(MultipleLocator(256))
    ax.yaxis.set_major_locator(MultipleLocator(1024))
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} GB/s".format(x / 1024))
    # ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_ylim([0, 4096])
    ax.set_ylim([0, 1024 * 4.5])
    ax.yaxis.grid(visible=True, which="major", color="#aaa")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")
    # ax.set_title('Scaling Ranks (CARP vs Others)')
    # ax.legend()
    pass


def plot_roofline():
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # fig.clear()
    # ax = fig.add_subplot(111)
    plot_roofline_inner(ax)
    fig.tight_layout()

    ax.get_children()
    sb = StagedBuildout(ax)

    alx = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [], [])

    alx_nwb = ([0], [], [])
    alx_stb = ([1, 2], [], [])
    alx_dfs = ([3, 4, 5, 6], [], [])
    alx_carp = ([7, 8, 9, 10], [], [])
    alx_ts = ([11], [], [])
    alx_fq = ([12], [], [])

    sb.disable_alx(*alx_nwb)
    sb.disable_alx(*alx_stb)
    sb.disable_alx(*alx_dfs)
    sb.disable_alx(*alx_carp)
    sb.disable_alx(*alx_ts)
    sb.disable_alx(*alx_fq)

    plot_dir = "/Users/schwifty/CMU/18911/Documents/20240901_CARP_SC24/plots"
    PlotSaver.save(fig, plot_dir, "roofline.f0")

    sb.enable_alx(*alx_stb)
    PlotSaver.save(fig, plot_dir, "roofline.f1")

    sb.enable_alx(*alx_nwb)
    PlotSaver.save(fig, plot_dir, "roofline.f2")

    sb.enable_alx(*alx_fq)
    PlotSaver.save(fig, plot_dir, "roofline.f3")

    sb.enable_alx(*alx_ts)
    PlotSaver.save(fig, plot_dir, "roofline.f4")

    sb.enable_alx(*alx_dfs)
    PlotSaver.save(fig, plot_dir, "roofline.f5")

    sb.enable_alx(*alx_carp)
    PlotSaver.save(fig, plot_dir, "roofline.f6")

    plt.close(fig)


def run():
    plot_roofline()


if __name__ == "__main__":
    run()
