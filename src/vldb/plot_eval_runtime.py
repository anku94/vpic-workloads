""" Created Sep 15, 2022 """

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
import glob
import numpy as np
import os
import pandas as pd
import re
import sys

from common import plot_init_bigfont, plot_init_bigfont_singlecol
from common import PlotSaver


def std(s):
    return np.std(s)


def customavg(ser):
    ser = sorted(list(ser))
    ser_tmp = ser[1:-1]
    print(f'[Removing Outliers] Before: {len(ser)}, After: {len(ser_tmp)}')
    return np.avg(ser_tmp)


def get_bw_mbps(run_df):
    run_nranks = run_df['nranks']
    if 'total_io_time_mean' in run_df.columns:
        run_time = run_df['total_io_time_mean']
    else:
        run_time = run_df['total_io_time']
    run_epcnt = np.array(run_df['epcnt'], dtype=float)

    for idx, r_eps in enumerate(zip(run_nranks, run_epcnt)):
        r, eps = r_eps
        if r > 512:
            print(f"[get_bw] Truncating epcnt for {r} ranks")
            run_epcnt[idx] = 512.0 / r
        # print(idx, r, eps)

    data_1r1ep_bytes = (6.55 * 1e6 * 60)

    time_s = run_time / 1e3
    data_bytes = run_nranks * run_epcnt * data_1r1ep_bytes
    data_mb = data_bytes / (2 ** 20)
    bw_mbps = data_mb / time_s

    data_x = run_nranks
    data_y = bw_mbps

    time_s_str = ', '.join(
        time_s.astype(int).astype(str).map(lambda x: x + 's'))
    runtype = run_df["runtype"].unique()[0]
    print(f"[df_getrun] {runtype}: {time_s_str}")
    bw_x = ', '.join(data_x.astype(str))
    bw_y = ', '.join(data_y.astype(int).astype(str))
    print(f"[df_getrun] [bw_x] {bw_x}")
    print(f"[df_getrun] [bw_y] {bw_y}")
    return data_x, data_y


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


def plot_allrun_intvlbar(run_df, plot_dir, save=False):
    run_df = run_df[run_df['epcnt'] == 12]

    fig, ax = plt.subplots(1, 1)

    data_x = run_df['intvl']
    data_y1a = run_df['wr_min_mean']
    data_y1a_err = run_df['wr_min_std']
    data_y1b = run_df['wr_max_mean']
    data_y1b_err = run_df['wr_max_std']

    pts_x = np.arange(len(data_x))

    width = 0.4
    ax.bar(pts_x - width / 2, data_y1a, yerr=data_y1a_err, width=width,
           capsize=5, label='Min Item Count')
    ax.bar(pts_x + width / 2, data_y1b, yerr=data_y1b_err, width=width,
           capsize=5, label='Max Item Count')

    ax.set_xlabel('Renegotiation Interval')
    ax.set_ylabel('Partition Size (#items)')
    ax.set_title('Impact Of Reneg Interval (CARP)')

    ax.set_xticks(pts_x)
    ax.set_xticklabels(data_x)

    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}M'.format(x / 1e6))
    # ax.yaxis.set_minor_locator(MultipleLocator(1e3 * 125))
    ax.yaxis.grid(b=True, which='major', color='#aaa')
    ax.yaxis.grid(b=True, which='minor', color='#ddd')
    ax.set_axisbelow(True)
    ax.legend()

    twinx_mode = "load"
    twinx_mode = "runtime"
    twinx_mode = "runtime_err"

    ax2 = ax.twinx()

    if twinx_mode.startswith("load"):
        data_y2 = run_df['load_std_mean']
        ax2.plot(pts_x, data_y2, marker='o')
        ax2.set_ylabel("Load Std-Dev (%)")
        ax2.yaxis.set_major_formatter(lambda x, pos: '{:.0f}%'.format(x * 100))
        ax2.set_ylim([0, ax2.get_ylim()[1] * 1.05])
    elif twinx_mode.startswith("runtime"):
        data_y2 = run_df['total_io_time_mean']
        data_y2_err = run_df['total_io_time_std']
        if 'err' in twinx_mode:
            ax2.errorbar(pts_x, data_y2, yerr=data_y2_err, marker='x')
        else:
            ax2.plot(pts_x, data_y2, marker='o')
        ax2.set_ylabel("Mean Runtime (s)")
        ax2.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x / 1e3))
        ax2.set_ylim([0, ax2.get_ylim()[1] * 1.25])

    plot_path = f"{plot_dir}/intvl_vs_partsz_{twinx_mode}.pdf"
    fig.tight_layout()
    if save:
        fig.savefig(plot_path, dpi=300)
    else:
        fig.show()
    return


def plot_intvls_addpt(plot_key, df, intvl, ax, **kwargs):
    print(df[df["intvl"] == intvl][["pvtcnt", "total_io_time"]].to_string())
    df_intvl = df[df["intvl"] == intvl].groupby("pvtcnt", as_index=False).agg({
        "total_io_time": "mean",
        "max_fin_dura": "mean",
        "load_std": "mean"
    })
    data_pvtcnt = df_intvl["pvtcnt"]
    if plot_key == "runtime":
        data_y = df_intvl["total_io_time"]
    else:
        data_y = df_intvl["load_std"]
    print(f"[Intvl] {intvl}, dx: {data_pvtcnt.tolist()}")
    print(f"[Intvl] {intvl}, dy: {(data_y / 1000).astype(int).tolist()}")

    data_x = range(len(data_y))
    ax.plot(data_x, data_y, '-o', **kwargs)
    pass


def plot_intvls(df_raw, plot_dir):
    cm = plt.cm.get_cmap('Dark2')
    colors = list(cm.colors)

    def intvl_map(intvl):
        if intvl < 100:
            return 1 / intvl
        num_writes = 6500000
        return num_writes / intvl

    mask_a = df_raw["runtype"] == "carp-suite-intraepoch-skipsort"
    mask_b = df_raw["nranks"] == 512
    mask = mask_a & mask_b
    df = df_raw[mask]

    df = df.astype({
        "intvl": int,
        "pvtcnt": int
    }, copy=True)

    df['intvl_norm'] = df['intvl'].map(intvl_map)
    df.sort_values('intvl_norm', inplace=True)
    intvls = df['intvl'].unique()
    pvtcnt = df['pvtcnt'].sort_values().unique()

    run_params = {
        'runtime': {
            'ykey': 'total_io_time_mean',
            'ylabel': 'Runtime (seconds)',
            'title': 'Pivot Count vs Runtime as f(intvl) - NoWrite',
            'majorfmt': lambda x, pos: '{:.0f}s'.format(x / 1e3),
            'majorloc': MultipleLocator(20000),
            'minorloc': MultipleLocator(5000),
            'ylim': 85 * 1e3,
            'fname': 'runtime.vs.params'
        },

        'load_std': {
            'ykey': 'load_std_mean',
            'ylabel': 'Load Std-Dev (\%)',
            'title': 'Pivot Count vs Partition Load Std-Dev as f(intvl) - NoWrite',
            'majorfmt': lambda x, pos: '{:.0f}%'.format(x * 100),
            'majorloc': MultipleLocator(0.1),
            'minorloc': MultipleLocator(0.05),
            'ylim': 0.55,
            'fname': 'load.vs.params'
        }
    }

    # plot_key = "load_std"
    plot_key = "runtime"
    params = run_params[plot_key]

    def intvl_str_func(nep):
        if nep < 1:
            return f'1X/{int(1 / nep)} Epochs'
        else:
            return f'{int(nep)}X Reneg./Epoch'

    fig, ax = plt.subplots(1, 1)

    for idx, intvl in enumerate(intvls):
        plot_intvls_addpt(plot_key, df, intvl, ax,
                          label=intvl_str_func(intvl_map(intvl)),
                          color=colors[idx + 1])

    xticks = list(range(len(pvtcnt)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in pvtcnt])

    ax.set_xlabel('Pivot Count')
    ax.set_ylabel(params['ylabel'])

    ax.yaxis.set_major_formatter(params['majorfmt'])
    ax.yaxis.set_major_locator(params['majorloc'])
    ax.yaxis.set_minor_locator(params['minorloc'])
    ax.yaxis.grid(visible=True, which='major', color='#aaa')
    ax.yaxis.grid(visible=True, which='minor', color='#ddd')

    ax.set_ylim([0, params['ylim']])

    ratio = 0.33
    ax.set_aspect(1.0 / ax.get_data_ratio() * ratio)

    if plot_key == "runtime":
        ax.legend(ncol=2, fontsize=15, loc="upper left", bbox_to_anchor=(0.01, 0.65), framealpha=0.6)
    else:
        ax.legend(ncol=2, fontsize=15, loc="upper left", bbox_to_anchor=(0.01, 1.07), framealpha=0.6)

    fig.tight_layout()

    PlotSaver.save(fig, plot_dir, params["fname"])


def preprocess_allrun_df(run_df):
    run_df = run_df.fillna(0)

    params_agg = [
        p
        for p in list(run_df.columns)
        if p not in ["Unnamed: 0", "runtype", "nranks", "epcnt", "run", "intvl",
                     "pvtcnt", "drop"]
    ]
    agg_ops = {p: ["mean", std] for p in params_agg}

    params_agg_key = [
        p
        for p in list(run_df.columns)
        if p in ["runtype", "nranks", "epcnt", "intvl", "pvtcnt", "drop"]
    ]
    run_df = run_df.groupby(params_agg_key, as_index=False).agg(
        agg_ops)
    run_df.columns = ["_".join(col).strip("_") for col in run_df.columns]

    cols_to_keep = [col for col in run_df.columns if 'epoch' not in col]
    props_agg = set()
    for col in run_df.columns:
        if not col.startswith('epoch'):
            continue

        mobj = re.match('epoch[0-9]+_(.*)', col)
        if mobj:
            prop = mobj.group(1)
            props_agg.add(prop)

    df_final = run_df[cols_to_keep].copy()

    for prop in props_agg:
        all_cols = run_df.columns

        prop_cols = [col for col in all_cols if prop in col]
        df_final[prop] = run_df[prop_cols].mean(axis=1)

    # print(df_final.columns)
    # dropzero_df = run_df[run_df['drop'] == 0]
    # print(dropzero_df.columns)

    return df_final


def plot_datascal(plot_dir, df_carp, df_dfs, save=False):
    fig, ax = plt.subplots(1, 1)

    def rundf_axes(run_df):
        run_df = run_df.groupby(['epcnt'], as_index=False).agg({
            'total_io_time_mean': 'mean',
            'total_io_time_std': 'mean'
        })
        print(run_df)
        data_x = run_df['epcnt'].astype(str)
        data_y = run_df['total_io_time_mean']
        data_yerr = run_df['total_io_time_std']
        return data_x, data_y, data_yerr

    for intvl in df_carp["intvl"].unique():
        df_intvl = df_carp[df_carp["intvl"] == intvl]
        dx, dy, dyerr = rundf_axes(df_intvl)
        intvl_map = {
            '250000': '250K',
            '500000': '500K',
            '750000': '750K',
            '1000000': '1M',
        }
        intvl = str(intvl)
        dx = dx.tolist()
        dy = dy.tolist()
        dyerr = dyerr.tolist()
        print(intvl, dx, dy)
        ax.errorbar(dx, dy, yerr=dyerr, label=f'CARP {intvl_map[intvl]}')
        # ax.plot(dx, dy)
        pass

    # ax.plot(data_x_carp, data_y_carp, label='CARP')
    dx_carp, dy_carp, dyerr_carp = rundf_axes(df_carp)
    # ax.errorbar(dx_carp, dy_carp, yerr=dyerr_carp, label='CARP')

    dx_dfs, dy_dfs, dyerr_dfs = rundf_axes(df_dfs)
    ax.errorbar(dx_dfs, dy_dfs, yerr=dyerr_dfs, label='DeltaFS')
    # data_y_ior = np.array([100, 300, 600, 1200]) * 1e3
    # ax.plot(data_x_carp, data_y_ior, '--', label='IOR')

    """
    Formatting
    """
    ax.set_xlabel('Number Of Epochs')
    ax.set_ylabel('Runtime')
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x / 1000))
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.yaxis.grid(b=True, which='major', color='#aaa')
    ax.yaxis.grid(b=True, which='minor', color='#ddd')
    ax.set_title('Data Scalability (CARP vs Others)')
    # ax.legend()

    ax2 = ax.twinx()
    data_x = dx_carp.astype(int)
    ax2.plot(dx_carp, dy_carp / data_x, linestyle='--', label='CARP')
    ax2.plot(dx_carp, dy_dfs / data_x, linestyle='--', label='DeltaFS')
    ax2.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x / 1000))
    ax2.set_ylabel('Avg Time Per Epoch (s)')
    ax2.set_ylim([0, ax2.get_ylim()[1] * 1.25])

    fig.tight_layout()
    fig.legend(ncol=4)

    if save:
        fig.savefig('{}/datascal.pdf'.format(plot_dir), dpi=300)
    else:
        fig.show()

    pass


def plot_datascal_nrankwise(plot_dir, df_carp, df_dfs, save=False):
    fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()

    def rundf_axes(run_df):
        run_df = run_df.groupby(['epcnt'], as_index=False).agg({
            'total_io_time_mean': 'mean',
            'total_io_time_std': 'mean'
        })
        print(run_df)
        data_x = run_df['epcnt'].astype(str)
        data_y = run_df['total_io_time_mean']
        data_yerr = run_df['total_io_time_std']
        return data_x, data_y, data_yerr

    for nranks in df_carp["nranks"].unique():
        df_nranks = df_carp[df_carp["nranks"] == nranks]
        dx, dy, dyerr = rundf_axes(df_nranks)
        ax.errorbar(dx, dy, yerr=dyerr, label=f'{nranks} ranks')
        # ax.plot(dx, dy)
        dxi = dx.astype(int)
        ax2.plot(dx, dy / dxi, linestyle='--')
        pass

    ax.legend()

    # ax.plot(data_x_carp, data_y_carp, label='CARP')
    dx_carp, dy_carp, dyerr_carp = rundf_axes(df_carp)
    # ax.errorbar(dx_carp, dy_carp, yerr=dyerr_carp, label='CARP')

    # dx_dfs, dy_dfs, dyerr_dfs = rundf_axes(df_dfs)
    # ax.errorbar(dx_dfs, dy_dfs, yerr=dyerr_dfs, label='DeltaFS')

    """
    Formatting
    """
    ax.set_xlabel('Number Of Epochs')
    ax.set_ylabel('Runtime')
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x / 1000))
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.yaxis.grid(b=True, which='major', color='#aaa')
    ax.yaxis.grid(b=True, which='minor', color='#ddd')
    ax.set_title('Scaling Ranks (CARP vs Others)')
    # ax.legend()

    ax2.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x / 1000))
    ax2.set_ylabel('Avg Time Per Epoch (s)')
    ax2.set_ylim([0, ax2.get_ylim()[1] * 1.25])

    fig.tight_layout()
    if save:
        fig.savefig('{}/datascal_nranks.pdf'.format(plot_dir), dpi=300)
    else:
        fig.show()
    pass


def plot_roofline_internal_vldb(df, ax):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]

    cm = plt.cm.get_cmap('Dark2')
    cm = plt.cm.get_cmap('Set3')
    colors = list(cm.colors)

    all_labels = {
        "dfs-reg-suite": "DeltaFS",
        "carp-suite-intraepoch-nowrite": "CARP - Shuffle Only",
        "carp-suite-intraepoch": "CARP - IntraEpoch",
        "carp-suite-intraepoch-skipsort": "CARP - IntraEpoch/NoSort",
        "carp-suite-intraepoch-nowrite": "CARP - IntraEpoch/NoWrite",
        "carp-suite-interepoch": "CARP",  # renamed from CARP-InterEpoch
        "carp-suite-interepoch-skipsort": "CARP - InterEpoch/NoSort",
        "carp-suite-interepoch-nowrite": "CARP/ShuffleOnly", # renamed
        "dfs-ioonly": "Storage Bound - Line",
        "network-suite": "Network Bound - Line",
        "network-suite-psm": "Max Shuffle Xput (PSM)",
        "network-suite-bigrpc": "Max Shuffle Xput (IPoIB, RPC32K)",
        "network-suite-1hopsim": "Max Shuffle Xput (IPoIB, 1HOPSIM)",
        "network-suite-1hopsim-node2x": "Max Shuffle Xput (IPoIB, 1HOPSIM, PPN1/2)",
    }

    keys_to_plot = [
        "dfs-reg-suite", "carp-suite-intraepoch-skipsort",
        "carp-suite-interepoch",
        "carp-suite-interepoch-nowrite",
    ]

    keys_to_plot = [
        "carp-suite-interepoch-nowrite",
        "dfs-reg-suite",
        "carp-suite-interepoch",
    ]

    markers = [
        'X',
        's',
        'o'
    ]

    for kidx, key in enumerate(keys_to_plot):
        print(f"Plotting datapoints: {key}")
        plot_roofline_util_addpt(ax, df, key,
                                 all_labels[key], colors, marker=markers[kidx])


def plot_roofline_internal_addshadedreg(df, ax):
    cm = plt.cm.get_cmap('Pastel1')
    colors = list(cm.colors)

    key = "network-suite"
    _, df_b = filter_df_by_run(df, key)
    ax.fill_between(df_b["x"].astype(str), df_b["y"], 0, color=colors[6],
                    edgecolor="#333",
                    linewidth=0, alpha=0.6, label="Network Bound", hatch='\\')

    key = "dfs-ioonly"
    _, df_a = filter_df_by_run(df, key)
    ax.fill_between(df_a["x"].astype(str), df_a["y"], 0, color=colors[1],
                    edgecolor="#777",
                    linewidth=0, alpha=0.6, label="Storage Bound", hatch='/')

    dy_min = np.minimum(df_a["y"].tolist(), df_b["y"].tolist())
    ax.fill_between(df_a["x"].astype(str), dy_min, 0, color='#bbb',
                    edgecolor="b",
                    linewidth=0, alpha=0.5, label="")


def filter_df_by_run(df, runtype):
    df_run = df[df["runtype"] == runtype]
    dx, dy = get_bw_mbps(df_run)
    df_bw = pd.DataFrame({'x': dx, 'y': dy})
    df_bw_aggr = df_bw.groupby('x', as_index=False).agg({'y': ['mean', 'std']})
    df_bw_aggr.columns = ['x', 'y', 'yerr']
    print(df_bw_aggr)
    return df_bw, df_bw_aggr


def plot_roofline_util_addpt(ax, df, key, label,
                             linecolor, marker='o'):
    df_bw, df_bw_aggr = filter_df_by_run(df, key)

    # ax.plot(df_bw['x'].astype(str), df_bw['y'], 'x', ms=6, color=linecolor)
    # ax.plot(df_bw_aggr['x'].astype(str), df_bw_aggr['y'], label=label,
    #         marker='o', linewidth=2, ms=6,
    #         color=linecolor)

    if label == "CARP":
        ax.errorbar(df_bw_aggr['x'].astype(str), df_bw_aggr['y'],
                    yerr=df_bw_aggr['yerr'], label=label,
                    marker='o', linewidth=2, ms=10,
                    color="#4E9B8F", capsize=4, mec='black', mew='1', mfc=linecolor[0])
    elif label == "DeltaFS":
        ax.errorbar(df_bw_aggr['x'].astype(str), df_bw_aggr['y'],
                    yerr=df_bw_aggr['yerr'], label=label,
                    marker='s', linewidth=2, ms=13,
                    color="#C1443C", capsize=4, mec='black', mew='1', mfc=linecolor[3])
    else:
        ax.errorbar(df_bw_aggr['x'].astype(str), df_bw_aggr['y'],
                    yerr=df_bw_aggr['yerr'], label=label,
                    marker=marker, linewidth=2, ms=13,
                    color="#4F7697", capsize=4, mec='black', mew='1', mfc=linecolor[4])


def plot_tritonsort(df, ax):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]

    cm = plt.cm.get_cmap('Set3')
    colors = list(cm.colors)

    ts_sort_per200 = 3360 / 12.0
    size_ep_g = (6.55 * 1e6 * 60 * 512) / (2 ** 30)
    time_ts_sort = ts_sort_per200 * size_ep_g / 200.0
    print(time_ts_sort)  # ---- time needed for one epoch

    time_ioonly = df[(df['runtype'] == 'dfs-ioonly') & (df['nranks'] == 512)][
        'total_io_time'].mean().tolist()
    time_ioonly /= 1000
    # time_ioonly = time_ioonly[4:]
    print(time_ioonly)
    data_mb = size_ep_g * (2 ** 10)
    time_total = time_ioonly + time_ts_sort

    bw_ts_mbps = data_mb / time_total
    _, df = filter_df_by_run(df, "dfs-ioonly")
    data_x = df["x"]
    ax.plot(data_x.astype(str), [bw_ts_mbps] * len(data_x), '-D',
            label='TritonSort', color="#C97F3C", ms=11, mec='black', mfc=colors[5])
    print(f'[TritonSort] {bw_ts_mbps} MB/s')
    pass


def plot_fq(df, ax):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]

    cm = plt.cm.get_cmap('Set3')
    colors = list(cm.colors)

    # fq, 320 ranks, time = 121s, 128s, 124s
    times_fq = [121, 128, 124]
    time_fq_mean = np.mean(times_fq)
    time_fq_std = np.std(times_fq)

    print(time_fq_mean)  # ---- time needed for one epoch

    time_ioonly = df[(df['runtype'] == 'dfs-ioonly') & (df['nranks'] == 512)][
        'total_io_time'].mean().tolist()
    time_ioonly /= 1000
    # time_ioonly = time_ioonly[4:]
    print(time_ioonly)

    size_ep_g = (6.55 * 1e6 * 60 * 512) / (2 ** 30)
    data_mb = size_ep_g * (2 ** 10)
    time_total = time_ioonly + time_fq_mean

    bw_ts_mbps = data_mb / time_total
    _, df = filter_df_by_run(df, "dfs-ioonly")
    data_x = df["x"]
    ax.plot(data_x.astype(str), [bw_ts_mbps] * len(data_x), '-^',
            label='FastQuery', color="#7D7999", ms=12, mec='black', mfc=colors[2])
    print(f'[FastQuery] {bw_ts_mbps} MB/s')
    pass


def plot_roofline(plot_dir, df):
    figsize=[9, 5]
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    file_name_noext = "runtime.v4.werr"

    plot_roofline_internal_addshadedreg(df, ax)
    plot_roofline_internal_vldb(df, ax)
    plot_tritonsort(df, ax)
    plot_fq(df, ax)

    handles, labels = ax.get_legend_handles_labels()
    new_handles = []

    for h in handles:
        if isinstance(h, ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)

    handle_order = [0, 1, 3, 2, 4, 5, 6]
    new_handles = [new_handles[i] for i in handle_order]
    labels = [labels[i] for i in handle_order]

    leg = ax.legend(new_handles, labels, ncol=1, fontsize=17, framealpha=0.5,
                    loc="upper left",
                    bbox_to_anchor=(-0.02, 1.03))

    for h in leg.legendHandles:
        if isinstance(h, Rectangle):
            h.set(ec='black', linewidth=1)

    """
    Formatting
    """
    x_label = "Number of Ranks"
    ax.set_xlabel(x_label)
    ax.set_ylabel('Effective I/O Throughput')
    ax.yaxis.set_minor_locator(MultipleLocator(256))
    ax.yaxis.set_major_locator(MultipleLocator(1024))
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f} GB/s'.format(x / 1024))
    # ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_ylim([0, 4096])
    ax.set_ylim([0, 1024 * 7])
    ax.yaxis.grid(visible=True, which='major', color='#aaa')
    ax.yaxis.grid(visible=True, which='minor', color='#ddd')
    # ax.set_title('Scaling Ranks (CARP vs Others)')
    # ax.legend()

    fig.tight_layout()
    PlotSaver.save(fig, plot_dir, file_name_noext)


def filter_params(df, params):
    df_params = pd.DataFrame(params)
    df_params = df_params.merge(df, on=df_params.columns.to_list(),
                                how="left")
    return df_params


def filter_strongscale(df):
    print("\n--- Applying Strongscale Filter ---")
    params_strongscale = {
        'nranks': [32, 64, 128, 256, 512, 1024],
        'epcnt': [12, 12, 6, 3, 1, 1]
    }

    df = df.astype({
        'nranks': int,
        'epcnt': int
    })

    df_params = filter_params(df, params_strongscale)

    # df_params[df_params['nranks'] == 1024]['epcnt'] = 0.5

    return df_params


def filter_weakscale(df):
    print("\n--- Applying Weakscale Filter ---")
    params_weakscale = {
        'nranks': [16, 32, 64, 128, 256, 512],
        'epcnt': [12, 12, 12, 12, 12, 12]
    }

    df_params = filter_params(df, params_weakscale)

    return df_params


def filter_carp_params(df):
    df_carp = df[df["runtype"].str.contains("skipsort")]
    df_carp = df[df["runtype"].str.contains("intraepoch")]
    print(df_carp)
    return df_carp


def prep_data_sources(rootdir=None):
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
            fpath = f'{rootdir}/{os.path.basename(fname)}'
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
    df_dir = "/Users/schwifty/Repos/workloads/rundata/20221128-roofline-ss1024-re"
    df = prep_data_sources(df_dir)

    all_masks = [
        df["run"] >= 40,
        df["runtype"] == "network-suite",
        df["runtype"].str.contains("nowrite")
    ]

    df_plot = pd.concat(map(lambda x: df[x], all_masks))

    runs_in_suite = '\n'.join(df_plot['runtype'].unique())
    print(f'\n[Runs in suite]: \n{runs_in_suite}\n----')

    return df_plot


def run_plot_roofline():
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20221128-roofline-ss1024-4gbps"
    plot_init_bigfont()
    df_plot = aggr_data_sources()
    df_ss = filter_strongscale(df_plot)
    plot_roofline(plot_dir, df_ss)


def run_plot_intvls():
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20221127-roofline-ss1024-4gbps"
    plot_init_bigfont_singlecol()
    df_plot = aggr_data_sources()
    df_carp = filter_carp_params(df_plot).copy()
    print(df_carp)
    plot_intvls(df_carp, plot_dir)


if __name__ == "__main__":
    # if not os.path.exists(plot_dir):
    #     os.mkdir(plot_dir)

    run_plot_roofline()
    # run_plot_intvls()
