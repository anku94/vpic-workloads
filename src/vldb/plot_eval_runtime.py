""" Created Sep 15, 2022 """

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
import pandas as pd
import re
import sys

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


def plot_allrun_intvlwise(run_df, ax, plot_key, label_fmt):
    run_df = run_df.groupby(['intvl', 'pvtcnt'], as_index=False).agg(
        {'total_io_time_mean': 'mean',
         'total_io_time_std': 'mean',
         'load_std_mean': 'mean',
         # 'max_fin_dura_mean': 'mean',
         # 'wr_min_mean': 'mean',
         # 'wr_max_mean': 'mean'
         }).astype({
        'pvtcnt': int,
        'intvl': int
    })

    labels_x = None
    data_x = None

    run_params = {
        'runtime': {
            'ykey': 'total_io_time_mean',
            'ylabel': 'Runtime (seconds)',
            'title': 'Pivot Count vs Runtime as f(intvl) - NoWrite',
            'majorfmt': lambda x, pos: '{:.0f}s'.format(x / 1e3),
            'minorfmt': MultipleLocator(100000),
            'ylim': 1500 * 1e3
        },

        'load_std': {
            'ykey': 'load_std_mean',
            'ylabel': 'Load Std-Dev (%)',
            'title': 'Pivot Count vs Partition Load Std-Dev as f(intvl) - NoWrite',
            'majorfmt': lambda x, pos: '{:.0f}%'.format(x * 100),
            'minorfmt': MultipleLocator(0.05),
            'ylim': 2
        }
    }

    params = run_params[plot_key]

    def intvl_map(intvl):
        if intvl < 100:
            return 1 / intvl
        num_writes = 6500000
        return num_writes / intvl

    run_df['intvl_norm'] = run_df['intvl'].map(intvl_map)
    run_df.sort_values('intvl_norm', inplace=True)
    intvls = run_df['intvl'].unique()

    intvl_str_dict = {
        1: '1X/Epoch',
        3: '1X/3 Epochs',
        750000: '750K Writes (8X/Epoch)',
        1000000: '1M Writes (6X/Epoch)',
        1500000: '1.5M Writes (~4X/Epoch)',
        2000000: '2M Writes (~3X/Epoch)',
        3000000: '3M Writes (~2X/Epoch)'
    }

    def intvl_str_func(nep):
        if nep < 1:
            return f'1X/{int(1 / nep)} Epochs'
        else:
            return f'{int(nep)}X/Epoch'

    for intvl in intvls:
        intvl_df = run_df[run_df['intvl'] == intvl].sort_values(['pvtcnt'])
        intvl_norm = list(intvl_df['intvl_norm'].unique())[0]
        labels_x = intvl_df['pvtcnt']
        data_x = np.arange(len(labels_x))
        # data_y = intvl_df['total_io_time_mean']
        data_y = intvl_df[params['ykey']]
        # data_y_err = intvl_df['total_io_time_std']

        linestyle = '-o'
        if len(data_x) == 1:
            data_x = list(data_x)
            data_y = list(data_y)
            data_x += [1, 2]
            y0 = data_y[0]
            data_y += [y0, y0]
            linestyle = '--x'

        # intvl_str = intvl_str_dict[intvl]
        # ax.errorbar(data_x, data_y, yerr=data_y_err, label=label_fmt.format(intvl),
        #             capsize=8)
        # ax.plot(data_x, data_y, linestyle, label=label_fmt.format(intvl_str))
        ax.plot(data_x, data_y, linestyle, label=intvl_str_func(intvl_norm))

    xticklabels = sorted(run_df["pvtcnt"].unique())
    xticks = list(range(len(xticklabels)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticklabels])

    ax.set_xlabel('#pivots')
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])

    ax.yaxis.set_major_formatter(params['majorfmt'])
    ax.yaxis.set_minor_locator(params['minorfmt'])
    ax.yaxis.grid(b=True, which='major', color='#aaa')
    ax.yaxis.grid(b=True, which='minor', color='#ddd')

    ax.set_ylim([0, params['ylim']])
    ax.legend()

    return ax


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


def plot_roofline_internal_nw(df, df_getrun, ax):
    file_name_noext = "roofline.shufs"
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]

    key = "network-suite-psm"
    label = "Max Shuffle Xput (PSM)"
    dx, dy = df_getrun(df, key)
    linecolor = colors[0]
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    key = "network-suite"
    label = "Max Shuffle Xput (IPoIB, RPC16K)"
    dx, dy = df_getrun(df, key)
    linecolor = colors[1]
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    key = "network-suite-bigrpc"
    label = "Max Shuffle Xput (IPoIB, RPC32K)"
    dx, dy = df_getrun(df, key)
    linecolor = colors[2]
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    key = "network-suite-1hopsim"
    label = "Max Shuffle Xput (IPoIB, 1HOPSIM)"
    dx, dy = df_getrun(df, key)
    linecolor = colors[3]
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    key = "network-suite-1hopsim-node2x"
    label = "Max Shuffle Xput (IPoIB, 1HOPSIM, PPN1/2)"
    dx, dy = df_getrun(df, key)
    linecolor = colors[4]
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)
    pass


def plot_roofline_internal_io(df, df_getrun, ax):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]

    key = "dfs-ioonly"

    def df_getrun_io(df, param_dict, premask=None):
        if premask is not None:
            df_tmp = df[premask]
        else:
            df_tmp = df
        df_params = pd.DataFrame(param_dict)
        df_params = df_params.merge(df_tmp, on=df_params.columns.to_list(), how="left")
        # print(df_params)
        dx, dy = df_getrun(df_params, key, mask_in=None)
        return dx, dy

    params_strongscale = {
        'nranks': [16, 32, 64, 128, 256, 512],
        'epcnt': [12, 12, 12, 6, 3, 1]
    }

    params_weakscale = {
        'nranks': [16, 32, 64, 128, 256, 512],
        'epcnt': [12, 12, 12, 12, 12, 12]
    }

    df_getrun_io(df, params_strongscale, premask=None)

    mask_old = (df['run'] <= 6)
    mask_new = (df['run'] >= 6)

    dx, dy = df_getrun_io(df, params_weakscale, premask=mask_old)
    linecolor = colors[4]
    label = "WeakScale (12 eps), OldData"
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    dx, dy = df_getrun_io(df, params_weakscale, premask=mask_new)
    linecolor = colors[5]
    label = "WeakScale (12 eps), NewData"
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    dx, dy = df_getrun_io(df, params_strongscale, premask=mask_new)
    linecolor = colors[6]
    label = "StrongScale (128/6, 256/3, 512/1)"
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    ax.set_title('IO-Only Xput W/ DeltaFS-IMD')


def plot_roofline_internal_wcarp(df, df_getrun, ax):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]

    key = "dfs-ioonly"
    label = "Max I/O Xput"
    dx, dy = df_getrun(df, key)
    linecolor = colors[5]
    ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    key = "dfs-seq-suite"
    label = "DeltaFS + IMD + NoFilter"
    dx, dy = df_getrun(df, key)
    linecolor = colors[6]
    # ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    key = "dfs-reg-suite"
    label = "DeltaFS + IMD + BloomFilter"
    dx, dy = df_getrun(df, key)
    linecolor = colors[7]
    # ax.plot(dx.astype(str), dy, label=label, marker='o', color=linecolor)

    key = "carp-suite"
    label = "CARP"
    dx, dy = df_getrun(df, key)
    linecolor = colors[8]
    # ax.plot(dx.astype(str), dy, label=label, marker='^', linewidth=2, ms=6,
    #         color=linecolor)

    key = "carp-suite-intraepoch-nowrite"
    label = "CARP - NoWrite"
    dx, dy = df_getrun(df, key)
    linecolor = colors[9]
    # ax.plot(dx.astype(str), dy, label=label, marker='^', linewidth=2, ms=6,
    #         color=linecolor)
    pass


def plot_roofline(plot_dir, df, save=False):
    fig, ax = plt.subplots(1, 1)

    plot_type = 0

    def df_getrun(run_df, runtype, mask_in=None):
        mask = (run_df["runtype"] == runtype)
        if mask_in is not None:
            mask = mask & mask_in

        run_df = run_df[mask]

        run_df = preprocess_allrun_df(run_df)
        run_df = run_df.groupby(['nranks', 'epcnt'], as_index=False).agg({
            'total_io_time_mean': 'mean'
        })

        # print(run_df)

        run_nranks = run_df['nranks']
        run_time = run_df['total_io_time_mean']
        run_epcnt = run_df['epcnt']

        time_s = run_time / 1e3
        data_bytes = run_nranks * (6.55 * 1e6 * 60) * run_epcnt
        data_mb = data_bytes / (2 ** 20)
        bw_mbps = data_mb / time_s

        data_x = run_nranks if plot_type == 0 else run_epcnt
        data_y = bw_mbps

        time_s_str = ', '.join(
            time_s.astype(int).astype(str).map(lambda x: x + 's'))
        print(f"[df_getrun] {runtype}: {time_s_str}")
        bw_x = ', '.join(data_x.astype(str))
        bw_y = ', '.join(data_y.astype(int).astype(str))
        print(f"[df_getrun] [bw_x] {bw_x}")
        print(f"[df_getrun] [bw_y] {bw_y}")
        # print(data_x, data_y)
        return data_x, data_y

    file_name_noext = "roofline.wcarp"
    # file_name_noext = "roofline.shufs"
    file_name_noext = "roofline.ioonly"
    save = True

    # plot_roofline_internal_nw(df, df_getrun, ax)
    plot_roofline_internal_io(df, df_getrun, ax)
    # plot_roofline_internal_warp(df, df_getrun, ax)

    ax.legend()

    """
    Formatting
    """
    x_label = "Num Ranks" if plot_type == 0 else "Num Epochs"
    ax.set_xlabel(x_label)
    ax.set_ylabel('Effective Bandwidth')
    ax.yaxis.set_minor_locator(MultipleLocator(256))
    ax.yaxis.set_major_locator(MultipleLocator(1024))
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f} GB/s'.format(x / 1024))
    # ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_ylim([0, 4096])
    ax.set_ylim([0, 8192 + 1024])
    ax.yaxis.grid(b=True, which='major', color='#aaa')
    ax.yaxis.grid(b=True, which='minor', color='#ddd')
    # ax.set_title('Scaling Ranks (CARP vs Others)')
    # ax.legend()

    fig.tight_layout()
    if save:
        fig.savefig(f'{plot_dir}/{file_name_noext}.png', dpi=300)
    else:
        fig.show()
    # fig.show()

    pass


def prep_data_sources(all_files, rootdir=None):
    all_dfs = []

    for fname in all_files:
        if rootdir:
            fpath = f'{rootdir}/{fname}'
        else:
            fpath = fname
        file_df = pd.read_csv(fpath, index_col=None)
        file_df = file_df.fillna(0)
        # if 'carp' in fname:
        #     file_df['runtype'] = 'carp'
        all_dfs.append(file_df)

    df = pd.concat(all_dfs)
    runs_in_suite = '\n'.join(df['runtype'].unique())
    print(f'[Runs in suite]: \n{runs_in_suite}\n----')

    return df


def run_plot_intvlwise_from_dfall(df_all, plot_dir):
    # mask = ((df_all["runtype"] == "carp-suite-intraepoch")
    #         | (df_all["runtype"] == "carp-suite-interepoch")
    #         | (df_all["runtype"] == "carp-suite")
    #         & (df_all["nranks"] == 512))
    # nowrite = False
    mask = ((df_all["runtype"] == "carp-suite-intraepoch-nowrite")
            | (df_all["runtype"] == "carp-suite-interepoch-nowrite")
            & (df_all["nranks"] == 512))
    nowrite = True
    df_all = df_all[mask]
    rel_columns = ["runtype", "nranks", "epcnt", "intvl", "pvtcnt",

                   "total_io_time_mean", "total_io_time_std",
                   "max_fin_dura_mean", "wr_min_mean", "wr_max_mean"]
    rel_columns = ["runtype", "nranks", "epcnt", "intvl", "pvtcnt",
                   "total_io_time_mean", "total_io_time_std", "load_std_mean"]
    df_all = df_all[rel_columns]
    print(df_all.to_string())
    fig, ax = plt.subplots(1, 1)
    # plot_key = "load_std"
    plot_key = "runtime"
    plot_name = f"{plot_dir}/carp_intvlwise_{plot_key}"
    if nowrite:
        plot_name = f"{plot_name}_nowrite"
    plot_allrun_intvlwise(df_all, ax, plot_key, "Interval: {}")
    print(plot_name)
    fig.tight_layout()
    # fig.show()
    fig.savefig(plot_name + ".pdf")

    return


def run_plot_roofline(plot_dir):
    df_all_dir = "/Users/schwifty/Repos/workloads/rundata/20220926-roofline"
    df_all_dir = "/Users/schwifty/Repos/workloads/rundata/20221020-roofline-throttlecheck"
    df_all_dir = "/Users/schwifty/Repos/workloads/rundata/20221025-roofline-ioexp"

    all_files = [
        "lt20ad1_carp_jobdir_throttlecheck.csv",
        "lt20ad1_deltafs_jobdir_throttlecheck.csv",
        "lt20ad2_carp_jobdir_throttlecheck.csv",
        "lt20ad2_deltafs_jobdir_throttlecheck.csv",
    ]

    df_all = prep_data_sources(all_files, df_all_dir)
    # df_all_dropcond = df_all[(df_all["run"] == 1) & (df_all["runtype"] == "network-suite")].index
    # df_all.drop(df_all_dropcond, inplace=True)
    # df_all = preprocess_allrun_df(df_all)

    # print(df_all)
    plot_roofline(df_all_dir, df_all, save=False)
    # TODO: df_all no longer preprocessed outside
    # run_plot_intvlwise_from_dfall(df_all, df_all_dir)


if __name__ == "__main__":
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20221020-roofline-throttlecheck"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot_init()
    run_plot_roofline(plot_dir)
