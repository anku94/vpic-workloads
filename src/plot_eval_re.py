import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FuncFormatter)
from typing import List, Tuple
from vldb.common import PlotSaver, plot_init_bigfont as plot_init


def plot_runtime_alt_2(dir: str, save: bool = False) -> None:
    # plt.figure(figsize=(16, 9))
    # plt.gca().set_aspect(4.0)
    x = ['CARP', 'DELTAFS', 'NOSHUF', 'SORT', 'NOSHUF/SORT']
    ymin = np.array([1428, 1228, 1276, 3360, 1276 + 3360])
    ymax = np.array([1630, 1295, 1279, 3360, 1279 + 3360])

    # x = ["NoShuffle", "DeltaFS", "NoShuffle \n+ TritonSort", "CARP"]
    x = ["DeltaFS", "   TritonSort", "CARP", "FastQuery"]
    ymin = [1228, 1276, 1518, 1276]
    ymax = [1295, 1279, 1604, 1279]

    carp_datapoints = [1518, 1576, 1604]
    carp_mean = np.mean(carp_datapoints)
    print('CARP mean: ', carp_mean)

    ytop = [0, 3360, 0, 759]

    ymin = np.array(ymin)
    ymax = np.array(ymax)

    y = (ymin + ymax) / 2
    y[2] = carp_mean

    errmin = y - ymin
    errmax = ymax - y

    cmap = plt.get_cmap('Pastel2')
    print(cmap(0))

    # order = [0, 3, 1, 2]
    order = [0, 1, 2, 3]
    order = [0, 3, 1, 2]
    xo = [x[o] for o in order]
    yo = [y[o] for o in order]
    errmino = [errmin[o] for o in order]
    errmaxo = [errmax[o] for o in order]
    ytopo = [ytop[o] for o in order]

    fig, ax = plt.subplots(1, 1)
    # eb = plt.errorbar(x, y, yerr=[errmin, errmax], fmt='.')
    bars = ax.bar(xo, yo, width=0.5, yerr=[errmino, errmaxo], capsize=10,
                  color=cmap(0),
                  ec='black')
    bars[1].set_hatch('/')
    bars[2].set_hatch('/')
    bars = ax.bar(xo, ytopo, width=0.5, yerr=[errmino, errmaxo], bottom=yo,
                  color=cmap(0),
                  ec='black')
    bars[1].set_hatch('\\')
    bars[2].set_hatch('\\')

    # ax.annotate('VPIC', xy=(0.5, 300), rotation=90, fontsize=18)
    # ax.annotate('PostProcessing', xy=(0.5, 1800), rotation=90, fontsize=18)
    #
    # ax.annotate('VPIC', xy=(2.5, 300), rotation=90, fontsize=18)
    # ax.annotate('FQ', xy=(2.5, 1600), rotation=90, fontsize=18)

    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(250))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#777', which='major')
    ax.yaxis.grid(True, color='#ccc', which='minor')
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.set_title('CARP vs Everything - Runtime')
    base_fontsz = 20
    ax.set_ylabel('Time To Complete (seconds)', fontsize=base_fontsz)
    ax.set_xlabel('Run Type', fontsize=base_fontsz)

    # ax.set_xticklabels(xo, rotation=10)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(base_fontsz - 4)

    patch_blank = mpatches.Patch(color=cmap(0), label='Runtime')

    patch_top = mpatches.Patch(color=cmap(0), label='Post Processing')
    patch_top.set_hatch('\\\\')

    patch_bottom = mpatches.Patch(color=cmap(0), label='VPIC')
    patch_bottom.set_hatch('//')

    patch_saf = plt.Line2D([0], [0], marker='o', label='Space Amplification',
                           markersize=10, color='#666', mec='#333', mew=10,
                           lw=6)
    patches = [patch_blank, patch_top, patch_bottom, patch_saf]

    for patch in patches[:-1]:
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
    labels = [patch.get_label() for patch in patches]
    plt.legend(patches, labels, fontsize=14, loc="lower left",
               bbox_to_anchor=(-0.02, 0.66), framealpha=0.8)

    ax2 = ax.twinx()
    x = [0, 1, 2, 3]
    carp_mfsz = 16341584
    carp_rdbsz = 2.13154e+12
    y = [14 / (48 * 8.0), carp_mfsz / carp_rdbsz, carp_mfsz / carp_rdbsz,
         81.0 / 301]
    y = np.array(y) + 1
    yo = [y[o] for o in order]
    ax2.plot(x, yo, marker='o', lw=6, mew=10, mec='#333', color='#666')
    ax2.set_ylabel('Space Amplification', fontsize=base_fontsz)
    # ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter('{x:.1f}X')
    ax2.yaxis.set_major_locator(MultipleLocator(.1))
    # ax2.set_yticklabels(['.01%', '.1%', '1%', '10%', '30%'])
    ax2.set_ylim([1, 1.49])

    for label in (ax2.get_yticklabels()):
        label.set_fontsize(base_fontsz - 2)

    # plt.axes().set_aspect(1.0)
    figw, figh = fig.get_size_inches()
    fig.set_size_inches(figw * 1.1, figh)
    fig.tight_layout()

    if save:
        fig.savefig(dir + '/runtime.v3.pdf', dpi=300)
    else:
        fig.show()


def read_query_fq(fpath: str):
    df = pd.read_csv(fpath)
    df.columns = [i.strip() for i in df.columns]
    df = df[['qbeg', 'qend', 'qsel', 'numhits', 'time']]
    df = df.groupby(['qbeg', 'qend', 'qsel', 'numhits']).agg(
        ['mean', 'std']).reset_index()
    df.columns = df.columns.to_flat_index().str.join('_')
    df.columns = [i.strip('_') for i in df.columns]
    df['qsel_actual'] = df['numhits'] / 3584000000
    return df


def read_query_carpdb(fpath):
    data = pd.read_csv(fpath)
    # data = data.sort_values('qselectivity')

    all_plfs = sorted(data['plfspath'].unique())
    type_carp = all_plfs[1]
    type_flat = all_plfs[0]

    print(type_carp)
    print(type_flat)

    assert (not type_carp.endswith('.merged'))
    assert (type_flat.endswith('.merged'))

    data.loc[data['plfspath'] == type_flat, 'qsortus'] = 0
    data['qreadus'] += data['qsortus']

    data['qreadus'] /= 1e6
    data['qkeyselectivity'] *= 100

    data_aggr = data.groupby(['plfspath', 'qbegin', 'qend', 'qkeyselectivity'],
                             as_index=False).agg({'qreadus': ['mean', 'std']})

    data_aggr = data_aggr.reset_index()
    data_aggr.columns = data_aggr.columns.to_flat_index().str.join('_')
    data_aggr.columns = [i.strip('_') for i in data_aggr.columns]

    df_carp = data_aggr[data_aggr['plfspath'] == type_carp].copy()
    df_flat = data_aggr[data_aggr['plfspath'] == type_flat].copy()

    df_carp.drop(['index', 'plfspath'], axis=1, inplace=True)
    df_flat.drop(['index', 'plfspath'], axis=1, inplace=True)

    # df_carp = df_carp.sort_values('qkeyselectivity')
    # df_flat = df_flat.sort_values('qkeyselectivity')

    return df_carp, df_flat


def read_query_csvs(fpath: str):
    basedir = '/Users/schwifty/Repos/vpic-workloads/rundata/eval.re'
    csv_path = basedir + '/querylog.csv'

    fq_path = f'{basedir}/20220328.queries.fastquery.aggr.csv'
    df_fq = pd.read_csv(fq_path)

    data = pd.read_csv(csv_path)
    data = data.sort_values('qselectivity')

    all_plfs = sorted(data['plfspath'].unique())
    type_carp = all_plfs[1]
    type_flat = all_plfs[0]

    print(type_carp)
    print(type_flat)

    assert (not type_carp.endswith('.merged'))
    assert (type_flat.endswith('.merged'))

    data.loc[data['plfspath'] == type_flat, 'qsortus'] = 0
    data['qreadus'] += data['qsortus']

    data['qreadus'] /= 1e6
    data['qkeyselectivity'] *= 100

    data_aggr = data.groupby(['plfspath', 'qbegin', 'qend', 'qkeyselectivity'],
                             as_index=False).agg({'qreadus': ['mean', 'std']})

    df_carp = data_aggr[data_aggr['plfspath'] == type_carp].copy()
    df_flat = data_aggr[data_aggr['plfspath'] == type_flat].copy()

    df_carp = df_carp[['qbegin', 'qend', 'qkeyselectivity', 'qreadus']]
    df_flat = df_flat[['qbegin', 'qend', 'qkeyselectivity', 'qreadus']]

    df_carp.sort_values('qbegin', inplace=True)
    df_flat.sort_values('qbegin', inplace=True)

    df_carp.rename(columns={'qkeyselectivity': 'qsel', 'qreadus': 'time'},
                   inplace=True)
    df_flat.rename(columns={'qkeyselectivity': 'qsel', 'qreadus': 'time'},
                   inplace=True)

    df_carp.columns = list(map(''.join, df_carp.columns.values))
    df_flat.columns = list(map(''.join, df_flat.columns.values))

    df_carp['selrat'] = np.array(df_carp['qsel'].tolist()) / np.array(
        df_fq['qsel'].tolist())
    df_carp['qsel'] = df_fq['qsel'].tolist()
    df_flat['qsel'] = df_fq['qsel'].tolist()

    drop_idx = []
    keep_idx = [0, 4, 7, 15, 18, 21, 22, 24]
    drop_idx = [i for i in range(25) if i not in keep_idx]

    for df in [df_carp, df_flat, df_fq]:
        df.sort_values('qsel', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop([df.index[x] for x in drop_idx], inplace=True)

    scan_fpath = '/Users/schwifty/Repos/vpic-workloads/rundata/eval/runs.uniform/querylog.scan.csv'
    df_scan = pd.read_csv(scan_fpath)
    df_scan['qreadus'] += df_scan['qsortus']
    df_scan = df_scan[['qbegin', 'qend', 'qkeyselectivity', 'qreadus']]
    df_scan['qreadus'] /= 1e6
    df_scan['qkeyselectivity'] *= 100
    df_scan.rename(columns={'qkeyselectivity': 'qsel', 'qreadus': 'time'},
                   inplace=True)
    df_scan['qsel'] = list(df_carp['qsel'])

    with pd.option_context('display.float_format', '{:,.2f}'.format):
        print(df_carp)
        print(df_flat)
        print(df_fq)
        print(df_scan)

    return df_carp, df_flat, df_fq, df_scan


def plot_query_latvssel_unified():
    basedir = '/Users/schwifty/Repos/vpic-workloads/rundata/eval.re'
    csv_path = basedir + '/querylog.csv'
    csv_scan = basedir + '/querylog.scan.csv'

    df_carp, df_flat, df_fq, df_scan = read_query_csvs(csv_path)

    p = df_carp['timemean']
    q = df_fq['timemean']
    print(max(q / p), min(q / p))

    fig, ax = plt.subplots(1, 1, figsize=[7, 6])
    cm = plt.cm.get_cmap('Set2')
    cm = plt.cm.get_cmap('Set3')

    markers = ['o', 'D', '^', 's']
    all_labels = [
        "DeltaFS/FullScan",
        "FastQuery", "TritonSort", "CARP"
    ]
    all_colors = list(cm.colors)
    all_colors = [
        all_colors[0],
        all_colors[5],
        all_colors[2],
        all_colors[3],
    ]
    all_msz = [16, 14, 14, 14]

    for type, df in enumerate([df_carp, df_flat, df_fq, df_scan]):
        rowidx = 0
        for index, row in df.iterrows():
            data_x = row['qsel']
            if type < 3:
                data_y = row['timemean']
                data_err = row['timestd']
            else:
                data_y = row['time']

            marker = markers[type]
            # color per query
            color = cm.colors[rowidx % 8]
            # color per type
            color = all_colors[type]

            if type == 0:
                zorder = 3
            else:
                zorder = 2
            ax.plot(data_x, data_y, marker=marker, mec='black', mfc=color,
                    markersize=all_msz[type], label=all_labels[type], zorder=zorder)
            if type < 3:
                ax.errorbar(data_x, data_y, yerr=data_err, color=color)

            rowidx += 1

    legend_items = []
    legend_items.append(
        plt.Line2D([0], [0], marker='s', mfc=all_colors[3],
                   label='DeltaFS/FullScan', mec='black',
                   markersize=12))
    legend_items.append(
        plt.Line2D([0], [0], marker='^', mfc=all_colors[2], mec='black',
                   label='FastQuery',
                   markersize=12))
    legend_items.append(
        plt.Line2D([0], [0], marker='D', mfc=all_colors[1], mec='black',
                   label='TritonSort',
                   markersize=12))
    legend_items.append(
        plt.Line2D([0], [0], marker='o', mfc=all_colors[0], mec='black',
                   label='CARP',
                   markersize=12))

    ax.legend(handles=legend_items, fontsize=18, loc="lower left",
              bbox_to_anchor=(0.08, 0.02), ncol=2)

    ax.set_yscale('log')
    yticks = [0.04, 0.2, 1, 5, 25, 125]
    yticks = [0.017, 0.07, 0.3, 1.25, 5, 20, 80, 320]
    yticks = [0.01, 0.1, 1, 10, 100, 330]
    # yticks = [0.04, 0.2, 1, 5, 20, 80, 320]
    ax.set_yticks(yticks)

    def tickfmt(x):
        if x - int(x) < 0.01:
            return '{:d}'.format(x)
        else:
            return '{:.2f}'.format(x)

    ax.set_yticklabels([tickfmt(y) for y in yticks])

    def latfmt(x, pos):
        if x < 1:
            return '{:.2f}s'.format(x)
        else:
            return '{:.0f}s'.format(x)

    # ax.yaxis.set_major_formatter('{x:.2f}s')
    ax.yaxis.set_major_formatter(FuncFormatter(latfmt))

    ax.set_xlabel('Query Selectivity')
    # fig.supylabel('Query Latency (seconds)', fontsize=base_fontsz, x=0.03,
    #               y=0.55)
    ax.set_ylabel('Query Latency')

    # ax.minorticks_off()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter('{x:.1f}\%')

    # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label.set_fontsize(base_fontsz - 1)

    ax.xaxis.grid(True, color='#777', which='major')
    ax.xaxis.grid(True, color='#ddd', which='minor')
    # ax.yaxis.grid(True, color='#bbb', which='major')
    ax.yaxis.grid(True, color='#777', which='major')
    ax.yaxis.grid(True, color='#ddd', which='minor')

    fig.tight_layout()

    plot_dir = "/Users/schwifty/CMU/18911/Documents/20240716_ASCR_CARP"
    fname = "qlatvssel.v6.mini"
    PlotSaver.save(fig, plot_dir, fname)


def plot_query_ycsb() -> None:
    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/YCSB.eval'
    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform/YCSB'
    widths = [5, 20, 50, 100]
    tags = ['carp', 'comp']
    qlog_fmt = '{0}/querylog.{1}.{2}.csv'

    all_dfs = []
    for tag in tags:
        dfs = []
        for width in widths:
            qlog_path = qlog_fmt.format(basedir, tag, width)
            df = pd.read_csv(qlog_path)
            df['width'] = width
            dfs.append(df)
        all_dfs.append(pd.concat(dfs))

    # all_dfs[1]['epoch'] -= 2
    print(all_dfs[0].describe())
    print(all_dfs[1].describe())

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[9, 5])
    axes = [axes[0], axes[1]]
    print(axes)
    epochs = [0, 11]
    epoch_labels = "200,7400,14600,19400".split(',')
    epoch_labels = "200,19400".split(',')
    epoch_labels = dict(zip(epochs, epoch_labels))

    x = np.arange(len(widths))
    width = 0.35

    odf, mdf = all_dfs
    odf['qreadus'] /= 1e6
    odf['qsortus'] /= 1e6
    mdf['qreadus'] /= 1e6

    cmap = plt.get_cmap('Pastel2')

    for epoch, ax in zip(epochs, axes):
        orig_df = odf[odf['epoch'] == epoch]
        merged_df = mdf[mdf['epoch'] == epoch]

        orig_df = orig_df.groupby(['width'], as_index=False).agg({
            'qreadus': 'sum',
            'qsortus': 'sum'
        })

        merged_df = merged_df.groupby(['width'], as_index=False).agg({
            'qreadus': 'sum',
        })

        print(orig_df)
        print(merged_df)
        ax.bar(x - width / 2, orig_df['qreadus'], width, label='CARP/Read',
               color=cmap(0), edgecolor='#000', hatch='.')
        ax.bar(x - width / 2, orig_df['qsortus'], width,
               bottom=orig_df['qreadus'], label='CARP/Sort', color=cmap(2),
               edgecolor='#000')
        ax.bar(x + width / 2, merged_df['qreadus'], width,
               label='TritonSort/Read', color=cmap(1), edgecolor='#000', hatch='X')
        ax.yaxis.grid(True, which='major', color='#aaa')
        ax.yaxis.grid(True, which='minor', color='#ddd')
        ax.yaxis.set_major_locator(MultipleLocator(300))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_ylim([0, 1200])
        ax.yaxis.set_major_formatter('{x:.0f}s')
        ax.set_title('Epoch {0}'.format(epoch_labels[epoch]))

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, ncol=1, bbox_to_anchor=(0.74, 0.89),
               fontsize=18, framealpha=0.8)

    fig.supxlabel('Query Width (\#SSTs)')
    fig.supylabel('Total Time Taken')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, left=0.14)
    PlotSaver.save(fig, "wherever", "query.ycsb.v3")


def plot_subpart_perf_abs(dir: str, save: bool = False) -> None:
    basedir = '/Users/schwifty/Repos/vpic-workloads/rundata/eval/big.2'
    basedir = '/Users/schwifty/Repos/vpic-workloads/rundata/eval/runs.uniform'
    csv50 = basedir + '/subpart_exps/olaps.50.csv'
    csv99 = basedir + '/subpart_exps/olaps.99.csv'

    cols = ['intvl', 'runidx', 'epochidx', 'olappct']
    df50 = pd.read_csv(csv50, header=None, names=cols)
    df99 = pd.read_csv(csv99, header=None, names=cols)

    def aggr(df, runs):
        print(runs)
        df = df[(df['runidx'] >= runs[0]) & (df['runidx'] <= runs[1])]
        df = df.groupby('epochidx', as_index=None).agg({
            'olappct': ['mean', 'std']
        })
        df.columns = list(map(''.join, df.columns.values))
        print(df)
        return df['olappctmean'] * 512 * 4 / 100.0

    # ysub4x50 = aggr(df50, [13, 15])
    # ysub2x50 = aggr(df50, [10, 12])
    # ysub1x50 = aggr(df50, [4, 6])
    # ysub0x50 = aggr(df50, [7, 9])
    #
    # print(ysub2x50)
    #
    # ysub4x99 = aggr(df99, [13, 15])
    # ysub2x99 = aggr(df99, [10, 12])
    # ysub1x99 = aggr(df99, [4, 6])
    # ysub0x99 = aggr(df99, [7, 9])
    ysub4x50_a = aggr(df50, [18, 21])
    ysub4x50_b = aggr(df50, [4, 6])
    print(ysub4x50_a)
    print(ysub4x50_b)
    w1 = 4
    w2 = 3
    ysub4x50 = (ysub4x50_a * w1 + ysub4x50_b * w2) / 7
    print(ysub4x50)
    ysub2x50 = aggr(df50, [7, 9])
    ysub1x50 = aggr(df50, [10, 12])
    ysub0x50 = aggr(df50, [13, 15])

    print(ysub0x50)

    # ysub4x99 = aggr(df99, [4, 6])
    ysub4x99 = aggr(df99, [18, 21])
    ysub2x99 = aggr(df99, [7, 9])
    ysub1x99 = aggr(df99, [10, 12])
    ysub0x99 = aggr(df99, [13, 15])

    x_ticks = range(12)
    x_ticklabels = "200,1400,2600,3800,5000,6200,7400,8600,9800,12200,15800,19400"
    x_ticklabels = "200,2000,3800,5600,7400,9200,11000,12800,14600,16400,18200,19400".split(
        ',')
    # x_ticklabels = x_ticklabels.split(',')

    fig, axes = plt.subplots(2, 1, sharex=False)

    # ax.plot(x_ticks y_sub4)
    # ax.plot(x_ticks, y_sub1)
    cm = plt.cm.get_cmap('tab20c')

    ax = axes[0]
    ax.plot(x_ticks, ysub0x50, 'o--', label='50\%ile, w/o repart.', color=cm(1))
    ax.plot(x_ticks, ysub1x50, 'o-', label='50\%ile, with repart.', color=cm(0))
    ax.plot(x_ticks, ysub0x99, 's--', label='99\%ile, w/o repart.', color=cm(5))
    ax.plot(x_ticks, ysub1x99, 's-', label='99\%ile, with repart.', color=cm(4))
    # ax.plot(x_ticks, ysub0x50 / ysub1x50, 'o-',
    #         label='Repartitioning (50 %ile)',
    #         color=cm(0))
    # ax.plot(x_ticks, ysub0x99 / ysub1x99, 's--',
    #         label='Repartitioning (99 %ile)',
    #         color=cm(1))

    ax = axes[1]
    # ax.plot(x_ticks, ysub1x50 / ysub2x50, 'o-', label='2x Subpart. (50 %ile)',
    #         color=cm(4))
    # ax.plot(x_ticks, ysub1x50 / ysub4x50, 'o-', label='4x Subpart. (50 %ile)',
    #         color=cm(8))
    #
    # ax.plot(x_ticks, ysub1x99 / ysub2x99, 's--', label='2x Subpart. (99 %ile)',
    #         color=cm(5))
    # ax.plot(x_ticks, ysub1x99 / ysub4x99, 's--', label='4x Subpart., (99 %ile)',
    #         color=cm(9))
    ax.plot(x_ticks, ysub1x50, '^--', label='50\%ile, no subpart.', color=cm(0),
            alpha=0.6)
    ax.plot(x_ticks, ysub2x50, 'o--', label='50\%ile, 2X subpart', color=cm(0),
            alpha=0.8)
    ax.plot(x_ticks, ysub4x50, 's-', label='50\%ile, 4X subpart', color=cm(0))
    # ax.plot(x_ticks, ysub1x99, 'o--', label='50%ile, 1X Subpart.', color=cm(6))
    # ax.plot(x_ticks, ysub2x99, 'o--', label='50%ile, 2X Subpart', color=cm(5))
    # ax.plot(x_ticks, ysub4x99, 's-', label='50%ile, 4X Subpart', color=cm(4))

    ax = axes[0]

    base_fontsz = 16
    # ax.set_ylabel('Multiplier Gains in Max DPS', fontsize=base_fontsz)
    # fig.supylabel('DPS', x=0.03, y=0.57,
    #               fontsize=base_fontsz)

    # y1_ticks = [0, 10, 20, 30, 40, 50, 60]
    y2_ticks = [0, 1, 2, 3, 4]
    # axes[0].set_yticks(y1_ticks)
    # axes[0].set_yticklabels([str(i) + 'X' for i in y1_ticks],
    #                         fontsize=base_fontsz - 4)
    # axes[1].set_yticks(y2_ticks)
    # axes[1].set_yticklabels([str(i) + 'X' for i in y2_ticks],
    #                         fontsize=base_fontsz - 4)

    # ax.legend(fontsize=base_fontsz - 4)

    for ax in axes:
        ax.set_ylabel('Read\nAmplification', fontsize=base_fontsz - 1)
        ax.set_xlabel('Simulation Timestep', fontsize=base_fontsz)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, rotation=30, fontsize=base_fontsz - 1)
        ax.xaxis.grid(True, color='#bbb')
        ax.yaxis.grid(True, color='#bbb')

    axes[0].legend(ncol=2, fontsize=base_fontsz - 3, loc="lower left",
                   bbox_to_anchor=(0.0, 0.722))
    axes[0].set_yscale('log')
    # axes[0].yaxis.set_major_locator(MultipleLocator(2))
    # yticks0 = [0.1, 0.4, 1.6, 6.4, 25.6, 100]
    # yticks0 = [0.1, 0.5, 2.5, 10, 40]
    yticks0 = [1, 4, 16, 64, 256]
    yticks0 = [4, 16, 64, 256, 1024]
    yticklabels0 = ['{:.0f}X'.format(x / 4) for x in yticks0]
    axes[0].set_yticks(yticks0)
    axes[0].set_yticklabels(yticklabels0, fontsize=base_fontsz - 2)
    # axes[0].yaxis.set_major_formatter('{x:.0f}X')
    axes[0].minorticks_off()
    axes[0].set_ylim([0.5, 512])
    axes[0].set_ylim([2, 2048])

    axes[1].legend(ncol=2, fontsize=base_fontsz - 3, loc="lower left",
                   bbox_to_anchor=(0.0, 0.62))
    axes[1].set_ylim([0.5, 16])
    axes[1].set_yscale('log')
    axes[1].minorticks_off()
    # axes[1].yaxis.set_major_locator(MultipleLocator(2))
    # axes[1].yaxis.set_minor_locator(MultipleLocator(1))
    yticks1 = [1, 2, 4, 8]
    axes[1].set_yticks(yticks1)
    axes[1].yaxis.set_tick_params(labelsize=base_fontsz - 2)
    # axes[1].yaxis.grid(True, color='#ddd', which='minor')
    axes[1].yaxis.set_major_formatter('{x:.0f}X')

    fig.tight_layout()
    PlotSaver.save(fig, dir, 'carpdb.impact.alt4')


def aggr_intvl_olaps(intvls: List, csv_path: str) -> List:
    olap_df = pd.read_csv(csv_path)
    olap_df = olap_df.groupby(['intvl', 'ptile'], as_index=None).agg({
        'olappctmean': 'mean'
    })

    olap_df.columns = list(map(''.join, olap_df.columns.values))
    all_data = []
    for label in [50, 75, 99]:
        pdf = olap_df[olap_df['ptile'] == label]
        data = [pdf[pdf['intvl'] == i]['olappctmean'].iloc[0] for i in intvls]
        all_data.append(data)

    return all_data


def gen_intvl_std(intvls: List) -> List:
    csv_path = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/std.csv'
    data = pd.read_csv(csv_path)
    data = data.groupby('intvl', as_index=False).agg({
        'std': ['mean']
    })
    print(data)
    data.columns = list(map(''.join, data.columns.values))
    data = data.astype({'intvl': 'str'})
    print(data)

    data_y = []

    for x in intvls:
        y = data[data['intvl'] == x]['stdmean'].values[0]
        data_y.append(y * 100)

    return data_y


def aggr_intvl_runtime_2(intvls: List, rtdata_path: str) -> Tuple[List, List]:
    get_ci = lambda a: st.t.interval(0.95, len(a) - 1, loc=np.mean(a),
                                     scale=st.sem(a))

    data = pd.read_csv(rtdata_path)

    data = data.groupby(['intvl'])['time'].apply(list)
    print(data)

    data_y = []
    data_ci = []

    for intvl in intvls:
        vals_y = data.loc[intvl]
        mean_y = np.mean(vals_y)
        data_y.append(mean_y)
        ci_y = get_ci(vals_y)
        data_ci.append(ci_y)

    data_err = np.array(data_ci)
    data_err = np.transpose(data_err)
    data_err[0] = data_y - data_err[0]
    data_err[1] = data_err[1] - data_y
    return (data_y, data_err)


def plot_intvl_runtime_2(dir: str, save: bool = False) -> None:
    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2'
    rtdata_path = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/rt.csv'
    fig, axes = plt.subplots(2, 1)
    ax, ax2 = axes[0], axes[1]

    intvls = [62500, 125000, 250000, 500000, 750000, 1000000, 'everyepoch']
    intvls = [str(i) for i in intvls]

    csv_path_woob = basedir + '/intvls.olapmean.with.oob.csv'
    csv_path_wo_oob = basedir + '/intvls.olapmean.wo.oob.csv'
    # gen_intvl_runtime_2(intvls, csv_path)

    cm = plt.cm.get_cmap('tab20c')
    base_fontsz = 16

    data_x = intvls
    data_y, data_err = aggr_intvl_runtime_2(intvls, rtdata_path)
    ax.errorbar(data_x, data_y, yerr=data_err, capsize=3, ecolor='#555',
                color=cm(4), lw=2, label='runtime')

    axx = ax.twinx()
    axx.plot(intvls, gen_intvl_std(intvls), 's-', color=cm(12), label='load',
             lw=2)
    axx.yaxis.set_major_formatter('{x:.0f}%')

    legend_items = []
    legend_items.append(
        plt.Line2D([0], [0], label='Runtime', color=cm(4), lw=2))
    legend_items.append(
        plt.Line2D([0], [0], marker='s', lw=2, label='Load Stddev',
                   color=cm(12)))
    ax.legend(handles=legend_items, ncol=2, fontsize=base_fontsz - 4,
              loc="lower left",
              bbox_to_anchor=(0.0, -0.03))

    all_data_woob = aggr_intvl_olaps(intvls, csv_path_woob)
    all_data_wo_oob = aggr_intvl_olaps(intvls, csv_path_wo_oob)

    cm_ptile = {50: 0, 75: 4, 99: 8}

    data_x = list(range(len(intvls)))

    legend_items = []
    for ptile, data in zip([50, 75, 99], all_data_woob):
        if ptile == 75: continue
        data = np.array(data) * 2048 / 100.0
        print(data)
        label = '{0} %ile'.format(ptile)
        linecol = cm(cm_ptile[ptile])
        ax2.plot(data_x, data, 'o-', label=label, color=linecol)
        legend_items.append(
            plt.Line2D([0], [0], marker='o', label=label, color=linecol))

    legend_items.append(
        plt.Line2D([0], [0], marker='o', label='With Out-Of-Order Keys',
                   color='black'))
    legend_items.append(
        plt.Line2D([0], [0], marker='s', linestyle='-.',
                   label='Without Out-Of-Order Keys',
                   color='gray'))

    for ptile, data in zip([50, 75, 99], all_data_wo_oob):
        if ptile == 75: continue
        data = np.array(data) * 2048 / 100.0
        print(data)
        label = '{0} %ile'.format(ptile)
        ax2.plot(data_x, data, 's-.', color=cm(cm_ptile[ptile] + 1))

    runtime_yticks = [300, 600, 900, 1200, 1500, 1800]
    ax.set_yticks(runtime_yticks)
    ax.set_yticklabels(['{:.0f}'.format(i / 60) for i in runtime_yticks])
    ax2.set_xticks(data_x)
    # ax2.yaxis.set_major_formatter('{x:.0f}X')
    intvls = ['62.5K', '125K', '250K', '500K', '750K', '1M', 'once_ts']
    ax.set_xticklabels(intvls, fontsize=base_fontsz - 4)
    ax2.set_xticklabels(intvls, fontsize=base_fontsz - 4)
    ax2.tick_params(axis='y', labelsize=base_fontsz - 4)
    ax.tick_params(axis='y', labelsize=base_fontsz - 4)
    axx.tick_params(axis='y', labelsize=base_fontsz - 4)

    ax.set_ylabel('Runtime (mins)', fontsize=base_fontsz - 2)
    axx.set_ylabel('Load Stddev', fontsize=base_fontsz - 2)
    ax.set_ylim([0, ax.get_ylim()[1] + 100])
    axx.set_ylim([-5, 100])
    axx.yaxis.set_major_locator(MultipleLocator(25))
    ax.set_xlabel('Renegotiation Interval', fontsize=base_fontsz - 2)

    ax2.set_ylabel('RAF', fontsize=base_fontsz - 2)
    ax2.set_xlabel('Renegotiation Interval', fontsize=base_fontsz - 2)
    ax2.set_yscale('log')
    ax2_yticks = [1, 2, 4, 8, 16]
    ax2.set_yticks(ax2_yticks)

    def tickfmt(x):
        if x < 4:
            return '{:.2f}X'.format(x / 4)
        else:
            return '{:.0f}X'.format(x / 4)

    ax2.set_yticklabels([tickfmt(x) for x in ax2_yticks])
    ax2.set_ylim([1, 18])
    # ax2.yaxis.set_major_locator(MultipleLocator(4))
    # ax2.yaxis.set_minor_locator(MultipleLocator(2))
    # ax2.yaxis.set_major_formatter('{x:.0f}')
    ax2.minorticks_off()

    for ax in axes:
        ax.xaxis.grid(True, color='#bbb')
        ax.yaxis.grid(True, color='#bbb')
    # ax2.yaxis.grid(True, color='#ddd', which='minor')

    ax2.legend(handles=legend_items, ncol=2, fontsize=base_fontsz - 4,
               loc="lower left",
               bbox_to_anchor=(0.0, -0.10), framealpha=0.6)
    # ax.legend()

    # ax.set_title(
    #     'Renegotiation Interval vs Runtime/Partition Quality (w/o OOB)')
    fig.tight_layout()

    if save:
        fig.savefig(dir + '/intvl.v2.pdf', dpi=600)
    else:
        fig.show()


def run(eval_dir):
    # plot_runtime_alt_2(eval_dir, False)
    plot_query_latvssel_unified()
    # plot_query_ycsb()
    # plot_subpart_perf_abs(eval_dir, True)
    # plot_intvl_runtime_2(eval_dir, True)
    # plot_rtp_lat(eval_dir, True)


if __name__ == '__main__':
    eval_dir = '/Users/schwifty/Repos/carp/carp-paper/figures/eval'
    plot_init()
    run(eval_dir)
