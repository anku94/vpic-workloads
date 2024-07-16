import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from typing import List, Tuple

from matplotlib import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from plot_util import get_olaps, get_predictive_power


def plot_runtime(dir: str) -> None:
    data_x = ['fio', 'deltafs', 'CARP', 'Post-Processing/Sorting']
    data_y = [707, 750, 808, 707]
    data_yp = [0, 0, 0, data_y[3] * 4]
    fig, ax = plt.subplots(1, 1)
    bars = ax.bar(data_x, data_y)
    bars[1].set_color('grey')
    pbars = ax.bar(data_x, data_yp, bottom=data_y)
    pbars[3].set_color('grey')
    ax.set_title('Time taken to write 1TB data')
    ax.set_xlabel('I/O driver')
    ax.set_ylabel('Time (seconds)')
    fig.show()
    # fig.savefig(dir + '/runtime.comp.pdf', dpi=300)


def plot_runtime_alt(dir: str) -> None:
    x = ['250K', '500K', '750K', '1M', 'ONCE_EPOCH', 'DELTAFS']
    ymin = np.array([1408, 1419, 1599, 1423, 1805, 2048])
    ymax = np.array([1408, 1419, 1599, 1423, 1970, 2048])
    y = (ymin + ymax) / 2
    errmin = y - ymin
    errmax = ymax - y
    fig, ax = plt.subplots(1, 1)
    eb = plt.errorbar(x, y, yerr=[errmin, errmax], fmt='.')
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_title('CARP vs DeltaFS - Runtime')
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Run Type')
    # fig.show()
    fig.savefig(dir + '/carp_vs_deltafs.pdf', dpi=300)
    pass


def plot_runtime_alt_2(dir: str, save: bool = False) -> None:
    x = ['CARP', 'DELTAFS', 'NOSHUF', 'SORT', 'NOSHUF/SORT']
    ymin = np.array([1428, 1228, 1276, 3360, 1276 + 3360])
    ymax = np.array([1630, 1295, 1279, 3360, 1279 + 3360])

    # x = ["NoShuffle", "DeltaFS", "NoShuffle \n+ TritonSort", "CARP"]
    x = ["DirectWrite", "DeltaFS", "TritonSort", "CARP"]
    ymin = [1276, 1228, 1276, 1518]
    ymax = [1279, 1295, 1279, 1604]

    carp_datapoints = [1518, 1576, 1604]
    carp_mean = np.mean(carp_datapoints)

    ytop = [0, 0, 3360, 0]

    ymin = np.array(ymin)
    ymax = np.array(ymax)

    y = (ymin + ymax) / 2
    y[3] = carp_mean

    errmin = y - ymin
    errmax = ymax - y

    cmap = plt.get_cmap('Pastel2')
    print(cmap(0))

    fig, ax = plt.subplots(1, 1)
    # eb = plt.errorbar(x, y, yerr=[errmin, errmax], fmt='.')
    bars = ax.bar(x, y, width=0.5, yerr=[errmin, errmax], capsize=10,
                  color=cmap(0),
                  ec='black')
    bars[2].set_hatch('/')
    bars = ax.bar(x, ytop, width=0.5, yerr=[errmin, errmax], bottom=y,
                  color=cmap(0),
                  ec='black')

    ax.annotate('VPIC', xy=(1.5, 300), rotation=90, fontsize=18)
    ax.annotate('PostProcessing', xy=(1.5, 1800), rotation=90, fontsize=18)

    ax.yaxis.grid(True, color='#bbb')
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.set_title('CARP vs Everything - Runtime')
    base_fontsz = 20
    ax.set_ylabel('Time To Complete (seconds)', fontsize=base_fontsz)
    ax.set_xlabel('Run Type', fontsize=base_fontsz)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(base_fontsz - 2)
    fig.tight_layout()

    if save:
        fig.savefig(dir + '/runtime.pdf', dpi=300)
    else:
        fig.show()


def plot_olap(dir: str, ptile: int) -> None:
    fig, ax = plt.subplots(1, 1)

    # in_dir = "/Users/schwifty/Repos/workloads/rundata/olap_aug31"
    in_dir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2'
    fname = '/olaps.{0}.csv'.format(ptile)

    cols = ['intvl', 'ridx', 'epoch', 'olap']
    data = pd.read_csv(in_dir + fname, names=cols, header=None)
    data = data.groupby(['intvl', 'epoch'], as_index=False).agg({
        'olap': ['mean']
    })
    data.columns = list(map(''.join, data.columns.values))
    data = data.astype({'intvl': 'str'})
    print(data)
    intvls = ['62500', '125000', '250000', '500000', '750000', '1000000',
              'everyepoch']
    # IPython.embed()
    for intvl_idx, intvl in enumerate(intvls):
        print(intvl)
        y_intvl = (data[data['intvl'] == intvl]['olapmean'])
        x_intvl = range(len(y_intvl))
        if len(y_intvl) == 0: continue
        intvl_color = plt.cm.get_cmap('tab10')(intvl_idx)
        ax.plot(x_intvl, y_intvl, label=intvl, color=intvl_color)

    epoch_labels = ['T.200', 'T.1400', 'T.2600', 'T.3800', 'T.5000', 'T.6200',
                    'T.7400', 'T.8600', 'T.9800', 'T.11000', 'T.12200',
                    'T.14600']
    ax.set_xticks(range(12))
    ax.set_xticklabels(epoch_labels, rotation=30)

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Derived Partition Size (% of Total Data)')
    ax.set_title('Renegotiation Interval vs DPS ({0} %ile)'.format(ptile))
    fname = fname.replace('csv', 'pdf')
    print(fname)
    fig.savefig(dir + fname, dpi=300)
    # fig.show()


def plot_intvl_runtime(dir: str) -> None:
    fig, ax = plt.subplots(1, 1)

    in_dir = "/Users/schwifty/Repos/workloads/rundata/olap_aug31/runs.big"
    runtimes = {
        '62500': [1, 3],
        '125000': [4, 6],
        '250000': [4, 6],
        '500000': [4, 6],
        '750000': [1, 3],
        '1000000': [1, 3],
        'everyepoch': [1, 3]
    }

    all_x = []
    all_y = []
    for intvl in runtimes:
        rseqs = runtimes[intvl]
        seqs = range(rseqs[0], rseqs[1] + 1)
        for i in seqs:
            fpath = '/runtime.%s.%s' % (intvl, i)
            with open(in_dir + fpath, 'r') as f:
                rt = int(f.read())
                print(intvl, i, rt)
                all_x.append(intvl)
                all_y.append(rt)

    ax.plot(all_x, all_y, 'o')
    # ax.legend()
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlabel('Renegotiation Interval')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Impact of Reneg Interval on runtime')
    fig.savefig(dir + '/runtime.intvl.pdf', dpi=300)
    # fig.show()
    pass


def gen_intvl_runtime_2(intvls: List, csv_path: str) -> None:
    data_olaps = []
    for intvl in intvls:
        data_intvl = get_olaps(intvl, 5)
        data_olaps.append(data_intvl)

    epochcnt = data_olaps[0].shape[1]
    all_dfs = []

    for intvl, intvl_data in zip(intvls, data_olaps):
        intvl_col = [intvl] * (3 * epochcnt)
        ptile = [50] * epochcnt + [75] * epochcnt + [99] * epochcnt
        eidx = list(range(epochcnt)) * 3
        intvl_data = np.concatenate([*intvl_data])
        df = pd.DataFrame({
            'intvl': intvl_col,
            'ptile': ptile,
            'epochidx': eidx,
            'olappctmean': intvl_data
        })
        # print(df)
        all_dfs.append(df)

    df = pd.concat(all_dfs)
    df.to_csv(csv_path, index=None)


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
    ax2.yaxis.set_major_formatter('{x:.0f}X')
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
    ax2.set_ylim([-1, 12])
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(2))
    # ax2.yaxis.set_major_formatter('{x:.0f}')

    for ax in axes:
        ax.xaxis.grid(True, color='#bbb')
        ax.yaxis.grid(True, color='#bbb')
    ax2.yaxis.grid(True, color='#ddd', which='minor')

    ax2.legend(handles=legend_items, ncol=2, fontsize=base_fontsz - 4,
               loc="lower left",
               bbox_to_anchor=(0.0, -0.10), framealpha=0.6)
    # ax.legend()

    # ax.set_title(
    #     'Renegotiation Interval vs Runtime/Partition Quality (w/o OOB)')
    fig.tight_layout()

    if save:
        fig.savefig(dir + '/intvl.pdf', dpi=600)
    else:
        fig.show()


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


def plot_intvl_std(dir: str) -> None:
    data_x = [62500, 125000, 250000, 500000, 750000, 1000000, 'everyepoch']
    data_x = [str(i) for i in data_x]
    data_y = gen_intvl_std(data_x)

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_y)
    ax.set_xlabel('Renegotiation Interval')
    ax.set_ylabel('Normalized Load Stddev (%)')
    ax.set_title('Reneg Interval vs Load Std')

    ax.set_ylim([0, 100])

    fig.show()
    # fig.savefig(dir + '/intvl.std.pdf', dpi=300)


def plot_query_latency(dir: str) -> None:
    data = [10, 10, 10, 10]
    data_x = ['manifest_read', 'sst_read', 'sst_process', 'total']
    data_carp = [700, 800, 300, 0]
    data_sort = [700, 250, 0, 0]
    ticks_x = np.arange(len(data_x))
    width = 0.35
    fig, ax = plt.subplots(1, 1)
    ax.bar(ticks_x - width / 2, data_carp, width, label='CARP')
    ax.bar(ticks_x + width / 2, data_sort, width, label='Sorting')

    cm = plt.cm.get_cmap('tab20c')
    print(cm.colors)

    data_total = [0, 0, 0, 0]
    data_prev = [i for i in data_carp]
    for idx, data in enumerate(data_carp):
        data_total[3] = data
        print(data_total)
        ax.bar(ticks_x - width / 2, data_total, width, bottom=data_prev,
               color=cm.colors[idx])
        data_prev[3] += data

    data_total = [0, 0, 0, 0]
    data_prev = [i for i in data_sort]
    for idx, data in enumerate(data_sort):
        data_total[3] = data
        print(data_total)
        ax.bar(ticks_x + width / 2, data_total, width, bottom=data_prev,
               color=cm.colors[4 + idx])
        data_prev[3] += data

    ax.set_xticks(ticks_x)
    ax.set_xticklabels(data_x)
    ax.set_ylim([0, 3000])
    ax.set_xlabel('Query Stage')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('CARP vs Sorting: Query Latency')
    ax.legend()
    fig.show()
    # fig.savefig(dir + '/latency.pdf')


def plot_query_latvssel(dir: str, save: bool = False):
    csv_path = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/querylog_curated_pread.csv'
    csv_path = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform/querylog.csv'
    csv_scan = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform/querylog.scan.csv'

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

    df_carp = data_aggr[data_aggr['plfspath'] == type_carp]
    df_flat = data_aggr[data_aggr['plfspath'] == type_flat]

    df_carp = df_carp.sort_values('qkeyselectivity')
    df_flat = df_flat.sort_values('qkeyselectivity')

    drop_idx = [1, 2, 4, 6, 7, 8, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24]
    df_carp = df_carp.drop([df_carp.index[x] for x in drop_idx])
    df_flat = df_flat.drop([df_flat.index[x] for x in drop_idx])

    print(df_carp[['qbegin', 'qend']])

    fig, axes = plt.subplots(2, 1, sharex=True,
                             gridspec_kw={'height_ratios': [1, 3]})

    cm = plt.cm.get_cmap('Set2')
    labels = []
    ax = axes[1]
    print(df_carp['qreadus'])
    print(df_flat['qreadus'])
    for type, df in enumerate([df_carp, df_flat]):
        rowidx = 0
        for index, row in df.iterrows():
            # if rowidx in [1, 2, 4, 6]:
            #     rowidx += 1
            #     continue

            data_x = row['qkeyselectivity']
            data_y = row['qreadus']['mean']
            if (rowidx < 3): print('---->', data_y)
            data_err = row['qreadus']['std']

            marker = 'o' if type == 0 else 's'
            color = cm.colors[rowidx % 8]

            ax.plot(data_x, data_y, marker=marker, mec='black', mfc=color,
                    markersize=14)
            ax.errorbar(data_x, data_y, yerr=data_err, color=color)

            rowidx += 1

    df_scan = pd.read_csv(csv_scan)
    df_scan['qreadus'] += df_scan['qsortus']
    df_scan['qreadus'] /= 1e6
    df_scan['qkeyselectivity'] *= 100
    print(df_scan)

    df_scan.sort_values(by='qbegin', inplace=True)
    df_carp.sort_values(by='qbegin', inplace=True)
    df_scan['qkeyselectivity'] = list(df_carp['qkeyselectivity'])

    for rowidx, row in df_scan.iterrows():
        data_x = row['qkeyselectivity']
        data_y = row['qreadus']
        print(data_x, data_y)

        marker = '^'
        color = cm.colors[rowidx % 8]

        axes[0].plot(data_x, data_y, marker=marker, mec='black', mfc=color,
                     markersize=14)
        pass

    legend_items = []
    # num_rows
    # for i in range(6):
    #     item = plt.Line2D([0], [0], color=cm.colors[i],
    #                       label='Query {0}'.format(i + 1))
    #     legend_items.append(item)

    legend_items.append(
        plt.Line2D([0], [0], marker='^', label='FullScan', markersize=12))
    legend_items.append(
        plt.Line2D([0], [0], marker='s', label='TritonSort', markersize=12))
    legend_items.append(
        plt.Line2D([0], [0], marker='o', label='CarpDB', markersize=12))

    base_fontsz = 20
    ax.set_xlabel('Query Selectivity', fontsize=base_fontsz)
    fig.supylabel('Query Latency (seconds)', fontsize=base_fontsz, x=0.03,
                  y=0.55)
    axes[0].spines.bottom.set_visible(False)
    axes[0].xaxis.tick_top()
    axes[1].spines.top.set_visible(False)

    # ax.set_title('CarpDB vs TritonSort - TEMP', color='red')
    axes[1].legend(handles=legend_items, fontsize=18)
    # ax.set_xticks(np.arange(0, 2, 0.5))
    ax.minorticks_off()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter('{x:.1f}%')
    # ax.set_yticklabels()

    axes[0].set_ylim([330, 350])

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)

    for ax in axes:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(base_fontsz - 1)

        ax.xaxis.grid(True, color='#bbb', which='major')
        ax.xaxis.grid(True, color='#ddd', which='minor')
        ax.yaxis.grid(True, color='#bbb', which='major')
        # ax.yaxis.grid(True, color='#ddd', which='minor')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    if save:
        fig.savefig(dir + '/qlatvssel.v2.pdf', dpi=600)
    else:
        fig.show()


def plot_query_selectivity(dir: str) -> None:
    data_read_pct = np.arange(0, 1, 0.01)
    data_read_gb = data_read_pct * 350 / 100
    print(data_read_pct)
    carp_bumps = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    sort_bumps = np.arange(0, 1, 0.05)

    carp_y = []
    sort_y = []

    for data_pct in data_read_pct:
        len_carp_y = len(carp_y)
        for bump in carp_bumps:
            if bump > data_pct:
                carp_y.append(bump)
                break
        if len(carp_y) == len_carp_y:
            carp_y.append(data_pct)

        len_sort_y = len(sort_y)
        for bump in sort_bumps:
            if bump > data_pct:
                sort_y.append(bump)
                break

        if len(sort_y) == len_sort_y:
            sort_y.append(data_pct)

    # carp_y = np.array(carp_y) * 4000
    # sort_y = np.array(sort_y) * 3000
    # carp_y += 800
    # sort_y += 800

    print(carp_y)
    print(sort_y)

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_read_pct, carp_y, label='CARP')
    ax.plot(data_read_pct, sort_y, label='Sorting')
    ax.set_xlabel('Query Selectivity')
    ax.set_ylabel('Query Latency (s)')
    ax.set_title('Query Selectivity vs Latency')
    fig.show()
    # fig.savefig(dir + '/selectivity_vs_latency.pdf')


def plot_query_selectivity_obs(dir: str) -> None:
    data_read_pct = np.arange(0, 1, 0.01)
    data_read_gb = data_read_pct * 350 / 100
    print(data_read_pct)
    carp_bumps = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    sort_bumps = np.arange(0, 1, 0.05)

    carp_y = []
    sort_y = []

    for data_pct in data_read_pct:
        len_carp_y = len(carp_y)
        for bump in carp_bumps:
            if bump > data_pct:
                carp_y.append(bump)
                break
        if len(carp_y) == len_carp_y:
            carp_y.append(data_pct)

        len_sort_y = len(sort_y)
        for bump in sort_bumps:
            if bump > data_pct:
                sort_y.append(bump)
                break

        if len(sort_y) == len_sort_y:
            sort_y.append(data_pct)

    for idx, y in enumerate(carp_y):
        if y > 0.7:
            ydel = y - 0.7
            ydel = ydel * ydel
            carp_y[idx] = 0.7 + ydel

    carp_y = np.array(carp_y) * 11
    sort_y = np.array(sort_y) * 10

    carp_y += .350
    sort_y += .150

    print(carp_y)
    print(sort_y)

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_read_pct, carp_y, label='CARP')
    ax.plot(data_read_pct, sort_y, label='Sorting')
    ax.set_xlabel('Query Selectivity')
    ax.set_ylabel('Query Latency (s)')
    ax.set_title('Query Selectivity vs Latency')
    fig.show()
    # fig.savefig(dir + '/selectivity_vs_latency_obs.pdf')
    # fig.savefig(dir + '/selectivity_vs_amplification.pdf')


def plot_carp_olap(dir: str) -> None:
    data_x = ['T.' + str(x) for x in np.arange(1, 6) * 2400]
    data_y = [0.41, 0.46, 0.41, 0.38, 0.35]
    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_y)
    ax.plot(data_x, [100 / 512.0] * len(data_x), 'r--')
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_ylabel('Max Overlap (%)')
    ax.set_xlabel('VPIC Timestep')
    ax.set_title('Partitioning Quality vs Epoch')
    fig.show()
    # fig.savefig(dir + '/quality.pdf')


def plot_stat_trigger(dir: str) -> None:
    stat_path = '/Users/schwifty/Repos/workloads/rundata/carp_stat_trigger_apr26'
    stat_inf = pd.read_csv(stat_path + '/stat.xinf.csv', header=None)
    stat_x4 = pd.read_csv(stat_path + '/stat.x4.0.csv', header=None)

    XOLD = 4.0

    fig, ax = plt.subplots(1, 1)

    data_xinf = stat_inf[0]
    data_x4 = stat_x4[0]
    markers_x4 = stat_x4[stat_x4[0] > XOLD]

    data_x = stat_x4.index.values

    ax.plot(data_x, data_x4, label='Threshold: 4.0')
    ax.plot(data_x, data_xinf, label='Threshold: Inf')
    ax.plot(markers_x4.index.values, markers_x4, 'o')
    ax.plot([0, data_x[-1]], [XOLD, XOLD], '--')

    num_epochs = 5
    num_points = len(data_x)
    epoch_length = num_points / num_epochs
    data_min = 0
    data_max = max(max(data_x4), max(data_xinf))
    for ep_start in np.arange(0, num_points, epoch_length):
        ax.fill_between([ep_start, ep_start + epoch_length], data_min, data_max,
                        alpha=0.2)

    ax.legend()
    ax.set_title('StatTrigger Invocations vs Trigger Value')
    ax.set_xlabel('Experiment Progress')
    ax.set_ylabel('Trigger Value')

    # fig.show()
    fig.savefig(dir + '/stat_trigger.pdf', dpi=300)
    pass


def plot_predictive_power(dir: str, save: bool = False) -> None:
    fig, ax = plt.subplots(1, 1)

    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2'
    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform'
    oracle_csv = basedir + '/oracle.predictive.olap.csv'

    # trace_dir = '/Users/schwifty/Repos/workloads/data/eval_trace'
    # trace_dir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform/hist_trace_uniform'
    # epochs, oracle_olap = get_predictive_power(trace_dir)
    # oracle_df = pd.DataFrame(zip(epochs, oracle_olap), columns=['timestep', 'oracolap'])
    # oracle_df.to_csv(oracle_csv, index=None)
    #
    # return

    oracle_df = pd.read_csv(oracle_csv)
    epochs = oracle_df['timestep']
    y_orac = oracle_df['oracolap']
    print(y_orac)

    # ax.plot(epochs, oracle_olap)

    actual_csv = basedir + '/subpart.250k.olap.final.3runs.csv'
    actual_df = pd.read_csv(actual_csv)
    actual_df = actual_df.groupby(['epochidx', 'timestep'], as_index=False).agg(
        {
            'olappct': ['mean', 'std'],
        })
    actual_df.columns = list(map(''.join, actual_df.columns.values))
    print(actual_df)

    y_act = actual_df['olappctmean']

    print(epochs)

    norm_fact = 0.18 / 4

    y_orac /= norm_fact
    y_act /= norm_fact

    x_ticks = range(len(epochs))
    ax.plot(x_ticks, y_orac, 'o-', label='Static', markersize=10,
            linewidth=2)
    ax.plot(x_ticks, y_act, 's-', label='CARP (Dynamic)', markersize=10,
            linewidth=2)

    base_fontsz = 20
    ax.set_xlabel('Simulation Timestep', fontsize=base_fontsz)
    ax.set_ylabel('Max Read Amplification',
                  fontsize=base_fontsz)
    ax.set_yscale('log')

    ticks = [1, 3, 10, 30, 150, 750]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) + '%' for t in ticks], fontsize=base_fontsz - 1)
    ax.set_yticklabels([str(t) + 'X' for t in ticks], fontsize=base_fontsz - 1)

    ax.yaxis.grid(True, color='#bbb')
    ax.xaxis.grid(True, color='#bbb')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(epochs, rotation=30, fontsize=base_fontsz - 4)

    ax.minorticks_off()
    ax.legend(fontsize=base_fontsz - 2)

    fig.tight_layout()
    if save:
        fig.savefig(dir + '/olapcomp.v2.pdf', dpi=600)
    else:
        fig.show()


def plot_subpart_perf_abs(dir: str, save: bool = False) -> None:
    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/big.2'
    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform'
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

    print(ysub0x50 / ysub1x50)
    print(ysub0x99 / ysub1x99)

    return

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
    ax.plot(x_ticks, ysub0x50, 'o--', label='50%ile, w/o repart.', color=cm(1))
    ax.plot(x_ticks, ysub1x50, 'o-', label='50%ile, with repart.', color=cm(0))
    ax.plot(x_ticks, ysub0x99, 's--', label='99%ile, w/o repart.', color=cm(5))
    ax.plot(x_ticks, ysub1x99, 's-', label='99%ile, with repart.', color=cm(4))
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
    ax.plot(x_ticks, ysub1x50, '^--', label='50%ile, no subpart.', color=cm(0), alpha=0.6)
    ax.plot(x_ticks, ysub2x50, 'o--', label='50%ile, 2X subpart', color=cm(0), alpha=0.8)
    ax.plot(x_ticks, ysub4x50, 's-', label='50%ile, 4X subpart', color=cm(0))
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
        ax.set_ylabel('RAF', fontsize=base_fontsz - 1)
        ax.set_xlabel('Simulation Timestep', fontsize=base_fontsz - 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, rotation=30, fontsize=base_fontsz - 4)
        ax.xaxis.grid(True, color='#bbb')
        ax.yaxis.grid(True, color='#bbb')

    axes[0].legend(ncol=2, fontsize=base_fontsz - 4, loc="lower left",
                   bbox_to_anchor=(0.0, 0.722))
    axes[0].set_yscale('log')
    # axes[0].yaxis.set_major_locator(MultipleLocator(2))
    # yticks0 = [0.1, 0.4, 1.6, 6.4, 25.6, 100]
    # yticks0 = [0.1, 0.5, 2.5, 10, 40]
    yticks0 = [1, 4, 16, 64, 256]
    yticks0 = [4, 16, 64, 256, 1024]
    axes[0].set_yticks(yticks0)
    axes[0].yaxis.set_major_formatter('{x:.0f}X')
    axes[0].minorticks_off()
    axes[0].set_ylim([0.5, 512])
    axes[0].set_ylim([2, 2048])

    axes[1].legend(ncol=2, fontsize=base_fontsz - 4, loc="lower left",
                   bbox_to_anchor=(0.0, 0.62))
    axes[1].set_ylim([0.5, 16])
    axes[1].set_yscale('log')
    axes[1].minorticks_off()
    # axes[1].yaxis.set_major_locator(MultipleLocator(2))
    # axes[1].yaxis.set_minor_locator(MultipleLocator(1))
    yticks1 = [1, 2, 4, 8]
    axes[1].set_yticks(yticks1)
    # axes[1].yaxis.grid(True, color='#ddd', which='minor')
    axes[1].yaxis.set_major_formatter('{x:.0f}X')

    fig.tight_layout()
    if save:
        fig.savefig(dir + '/carpdb.impact.alt2.pdf', dpi=600)
    else:
        fig.show()


def plot_subpart_perf(dir: str, save: bool = False) -> None:
    csv50 = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/subpart_exps/olaps.50.csv'
    csv99 = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/subpart_exps/olaps.99.csv'

    cols = ['intvl', 'runidx', 'epochidx', 'olappct']
    df50 = pd.read_csv(csv50, header=None, names=cols)
    df99 = pd.read_csv(csv99, header=None, names=cols)

    def aggr(df, runs):
        df = df[(df['runidx'] >= runs[0]) & (df['runidx'] <= runs[1])]
        df = df.groupby('epochidx', as_index=None).agg({
            'olappct': ['mean', 'std']
        })
        df.columns = list(map(''.join, df.columns.values))
        return df['olappctmean']

    ysub4x50 = aggr(df50, [13, 15])
    ysub2x50 = aggr(df50, [10, 12])
    ysub1x50 = aggr(df50, [4, 6])
    ysub0x50 = aggr(df50, [7, 9])

    print(ysub2x50)

    ysub4x99 = aggr(df99, [13, 15])
    ysub2x99 = aggr(df99, [10, 12])
    ysub1x99 = aggr(df99, [4, 6])
    ysub0x99 = aggr(df99, [7, 9])

    x_ticks = range(12)
    x_ticklabels = "200,1400,2600,3800,5000,6200,7400,8600,9800,12200,15800,19400"
    x_ticklabels = x_ticklabels.split(',')

    fig, axes = plt.subplots(2, 1, sharex=True)

    # ax.plot(x_ticks, y_sub4)
    # ax.plot(x_ticks, y_sub1)
    cm = plt.cm.get_cmap('tab20c')

    ax = axes[0]
    ax.plot(x_ticks, ysub0x50 / ysub1x50, 'o-',
            label='Repartitioning (50 %ile)',
            color=cm(0))
    ax.plot(x_ticks, ysub0x99 / ysub1x99, 's--',
            label='Repartitioning (99 %ile)',
            color=cm(1))

    ax = axes[1]
    ax.plot(x_ticks, ysub1x50 / ysub2x50, 'o-', label='2x Subpart. (50 %ile)',
            color=cm(4))
    ax.plot(x_ticks, ysub1x50 / ysub4x50, 'o-', label='4x Subpart. (50 %ile)',
            color=cm(8))

    ax.plot(x_ticks, ysub1x99 / ysub2x99, 's--', label='2x Subpart. (99 %ile)',
            color=cm(5))
    ax.plot(x_ticks, ysub1x99 / ysub4x99, 's--', label='4x Subpart., (99 %ile)',
            color=cm(9))

    ax = axes[0]

    base_fontsz = 16
    # ax.set_ylabel('Multiplier Gains in Max DPS', fontsize=base_fontsz)
    fig.supylabel('Multiplier Gains in DPS', x=0.03, y=0.57,
                  fontsize=base_fontsz)

    y1_ticks = [0, 10, 20, 30, 40, 50, 60]
    y2_ticks = [0, 1, 2, 3, 4]
    axes[0].set_yticks(y1_ticks)
    axes[0].set_yticklabels([str(i) + 'X' for i in y1_ticks],
                            fontsize=base_fontsz - 4)
    axes[1].set_yticks(y2_ticks)
    axes[1].set_yticklabels([str(i) + 'X' for i in y2_ticks],
                            fontsize=base_fontsz - 4)

    # ax.legend(fontsize=base_fontsz - 4)

    for ax in axes:
        ax.legend(ncol=2, fontsize=base_fontsz - 4, loc="lower left",
                  bbox_to_anchor=(0.0, -0.07))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, rotation=30, fontsize=base_fontsz - 4)
        ax.xaxis.grid(True, color='#bbb')
        ax.yaxis.grid(True, color='#bbb')

    axes[0].legend(ncol=1, fontsize=base_fontsz - 4, loc="lower left",
                   bbox_to_anchor=(0.5, -0.07))
    axes[1].set_xlabel('Timestep', fontsize=base_fontsz)

    fig.tight_layout()
    if save:
        fig.savefig(dir + '/carpdb.impact.alt.pdf', dpi=600)
    else:
        fig.show()


def plot_query_ycsb(dir: str, save: bool = False) -> None:
    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/YCSB.eval'
    basedir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform/YCSB'
    widths = [5, 20, 50, 100]
    tags = ['carp', 'comp']
    qlog_fmt = '{0}/querylog.{1}.{2}.csv'
    base_fontsz = 12

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

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes = [*axes[0], *axes[1]]
    print(axes)
    epochs = [0, 4, 7, 11]
    epoch_labels = "200,2600,6200,9800,19400".split(',')
    epoch_labels = "200,7400,14600,19400".split(',')
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
               color=cmap(0), edgecolor='#000')
        ax.bar(x - width / 2, orig_df['qsortus'], width,
               bottom=orig_df['qreadus'], label='CARP/Sort', color=cmap(1),
               edgecolor='#000')
        ax.bar(x + width / 2, merged_df['qreadus'], width,
               label='TritonSort/Read', color=cmap(2), edgecolor='#000')
        ax.yaxis.grid(True, color='#bbb')
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in widths], fontsize=base_fontsz - 1)
        yticklabels = [0, 300, 600, 900, 1200]
        ax.set_yticks(yticklabels)
        ax.set_yticklabels(yticklabels, fontsize=base_fontsz - 1)
        ax.set_title('Epoch {0}'.format(epoch_labels[epoch]),
                     fontsize=base_fontsz)

    # axes[0].legend(loc="upper right", ncol=3, bbox_to_anchor=(1, 2))
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, ncol=3, bbox_to_anchor=(0.94, 0.93),
               fontsize=base_fontsz - 1, framealpha=0.8)

    fig.supxlabel('Query Width (#SSTs)', x=0.55, y=0.04,
                  fontsize=base_fontsz + 1)
    fig.supylabel('Total Time Taken (seconds)', x=0.03, y=0.55,
                  fontsize=base_fontsz + 1)
    fig.tight_layout()
    if save:
        fig.savefig(dir + '/query.ycsb.v2.pdf', dpi=600)
    else:
        fig.show()


def plot_subpart_query(dir: str, save: bool = False) -> None:
    data_path = "/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform/YCSB.eval.subpart"
    all_subs = {}
    for sub in [1, 2, 4]:
        df = pd.read_csv(
            '{0}/querylog.sub{1}.carp.5.csv'.format(data_path, sub))
        df = df.groupby('epoch', as_index=None).agg({
            'qreadus': ['sum']
        })
        df.columns = list(map(''.join, df.columns.values))
        print(df)
        all_subs[sub] = df['qreadussum']

    fig, axes = plt.subplots(2, 1)
    base_fontsz = 16
    ax = axes[0]
    x_ticklabels = "200,2000,3800,5600,7400,9200,11000,12800,14600,16400,18200,19400".split(
        ',')
    xticks = range(len(x_ticklabels))

    labels = {
        1: 'No Subpart',
        2: '2X Subpart',
        4: '4X Subpart'
    }
    cm = plt.cm.get_cmap('tab20c')
    ax.plot(xticks, all_subs[1] / 1e6, '^--', label='No subpart.', color=cm(0), alpha=0.6)
    ax.plot(xticks, all_subs[2] / 1e6, 'o--', label='2X subpart.', color=cm(0), alpha=0.8)
    ax.plot(xticks, all_subs[4] / 1e6, 's-', label='4X subpart.', color=cm(0))

    p = np.array(all_subs[1])
    q = np.array(all_subs[2])
    r = np.array(all_subs[4])

    print(max(q/p), min(q/p))
    print(max(r/p), min(r/p))

    ax.set_xticks(xticks)
    ax.set_xticklabels(x_ticklabels, rotation=30, fontsize=base_fontsz - 4)
    ax.set_ylim([0, 25])

    for a in axes:
        a.set_xlabel('Simulation Timestep', fontsize=base_fontsz - 1)
        a.set_ylabel('Latency (s)', fontsize=base_fontsz - 1)

    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(2.5))
    ax.xaxis.grid(True, color='#bbb')
    ax.yaxis.grid(True, color='#bbb')
    ax.yaxis.grid(True, color='#ddd', which='minor')

    ax.legend(ncol=3, fontsize=base_fontsz - 4)

    fig.tight_layout()
    if save:
        fig.savefig(dir + '/subpart.query.pdf', dpi=600)
    else:
        fig.show()


def eval(dir: str) -> None:
#     # plot_runtime_alt(dir)
#     # plot_olap(dir, 50)
#     # plot_olap(dir, 75)
#     # plot_olap(dir, 99)
#     # plot_intvl_runtime(dir)
#     # plot_intvl_std(dir)
#     # plot_query_latency(dir)
#     # plot_query_selectivity(dir)
#     # plot_query_selectivity_obs(dir)
#     # plot_carp_olap(dir)
#     # plot_stat_trigger(dir)
#     plot_runtime_alt_2(dir, False)
#     # plot_query_latvssel(dir, False)
#     # plot_predictive_power(dir, True)
#     plot_query_ycsb(dir, False)
#     # plot_intvl_runtime_2(dir, True)
#     # plot_subpart_perf(dir, False)
    plot_subpart_perf_abs(dir, False)
    #  plot_subpart_query(dir, False)


if __name__ == '__main__':
    eval_dir = '/Users/schwifty/Repos/carp/carp-paper/figures/eval'
    eval(eval_dir)
