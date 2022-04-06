import numpy as np
import pandas as pd
import sys

import IPython

from sklearn.linear_model import LinearRegression

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr


def plot_sst_multipar_util(ax, par, bar_a, bar_b, bar_width):
    basedir = '/Users/schwifty/Repos/workloads/rundata/query_latency_june1/YCSB'

    read_carp = []
    sort_carp = []
    read_flat = []

    all_widths = [20, 50, 100]

    for width in all_widths:
        df_carp = pd.read_csv(
            basedir + '/querylog.orig.{0}.p{1}.csv'.format(width, par))
        df_flat = pd.read_csv(
            basedir + '/querylog.merged.{0}.p{1}.csv'.format(width, par))

        a = df_carp['qreadus'].sum()
        b = df_carp['qsortus'].sum()
        read_carp.append(a)
        sort_carp.append(b)

        c = df_flat['qreadus'].sum()
        read_flat.append(c)

    read_carp = np.array(read_carp) / 1e6
    sort_carp = np.array(sort_carp) / 1e6
    read_flat = np.array(read_flat) / 1e6

    ticks_x = np.arange(len(read_flat))
    ticklabels_x = all_widths

    ax.bar(ticks_x + bar_a, read_carp, bar_width, label='CarpDB - Read')
    ax.bar(ticks_x + bar_a, sort_carp, bar_width, bottom=read_carp,
           label='CarpDB - Sort')
    ax.bar(ticks_x + bar_b, read_flat, bar_width,
           label='CompactDB - Read')

    # fig.savefig(basedir + '/../ycsb.query.p{0}.pdf'.format((par,)), dpi=600)


def plot_sst_multipar():
    fig, ax = plt.subplots(1, 1)

    all_widths = [20, 50, 100]

    ticks_x = np.arange(len(all_widths))
    ticklabels_x = all_widths
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticklabels_x)

    bar_width = 0.15
    bar_half = bar_width / 2
    plot_sst_multipar_util(ax, 16, -3 * bar_half, -bar_half, bar_width)
    plt.gca().set_prop_cycle(None)
    plot_sst_multipar_util(ax, 64, bar_half, 3 * bar_half, bar_width)

    ax.set_xlabel('Query Width (Perfectly Partitioned SSTs Read)')
    ax.set_ylabel('Total Query Latency (seconds, 1000 queries)')
    ax.set_title('Batch Query Latency for a YCSB-Zipfian Query Workload')
    ax.legend()
    fig.show()
    # fig.savefig(basedir + '/../ycsb.query.p{0}.pdf'.format((par,)), dpi=600)


def plot_sst_bar():
    basedir = '/Users/schwifty/Repos/workloads/rundata/query_latency_june1/YCSB'

    fig, ax = plt.subplots(1, 1)

    read_carp = []
    sort_carp = []
    read_flat = []

    all_widths = [20, 50, 100]
    par = 16

    for width in all_widths:
        df_carp = pd.read_csv(
            basedir + '/querylog.orig.{0}.p{1}.csv'.format(width, par))
        df_flat = pd.read_csv(
            basedir + '/querylog.merged.{0}.p{1}.csv'.format(width, par))

        a = df_carp['qreadus'].sum()
        b = df_carp['qsortus'].sum()
        read_carp.append(a)
        sort_carp.append(b)

        c = df_flat['qreadus'].sum()
        read_flat.append(c)

    read_carp = np.array(read_carp) / 1e6
    sort_carp = np.array(sort_carp) / 1e6
    read_flat = np.array(read_flat) / 1e6

    ticks_x = np.arange(len(read_flat))
    ticklabels_x = all_widths
    bar_width = 0.35

    ax.bar(ticks_x - bar_width / 2, read_carp, bar_width, label='CarpDB - Read')
    ax.bar(ticks_x - bar_width / 2, sort_carp, bar_width, bottom=read_carp,
           label='CarpDB - Sort')
    ax.bar(ticks_x + bar_width / 2, read_flat, bar_width,
           label='CompactDB - Read')

    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticklabels_x)

    ax.set_xlabel('Query Width (Perfectly Partitioned SSTs Read)')
    ax.set_ylabel('Total Query Latency (seconds, 1000 queries)')

    ax.set_title('Batch Query Latency for a YCSB-Zipfian Query Workload')

    ax.legend()

    fig.show()
    # fig.savefig(basedir + '/../ycsb.query.p{0}.pdf'.format((par,)), dpi=600)


def plot_sst_read_distrib():
    all_widths = [20, 50, 100]
    basedir = '/Users/schwifty/Repos/workloads/rundata/query_latency_june1/YCSB'

    for width in all_widths:
        fig, ax = plt.subplots(1, 1)

        df_carp = pd.read_csv(
            basedir + '/querylog.orig.{0}.csv'.format(width, ))
        df_flat = pd.read_csv(
            basedir + '/querylog.merged.{0}.csv'.format(width, ))

        y_carp = df_carp['qreadus'] / 1e6
        y_flat = df_flat['qreadus'] / 1e6

        ax.hist(y_carp, histtype='step', bins=100, label='CarpDB')
        ax.hist(y_flat, histtype='step', bins=100, label='CompactDB')

        ax.legend()

        ax.set_xlabel('Query Latency (seconds)')
        ax.set_ylabel('Query Count')
        ax.set_title(
            'Distribution of SST Read Times (Zipfian Batch Queries, Width={0})'.format(
                width, ))

        # fig.show()
        fig.savefig(basedir + '/../ycsb.distrib.{0}.pdf'.format(width, ),
                    dpi=600)


def plot_sst_read_distrib_adhoc():
    basedir = '/Users/schwifty/Repos/workloads/rundata/query_latency_june1/YCSB'
    width = 100

    fig, ax = plt.subplots(1, 1)

    df_carp = pd.read_csv(
        basedir + '/querylog.orig.{0}.p128.csv'.format(width, ))

    y_carp = df_carp['qreadus'] / 1e6

    ax.hist(y_carp, histtype='step', bins=100, label='CarpDB')

    ax.legend()

    ax.set_xlabel('Query Latency (seconds)')
    ax.set_ylabel('Query Count')
    ax.set_title(
        'Distribution of SST Read Times (Zipfian Batch Queries, Width={0})'.format(
            width, ))

    fig.show()
    # fig.savefig(basedir + '/../ycsb.distrib.{0}.pdf'.format(width, ),
    #             dpi=600)


def plot_sst_read_unified():
    basedir = '/Users/schwifty/Repos/workloads/rundata/query_latency_june1'
    csv = basedir + '/querylog.csv'

    data = pd.read_csv(csv)
    data = data.sort_values('qselectivity')

    all_plfs = sorted(data['plfspath'].unique())
    type_carp = all_plfs[0]
    type_flat = all_plfs[1]

    # data.loc[data['plfspath'] == type_flat, 'qsortus'] = 0
    # data['qreadus'] += data['qsortus']

    data['qreadus'] /= 1e6
    data['qselectivity'] *= 100

    data_aggr = data.groupby(['plfspath', 'qbegin', 'qend', 'qselectivity'],
                             as_index=False).agg({'qreadus': ['mean', 'std']})

    df_carp = data_aggr[data_aggr['plfspath'] == type_carp]
    df_flat = data_aggr[data_aggr['plfspath'] == type_flat]

    df_carp = df_carp.sort_values('qselectivity')
    df_flat = df_flat.sort_values('qselectivity')

    df_carp = df_carp.drop([df_carp.index[x] for x in [1, 2, 4, 6]])
    df_flat = df_flat.drop([df_flat.index[x] for x in [1, 2, 4, 6]])

    fig, ax = plt.subplots(1, 1)

    cm = plt.cm.get_cmap('Dark2')

    labels = []
    for type, df in enumerate([df_carp, df_flat]):
        rowidx = 0
        for index, row in df.iterrows():
            # if rowidx in [1, 2, 4, 6]:
            #     rowidx += 1
            #     continue

            data_x = row['qselectivity']
            data_y = row['qreadus']['mean']
            data_err = row['qreadus']['std']

            marker = 'o' if type == 0 else 's'
            color = cm.colors[rowidx]

            ax.plot(data_x, data_y, marker=marker, mec='black', mfc=color,
                    markersize=10)
            ax.errorbar(data_x, data_y, yerr=data_err, color=color)

            rowidx += 1

    legend_items = []
    # num_rows
    for i in range(4):
        item = plt.Line2D([0], [0], color=cm.colors[i],
                          label='Query {0}'.format(i))
        legend_items.append(item)

    legend_items.append(plt.Line2D([0], [0], marker='o', label='CarpDB'))
    legend_items.append(plt.Line2D([0], [0], marker='s', label='CompactDB'))

    ax.set_xlabel('Data Read for Query (percent)')
    ax.set_ylabel('Query Latency (seconds)')
    ax.set_title('CarpDB vs CompactDB: Index Selectivity and Query Latency')

    ax.legend(handles=legend_items)
    fig.show()
    # fig.savefig(basedir + '/query.latency.wo.sort.pdf', dpi=600)


def plot_sst_read_subpart_worker(queries, ax, cm, legend_items, marker, label,
                                 csv):
    data = pd.read_csv(csv)
    data = data.sort_values('qselectivity')

    data['qreadus'] += data['qsortus']
    data['qreadus'] /= 1e6
    data['qselectivity'] *= 100

    df = data.groupby(['epoch', 'qbegin', 'qend', 'qselectivity'],
                      as_index=False).agg({'qreadus': ['mean', 'std']})

    df = df.sort_values('qselectivity')
    dflen = len(df)

    df = pd.merge(df, queries, on=['epoch', 'qbegin', 'qend'],
                  how='inner')

    rowidx = 0
    for index, row in df.iterrows():
        data_x = row['qselectivity_x']
        data_y = row['qreadus_x']['mean']
        data_err = row['qreadus_x']['std']

        print(rowidx)
        color = cm.colors[rowidx % len(cm.colors)]

        ax.plot(data_x, data_y, marker=marker, mec='black', mfc=color,
                markersize=10)
        ax.errorbar(data_x, data_y, yerr=data_err, color=color)

        rowidx += 1

    # num_rows
    legend_items.append(plt.Line2D([0], [0], marker=marker, label=label))


def plot_sst_read_subpart():
    basedir = '/Users/schwifty/Repos/workloads/rundata/subpart_jun15'
    csv4 = basedir + '/SUBPART/querylog.74574.csv'
    csv2 = basedir + '/SUBPART/querylog.74572.csv'
    csv1 = basedir + '/SUBPART/querylog.74571.csv'

    data = pd.read_csv(csv4)
    data = data.sort_values('qselectivity')

    data['qreadus'] /= 1e6
    data['qselectivity'] *= 100

    data_aggr = data.groupby(
        ['plfspath', 'epoch', 'qbegin', 'qend', 'qselectivity'],
        as_index=False).agg({'qreadus': ['mean', 'std']})

    data_aggr = data_aggr.sort_values('qselectivity').reset_index(drop=True)

    queries = data_aggr.iloc[[0, 5, 11, 18]]
    queries = data_aggr.iloc[[0, 4, 7, 12, 17]]

    fig, ax = plt.subplots(1, 1)
    cm = plt.cm.get_cmap('Dark2')
    legend_items = []

    for i in range(len(queries)):
        item = plt.Line2D([0], [0], color=cm.colors[i],
                          label='Query {0}'.format(i + 1))
        legend_items.append(item)

    plot_sst_read_subpart_worker(queries, ax, cm, legend_items, 'o', 'x1', csv1)
    plot_sst_read_subpart_worker(queries, ax, cm, legend_items, 's', 'x2', csv2)
    plot_sst_read_subpart_worker(queries, ax, cm, legend_items, '>', 'x4', csv4)

    ax.set_xlabel('Data Read for Query (percent)')
    ax.set_ylabel('Query Latency (seconds)')
    ax.set_title('Impact of Subpartitioning on Query Latency')

    ax.legend(handles=legend_items)
    # fig.show()
    fig.savefig(basedir + '/query.latency.subpart.pdf', dpi=300)


def plot_sst_read():
    basedir = '/Users/schwifty/Repos/workloads/rundata/query_latency_may28'
    csv_carp = basedir + '/carpdb.querylog.csv'
    csv_flat = basedir + '/compactdb.querylog.csv'

    df_carp = pd.read_csv(csv_carp)
    df_flat = pd.read_csv(csv_flat)

    x_carp = df_carp['qselectivity'] * 100
    y_carp = (df_carp['qreadus'] + df_carp['qsortus']) / 1e6

    x_flat = df_flat['qselectivity'] * 100
    y_flat = df_flat['qreadus'] / 1e6

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_carp, y_carp, label='CarpDB')
    ax.scatter(x_flat, y_flat, label='CompactDB')

    ax.set_xlabel('Data Read - percent')
    ax.set_ylabel('SST Read Latency - seconds')
    ax.set_title('CarpDB vs CompactDB: Impact of Selectivity')
    ax.legend()
    # fig.show()
    fig.savefig(basedir + '/sst.read.scatter.pdf', dpi=600)

    pass


def plot_mf_bar():
    basedir = '/Users/schwifty/Repos/workloads/rundata/mar18_compact-data'

    labels = ["FULLREAD", "FULLREAD_CACHED", "READ_ONE"]
    carp_ms = [319.91, 336.83, 20]
    flat_ms = [136.57, 111, 24]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, carp_ms, width, label='CarpDB')
    rects2 = ax.bar(x + width / 2, flat_ms, width, label='CompactDB')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (ms)')
    ax.set_title('Manifest Read Latency - CarpDB vs CompactDB')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # plt.show()
    fig.savefig(basedir + '/manifest.read.latency.pdf', dpi=300)


def plot_sample_queries():
    basedir = '/Users/schwifty/Repos/workloads/rundata/mar18_compact-data'
    labels = ['BigQ (1.6%/1.49%)', 'SmallQ (0.02%/0.22%)']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    mfread_carp = [319, 336]
    sstread_carp = [6575, 4746]
    sort_carp = [853, 90]

    mfread_flat = [136, 144]
    sstread_flat = [9822, 190]
    sort_flat = [831, 21]

    mfread_carp = np.array(mfread_carp)
    sstread_carp = np.array(sstread_carp)
    sort_carp = np.array(sort_carp)

    mfread_flat = np.array(mfread_flat)
    sstread_flat = np.array(sstread_flat)
    sort_flat = np.array(sort_flat)

    fig, ax = plt.subplots(1, 1)
    x_carp = x - width / 2 - 0.01
    x_flat = x + width / 2 + 0.01

    bars_1 = plt.bar(x_carp, mfread_carp, width, label='ManifestRead',
                     color=plt.cm.tab10(0))
    bars_2 = plt.bar(x_carp, sstread_carp, width, bottom=mfread_carp,
                     label='SST Read',
                     color=plt.cm.tab10(1))
    bars_3 = plt.bar(x_carp, sort_carp, width,
                     bottom=mfread_carp + sstread_carp,
                     label='Sort', color=plt.cm.tab10(2))

    bars_4 = plt.bar(x_flat, mfread_flat, width, color=plt.cm.tab10(0))
    bars_5 = plt.bar(x_flat, sstread_flat, width, bottom=mfread_flat,
                     color=plt.cm.tab10(1))
    bars_6 = plt.bar(x_flat, sort_flat, width,
                     bottom=mfread_flat + sstread_flat, color=plt.cm.tab10(2))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.set_title('Query Latency - CarpDB vs CompactDB - Breakdown')
    ax.set_ylabel('Time (ms)')

    # plt.show()
    plt.savefig(basedir + '/query.latency.breakdown.pdf', dpi=600)
    pass


def plot_sst_subpart_parcomp():
    basedir = '/Users/schwifty/Repos/workloads/rundata/subpart_jun15'
    path_a = '/SUBPART/rankwise.csv'
    path_b = '/SUBPART/fullpar.csv'

    path_a = '/SUBPART/querylog.orig.p16.csv'
    path_b = '/SUBPART/querylog.merged.subbatch.p32.csv'
    path_c = '/SUBPART/querylog.74571.fullpar.csv'
    path_d = '/SUBPART/querylog.74574.fullpar.p32.csv'
    path_e = '/SUBPART/querylog.74572.fullpar.p16.csv'
    path_f = '/SUBPART/querylog.74574.fullpar.p32.schedoptoff.csv'
    path_g = '/SUBPART/querylog.merged.p16.keyonly.mmapio.csv'
    path_h = '/SUBPART/querylog.merged.p16.keyonly.preadio.csv'
    path_i = '/SUBPART/querylog.74574.fullpar.p16.keyonly.mmapio.csv'
    path_j = '/SUBPART/querylog.74574.fullpar.p16.keyonly.preadio.csv'
    path_k = '/SUBPART/querylog.74574.fullpar.p16.keyonly.preadio.wremount.csv'

    fig, ax = plt.subplots(1, 1)

    labels = ['CarpDB', 'CompactDB', 'CarpDB.SUB4', 'CarpDB.SUB4/SchedOpt',
              'MMAP']
    labels = {
        path_a: 'CarpDB',
        path_b: 'CompactDB',
        path_c: 'CarpDB.SUB1',
        path_d: 'CarpDB.SUB4',
        path_e: 'CarpDB.SUB2',
        path_f: 'CarpDB.SUB4/Rerun',
        path_g: 'CompactDB/KeyOnly/MMAP',
        path_h: 'CompactDB/KeyOnly/pread',
        path_i: 'CarpDB.SUB4/KeyOnly/MMAP',
        path_j: 'CarpDB.SUB4/KeyOnly/pread',
        path_k: 'CarpDB.SUB4/KeyOnly/pread/remount'
    }

    fig_path = '/JUN29/qlat.jun29.fullpar.best.wsort.pdf'
    # fig_path = None

    all_paths = [path_g, path_h, path_i, path_j]
    sort_paths = [path_a, path_c, path_d, path_e, path_f, path_i, path_j,
                  path_k]
    # sort_paths = []

    for csv_idx, csv_path in enumerate(all_paths):
        df = pd.read_csv(basedir + csv_path)
        data_x = df['qselectivity'] * 100
        data_x = data_x.values[:, np.newaxis]
        data_y = df['qreadus'] / 1e6

        if csv_path in sort_paths:
            data_y = (df['qreadus'] + df['qsortus']) / 1e6
        else:
            data_y = df['qreadus'] / 1e6

        data_y = data_y.values
        lr = LinearRegression().fit(data_x, data_y)
        dots = ax.plot(data_x, data_y, 'o', label=labels[csv_path])
        line_x = np.array([0.1, 2.0])
        line_y = lr.predict(line_x[:, np.newaxis])
        ax.plot(line_x, line_y, dots[0].get_color())

    ax.legend()
    ax.set_ylim([0, 8])
    ax.set_ylabel('Latency (seconds)')
    ax.set_xlabel('Data Read (%)')
    if len(sort_paths) == 0:
        ax.set_title('Query Latency: CarpDB vs CompactDB - SST Read Only')
    else:
        ax.set_title('Query Latency: CarpDB vs CompactDB - SST Read + Sort')
    if fig_path:
        fig.savefig(basedir + fig_path, dpi=300)
    else:
        fig.show()

    pass


def plot_sst_query_final():
    fig, ax = plt.subplots(1, 1)

    # fig_path = '/JUN29/qlat.jun29.fullpar.best.wsort.pdf'
    fig_path = None

    # df_path = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/querylog_curated_pread.csv'
    # df_path_2 = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/querylog_everyepoch_curated_pread.csv'
    df_path_cp = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform/querylog.carp.csv'
    df_path_ts = '/Users/schwifty/Repos/workloads/rundata/eval/runs.uniform/querylog.comp.csv'
    # df = pd.read_csv(df_path_3)
    # paths = df['plfspath'].unique()
    # print(paths)
    # df_cp = df[df['plfspath'] == paths[0]]
    # df_ts = df[df['plfspath'] == paths[1]]
    # df_cp_2 = pd.read_csv(df_path_2)
    df_cp = pd.read_csv(df_path_cp)
    df_ts = pd.read_csv(df_path_ts)

    for idx, df in enumerate([df_cp, df_ts]):
        data_x = df['qselectivity'] * 100
        data_x = data_x.values[:, np.newaxis]
        data_y = df['qreadus'] / 1e6

        if idx == 0:
            data_y = (df['qreadus'] + df['qsortus']) / 1e6
        else:
            data_y = df['qreadus'] / 1e6

        labels = ["CARP", "TritonSort"]

        data_y = data_y.values
        lr = LinearRegression().fit(data_x, data_y)
        label = labels[idx]
        dots = ax.plot(data_x, data_y, 'o', label=label)
        line_x = np.array([0.1, 2.0])
        line_y = lr.predict(line_x[:, np.newaxis])
        ax.plot(line_x, line_y, dots[0].get_color())

    ax.legend()
    ax.set_ylim([0, 8])
    ax.set_ylabel('Latency (seconds)')
    ax.set_xlabel('Data Read (%)')
    ax.set_title('Query Latency: CarpDB vs TritonSort - SST Read + Sort')

    if fig_path:
        fig.savefig(fig_path, dpi=300)
    else:
        fig.show()

    pass


if __name__ == '__main__':
    # plot_sst_multipar()
    # plot_sst_bar()
    # plot_sst_read_distrib()
    # plot_sst_read_distrib_adhoc()
    # plot_sst_read_unified()
    # plot_sst_read()
    # plot_mf_bar()
    # plot_sample_queries()
    # plot_sst_read_subpart()
    # plot_sst_subpart_parcomp()
    plot_sst_query_final()
