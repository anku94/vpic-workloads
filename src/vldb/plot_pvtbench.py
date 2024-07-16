import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import glob
import numpy as np
import os
import pandas as pd
import re
import sys
from common import plot_init_bigfont, plot_init_bigfont_singlecol, PlotSaver


def multiline_label(str_a, str_b):
    label = r'{\fontsize{13pt}{3em}\selectfont{}' + \
            str_a + '}\n' + \
            r'{\fontsize{12pt}{3em}\selectfont{}\emph{' + \
            str_b + '}}'

    label = r'{\fontsize{15pt}{3em}\selectfont{}' + \
            str_b.strip('()') + '}'

    return label


def plot_pvtbench(df_path, plot_dir, save=False):
    df = pd.read_csv(df_path, index_col=None)
    df = df[df['nranks'] == 512]
    df.sort_values(['runtype', 'pvtcnt'], inplace=True)
    print(df)

    all_pvtcnt = df['pvtcnt'].unique()
    num_ep = 12

    fig, ax = plt.subplots(1, 1)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]
    cm = plt.cm.get_cmap('Dark2')
    colors = list(cm.colors)
    print('Num colors: {}'.format(len(colors)))

    def color(x):
        colidx = hash(str(x)) % len(colors)
        return colors[colidx]

    runtypes_to_skip = ['ownpvt', 'initpvt']

    marker_styles = [
                        "D", "h", "^", "v", "X", "X", "*", "s", "P"
                    ]

    for runtype in df['runtype'].unique():
        df_run = df[df['runtype'] == runtype]
        if runtype == 'initpvt':
            linestyle = '--'
        else:
            linestyle = '-'

        colidx = 0
        for pvtidx, pvtcnt in enumerate(all_pvtcnt):
            if pvtcnt == 128: continue

            df_tmp = df_run[df_run['pvtcnt'] == pvtcnt]
            data_y = df_tmp['load_std']
            data_x = df_tmp['epidx']
            print(','.join(data_x.astype(str)))
            print(','.join(data_y.astype(str)))

            if runtype == 'initpvt':
                label = ''
            else:
                label = f'{pvtcnt} pivots'

            if runtype == 'epxpp':
                ax.plot(data_x, data_y, marker=marker_styles[pvtidx], linestyle='-',
                        color=colors[colidx],
                        label=label, ms=7)
                pass

            colidx += 1

    tag = 'expsub1pp'
    tag = 'epxpp'
    save = True,

    ax.set_ylim([0, 1])
    ax.set_xlabel('Simulation Timestep')
    ax.set_ylabel('Load Std-Dev (\%)')

    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}%'.format(x * 100))
    timesteps = [200, 2000, 3800, 5600, 7400, 9200, 11000, 12800, 14600, 16400,
                 18200, 19400]
    # ax.xaxis.set_major_formatter(lambda x, pos: 'Ep{}'.format(int(x) + 1))
    # ax.xaxis.set_major_formatter(lambda x, pos: '{}'.format(timesteps[int(x)]))
    ax.set_xticks(data_x)
    ax.set_xticklabels([str(t) for t in timesteps], rotation=40)

    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.grid(visible=True, which='major', color='#aaa')
    ax.yaxis.grid(visible=True, which='minor', color='#ddd')
    # fig.legend()

    # ax.set_title('Fit Of Initially Sampled Pivots at Different PivotCounts')
    custom_lines = [
        Line2D([0], [0], linestyle='--', label='Epoch0 Samples'),
        Line2D([0], [0], linestyle='-', label='CurEpoch Samples')
    ]

    custom_lines = []

    handles, lables = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:9] + custom_lines, ncol=3, loc="upper left",
              bbox_to_anchor=(0.01, 1.05), fontsize=13)

    ratio = 0.33
    ax.set_aspect(1.0 / ax.get_data_ratio() * ratio)

    fig.tight_layout()

    PlotSaver.save(fig, "wherever", f"pvtbench.{tag}")


def plot_pvtbench_adaptivity(df_path, plot_dir, save=False):
    df = pd.read_csv(df_path, index_col=None)
    df = df[df['nranks'] == 512]
    df.sort_values(['runtype', 'pvtcnt'], inplace=True)
    print(df)

    all_pvtcnt = df['pvtcnt'].unique()
    num_ep = 12

    fig, ax = plt.subplots(1, 1)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]
    cm = plt.cm.get_cmap('Dark2')
    colors = list(cm.colors)
    print('Num colors: {}'.format(len(colors)))

    def color(x):
        colidx = hash(str(x)) % len(colors)
        return colors[colidx]

    runtypes_to_skip = ['ownpvt', 'initpvt']

    for runtype in df['runtype'].unique():
        df_run = df[df['runtype'] == runtype]
        if runtype == 'initpvt':
            linestyle = '--'
        else:
            linestyle = '-'

        colidx = 0
        for pvtidx, pvtcnt in enumerate(all_pvtcnt):
            if pvtcnt != 4096: continue

            df_tmp = df_run[df_run['pvtcnt'] == pvtcnt]
            data_y = df_tmp['load_std']
            data_x = df_tmp['epidx']
            print(','.join(data_x.astype(str)))
            print(','.join(data_y.astype(str)))

            if runtype == 'initpvt':
                label = ''
            else:
                label = f'{pvtcnt} pivots'

            if runtype == 'ep0pp':
                ax.plot(data_x, data_y, linestyle='--',
                        color=colors[0], ms=11, marker='X',
                        label=multiline_label("Static Partitioning",
                                              "(Partition table from first timestep)")
                        )
                pass
            elif runtype == 'epxsub1pp':
                ax.plot(data_x, data_y, linestyle='-.',
                        color=colors[1], ms=7, marker='s',
                        label=multiline_label("Partially Dynamic Partitioning",
                                              "(Partition table from prev. timestep)")
                        )
                pass
            elif runtype == 'epxpp':
                ax.plot(data_x, data_y, marker='o', linestyle='-',
                        color=colors[2], ms=11,
                        label=multiline_label("Fully Dynamic Partitioning",
                                              "(Partition table from current timestep)")
                        )
                pass

            colidx += 1

    tag = 'expsub1pp'
    tag = 'epxpp'
    save = True,

    ax.set_aspect(4)

    ax.set_ylim([0, 1])
    ax.set_xlabel('Simulation Timestep')
    ax.set_ylabel('Load Std-Dev (\%)')

    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}%'.format(x * 100))
    timesteps = [200, 2000, 3800, 5600, 7400, 9200, 11000, 12800, 14600, 16400,
                 18200, 19400]
    ax.set_xticks(data_x)
    ax.set_xticklabels([str(t) for t in timesteps], rotation=40)

    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.grid(visible=True, which='major', color='#aaa')
    ax.yaxis.grid(visible=True, which='minor', color='#ddd')
    # fig.legend()

    # ax.set_title('Fit Of Initially Sampled Pivots at Different PivotCounts')
    custom_lines = [
        Line2D([0], [0], linestyle='--', label='Epoch0 Samples'),
        Line2D([0], [0], linestyle='-', label='CurEpoch Samples')
    ]

    custom_lines = []

    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    # ax.legend(handles=handles[:9] + custom_lines, ncol=1, loc="upper left",
    #           bbox_to_anchor=(0.1, 1.01))
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.27, 0.81),
              fontsize=11, framealpha=0.5)

    ratio = 0.33
    ax.set_aspect(1.0 / ax.get_data_ratio() * ratio)

    fig.tight_layout()

    PlotSaver.save(fig, plot_dir, f"pvtbench.adapt")


def run_plot():
    df_path = "/Users/schwifty/repos/workloads/rundata/20221027-roofline-strongscale/pvtbench.csv"
    plot_dir = os.path.dirname(df_path)
    plot_pvtbench(df_path, plot_dir, save=False)
    # plot_pvtbench_adaptivity(df_path, plot_dir, save=False)


if __name__ == '__main__':
    # plot_init()
    plot_init_bigfont_singlecol()
    run_plot()
