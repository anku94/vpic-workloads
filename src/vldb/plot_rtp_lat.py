import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import os
import pandas as pd

from common import plot_init_bigfont as plot_init


def plot_rtp_lat_orig(eval_dir: str, save: False):
    latdata_path = '/Users/schwifty/Repos/workloads/rundata/20220915-rtpbench-throttledruns/rtp-bench-runs-orig.csv'
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


def plot_rtp_lat_wpvtcnt(plot_dir: str, save: False):
    # latdata_path = '/Users/schwifty/Repos/workloads/rundata/20220915-rtpbench-throttledruns/rtp-bench-runs-ipoib.csv'
    latdata_path = '/Users/schwifty/Repos/workloads/rundata/20221020-roofline-throttlecheck/rtp-bench-runs-all.csv'
    latdata_path = "/Users/schwifty/Repos/workloads/rundata/20221027-roofline-strongscale/rtp-bench-runs-bmi.csv"

    df = pd.read_csv(latdata_path)
    print(df.columns)
    df_mask = (df['rounds'] == 10) & (df['rwarmup'] == 5)
    df = df[df_mask]

    hg_proto = "bmi+tcp"
    df_mask = (df['hg_proto'] == hg_proto)
    df = df[df_mask]
    print(df)

    hg_protostr = {
        "bmi+tcp": "ipoib_bmitcp",
        "psm+psm": "psm"
    }[hg_proto]
    print(hg_protostr)

    fig, ax = plt.subplots(1, 1)

    wo8192 = False
    plot_fname = f'rtp.latvspvtcnt.{hg_protostr}'
    plot_title = f'RTP Round Latency: {hg_protostr}'
    save = True

    hg_poll_linestyle = {
        0: '-o',
        1: '--s'
    }

    for hg_poll in df["hg_poll"].unique():
        df_plot = df[df["hg_poll"] == hg_poll]
        all_npivots = sorted(df['npivots'].unique())
        if wo8192:
            all_npivots = all_npivots[:-1]
            plot_fname = f'{plot_fname}.wo8192'

        for npivots in all_npivots:
            print(npivots)
            df_plot_pvt = df_plot[df_plot['npivots'] == npivots]
            data_x = df_plot_pvt['nranks']
            data_y = df_plot_pvt['mean']
            ls = hg_poll_linestyle[hg_poll]
            label = '{} pivots'.format(npivots)
            ax.plot(data_x, data_y, ls, label=label)

    # df_std = df[df['rounds'] == 100]
    # data_y1 = df_std['mean'] - df_std['std']
    # data_y2 = df_std['mean'] + df_std['std']
    # data_x = df_std['nranks']
    #
    # ax.fill_between(data_x, data_y1, data_y2, facecolor='green', alpha=0.1)

    ax.set_ylim([0, 600000])
    ax.set_xscale('log')
    xticks = df['nranks'].unique()
    ax.set_xticks(xticks)
    ax.minorticks_off()
    ax.set_xticklabels([str(i) for i in xticks])
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}ms'.format(x / 1000))

    # ax.set_title(plot_title)
    ax.set_xlabel('Number of Ranks')
    ax.set_ylabel('Renegotiation Round Latency')

    # ax.legend(loc='upper left', ncol=3)
    custom_lines = [
        Line2D([0], [0], linestyle='-', label='HG_POLL=OFF'),
        Line2D([0], [0], linestyle='--', label='HG_POLL=ON')
    ]

    if (hg_proto == "bmi+tcp"):
        ax.legend(loc='upper left', fontsize=14)
    else:
        ax.legend(handles=custom_lines, loc='upper left')

    fig.tight_layout()

    ax.yaxis.set_minor_locator(MultipleLocator(25000))
    ax.yaxis.grid(b=True, which="major", color="#aaa")
    ax.yaxis.grid(b=True, which="minor", color="#ddd")

    if save:
        fig.savefig('{}/{}.pdf'.format(plot_dir, plot_fname), dpi=600)
        # fig.savefig('{}/{}.png'.format(plot_dir, plot_fname), dpi=600)
    else:
        fig.show()


def run_plot_rtpbench(plot_dir):
    # plot_rtp_lat_orig(plot_dir, False)
    plot_rtp_lat_wpvtcnt(plot_dir, False)
    pass


if __name__ == "__main__":
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20220915-rtpbench-throttledruns"
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20221020-roofline-throttlecheck"
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20221030-misc-plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot_init()
    run_plot_rtpbench(plot_dir)
