import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import multiprocessing
import numpy as np
import os
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
    latdata_path = '/Users/schwifty/Repos/workloads/rundata/20220915-rtpbench-throttledruns/rtp-bench-runs-ipoib.csv'
    df = pd.read_csv(latdata_path)
    print(df)

    fig, ax = plt.subplots(1, 1)
    all_npivots = sorted(df['npivots'].unique())

    wo8192 = False
    plot_fname = 'rtp.latvspvtcnt.pdf'

    if wo8192:
        all_npivots = all_npivots[:-1]
        plot_fname = 'rtp.latvspvtcnt.wo8192.pdf'

    for npivots in all_npivots:
        print(npivots)
        df_plot = df[df['npivots'] == npivots]
        data_x = df_plot['nranks']
        data_y = df_plot['mean']
        ls = '-o'
        label = '{} pivots'.format(npivots)
        ax.plot(data_x, data_y, ls, label=label)

    # df_std = df[df['rounds'] == 100]
    # data_y1 = df_std['mean'] - df_std['std']
    # data_y2 = df_std['mean'] + df_std['std']
    # data_x = df_std['nranks']
    #
    # ax.fill_between(data_x, data_y1, data_y2, facecolor='green', alpha=0.1)

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
        fig.savefig('{}/{}'.format(plot_dir, plot_fname), dpi=600)
    else:
        fig.show()


def run_plot_rtpbench(plot_dir):
    # plot_rtp_lat_orig(plot_dir, False)
    plot_rtp_lat_wpvtcnt(plot_dir, True)
    pass


if __name__ == "__main__":
    plot_dir = "/Users/schwifty/Repos/workloads/rundata/20220915-rtpbench-throttledruns"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot_init()
    run_plot_rtpbench(plot_dir)
