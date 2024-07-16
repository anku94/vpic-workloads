import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import glob
import numpy as np
import os
import pandas as pd
import re
import sys

from common import plot_init_bigfont as plot_init
from common import PlotSaver


def make3x3(ax, colors_x, colors_y):
    for idx, color in enumerate(colors_x):
        chunk = 1.0 / 3
        xbeg = chunk * idx
        xend = chunk * (idx + 1)
        ax.fill_between([xbeg, xend], 0, 1, facecolor=color, alpha=0.1)

    for idx, color in enumerate(colors_x):
        chunk = 1.0 / 3
        xbeg = chunk * idx
        xend = chunk * (idx + 1)
        ax.fill_between([0, 1], xbeg, xend, facecolor=color, alpha=0.1)
    pass


def make3x3a(ax):
    colors_x = ['#ddd', '#aaa', '#ddd']
    colors_y = ['green', 'yellow', 'orange']
    make3x3(ax, colors_x, colors_y)


def make3x3b(ax):
    colors_x = ['#ddd', '#aaa', '#ddd']
    make3x3(ax, colors_x, colors_x)


def make3x3c(ax):
    colors_x = ['green', 'yellow', 'red']
    make3x3(ax, colors_x, colors_x)


def set_3levelticks(ax):
    ticks = [0.166, 0.5, 0.833]
    labels = ['Low', 'Mid', 'Hi']

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def set_simpleticks(fig, ax):
    lib_str = [r"$\it{(Lower}$ $\it{Is}$ $\it{Better})$"]
    ax.set_xlabel(
        r"{\fontsize{18}{20} \selectfont (\textit{Lower Is Beter})}"
        "\nQuery Latency")
    ax.set_ylabel('I/O Amplification\n'
                  r"{\fontsize{18}{20} \selectfont (\textit{Lower Is Beter})}")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.text(0.17, 0.15, "\emph{Low}", fontsize=18, horizontalalignment="left")
    fig.text(0.95, 0.15, "\emph{High}", fontsize=18,
             horizontalalignment="right")
    fig.text(0.12, 0.23, "\emph{Low}", fontsize=18, horizontalalignment="left",
             rotation="vertical")
    fig.text(0.12, 0.85, "\emph{High}", fontsize=18,
             horizontalalignment="left", rotation="vertical")


def plot_tradeoff():
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    make3x3a(ax)

    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=True, labelbottom=True)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]

    ax.plot([
        0, 0.3,
        0.36, 0.63,
        0.69, 1
    ], [
        0.01, 0.01,
        0.9, 0.9,
        0.1, 0.1
    ], label='Bulk Sorting', color=colors[0])

    ax.plot([
        0, 0.3,
        0.36, 0.63,
        0.69, 1
    ], [
        0.9, 0.9,
        0.01, 0.01,
        0.16, 0.16
    ], label='Online Indexes', color=colors[1])

    # ax.plot([
    #     0, 0.3,
    #     0.36, 0.63,
    #     0.69, 1
    # ], [
    #     0.02, 0.02,
    #     0.8, 0.8,
    #     0.4, 0.4
    # ], label='Bitmap Indexes')

    ax.plot([
        0, 0.3,
        0.36, 0.63,
        0.69
    ], [
        0.02, 0.02,
        0.02, 0.02,
        0.8
    ], label='Slalom', color=colors[2])

    ax.plot([
        0.69, 1
    ], [
        0.8, 0.4
    ], linestyle='--', color=colors[2])

    ax.plot([
        0, 0.3,
        0.36, 0.63,
        0.69, 1
    ], [
        0.2, 0.2,
        0.03, 0.03,
        0.14, 0.14
    ], label='CARP', color=colors[3], linewidth='3')

    ax.set_xticks([0.166, 0.5, 0.833])
    ax.set_xticklabels(['Write Phase', 'Post Processing', 'Read Phase'])
    ax.set_yticks([0.166, 0.5, 0.833])
    ax.set_yticklabels(['Low', 'Mid', 'Hi'])

    ax.set_xlabel('Timeline Of Data Lifecycle')
    ax.set_ylabel('I/O Cost (Bandwidth and/or Space)')

    fig.suptitle('Different Indexing Approachs and Costs')
    fig.legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 0.83))
    # ax.legend()
    plt.subplots_adjust(top=0.8)

    save = True
    if save:
        plot_dir = "/Users/schwifty/Repos/workloads/rundata/20221030-misc-plots"
        fig.savefig(f"{plot_dir}/tradeoff_matrix.png", dpi=300)
    else:
        fig.show()


def plot_tradeoff_2():
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # make3x3c(ax)

    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=True, labelbottom=True)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c for c in prop_cycle.by_key()['color']]

    # ----- Online Indexes -----

    ax.plot([0.1], [0.7], marker='o', mew='2', mec='black', ms='12',
            label='TritonSort', color=colors[0])
    ax.annotate('TritonSort', xy=(0.1, 0.7), xycoords='data',
                xytext=(0.25, 0.7), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.2),
                ha='left', va='center',
                )

    # ----- Online Indexes -----

    ax.plot([0.15], [0.9], marker='o', mew='2', mec='black', ms='12',
            label='Online Indexes', color=colors[1])
    ax.annotate('Distributed DBs', xy=(0.15, 0.9),
                xycoords='data',
                xytext=(0.3, 0.9), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.2),
                ha='left', va='center',
                )

    # ----- FastQuery -----

    ax.plot([0.55], [0.45], marker='o', mew='2', mec='black', ms='12',
            label='FastQuery', color=colors[2])
    ax.annotate('FastQuery', xy=(0.55, 0.45), xycoords='data',
                xytext=(0.7, 0.45), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.2),
                ha='left', va='center',
                )

    # ----- Slalom -----

    ax.plot([0.8], [0.05], marker='o', mew='2', mec='black', ms='12',
            label='Slalom', color=colors[3])
    ax.annotate('', xy=(0.65, 0.10), xycoords='data',
                xytext=(0.8, 0.05), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.05",
                                linestyle='--', shrinkA=10, shrinkB=10),
                ha='left', va='center',
                )
    ax.plot([0.65], [0.10], marker='o', mew='2', mec='black', ms='12',
            label='Slalom', color=colors[3], alpha=0.7)
    ax.annotate('', xy=(0.5, 0.18), xycoords='data',
                xytext=(0.65, 0.10), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.05",
                                linestyle='--', shrinkA=10, shrinkB=10),
                ha='left', va='center',
                )
    ax.plot([0.5], [0.18], marker='o', mew='2', mec='black', ms='12',
            label='Slalom', color=colors[3], alpha=0.5)

    # ax.annotate('Slalom ($t_0$)', xy=(0.8, 0.05), xycoords='data',
    #             xytext=(0.8, 0.22), textcoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.2),
    #             ha='center', va='center',
    #             )
    #
    # ax.annotate('Slalom ($t_1$)', xy=(0.65, 0.10), xycoords='data',
    #             xytext=(0.65, 0.30), textcoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.2),
    #             ha='center', va='center',
    #             )
    #
    # ax.annotate('Slalom ($t_2$)', xy=(0.5, 0.18), xycoords='data',
    #             xytext=(0.5, 0.38), textcoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.2),
    #             ha='center', va='center',
    #             )

    ax.annotate("Slalom\n"r"\textit{(over time)}", xy=(0.65, 0.2),
                xycoords='data',
                xytext=(0.75, 0.2), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.2),
                ha='left', va='center',
                )

    # ----- DeltaFS -----
    # ax.plot([0.12], [0.04], marker='o', mew='2', mec='black', ms='12',
    #         label='DeltaFS', color=colors[5])
    # ax.annotate('DeltaFS (Point Queries)', xy=(0.12, 0.03), xycoords='data',
    #             xytext=(0.25, 0.03), textcoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.25),
    #             ha='left', va='center',
    #             )

    # ----- CARP -----
    ax.plot([0.15], [0.1], marker='o', mew='2', mec='black', ms='16',
            label='CARP', color=colors[4])
    ax.annotate(r'\textbf{CARP}', xy=(0.15, 0.1), xycoords='data',
                xytext=(0.3, 0.1), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.25),
                ha='left', va='center',
                )

    # ------ Ticks -----
    # set_3levelticks(ax)
    set_simpleticks(fig, ax)

    # ax.set_xlabel('Query Latency')
    # ax.set_ylabel('I/O Amplification')

    # fig.suptitle('Different Indexing Approachs and Costs')
    # fig.legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 0.83))
    # ax.legend()
    # plt.subplots_adjust(top=0.8)
    ax.xaxis.set_major_locator(MultipleLocator(0.34))
    ax.yaxis.set_major_locator(MultipleLocator(0.34))
    ax.xaxis.grid(visible=True, which='major', color='#aaa')
    ax.yaxis.grid(visible=True, which='major', color='#aaa')

    fig.tight_layout()

    PlotSaver.save(fig,
                   "/Users/schwifty/Repos/workloads/rundata/20221127-roofline-ss1024-4gbps",
                   "tradeoff_matrix_3")


def run_plot():
    # plot_tradeoff()
    plot_tradeoff_2()


if __name__ == "__main__":
    # if not os.path.exists(plot_dir):
    #     os.mkdir(plot_dir)

    plot_init()
    run_plot()
