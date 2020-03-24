#!/usr/bin/env python
# coding: utf-8

import struct
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import sys
import functools
import operator
from itertools import zip_longest

from util import VPICReader, Histogram

all_energies = []


def arrf(arr):
    arr_str = ', '.join(map(lambda x: "%.2f" % (x, ), arr))
    return '[ %s ]' % (arr_str, )


def parse_glob(glob_pattern):
    ts = re.findall('T\.(\d+)', glob_pattern)[0]
    e_or_h = glob_pattern.split('/')[-1][0]
    title = "TS %s (%s-part)" % (ts, e_or_h)
    fname = "data.t%s.%spart.gif" % (ts, e_or_h)
    print(title, fname)
    return title, fname


def get_chart_title(ts_int, e_or_h='e'):
    title = "TS %s (%s-part)" % (ts_int, e_or_h)
    return title


def get_chart_fname(ts_int, e_or_h='e'):
    fname = "data.t%s.%spart.gif" % (ts_int, e_or_h)
    return fname


def read_into_global(reader):
    global all_data
    #all_data = reader.read_global(TSIDX)
    all_data = reader.read_a_rank(TSIDX, 0)


def gen_anim_frames():
    global anim_frames
    anim_frames = [1] + list(range(5, 105, 5))
    print(anim_frames)


def gen_hist(percent_particles):
    all_data_len = len(all_data)

    data_len_1p = int(all_data_len * percent_particles)

    sub_data = all_data[:data_len_1p]

    data_hist = Histogram(sub_data, 128)

    global accurate_hist
    accurate_hist, mass_per_bin = data_hist._rebalance(32)


def plot_load_view(ax, loads, lines):
    x = np.arange(0, max(lines) + 1, (max(lines) + 1) / len(loads))
    print(len(x))
    print(len(loads))

    ax.bar(x, loads, width=0.2)
    for line in lines:
        ax.plot([line, line], [0, max(loads)], color='red', linestyle='dashed')

    mean_load = np.mean(loads)
    ax.plot([0, max(lines + 1)], [mean_load, mean_load], color='orange')
    mean_load *= 1.15
    ax.plot([0, max(lines + 1)], [mean_load, mean_load], color='black')


def animate(reader, ax, i):
    global accurate_hist
    global all_data

    global anim_frames

    len_all_data = len(all_data)
    len_clipped = int(len_all_data * anim_frames[i] / 100.0)
    data = all_data[:len_clipped]

    mean = np.mean(data)
    var = np.var(data)

    loads, lines = np.histogram(data, accurate_hist)
    ax.clear()

    plot_load_view(ax, loads, lines)
    #n, bins, patches = ax1.hist(full_data, data_hist.bin_edges, rwidth=0.8)

    #print("Plotting Total: ", percent_particles)
    percent_particles = anim_frames[i]

    chart_title = get_chart_title(reader.get_ts(0))
    plt.title("%s: %0.2f%% particles (mean: %.2f, var: %.2f)" %
              (chart_title, percent_particles, mean, var))

    plt.draw()

#ani.save(gif_fname, writer='imagemagick', fps=3)

#v.read_global(0)


def run():
    v = VPICReader(sys.argv[1])
    n = v.get_num_ranks()
    print("Ranks: ", n)

    global TSIDX
    TSIDX = 0

    read_into_global(v)
    gen_anim_frames()
    gen_hist(0.01)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    #animate(v, ax1, 10)
    ax1.clear()

    ani = animation.FuncAnimation(fig, lambda x: animate(
        v, ax1, x), interval=10, frames=len(anim_frames))
    plt.show()


if __name__ == "__main__":
    run()
