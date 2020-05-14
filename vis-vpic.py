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
    all_data = reader.read_global(TSIDX)
    #all_data = reader.read_a_rank(TSIDX, 0)


def gen_anim_frames():
    global anim_frames
    anim_frames = [1] + list(range(5, 105, 5))
    anim_frames = list(range(5, 105, 5))
    print(anim_frames)


def gen_hist(percent_particles):
    all_data_len = len(all_data)

    data_len_1p = int(all_data_len * percent_particles)

    print("Generating hist for %s, %s" % (all_data_len * percent_particles,
                                          percent_particles))

    sub_data = all_data[:data_len_1p]

    data_hist = Histogram(sub_data, 512)
    #  print("Sub data: ", len(sub_data))
    #  print("Data Hist: ", data_hist.bin_edges[-1],
          #  data_hist.hist[-1], sum(data_hist.hist))

    global accurate_hist
    accurate_hist, mass_per_bin = data_hist._rebalance(32)

    #  print("Accuurate Hist: ", accurate_hist[-1],
          #  mass_per_bin, sum(data_hist.hist))
    #  print(data_hist.bin_edges, data_hist.hist)
    #  print(accurate_hist)
    del data_hist


def plot_load_view(ax, loads, lines):
    x = np.arange(0, max(lines) + 1, (max(lines) + 1) / len(loads))
    print(len(x))
    print(len(loads))

    global TSIDX

    widths = {
        0: 0.3,
        1: 0.8,
        2: 3,
        3: 5
    }

    ax.bar(x, loads, width=widths[TSIDX])
    #  for line in lines:
        #  ax.plot([line, line], [0, max(loads)], color='red', linestyle='dashed')

    #  mean_load = np.mean(loads)
    #  ax.plot([0, max(lines + 1)], [mean_load, mean_load], color='orange')
    #  mean_load *= 1.15
    #  ax.plot([0, max(lines + 1)], [mean_load, mean_load], color='black')


def animate(reader, ax, i):
    global accurate_hist
    global all_data
    global anim_frames
    global TSIDX

    len_all_data = len(all_data)
    len_clipped = int(len_all_data * anim_frames[i] / 100.0)
    print("Len Before, After: ", len_all_data, len_clipped)

    data = all_data[:len_clipped]

    mean = np.mean(data)
    var = np.var(data)

    loads, lines = np.histogram(data, accurate_hist)
    ax.clear()

    plot_load_view(ax, loads, lines)

    percent_particles = anim_frames[i]

    chart_title = get_chart_title(reader.get_ts(TSIDX))
    plt.title("%s: %0.2f%% particles (mean: %.2f, var: %.2f)" %
              (chart_title, percent_particles, mean, var))

    plt.draw()


def animate_reneg(reader, ax, i):
    global accurate_hist
    global all_data
    global anim_frames
    global TSIDX

    frac_seen = anim_frames[i] / 100.0
    frac_prev = frac_seen - 0.05

    print(frac_prev, frac_seen)

    #  if i == 0:
        #  gen_hist(0.01)
    #  else:
        #  gen_hist(frac_prev)
    gen_hist(0.01)

    len_all_data = len(all_data)

    all_data_start_per = frac_prev
    all_data_end_per = frac_seen

    all_data_start = int(len_all_data * all_data_start_per)
    all_data_end = int(len_all_data * all_data_end_per)

    print("Len Before, After: ", len_all_data, all_data_start, all_data_end)
    print("Accurate Hist: ", accurate_hist[-1])

    data = all_data[all_data_start:all_data_end]
    data = all_data

    mean = np.mean(data)
    var = np.var(data)

    loads, lines = np.histogram(data, accurate_hist)

    global all_loads
    all_loads += loads

    ax.clear()

    plot_load_view(ax, all_loads, lines)

    percent_particles = anim_frames[i]

    #  chart_title = get_chart_title(reader.get_ts(TSIDX))
    #  plt.title("%s: %0.2f%% particles (mean: %.2f, var: %.2f)" %
              #  (chart_title, percent_particles, mean, var))

    plt.draw()


def run():
    v = VPICReader(sys.argv[1])
    n = v.get_num_ranks()
    print("Ranks: ", n)

    global TSIDX
    TSIDX = int(sys.argv[2])

    read_into_global(v)
    gen_anim_frames()
    gen_hist(1.0)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.clear()

    #animate(v, ax1, len(anim_frames) - 1)
    ani = animation.FuncAnimation(fig, lambda x: animate(
        v, ax1, x), interval=10, frames=len(anim_frames))
    #plt.show()
    fname = 'vis/loadbalance.finbins.%s.512samples.gif' % (v.get_ts(TSIDX))
    print(fname)
    ani.save(fname, writer='imagemagick', fps=30, dpi=200)


def run_reneg():
    v = VPICReader(sys.argv[1])
    n = v.get_num_ranks()
    print("Ranks: ", n)

    global TSIDX
    TSIDX = int(sys.argv[2])

    print("TS: ", v.get_ts(TSIDX))

    global all_loads
    all_loads = [0] * 32

    read_into_global(v)
    gen_anim_frames()
    gen_hist(0.01)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.clear()

    #animate(v, ax1, len(anim_frames) - 1)
    #  ani = animation.FuncAnimation(fig, lambda x: animate_reneg(
        #  v, ax1, x), interval=10, frames=len(anim_frames))
    animate_reneg(v, ax1, len(anim_frames) - 1)
    plt.show()
    #  fname = 'vis/loadbalance.renegsimple.%s.512samples.gif' % (v.get_ts(TSIDX))
    #  print(fname)
    #  ani.save(fname, writer='imagemagick', fps=30, dpi=200)


if __name__ == "__main__":
    run_reneg()
