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

all_energies = []


def read_a_file(fname):
    f = open(fname, 'rb')
    energies = []

    while True:
        raw = f.read(4)
        if not raw:
            break
        energies.append(struct.unpack('f', raw)[0])
    return energies


def read_zipped(glob_pattern):
    files = glob.glob(glob_pattern)
    print(files)
    temp_energies = []
    for f in files:
        energies = read_a_file(f)
        print(len(energies))
        temp_energies.append(energies)

    global all_energies
    all_energies = functools.reduce(
        operator.iconcat, zip_longest(*temp_energies), [])
    all_energies = list(filter(lambda x: x, all_energies))
    print(len(all_energies))
    return


def parse_glob(glob_pattern):
    ts = re.findall('T\.(\d+)', glob_pattern)[0]
    e_or_h = glob_pattern.split('/')[-1][0]
    title = "TS %s (%s-part)" % (ts, e_or_h)
    fname = "data.t%s.%spart.gif" % (ts, e_or_h)
    fname = "ptile.t%s.%spart.gif" % (ts, e_or_h)
    print(title, fname)
    return title, fname


def approx_equal(a, b):
    delta = 1e-4

    if a > (b - delta) and a < (b + delta):
        return True

    return False

def get_chi2_score(data_a, data_b):
    len_a = sum(data_a)
    len_b = sum(data_b)

    assert len_a < len_b

    data_a += np.array([10] * len(data_a))
    data_b += np.array([10] * len(data_b))

    len_a += len(data_a)
    len_b += len(data_b)

    data_a = data_a * (len_b / len_a)

    data_a += np.array([1] * len(data_a))
    data_b += np.array([1] * len(data_b))

    delta = data_b - data_a

    chi2 = sum(delta *delta / data_a)

    return chi2

def get_chi2_seq(data):
    bin_edges = np.histogram_bin_edges(data, 'fd')
    print(len(bin_edges))
    hist_a, _ = np.histogram(data[:10000], bin_edges)
    hist_b, _ = np.histogram(data[:20000], bin_edges)

    chi2 = get_chi2_score(hist_a, hist_b)
    print(chi2)

    data_len = len(data)

    scores = []

    for i in range(1, 100):
        cur_len = int(i * data_len / 100.0)
        next_len = int((i + 1) * data_len / 100.0)

        next_len = min(data_len, next_len)

        hist_a, _ = np.histogram(data[:cur_len], bin_edges)
        hist_b, _ = np.histogram(data[:next_len], bin_edges)

        chi2 = get_chi2_score(hist_a, hist_b)
        print(i, cur_len, next_len, chi2)
        scores.append(chi2)

    return scores

glob_pattern = sys.argv[1]
print(parse_glob(glob_pattern))
title, fname = parse_glob(glob_pattern)

read_zipped(glob_pattern)

energies = all_energies

#scores = get_chi2_seq(energies)
#plt.plot(range(1, 100), scores)

energies = sorted(energies)

per_50 = int(len(energies) * 0.5)
per_75 = int(len(energies) * 0.75)
per_90 = int(len(energies) * 0.9)
per_95 = int(len(energies) * 0.95)
per_99 = int(len(energies) * 0.99)
per_999 = int(len(energies) * 0.999)

with open(fname, 'w+') as f:
    print("50%ile: ", energies[per_50], file=f)
    print("75%ile: ", energies[per_75], file=f)
    print("90%ile: ", energies[per_90], file=f)
    print("95%ile: ", energies[per_95], file=f)
    print("99%ile: ", energies[per_99], file=f)
    print("99.9%ile: ", energies[per_999], file=f)
#  yvals = np.arange(len(energies)) / float(len(energies) - 1)

#  plt.plot(energies, yvals)

#  ts = re.findall(r'T\.(\d+)', glob_pattern)[0]
#  e_or_h = glob_pattern.split('/')[-1][0]
#  print(ts)
#  print(e_or_h)

#  plt.title("CDF, %spart, T.%s" % (e_or_h.upper(), ts,))
#  plt.savefig("cdf.%spart.t%s.png" % (e_or_h, ts,))
#plt.show()
