#!/usr/bin/env python
# coding: utf-8

import struct
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob, sys
import functools, operator

all_energies = []

def read_a_file(fname):
    f = open('eparticle.100.0', 'rb')
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
    all_energies = functools.reduce(operator.iconcat, zip(*temp_energies), [])
    print(len(all_energies))
    return

def parse_glob(glob_pattern):
    ts = re.findall('T\.(\d+)', glob_pattern)[0]
    e_or_h = glob_pattern.split('/')[-1][0]
    title = "TS %s (%s-part)" % (ts, e_or_h)
    fname = "data.t%s.%spart.gif" % (ts, e_or_h)
    print(title, fname)
    return title, fname

glob_pattern = sys.argv[1]
print(parse_glob(glob_pattern))

read_zipped(glob_pattern)

energies = all_energies
print(len(energies))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

prev_jump = 0.01
prev_sum = 0
chart_title, gif_fname = parse_glob(glob_pattern)

def approx_equal(a, b):
    delta = 1e-4

    if a > (b - delta) and a < (b + delta):
        return True

    return False

def animate(i):
    total_particles = len(energies)

    global prev_sum
    global prev_jump
    global chart_title

    if approx_equal(prev_sum, 0.1):
        prev_jump = 0.1
    elif approx_equal(prev_sum, 1):
        prev_jump = 1
    elif approx_equal(prev_sum, 10):
        prev_jump = 10

    prev_sum += prev_jump

    num_particles = int(prev_sum / 100.0 * total_particles)

    num_particles = min(num_particles, total_particles)

    percent_particles = 100.0 * num_particles / total_particles

    data = energies[:num_particles]
    mean = np.mean(data)
    var = np.var(data)

    ax1.clear()
    n, bins, patches = ax1.hist(energies[:num_particles], 500)

    print("Plotting Total: ", percent_particles)
    #  print(n, bins)

    plt.title("%s: %0.2f%% particles (mean: %.2f, var: %.2f)" % (chart_title, percent_particles,mean,var))

ani = animation.FuncAnimation(fig, animate, interval=30, frames=37)
#plt.show()
ani.save(gif_fname, writer='imagemagick', fps=3)
