import glob
import re
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def read_manifest_file(data_path: str, rank: int = 0):
    f = open(data_path).read().splitlines()
    all_items = []
    print('rank: ', rank)

    all_pol = 0
    all_total = 0

    for item in f:
        size_stats = re.findall('- (\d+) (\d+)', item)
        pol_size = int(size_stats[0][1])
        total_size = int(size_stats[0][0])

        all_pol += pol_size
        all_total += total_size

        pol_ratio = pol_size * 1.0 / total_size
        # print(pol_size, total_size)
        all_items.append(pol_ratio)

    all_pol_ratio = all_pol * 1.0 / all_total

    return all_items, all_pol_ratio

def read_entire_manifest(data_path: str):
    all_data = {}
    all_files = glob.glob(data_path + '/*manifest*')
    max_data_len = 0

    all_aggr_pol = {}

    for file in all_files:
        file_rank = int(file.split('.')[-1])
        data, aggr_pol_ratio = read_manifest_file(file, file_rank)
        all_data[file_rank] = data
        all_aggr_pol[file_rank] = aggr_pol_ratio
        max_data_len = max(max_data_len, len(data))

    print(max_data_len)

    ret_data = []

    for rank in sorted(all_data.keys()):
        all_data[rank] += [0] * (max_data_len - len(all_data[rank]))
        assert(len(all_data[rank]) == max_data_len)
        ret_data.append(all_data[rank])

    twod_poldata = ret_data
    oned_poldata = list(map(lambda x: x[1], sorted(all_aggr_pol.items())))

    return twod_poldata, oned_poldata

def colorbar_index(ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in xrange(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


def plot_pollution(data_path: str):
    data_2d, data_1d = read_entire_manifest(data_path)
    print(data_2d)
    data_2d = list(zip(*data_2d))
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(data_2d, aspect='auto')
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal', pad=0.15)

    ax.set_title('SST Pollution For Each Rank')

    ax.set_xlabel('Rank (0-512)')
    ax.set_ylabel('SSTable ID')

    ax2 = ax.twinx()
    ax2.plot(range(len(data_1d)), data_1d, color='#cccccc')

    ax2.set_ylim([0, 0.3])
    ax2.set_ylabel('Average Pollution (%)')

    fig.tight_layout()
    # plt.show()
    plt.savefig('../vis/manifest/pollution-bufferprev.pdf', dpi=600)
    print('Avg Pollution: ', np.mean(data_1d))


if __name__ == '__main__':
    data_path = '../rundata/vpic-pollution-bufferprev'
    plot_pollution(data_path)
    pass
