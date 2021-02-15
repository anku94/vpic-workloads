import glob
import re
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

def read_manifest_file(data_path: str, rank: int = 0):
    f = open(data_path).read().splitlines()
    all_items = []
    for item in f:
        item = re.split('\ |-', item)
        item = list(filter(lambda x: x != '', item))
        item[0] = float(item[0])
        item[1] = float(item[1])
        item[2] = int(item[2])
        if item[2] == 0: continue
        item = item[:3]
        print(item)
        item.append(rank)
        all_items.append(item)

    return all_items

def read_entire_manifest(data_path: str):
    all_data = []
    all_files = glob.glob(data_path + '/*manifest*')
    for file in all_files:
        file_rank = int(file.split('.')[-1])
        print(file, file_rank)
        data = read_manifest_file(file, file_rank)
        all_data += data

    return all_data

def get_stats(manifest):
    range_min = manifest[0][0]
    range_max = manifest[0][1]
    item_count = 0

    for item in manifest:
        range_min = min(item[0], range_min)
        range_max = max(item[1], range_max)
        item_count += item[2]

    print(range_min, range_max, item_count)
    return(range_min, range_max, item_count)


def get_overlapping_count(manifest, point):
    overlapping_ssts = 0
    overlapping_mass = 0

    overlapping_items = []
    overlapping_ranks = set()

    for item in manifest:
        if point >= item[0] and point <= item[1]:
            overlapping_ssts += 1
            overlapping_mass += item[2]
            overlapping_items.append(item)
            overlapping_ranks.add(item[3])

    print(overlapping_ranks)

    return overlapping_ssts, overlapping_mass


def do_ranges_overlap(a1, b1, a2, b2):
    if a1 >= a2 and a1 <= b2:
        return True

    if b1 >= a2 and b1 <= b2:
        return True

    if a2 >= a1 and a2 <= b2:
        return True

    if b2 >= a1 and b2 <= a2:
        return True

    return False
    

def get_overlapping_range_count(manifest, p, q):
    overlapping_ssts = 0
    overlapping_mass = 0

    for item in manifest:
        if do_ranges_overlap(item[0], item[1], p, q):
            overlapping_ssts += 1
            overlapping_mass += item[2]
            
    return overlapping_ssts, overlapping_mass

def run_manifest_analysis(data_path: str):
    mf_items = read_entire_manifest(data_path)
    # print(mf_items)
    range_min, range_max, item_sum = get_stats(mf_items)
    item_count = len(mf_items)

    count = get_overlapping_count(mf_items, 0.75)
    x_values_p1 = np.linspace(range_min, 2, num=100)
    x_values_p2 = np.linspace(2, range_max, num=100)
    x_values = np.concatenate([x_values_p1, x_values_p2])
    # x_values = x_values_p1
    overlap_stats = map(lambda x: get_overlapping_count(mf_items, x), x_values)
    overlap_stats = list(zip(*overlap_stats))
    y_count_values = np.array(overlap_stats[0])
    y_mass_values = np.array(overlap_stats[1])

    y_mass_percent = y_mass_values * 100.0 / item_sum
    # print(x_values)
    # print(y_count_values)
    # print(y_mass_values)
    # print(y_mass_percent)
    range_max = 2

    fig, ax = plt.subplots(1, 1)
    # ax.plot(x_values, y_count_values)
    ax.plot(x_values, y_mass_percent)
    # ax.plot([range_min, range_max], [item_count/512.0, item_count/512.0], '--')
    # ax.plot([range_min, range_max], [100.0/512.0, 100.0/512.0], '--')
    # ax.set_ylim(0, 1)
    ax.set_ylabel('Partitioning (% of total size)')
    ax.set_ylabel('Number of overlapping SSTs')
    ax.set_xlabel('Indexed Attribute (Energy)')
    ax.set_title('Selectivity (%age) of range-partitioning across keyspace (T1900)')
    ax.set_title('Overlapping SST Count vs Keyspace (Ranks 512, SSTs = 31,096)')
    timestep = data_path.split('.')[-1]
    print(timestep)
    fig.show()
    # fig.savefig('../vis/manifest/selectivity.512.seppol.fbar.pdf')
    print(len(mf_items))

def run_manifest_cdf_analysis(data_path: str):
    mf_items = read_entire_manifest(data_path)
    # print(mf_items)
    range_min, range_max, item_sum = get_stats(mf_items)
    item_count = len(mf_items)

    get_overlapping_count(mf_items, 0.5)
    x_values_p1 = np.linspace(range_min, 2, num=100)
    # x_values_p2 = np.linspace(2, range_max, num=100)
    # x_values = np.concatenate([x_values_p1, x_values_p2])
    x_values = x_values_p1
    overlap_stats = map(lambda x: get_overlapping_range_count(mf_items, 0, x), x_values)
    overlap_stats = list(zip(*overlap_stats))
    y_count_values = np.array(overlap_stats[0])
    y_mass_values = np.array(overlap_stats[1])

    y_mass_percent = y_mass_values * 100.0 / item_sum
    # print(x_values)
    # print(y_count_values)
    # print(y_mass_values)
    # print(y_mass_percent)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_values, y_count_values)
    ax.plot([range_min, range_max], [item_count/32.0, item_count/32.0], '--')
    # ax.set_ylim(0, 100)
    ax.set_ylabel('Partitioning (% of total size)')
    ax.set_ylabel('Number of overlapping SSTs')
    ax.set_xlabel('Indexed Attribute (Energy)')
    ax.set_title('Selectivity (%age) of range-partitioning across keyspace (T1900)')
    ax.set_title('Overlapping SST Count for query in (0, X)')
    timestep = data_path.split('.')[-1]
    fig.show()
    # fig.savefig('../vis/manifest/cdffromzero.T{0}.1.5M.32.32.pdf'.format(timestep))

def plot_misc_2():
    df = pd.read_csv('../rundata/rtp-params/pivot-std.csv')
    print(df)

    df = df[9:]

    df_1900_data = df[df['ts'] == 1900]
    df_2850_data = df[df['ts'] == 2850]

    df_1900_y = df_1900_data['stddev'] * 100
    df_1900_x = df_1900_data['renegcnt']

    df_2850_y = df_2850_data['stddev'] * 100
    df_2850_x = df_2850_data['renegcnt']

    fig, ax = plt.subplots(1, 1)
    ax.plot(df_1900_x, df_1900_y, label='Timestep 1900')
    ax.plot(df_2850_x, df_2850_y, label='Timestep 2850')
    ax.set_ylabel('Standard Deviation of Load (%)')
    ax.set_xlabel('Num. of Fixed RTP Rounds During I/O')
    ax.set_title('Renegotiation Freq vs Load Balance')
    ax.legend()

    # fig.show()
    fig.savefig('../vis/manifest/renegfreq.pdf')

def plot_misc():
    df = pd.read_csv('../rundata/rtp-params/pivot-std.csv')
    print(df)

    df1 = df[0:8]
    df_x = df1['pvtcnt']
    df_data = df1['stddev']
    df_xnum = range(len(df_data))
    print(df_data)
    fig, ax = plt.subplots(1, 1)
    ax.bar(df_xnum, df_data * 100)
    ax.set_ylabel('Standard deviation in load (%)')
    ax.set_xlabel('Number of pivots')
    ax.set_title('Pivots Sampled vs Load stddev')
    plt.xticks(df_xnum, df_x)
    # fig.show()
    fig.savefig('../vis/manifest/pvtcnt.pdf')
    return

if __name__ == '__main__':
    # plot_misc_2()
    data_path = '../rundata/vpic-512/data.3200.256/vpic-deltafs-run-2617962'
    data_path = '../rundata/vpic-pollution-nextsession'
    run_manifest_analysis(data_path)
    sys.exit(0)

    data_path_base = "../rundata/manifest-data/T."
    timesteps = ['100', '950', '1900', '2850']
    for timestep in timesteps:
        data_path = data_path_base + timestep
        print(data_path)
        run_manifest_cdf_analysis(data_path)
