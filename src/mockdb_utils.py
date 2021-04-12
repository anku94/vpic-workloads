import glob
import re
from typing import List, Tuple

import multiprocessing
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

from common import abbrv_path

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
        item.append(rank)
        all_items.append(item)

    return all_items


def read_entire_manifest(data_path: str, epoch: int):
    all_data = []
    all_files = glob.glob(data_path + '/*manifest.{0}*'.format(epoch))
    for file in all_files:
        file_rank = int(file.split('.')[-1])
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

    return (range_min, range_max, item_count)


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

    return overlapping_ssts, overlapping_mass


def get_overlapping_count_parutil(point):
    global mf_items
    return get_overlapping_count(mf_items, point)


def do_ranges_overlap(a1, b1, a2, b2):
    if a2 <= a1 <= b2:
        return True

    if a2 <= b1 <= b2:
        return True

    if a1 <= a2 <= b2:
        return True

    if a1 <= b2 <= a2:
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


def get_manifest_overlaps(data_path: str, epoch: int,
                          probe_points: List[float]) -> Tuple[int, List[int]]:
    mf_items = read_entire_manifest(data_path, epoch)
    if len(mf_items) == 0:
        return 0, []

    _, _, item_sum = get_stats(mf_items)

    print('\nReading MockDB Manifest (path: ... {0}): {1}M items'.format(
        abbrv_path(data_path),
        int(item_sum / 1e6)))

    overlap_stats = list(
            map(lambda x: get_overlapping_count(mf_items, x)[1], probe_points))
    return item_sum, overlap_stats


def worker_initialize(manifest):
    global mf_items
    #  mf_items = read_entire_manifest(data_path, epoch)
    mf_items = manifest


def gen_probe_points(pmin: float, pmax: float) -> List[float]:
    cutoffs = [
        [0, 0.1],
        [5, 0.5],
        [10, 1.0],
        [20, 2.0],
        [40, 4.0],
        [80, 8.0]
    ]

    cutoffs = sorted(cutoffs, reverse=True)
    all_points = np.array([], dtype=np.int64)

    for clim, cdel in cutoffs:
        if pmax > clim:
            cur_points = np.arange(clim, pmax, cdel)
            all_points = np.concatenate([all_points, cur_points])
            pmax = clim

    all_points = sorted(all_points)
    return all_points


def gen_overlaps(data_path: str) -> None:
    epoch = 0

    while True:
        mf_items = read_entire_manifest(data_path, epoch)
        if len(mf_items) == 0:
            break

        probe_min = min(mf_items, key = lambda x: x[0])[0]
        probe_max = max(mf_items, key = lambda x: x[1])[1]
        probe_points = gen_probe_points(probe_min, probe_max)

        _, _, item_sum = get_stats(mf_items)

        print('\nReading MockDB Manifest (path: ... {0}): {1}M items'.format(
            abbrv_path(data_path),
            int(item_sum / 1e6)))

        overlap_stats = None

        with multiprocessing.Pool(16,
                                  worker_initialize, 
                                  [mf_items]) as pool:
            point_overlaps = pool.map(get_overlapping_count_parutil, 
                                      probe_points)
            point_overlaps = map(lambda x: x[1], point_overlaps)
            overlap_stats = np.fromiter(point_overlaps, dtype=np.int64)

        overlap_pct = overlap_stats / item_sum * 100.0
        overlap_max = max(overlap_pct)
        overlap_fmt = ['{:.2f}'.format(x) for x in overlap_pct]
        print('Epoch {}, MDB Max Overlap: {:.2f}%, Avg Overlap: {:.2f}% ({} points)'.format(
            epoch, max(overlap_pct), np.mean(overlap_pct), len(probe_points)))
        epoch += 1
    pass


if __name__ == '__main__':
    gen_overlaps('.')
