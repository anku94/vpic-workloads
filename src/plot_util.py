import glob

import numpy as np
import pandas as pd

from typing import List, Tuple
from util import Histogram, VPICReader


def get_hist():
    pass


def get_points(dir: str, tsidx: int, npts: int = 99) -> List[float]:
    hists = glob.glob(dir + '/hist_trace/hist.data.*')
    hists = sorted(hists, key=lambda x: int(x.split('.')[-1]))
    hist = hists[tsidx]
    hist_data = np.load(hist, allow_pickle=True)
    _, hist_bins = np.histogram(range(1000), bins=100000, range=(0, 1000))

    hobj = Histogram(bin_edges=hist_bins, bin_weights=hist_data)
    hobj.rebalance(npts)

    return hobj.bin_edges


def get_points_2(dir: str, tsidx: int, npts: int = 99) -> List[float]:
    hists = glob.glob(dir + '/T*')
    hists = sorted(hists, key=lambda x: int(x.split('.')[-1]))
    hist = hists[tsidx]
    hist_data = np.load(hist + '/hist', allow_pickle=True)
    _, hist_bins = np.histogram(range(1000), bins=100000, range=(0, 1000))

    hobj = Histogram(bin_edges=hist_bins, bin_weights=hist_data)
    hobj.rebalance(npts)

    return hobj.bin_edges


def get_predictive_power(dir: str) -> Tuple[List[str], List[float]]:
    reader = VPICReader(dir)
    all_ts = [reader.get_ts(i) for i in range(reader.get_num_ts())]

    points = get_points_2(dir, 0, 512)

    all_olap_pct = []

    for epoch in range(len(all_ts)):
        data = reader.sample_hist(epoch)
        # print(points)
        # print(len(points))

        counts, edges = np.histogram(data, bins=points)
        max_count = max(counts)
        sum_count = sum(counts)
        print(np.median(counts) * 100.0 / sum_count)
        all_olap_pct.append(max_count * 100.0 / sum_count)

    print(all_olap_pct)
    return (all_ts, all_olap_pct)


def read_manifest(fpath, nranks):
    all_fpaths = [fpath.format(i) for i in range(nranks)]
    manifest_cols = ['epoch', 'offset', 'obsbeg', 'obsend', 'expbeg', 'expend',
                     'masstot', 'massoob', 'renegidx']
    all_dfs = [pd.read_csv(f, names=manifest_cols, header=None) for f in
               all_fpaths]
    manifest = pd.concat(all_dfs)
    return manifest


def get_mf_olap(manifest, point: float):
    matches = manifest[
        (manifest['obsbeg'] <= point) & (manifest['obsend'] >= point)]
    return matches['masstot'].sum()


def analyze_manifest(dir: str, hist_dir: str, nranks: int):
    mfpath = dir + '/vpic-manifest.{0}'
    manifest = read_manifest(mfpath, nranks)
    epochs = manifest['epoch'].unique()
    ptiles = [50, 75, 99]
    all_epoch_olaps = []
    for epoch in epochs:
        mf_epoch = manifest[manifest['epoch'] == epoch]
        mass_epoch = mf_epoch['masstot'].sum()
        epoch_points = get_points(hist_dir, epoch)
        epoch_olaps = [get_mf_olap(mf_epoch, point) for point in epoch_points]
        epoch_olaps = sorted(epoch_olaps)
        ptile_olaps = []
        for ptile in ptiles:
            ptile_olap_idx = int(ptile * len(epoch_olaps) / 100.0)
            ptile_olap = epoch_olaps[ptile_olap_idx] * 100.0 / mass_epoch
            ptile_olaps.append(ptile_olap)
        all_epoch_olaps.append(ptile_olaps)

    all_epoch_olaps = np.array(all_epoch_olaps)
    all_epoch_olaps = np.transpose(all_epoch_olaps)
    for olaps in all_epoch_olaps:
        print(olaps)

    # olap_means = np.mean(all_epoch_olaps, axis=1)
    # print(olap_means)
    return all_epoch_olaps


def adhoc_analysis():
    out_dir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/subpart_exps'
    intvls = [62500, 125000, 250000, 500000, 750000, 1000000, 'everyepoch']
    intvls = [ 250000 ]
    for intvl in intvls:
        hist_dir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2'
        dir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/runs.big.2/run.{0}.{1}/carp_P3584M_intvl{0}/plfs/particle'
        dir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/subpart_exps/runs.qlat.tar/run.{0}.{1}/carp_P3584M_intvl{0}/plfs/particle'
        for idx in range(4, 16):
            dpath = dir.format(intvl, idx)
            olaps = analyze_manifest(dpath, hist_dir, 512)
            print(olaps)
            for pidx, ptile in enumerate([50, 75, 99]):
                with open('{0}/olaps.{1}.csv'.format(out_dir, ptile),
                          'a+') as f:
                    oout = olaps[pidx]
                    for epoch, data in enumerate(oout):
                        f.write(
                            '{0},{1},{2},{3}\n'.format(intvl, idx, epoch, data))


def get_olaps(intvl, count):
    hist_dir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2'
    dir = '/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/runs.big.2/run.{0}.{1}/carp_P3584M_intvl{0}/plfs/particle'

    all_olap_means = []
    for idx in range(1, count + 1):
        dpath = dir.format(intvl, idx)
        olap_means = analyze_manifest(dpath, hist_dir, 512)
        all_olap_means.append(olap_means)

    all_olap_means = np.array(all_olap_means)
    res = np.mean(all_olap_means, axis=0)
    # res = res.mean(axis=1) # avg all epochs
    return res


if __name__ == '__main__':
    # olaps = get_olaps(62500, 2)
    adhoc_analysis()
    # dir = '/Users/schwifty/Repos/workloads/data/eval_trace'
    # get_predictive_power(dir)
