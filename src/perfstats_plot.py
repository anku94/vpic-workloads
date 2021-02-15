import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import os, re
import glob


def strip_perfdata(raw_data):
    header = []
    data = []
    for elem in raw_data:
        elem_new = re.sub('MANIFEST_ANALYTICS_E\d+_', '', elem)
        elem_new = elem_new.split(',')[-2:]
        header.append(elem_new[-2])
        data.append(elem_new[-1])

    return header, data


def read_perflog(fpath, hdr_pref, data_pref):
    data = open(fpath).read().split('\n')
    data = [i for i in data if 'MANIFEST_ANALYTICS_E' in i]

    epoch = 0

    hdr = None
    all_data = []

    while True:
        epoch_pref = 'MANIFEST_ANALYTICS_E{0}'.format(epoch)
        epoch_data = [i for i in data if epoch_pref in i]
        if len(epoch_data) == 0: break

        stat_hdr, stat_data = strip_perfdata(epoch_data)

        if hdr == None:
            hdr = hdr_pref + ['EPOCH_IDX'] + stat_hdr
        all_data.append(data_pref + [str(epoch)] + stat_data)
        epoch += 1

    return hdr, all_data


def gen_csv():
    intvl_arr = 250000 * np.array(range(1, 5))
    intvl_arr = [ 2 * 10**7 ]
    pvtcnt_arr = np.array([64, 128, 256])
    pvtcnt_arr = np.array([ 256 ])
    print(intvl_arr)
    basedir = '../rundata/e2e_reg_intvl_att3/intvl{0}.pvtcnt{1}/vpic-perfstats.log.0'
    basedir = '../rundata/bigcarp_rel_paramsweep_e1toe7/intvl{0}.pvtcnt{1}/vpic-perfstats.log.0'
    basedir = '../rundata/bigcarp_rel_oneneg_oob/vpic-perfstats.log.0.oob4k'

    header = None
    all_data = []

    for intvl in intvl_arr:
        for pvtcnt in pvtcnt_arr:
            fullpath = basedir.format(intvl, pvtcnt)
            print(fullpath)
            if not os.path.exists(fullpath): continue
            print(fullpath)
            cur_header, cur_data = read_perflog(fullpath, ['RENEG_INTERVAL',
                                                           'RENEG_PVTCNT'],
                                                [str(intvl), str(pvtcnt)])
            if header == None:
                header = cur_header
            all_data += cur_data

    data_csv = open('../rundata/bigcarp_rel_oneneg_oob/data.csv', 'w')
    data_csv.write(','.join(header) + '\n')
    for line in all_data:
        data_csv.write(','.join(line) + '\n')
    data_csv.close()

def gen_csv_2():
    intvl_arr = 250000 * np.array(range(1, 5))
    pvtcnt_arr = np.array([64, 128, 256])
    basedir = '../rundata/e2e_reg_intvl_att3/intvl{0}.pvtcnt{1}/vpic-perfstats.log.0'

    header = None
    all_data = []

    for intvl in intvl_arr:
        for pvtcnt in pvtcnt_arr:
            fullpath = basedir.format(intvl, pvtcnt)
            cur_header, cur_data = read_perflog(fullpath, ['RENEG_INTERVAL',
                                                           'RENEG_PVTCNT'],
                                                [str(intvl), str(pvtcnt)])
            if header == None:
                header = cur_header
            all_data += cur_data

    data_csv = open('../rundata/e2e.perfstats.csv', 'w+')
    data_csv.write(','.join(header) + '\n')
    for line in all_data:
        data_csv.write(','.join(line) + '\n')
    data_csv.close()

def gen_runtime_csv():
    basedir = '../rundata/bigcarp_rel_paramsweep_e1toe7'
    data = glob.glob(basedir + '/**/vpic-perfstats.log.0');
    print(data)

    csvpath = basedir + '/runtime.csv'
    df = pd.DataFrame()

    for file in data:
        params = re.findall('intvl(\d+).pvtcnt(\d+)', file)[0]
        rintvl, pvtcnt = params
        fdata = open(file).readlines()[:-10]
        fdata = [line for line in fdata if not line.startswith('0,')]
        max_ts = fdata[-1].split(',')[0]
        # rintvl = int(rintvl)
        max_ts = int(max_ts)
        data = {'rnum': int(rintvl), 'rintvl': rintvl, 'pvtcnt': pvtcnt, 'runtime': max_ts}
        print(data)
        if max_ts < 1e6: continue
        df = df.append(data, ignore_index=True)

    print(df)

    df = df.groupby('rintvl', as_index=False).mean()
    runtime_den = 60 * 1000
    df['runtime'] /= runtime_den

    df.sort_values('rnum', inplace=True)
    print(df)

    fig, ax = plt.subplots(1, 1)
    ax.plot(df['rintvl'], df['runtime'])
    ax.set_ylim([70, 130])
    ax.set_ylabel('Time (minutes)')
    ax.set_xlabel('Renegotiation Interval')
    ax.set_title('Reneg Interval vs Runtime (preliminary)')
    fig.savefig('../vis/bigcarp/runtime_pre.pdf', dpi=300)

def plot_pvtcnt():
    all_data = pd.read_csv('../rundata/bigcarp.e17.perfstats.csv')

    for intvl in all_data['RENEG_INTERVAL'].unique():
        fig, ax = plt.subplots(1, 1)

        data = all_data[all_data['RENEG_INTERVAL'] == intvl]
        for pvtcnt in data['RENEG_PVTCNT'].unique():
            cur_data = data[data['RENEG_PVTCNT'] == pvtcnt]
            cur_data = cur_data.groupby('EPOCH_IDX', as_index=False).mean()
            ax.plot(cur_data['EPOCH_IDX'], cur_data['OLAP_FRACPCT'],
                    label='Pivot Count {0}'.format(pvtcnt))
            print(cur_data)
            print(pvtcnt)


        base_overlap = 100.0/512
        ax.plot([0, 6], [base_overlap, base_overlap], 'r--')
        ax.set_xlabel('Epoch Index')
        ax.set_ylabel('Overlap Percent')
        ax.set_title('Partitioning Quality vs Epoch as f(pvtcnt) (intvl={0})'.format(intvl))
        ax.legend()
        # fig.show()
        # break
        fig.savefig('../vis/bigcarp/pvtcnt_vs_olap.rintvl{0}.pdf'.format(intvl), dpi=600)

def plot_intvl():
    # all_data = pd.read_csv('../rundata/bigcarp.e17.perfstats.csv')
    all_data = pd.read_csv('../rundata/bigcarp_rel_oneneg_oob/bigcarp.e17.perfstats.csv')

    param_filter = 'RENEG_PVTCNT'
    param_group = 'RENEG_INTERVAL'

    # for pvtcnt in all_data[param_filter].unique():
    for pvtcnt in [256]:
        fig, ax = plt.subplots(1, 1)

        data = all_data[all_data[param_filter] == pvtcnt]
        for intvl in data[param_group].unique():
            cur_data = data[data[param_group] == intvl]
            cur_data = cur_data.groupby('EPOCH_IDX', as_index=False).mean()
            ax.plot(cur_data['EPOCH_IDX'], cur_data['OLAP_FRACPCT'],
                    label='Reneg Interval {0}'.format(intvl))
            print(cur_data)
            print(pvtcnt)


        base_overlap = 100.0/512
        ax.plot([0, 6], [base_overlap, base_overlap], 'r--')
        ax.set_xlabel('Epoch Index')
        ax.set_ylabel('Overlap Percent')
        ax.set_title('Partitioning Quality vs Epoch as f(frequency) (pvtcnt={0})'.format(pvtcnt))
        ax.legend()
        # fig.show()
        # break
        fig.savefig('../vis/bigcarp/pvtcnt_vs_rintvl.pvtcnt{0}.extreme.oobsz.pdf'.format(pvtcnt), dpi=600)

def plot_real_vpic():
    data = [0.2631, 0.2626, 0.2626, 0.2628, 0.2626]
    pvtcnt = 256
    intvl = 500000
    partcnt=2e9

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(data)), data)
    base_overlap = 100.0 / 512
    ax.plot([0, 4], [base_overlap, base_overlap], 'r--')
    ax.set_ylim([0, 0.3])
    ax.set_title('Real VPIC, pvtcnt 256, intvl 500k, ppn 4M')
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Overlap Percent')
    fig.savefig('../vis/e2e/realvpic.pdf')

def plot_bigcarp_renegintvl():
    intervals = ['250K', '500K', '750K']

    all_data = pd.read_csv('../rundata/bigcarp.perfstats.csv')
    data = all_data[all_data['RENEG_PVTCNT'] == 128]

    olap_e0 = data[data['EPOCH_IDX'] == 0]['OLAP_FRACPCT']
    olap_e3 = data[data['EPOCH_IDX'] == 3]['OLAP_FRACPCT']

    print(olap_e0)
    print(olap_e3)

    fig, ax = plt.subplots(1, 1)
    ax.plot(intervals, olap_e0, label='Epoch 800')
    ax.plot(intervals, olap_e3, label='Epoch 3200')

    ax.set_ylim(0, 0.5)
    ax.set_xlabel('Reneg Frequency')
    ax.set_ylabel('Overlap Percent')
    ax.set_title(
        'BigCARP Partitioning Quality vs Reneg Freq (pvtcnt=128)')
    ax.legend()
    # fig.show()
    fig.savefig('../vis/bigcarp/reneg_intvl.pdf', dpi=300)

def plot_bigcarp_pvtcnt():
    pvtcnts = ['64', '128', '256']

    all_data = pd.read_csv('../rundata/bigcarp.e17.perfstats.csv')
    data = all_data[all_data['RENEG_INTERVAL'] == 750000]

    olap_e0 = data[data['EPOCH_IDX'] == 0]['OLAP_FRACPCT']
    olap_e3 = data[data['EPOCH_IDX'] == 3]['OLAP_FRACPCT']

    print(olap_e0)
    print(olap_e3)

    fig, ax = plt.subplots(1, 1)
    ax.plot(pvtcnts, olap_e0, label='Epoch 800')
    ax.plot(pvtcnts, olap_e3, label='Epoch 3200')

    ax.set_ylim(0, 0.5)
    ax.set_xlabel('Pivot Count')
    ax.set_ylabel('Overlap Percent')
    ax.set_title(
        'BigCARP Partitioning Quality vs Pivot Count (Freq=750K)')
    ax.legend()
    fig.show()
    # fig.savefig('../vis/bigcarp/reneg_pvtcnt.pdf', dpi=300)

if __name__ == '__main__':
    # gen_csv()
    # plot_bigcarp_renegintvl()
    # plot_bigcarp_pvtcnt()
    # plot_pvtcnt()
    # plot_intvl()
    # plot_real_vpic()
    gen_runtime_csv()
