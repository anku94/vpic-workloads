import glob
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter



def read_rtp_trace_file(fpath):
    fdata = open(fpath).read().splitlines()
    fdata = [l.strip() for l in fdata if l.startswith('RENEG')]
    num_rounds = len(fdata)
    rank_id = re.findall('(\d+)@', fdata[0])[0]
    print(rank_id)

    data = []

    for line in fdata:
        line = line.split(':')[1]
        line = filter(lambda x: x.strip() != '', line.split(' '))
        line = list(map(lambda x: float(x), line))
        data.append(line)

    data = np.array(data)
    return (rank_id, data)


def read_rtp_trace(trace_path):
    data = None
    perf_files = glob.glob(trace_path + '/*perfstats*')
    num_ranks = len(perf_files)

    all_data = map(lambda x: read_rtp_trace_file(x)[1], perf_files)
    all_data = list(all_data)

    aggr_data = np.sum(all_data, 0)
    cleaned_data = np.delete(aggr_data, (0), axis=0)
    cleaned_mean = np.mean(cleaned_data, 1)

    norm_data = cleaned_data / cleaned_mean[:, None]
    norm_std = np.std(norm_data, 1)
    norm_max = np.max(norm_data, 1)
    norm_min = np.min(norm_data, 1)

    round_sum = np.sum(cleaned_data, 1)

    num_rounds = len(cleaned_data)

    x_vals = range(num_rounds)
    y_std = norm_std
    y_min = norm_min
    y_max = norm_max

    return (x_vals, y_std, y_min, y_max, round_sum)


def plot_rtp_trace(x_vals, y_std, y_min, y_max, round_sum):
    print(y_min)
    print(y_max)

    fig, ax = plt.subplots(1, 1)
    plt.ticklabel_format(useOffset=False)

    ax.plot(x_vals, y_std, label = 'std (%)')
    ax.fill_between(x_vals, y_min, y_max, alpha=0.3, interpolate=True, label = 'min/max')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax.plot([0, 120], [0.2, 0.2])

    plt.yticks([0.05, 0.2, 1, 10], ['5%', '20%', '100%', '1000%'])

    plt.legend(loc='lower left')

    ax2 = ax.twinx()

    global TITLE
    plt.title(TITLE)
    ax.set_xlabel('Renegotiation Round Number')
    ax.set_ylabel('Percentage (Std/Min/Max)')
    ax2.set_ylabel('Load per round')

    ax2.bar(x_vals, round_sum, color='red', alpha=0.2)

    # ax.plot(x_vals, y_min)
    # ax.plot(x_vals, y_max)

    # fig.show()
    base_path = '../vis/rtp_traces/'
    fig.savefig(base_path + TITLE.lower() + '.pdf')


if __name__ == '__main__':
    data_path = '../rundata/rtp_traces/*'
    all_traces = glob.glob(data_path)
    for trace in all_traces:
        data = read_rtp_trace(trace)
        global TITLE
        TITLE = trace.split('/')[-1].upper()
        plot_rtp_trace(*data)
