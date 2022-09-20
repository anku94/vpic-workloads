import glob
import matplotlib.pyplot as plt
import pandas as pd
from common import plot_init

def plot_size():
    plot_dir = '/Users/schwifty/Repos/workloads/rundata/20220825-pvtcnt-analysis'
    all_sizes = glob.glob(plot_dir + '/sizes*txt')
    ts_func = lambda x: int(x.split('.')[-2])
    all_sizes = sorted(all_sizes, key=ts_func)

    fig, ax = plt.subplots(1, 1)

    fig_path = '{}/sizes.pdf'.format(plot_dir)

    data_x = None
    for size_file in all_sizes:
        print(size_file)
        data = pd.read_csv(size_file, names=['size'])
        data_x = range(len(data))
        data_y = data['size'].astype('float')
        ax.plot(data_x, data['size'], label='TS {}'.format(ts_func(size_file)))

    xmax = data_x[-1]
    ax.plot([0, xmax], [26, 26], linestyle='--', label='Mean')

    # ax.set_yticks([0, 5, 25])
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}M'.format(x/4.0))
    ax.set_xlabel('Rank ID')
    ax.set_ylabel('Number Of Particles Dumped')
    ax.set_title('Particles Emitted Per Rank')
    ax.legend()
    # fig.show()
    fig.savefig(fig_path, dpi=200)


def run():
    plot_init()
    plot_size()
    pass

if __name__ == '__main__':
    run()