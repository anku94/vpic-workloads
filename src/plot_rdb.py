import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import argparse
import sys
from pathlib import Path
import re

from common import abbrv_path


def plot_epoch(path_csv: str, path_plot: str, ax_aggr=None,
               plot_olap=False) -> None:
    data = pd.read_csv(path_csv)
    epoch = data['Epoch'][0]
    data_x = data['Point']
    data_y = data['MatchMass'] * 100.0 / data['TotalMass']

    print('Epoch {:d}, Max RDB: {:.3f}%'.format(epoch, max(data_y)))

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_y, label='Epoch {0}: Overlap %'.format(epoch))

    if ax_aggr:
        ax_aggr.plot(data_x, data_y,
                     label='Epoch {0}: RDB Overlap %'.format(epoch))

    if plot_olap:
        ax2 = ax.twinx()
        data_olap = data['MatchCount']
        ax2.plot(data_x, data_olap,
                 label='Epoch {0}: Overlap Count'.format(epoch))

    path_rtp_csv = re.sub('rdb\.olap', 'rtp.olap', path_csv)
    path_rtp_csv = Path(path_rtp_csv)

    if path_rtp_csv.exists():
        data = pd.read_csv(path_rtp_csv)
        data_x = data['Point']
        data_y = data['MatchMass'] * 100.0 / data['TotalMass']
        print('Epoch {:d}, Max RTP: {:.3f}%'.format(epoch, max(data_y)))
        ax.plot(data_x, data_y, label='Epoch {0}: RTP Overlap %'.format(epoch))

    path_mdb_csv = re.sub('rdb\.olap', 'mdb.olap', path_csv)
    path_mdb_csv = Path(path_mdb_csv)

    if path_mdb_csv.exists():
        data = pd.read_csv(path_mdb_csv)
        data_x = data['Point']
        data_y = data['MatchMass'] * 100.0 / data['TotalMass']
        print('Epoch {:d}, Max MDB: {:.3f}%'.format(epoch, max(data_y)))
        ax.plot(data_x, data_y, label='Epoch {0}: MDB Overlap %'.format(epoch))

    ax.legend()
    ax.set_xlabel('Attribute Range')
    ax.set_ylabel('Overlap Percent')
    ax.set_title('Manifest Stats for Epoch {0}'.format(epoch))

    # fig.show()
    fig.savefig(path_plot, dpi=300)
    #  print('Plot saved: ', abbrv_path(path_plot))


def run(path_in: str, path_out: str) -> None:
    print(path_in)
    desc = """
    [plot_rdb] parsing RDB to compute RDB overlaps
    """
    print(desc)
    manifest_files = glob.glob(path_in + '/rdb.olap.e*.csv')
    print('{0} epoch files found\n'.format(len(manifest_files)))
    manifest_files.sort()

    fig_aggr, ax_aggr = plt.subplots(1, 1)

    for fin in manifest_files:
        fout = re.sub('csv$', 'pdf', Path(fin).name)
        fout = Path(path_out) / fout
        plot_epoch(fin, fout, ax_aggr=ax_aggr)
        print('Plot saved: {0} ==> {1}\n'.format(abbrv_path(fin), abbrv_path(fout)))

    ax_aggr.set_xlabel('Attribute Range')
    ax_aggr.set_ylabel('Overlap Percent')
    ax_aggr.set_title('Aggregate Manifest Stats')
    ax_aggr.legend()

    # fig_aggr.show()
    aggr_out = Path(path_out) / 'manifest.aggr.pdf'
    fig_aggr.savefig(aggr_out, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Manifest Plotter',
        description='''
        Plot manifest CSVs generated by the RDB reader''',
        epilog='''
        TBD
        '''
    )

    parser.add_argument('--input-path', '-i', type=str,
                        help='Path to CSV input files', required=True)
    parser.add_argument('--output-path', '-o', type=str,
                        help='Destination for plotted graphs', required=True)

    options = parser.parse_args()
    if not options.input_path or not options.output_path:
        parser.print_help()
        sys.exit(0)

    run(options.input_path, options.output_path)
