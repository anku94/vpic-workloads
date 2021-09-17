from reneg_graphs import generate_distribution_violin_alt_2
import argparse
import sys


def run(data: str, image: str):
    generate_distribution_violin_alt_2(data, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='CARP Plotter',
        description='''
        Expose reusable CARP plotting options
        ''',
        epilog='''
        Dependencies: pyplot, numpy, plotly
        '''
    )

    parser.add_argument('--violin', action='store_true',
                        help='Plot Violin', default=False)
    parser.add_argument('-d', '--data', type=str, help='VPIC Trace Data')
    parser.add_argument('-o', '--output-image', type=str,
                        help='Filename for plot')

    options = parser.parse_args()
    if not options.violin or not options.output_image:
        parser.print_help()
        sys.exit(0)

    if options.violin:
        run(options.data, options.output_image)
