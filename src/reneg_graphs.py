import math

import numpy as np
import pandas as pd

from util import VPICReader, Histogram
from reneg import Renegotiation
from typing import List, Tuple

import plotly.graph_objects as go
import plotly.express as px


def log_tailed(x, cutoff):
    if x < cutoff:
        return x

    return cutoff + math.log(1 + (x - cutoff))


def log_tailed_reverse(x, cutoff):
    if x < cutoff:
        return x

    return (math.e ** (x - cutoff)) + (cutoff - 1)


def generate_distribution_box(data_path: str, num_ranks: int,
                                 timesteps: int):
    # num_ranks = 4
    vpic_reader = VPICReader(data_path, num_ranks=num_ranks)
    fig = go.Figure()

    for tsidx in range(0, timesteps):
        data = vpic_reader.read_global(tsidx)
        print(len(data))
        plotted_data = np.random.choice(data, 50000)

        head_cutoff = 2
        tail_cutoff = 10

        head_data = len([i for i in plotted_data if i < head_cutoff])
        tail_data = len([i for i in plotted_data if i > tail_cutoff])

        percent_head = head_data * 100.0 / len(plotted_data)
        percent_tail = tail_data * 100.0 / len(plotted_data)

        print('TS {0}, < {1}: {2:.2f}'.format(tsidx, head_cutoff, percent_head))
        print('TS {0}, > {1}: {2:.2f}'.format(tsidx, tail_cutoff, percent_tail))

        plotted_data = list(map(lambda x: log_tailed(x, 10), plotted_data))

        ts_name = 'Timestep {0}'.format(vpic_reader.get_ts(tsidx), )

        box_data = go.Box(y=plotted_data, name=ts_name, line_width=1,
                          marker=dict(
                              size=4,
                              line=dict(
                                  width=0
                              )))
        fig.add_trace(box_data)

    fig.update_layout(
        title_text='Energy distribution from 4 timesteps of a VPIC simulation'
                   ' (tail is logarithmic)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 18, 2)),
            ticktext=['{0:.0f}'.format(log_tailed_reverse(x, 10))
                      for x in range(0, 18, 2)]),
    )

    # fig.show()
    fig.write_image('../vis/poster/vpic32.distrib.box.pdf')


def generate_distribution_violin(data_path: str, num_ranks: int,
                                 timesteps: int, bw_value: float):
    num_ranks = 2
    vpic_reader = VPICReader(data_path, num_ranks=num_ranks)
    fig = go.Figure()

    for tsidx in range(0, timesteps):
        data = vpic_reader.read_global(tsidx)
        print(len(data))
        # plotted_data = np.random.choice(data, 50000)
        plotted_data = np.random.choice(data, 500)

        head_cutoff = 0.5
        head_cutoff_2 = 4
        tail_cutoff = 10

        head_data = len([i for i in plotted_data if i < head_cutoff])
        head_data_2 = len([i for i in plotted_data if i < head_cutoff_2])
        tail_data = len([i for i in plotted_data if i > tail_cutoff])

        percent_head = head_data * 100.0 / len(plotted_data)
        percent_head_2 = head_data_2 * 100.0 / len(plotted_data)
        percent_tail = tail_data * 100.0 / len(plotted_data)

        print('TS {0}, < {1}: {2:.2f}'.format(tsidx, head_cutoff, percent_head))
        print('TS {0}, < {1}: {2:.2f}'.format(tsidx, head_cutoff_2, percent_head_2))
        print('TS {0}, > {1}: {2:.2f}'.format(tsidx, tail_cutoff, percent_tail))

        plotted_data = list(map(lambda x: log_tailed(x, 10), plotted_data))

        ts_name = 'Timestep {0}'.format(vpic_reader.get_ts(tsidx), )

        violin_data = go.Violin(
            y=plotted_data,
            box_visible=False, meanline_visible=False,
            name=ts_name,
            side='positive',
            points=False,
            bandwidth=bw_value,
            scalemode='width',
            line=dict(
                width=1
            )
        )

        fig.add_trace(violin_data)

    fig.update_traces(width=1.8)
    fig.update_layout(
        title_text='Energy distribution from 4 timesteps of a VPIC simulation'
                   ' (tail is logarithmic)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 18, 2)),
            ticktext=['{0:.0f}'.format(log_tailed_reverse(x, 10))
                      for x in range(0, 18, 2)]),
    )

    fig.show()
    # fig.write_image('../vis/poster/vpic32.distrib.violin.{0}.pdf'.format(bw_value))

def gen_annotation(all_shapes, all_annotations, label, x, ymin, ymax):
    ref=1
    xref = 'x%s'%(ref,)
    yref = 'y%s'%(ref,)

    x0 = str(x - 0.04)
    x1 = str(x)

    yann = (ymin + ymax) * 0.5
    ymin = str(ymin)
    ymax = str(ymax)

    all_annotations.append(dict(
        x=x + 0.2,
        y=yann,
        text=label,
        showarrow=False,
        xref=xref, yref=yref,
        textangle=270))

    line_color = px.colors.qualitative.Bold[2]

    shapes=[
        dict(type='line', xref=xref, yref=yref,
             x0=x1, y0=ymin, x1=x1, y1=ymax, line_color=line_color, line_width=1),
        dict(type='line', xref=xref, yref=yref,
             x0=x0, y0=ymin, x1=x1, y1=ymin, line_color=line_color, line_width=1),
        dict(type='line', xref=xref, yref=yref,
             x0=x0, y0=ymax, x1=x, y1=ymax, line_color=line_color, line_width=1),
    ]

    all_shapes.extend(shapes)

def generate_distribution_violin_alt(data_path: str, num_ranks: int,
                                 timesteps: int, bw_value: float):
    # num_ranks = 1
    vpic_reader = VPICReader(data_path, num_ranks=num_ranks)
    fig = go.Figure()

    all_shapes = []
    all_annotations = []

    for tsidx in range(0, timesteps):
        data = vpic_reader.read_global(tsidx)
        print(len(data))
        plotted_data = np.random.choice(data, 50000)
        # plotted_data = np.random.choice(data, 500)

        head_cutoff = 0.5
        head_cutoff_2 = 4
        tail_cutoff = 10

        head_data = len([i for i in plotted_data if i < head_cutoff])
        head_data_2 = len([i for i in plotted_data if i < head_cutoff_2])
        tail_data = len([i for i in plotted_data if i > tail_cutoff])

        percent_head = head_data * 100.0 / len(plotted_data)
        percent_head_2 = head_data_2 * 100.0 / len(plotted_data)
        percent_tail = tail_data * 100.0 / len(plotted_data)

        print('TS {0}, < {1}: {2:.2f}'.format(tsidx, head_cutoff, percent_head))
        print('TS {0}, < {1}: {2:.2f}'.format(tsidx, head_cutoff_2, percent_head_2))
        print('TS {0}, > {1}: {2:.2f}'.format(tsidx, tail_cutoff, percent_tail))

        plotted_data = list(map(lambda x: log_tailed(x, 10), plotted_data))

        ts_name = 'Timestep {0}'.format(vpic_reader.get_ts(tsidx), )

        violin_data = go.Violin(
            y=plotted_data,
            box_visible=False, meanline_visible=False,
            name=ts_name,
            side='positive',
            points=False,
            bandwidth=bw_value,
            scalemode='width',
            line=dict(
                width=1
            )
        )

        fig.add_trace(violin_data)
        label = 'Tail Mass: <br />  {0:.1f}%'.format(100 - percent_head_2)
        print(label)
        gen_annotation(all_shapes, all_annotations, label, tsidx + 0.5, 4, max(plotted_data))

    fig.update_traces(width=1.8)
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Simulation Time (s)",
            ),
            ticktext=[ "100", "950", "1900", "2850"],
            tickvals=[x + 0.2 for x in range(4)],
            color='#000',
            linecolor='#444',
        ),
        yaxis=dict(
            title=dict(
                text='Particle Energy (eV)',
            ),
            tickmode='array',
            tickvals=list(range(0, 18, 2)),
            ticktext=['{0:.0f}'.format(log_tailed_reverse(x, 10))
                      for x in range(0, 18, 2)],
            linecolor='#444',
            showgrid=True,
            gridcolor='#ddd'
        ),
        plot_bgcolor='#fff',
        shapes=all_shapes,
        annotations=all_annotations,
        showlegend=False,
        legend=dict(x=0.87,y=1),
        font=dict(size=18)
    )

    # fig.show()
    fig.write_image('../vis/poster/vpic32.distrib.violin.alt.{0}.pdf'.format(bw_value))


def gen_skew_runtime_util(fig, data, num_ranks, percent):
    col_x = data['skewpct'].to_numpy()


    col_xrev = col_x[::-1]
    col_mean = data['runtime']['mean'].to_numpy()
    col_min = data['runtime']['min'].to_numpy()
    col_max = data['runtime']['max'].to_numpy()
    col_min = col_min[::-1]

    col_hash = lambda x: (x * (x - 1)) ^ (x % 3) % 657
    color_idx = col_hash(num_ranks) % (len(px.colors.qualitative.Dark2))

    line_color = px.colors.qualitative.Dark2[color_idx]
    band_color = px.colors.qualitative.Pastel2[color_idx]
    band_color = band_color[:-1] + ',0.0)'
    print(line_color, band_color)

    main_line = go.Scatter(x=col_x, y=col_mean, mode='lines',
                           line=dict(color=line_color),
                           name='{0} ranks ({1}%)'.format(num_ranks, percent))
    main_band = go.Scatter(x=np.concatenate([col_x, col_xrev]),
                           y=np.concatenate([col_max, col_min]), fill='tozerox',
                           fillcolor=band_color,
                           line=dict(color='rgba(255,255,255,0)'),
                           name='{0} ranks ({1}%)'.format(num_ranks, percent),
                           showlegend=False)

    # fig.add_traces([main_line])
    return main_line, main_band


def gen_skew_runtime():
    csv_file = '../vis/poster/skewruntimejun25.csv'

    data = pd.read_csv(csv_file)

    fig = go.Figure()
    all_lines = []
    all_bands = []

    for skewnodecnt in data['skewnodecnt'].unique():
        print(skewnodecnt)

        plot_data = data[data['skewnodecnt'] == skewnodecnt]
        skewnodepct = plot_data['skewnodepct'].iloc[0]
        print(plot_data)

        aggr_data = plot_data.groupby('skewpct').agg({
            'runtime': ['min', 'max', 'mean'],
        })
        aggr_data['skewpct'] = aggr_data.index
        # print(aggr_data)

        # skewnodepct = 10 if 26 else 20

        line, band = gen_skew_runtime_util(fig, aggr_data, skewnodecnt, skewnodepct)
        all_lines.append(line)
        all_bands.append(band)

    # fig.add_traces(all_bands)
    fig.add_traces(all_lines)

    fig.update_layout(title_text='Change in runtime as workload skew is varied',
                      yaxis=dict(
                          range=[90,130],
                          title='Time (seconds)'
                      ),
                      xaxis=dict(
                          title='Additional workload for skewed ranks (%)'
                      ))
    # fig.write_image('../vis/poster/runtimeskew.pdf')
    fig.show()
    return


if __name__ == '__main__':
    num_ranks = 32
    timesteps = 4
    data = '../data'
    #
    # # for bw_value in [0.0001, 0.001, 0.01, 0.1]:
    #
    bw_value = 0.03
    # generate_distribution_violin_alt(data, num_ranks, timesteps, bw_value)
    # generate_distribution_box(data, num_ranks, timesteps)
    gen_skew_runtime()

