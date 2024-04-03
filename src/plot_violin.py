import os

from reneg_graphs import (
    generate_distribution_violin_alt,
    log_tailed,
    log_tailed_reverse,
)
import argparse
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from util import VPICReader


"""
20240329: Generated for SC24, combines both plots into one
Derived from reneg_graphs/generate_distribution_violin_alt
"""


def generate_distribution_violin_merged(
    data_path: str, fig_path: str, num_ranks: int = None, bw_value: float = 0.03
):
    vpic_reader = VPICReader(data_path, num_ranks=num_ranks)
    ranks = vpic_reader.get_num_ranks()
    timesteps = vpic_reader.get_num_ts()
    print("[VPICReader] Ranks: {0}, ts: {1}".format(ranks, timesteps))

    fig = go.Figure()

    all_shapes = []
    all_annotations = []

    timestamps = [vpic_reader.get_ts(i) for i in range(vpic_reader.get_num_ts())]

    # fig = make_subplots(
    #     rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True, vertical_spacing=0.02
    # )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    tail_masses = []

    leftcolor = "rgba(0, 40, 40, 1.0)"
    leftcolor_light = "rgba(0, 100, 80, 0.5)"
    rightcolor = "rgba(80, 0, 0, 0.9)"
    annotcolor = "rgba(100, 100, 100, 0.9)"
    annottxtcolor = "rgba(50, 50, 50, 1.0)"

    for tsidx in range(0, timesteps):
        # data = vpic_reader.sample_global(tsidx, load_cached=True)
        # data = vpic_reader.sample_hist(tsidx)
        data = vpic_reader.read_sample(tsidx)

        print("Read: ", len(data))
        plotted_data = data

        head_cutoff = 0.5
        head_cutoff_2 = 4
        tail_cutoff = 10

        head_data = len([i for i in plotted_data if i < head_cutoff])
        head_data_2 = len([i for i in plotted_data if i < head_cutoff_2])
        tail_data = len([i for i in plotted_data if i > tail_cutoff])

        percent_head = head_data * 100.0 / len(plotted_data)
        percent_head_2 = head_data_2 * 100.0 / len(plotted_data)
        percent_tail = tail_data * 100.0 / len(plotted_data)
        tail_masses.append(100 - percent_head_2)

        print("TS {0}, < {1}: {2:.2f}".format(tsidx, head_cutoff, percent_head))
        print("TS {0}, < {1}: {2:.2f}".format(tsidx, head_cutoff_2, percent_head_2))
        print("TS {0}, > {1}: {2:.2f}".format(tsidx, tail_cutoff, percent_tail))

        plotted_data = list(map(lambda x: log_tailed(x, 10), plotted_data))

        ts_name = "Timestep {0}".format(
            vpic_reader.get_ts(tsidx),
        )

        violin_data = go.Violin(
            x0=tsidx,
            y=plotted_data,
            box_visible=False,
            meanline_visible=False,
            name=ts_name,
            side="positive",
            points=False,
            bandwidth=bw_value,
            scalemode="width",
            line=dict(width=8, color=leftcolor),
            fillcolor=leftcolor_light,
            showlegend=False,
        )

        fig.add_trace(violin_data, row=1, col=1)
        label = "Tail Mass: <br />  {0:.1f}%".format(100 - percent_head_2)
        print(label)
        # gen_annotation(all_shapes, all_annotations, label, tsidx + 0.1, 4,
        #                max(plotted_data))

    fig.update_traces(width=1.4)
    ytickvals = [0, 2, 4, 6, 8, 10, 12, 14, 15.9687]
    gridcolor = "#777"
    gridcolor = "#bbb"
    gridwidth = 2
    axislinewidth = 3
    fig.update_layout(
        # xaxis=dict(
        #     title=dict(
        #         text="Simulation Time (s)",
        #     ),
        #     ticktext=timestamps,
        #     tickvals=[x + 0.2 for x in range(timesteps)],
        #     color='#000',
        #     linecolor='#444',
        # ),
        xaxis=dict(
            title=dict(text="Simulation Timestep", standoff=30),
            ticktext=timestamps,
            tickvals=[x + 0.2 for x in range(timesteps)],
            color="#000",
            title_font=dict(color="#222"),
            tickfont=dict(color="#222"),
            linecolor="#444",
            linewidth=axislinewidth,
            range=[-0.5, timesteps + 0.3],
            showgrid=False,
            gridcolor=gridcolor,
            gridwidth=gridwidth,
        ),
        yaxis=dict(
            title=dict(text="Energy (γ)"),
            tickmode="array",
            # tickvals=list(range(0, 18, 2)),
            tickvals=ytickvals,
            ticktext=["{0:.0f}".format(log_tailed_reverse(x, 10)) for x in ytickvals],
            linecolor="#444",
            linewidth=axislinewidth,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=gridwidth,
            range=[-0.3, 18],
            title_font=dict(color=leftcolor),
            tickfont=dict(color=leftcolor),
        ),
        yaxis2=dict(
            title=dict(text="% of all data (tail)", standoff=50),
            linecolor="#777",
            linewidth=axislinewidth,
            showgrid=False,
            gridcolor="red",
            gridwidth=gridwidth,
            tickvals=list(np.arange(0, 0.32, 0.035)),
            tickformat=".0%",
            range=[0.0, 0.32],
            title_font=dict(color=rightcolor),
            tickfont=dict(color=rightcolor),
        ),
        plot_bgcolor="#fff",
        shapes=all_shapes,
        annotations=all_annotations,
        showlegend=True,
        legend=dict(x=0.04, y=1, bordercolor="#666", borderwidth=2),
        font=dict(size=52),
    )

    x2 = [x + 0.2 for x in range(timesteps)]
    y2 = np.array(tail_masses) / 100.0

    fig.add_trace(
        go.Scatter(
            x=x2,
            y=y2,
            yaxis="y2",
            line=dict(color=rightcolor, width=8),
            mode="lines+markers",
            marker=dict(size=42),
            name="Tail",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # fig.add_shape(
    #     type="rect",
    #     xref="x",
    #     yref="y",
    #     x0=0,
    #     y0=4,
    #     x1=7.3,
    #     y1=20,
    #     fillcolor="purple",
    #     row=1,
    #     col=1,
    #     opacity=0.12,
    # )

    fig.add_shape(
        type="line",
        x0=-2,
        y0=4,
        x1=7.3,
        y1=4,
        line=dict(
            color=annotcolor,
            width=12,
            dash="dot",
        ),
    )

    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    fig.add_annotation(
        x=-0.3,
        y=4,
        text="Cutoff (Tail)",
        font=dict(
            color=annottxtcolor,
            size=44,
        ),
        arrowwidth=5,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor=annotcolor,
        ax=120,
        ay=-140,
        bgcolor="#fff",
    )

    pio.write_image(fig, fig_path, width=2048, height=768)


def run(data: str, image: str):
    generate_distribution_violin_merged(data, image)


if __name__ == "__main__":
    data = "~/Repos/vpic-workloads/data/toplot"
    image = "~/Repos/carp-paper/figures/distrib.vpic.pdf"
    data = os.path.expanduser(data)
    image = os.path.expanduser(image)

    run(data, image)
