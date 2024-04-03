import hashlib
import math

import numpy as np
import pandas as pd
import sys

from util import VPICReader, Histogram
from typing import List, Tuple

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots


def log_tailed(x, cutoff):
    if x < cutoff:
        return x

    return cutoff + math.log(1 + (x - cutoff))


def log_tailed_reverse(x, cutoff):
    if x < cutoff:
        return x

    return (math.e ** (x - cutoff)) + (cutoff - 1)


def generate_distribution_box(data_path: str, num_ranks: int, timesteps: int):
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

        print("TS {0}, < {1}: {2:.2f}".format(tsidx, head_cutoff, percent_head))
        print("TS {0}, > {1}: {2:.2f}".format(tsidx, tail_cutoff, percent_tail))

        plotted_data = list(map(lambda x: log_tailed(x, 10), plotted_data))

        ts_name = "Timestep {0}".format(
            vpic_reader.get_ts(tsidx),
        )

        box_data = go.Box(
            y=plotted_data,
            name=ts_name,
            line_width=1,
            marker=dict(size=4, line=dict(width=0)),
        )
        fig.add_trace(box_data)

    fig.update_layout(
        title_text="Energy distribution from 4 timesteps of a VPIC simulation"
        " (tail is logarithmic)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 18, 2)),
            ticktext=[
                "{0:.0f}".format(log_tailed_reverse(x, 10)) for x in range(0, 18, 2)
            ],
        ),
    )

    # fig.show()
    fig.write_image("../vis/poster/vpic32.distrib.box.pdf")


def generate_distribution_violin(
    data_path: str, num_ranks: int, timesteps: int, bw_value: float
):
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

        print("TS {0}, < {1}: {2:.2f}".format(tsidx, head_cutoff, percent_head))
        print("TS {0}, < {1}: {2:.2f}".format(tsidx, head_cutoff_2, percent_head_2))
        print("TS {0}, > {1}: {2:.2f}".format(tsidx, tail_cutoff, percent_tail))

        plotted_data = list(map(lambda x: log_tailed(x, 10), plotted_data))

        ts_name = "Timestep {0}".format(
            vpic_reader.get_ts(tsidx),
        )

        violin_data = go.Violin(
            y=plotted_data,
            box_visible=False,
            meanline_visible=False,
            name=ts_name,
            side="positive",
            points=False,
            bandwidth=bw_value,
            scalemode="width",
            line=dict(width=1),
        )

        fig.add_trace(violin_data)

    fig.update_traces(width=1.8)
    fig.update_layout(
        title_text="Energy distribution from 4 timesteps of a VPIC simulation"
        " (tail is logarithmic)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 18, 2)),
            ticktext=[
                "{0:.0f}".format(log_tailed_reverse(x, 10)) for x in range(0, 18, 2)
            ],
        ),
    )

    fig.show()
    # fig.write_image('../vis/poster/vpic32.distrib.violin.{0}.pdf'.format(bw_value))


def gen_annotation(all_shapes, all_annotations, label, x, ymin, ymax):
    ref = 1
    xref = "x%s" % (ref,)
    yref = "y%s" % (ref,)

    x0 = str(x - 0.04)
    x1 = str(x)

    yann = (ymin + ymax) * 0.5
    ymin = str(ymin)
    ymax = str(ymax)

    all_annotations.append(
        dict(
            x=x + 0.3,
            y=yann,
            text=label,
            showarrow=False,
            xref=xref,
            yref=yref,
            font=dict(size=12),
            textangle=270,
        )
    )

    line_color = px.colors.qualitative.Bold[2]

    shapes = [
        dict(
            type="line",
            xref=xref,
            yref=yref,
            x0=x1,
            y0=ymin,
            x1=x1,
            y1=ymax,
            line_color=line_color,
            line_width=1,
        ),
        dict(
            type="line",
            xref=xref,
            yref=yref,
            x0=x0,
            y0=ymin,
            x1=x1,
            y1=ymin,
            line_color=line_color,
            line_width=1,
        ),
        dict(
            type="line",
            xref=xref,
            yref=yref,
            x0=x0,
            y0=ymax,
            x1=x,
            y1=ymax,
            line_color=line_color,
            line_width=1,
        ),
    ]

    all_shapes.extend(shapes)


def data_tmp():
    data = np.random.normal(1, 0.3, 1000)
    data2 = np.random.normal(30, 15, 200)
    data = np.concatenate((data, data2), axis=0)
    data = np.abs(data)
    return data


def generate_distribution_violin_alt(
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

    fig = make_subplots(
        rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True, vertical_spacing=0.02
    )

    tail_masses = []

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
            y=plotted_data,
            box_visible=False,
            meanline_visible=False,
            name=ts_name,
            side="positive",
            points=False,
            bandwidth=bw_value,
            scalemode="width",
            line=dict(width=8),
        )

        fig.add_trace(violin_data, row=1, col=1)
        label = "Tail Mass: <br />  {0:.1f}%".format(100 - percent_head_2)
        print(label)
        # gen_annotation(all_shapes, all_annotations, label, tsidx + 0.1, 4,
        #                max(plotted_data))

    fig.update_traces(width=1.8)
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
        xaxis2=dict(
            title=dict(text="Simulation Time (s)", standoff=30),
            ticktext=timestamps,
            tickvals=[x + 0.2 for x in range(timesteps)],
            color="#000",
            linecolor="#444",
            linewidth=axislinewidth,
            range=[-0.5, timesteps + 0.3],
            showgrid=True,
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
        ),
        yaxis2=dict(
            title=dict(text="Tail Mass", standoff=50),
            linecolor="#777",
            linewidth=axislinewidth,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=gridwidth,
            tickformat=".0%",
            range=[0, 0.32],
        ),
        plot_bgcolor="#fff",
        shapes=all_shapes,
        annotations=all_annotations,
        showlegend=False,
        legend=dict(x=0.87, y=1),
        font=dict(size=52),
    )

    x2 = [x + 0.2 for x in range(timesteps)]
    y2 = np.array(tail_masses) / 100.0

    fig.add_trace(
        go.Scatter(x=x2, y=y2, marker_size=40, line_width=8, marker_color="#4C78A8"),
        row=2,
        col=1,
    )

    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0,
        y0=4,
        x1=7.3,
        y1=20,
        fillcolor="purple",
        row=1,
        col=1,
        opacity=0.12,
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=4,
        x1=7.3,
        y1=4,
        line=dict(
            color="purple",
            width=12,
            dash="dot",
        ),
    )

    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    fig.add_annotation(
        x=6.8, y=5, text="Distribution<br />Tail", font=dict(color="#444", size=40)
    )

    # fig.show()
    # fig.write_image(
    #     '../vis/poster/vpic32.distrib.violin.alt.{0}.pdf'.format(bw_value))
    # fig.write_image(fig_path)
    pio.write_image(fig, fig_path, width=2048, height=1536)


def generate_distribution_violin_alt_2(
    data_path: str, fig_path: str, num_ranks: int = None, bw_value: float = 0.03
):
    num_ranks = 4
    vpic_reader = VPICReader(data_path, num_ranks=num_ranks)
    ranks = vpic_reader.get_num_ranks()
    timesteps = vpic_reader.get_num_ts()
    print("[VPICReader] Ranks: {0}, ts: {1}".format(ranks, timesteps))
    fig = go.Figure()

    all_shapes = []
    all_annotations = []

    timestamps = [vpic_reader.get_ts(i) for i in range(vpic_reader.get_num_ts())]

    # fig = make_subplots(rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True,
    #                     vertical_spacing=0.02)

    tail_masses = []

    for tsidx in range(0, timesteps):
        # for tsidx in range(0, 3):
        # data = vpic_reader.sample_global(tsidx, load_cached=True)
        data = vpic_reader.sample_hist(tsidx)

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

        plotted_data = list(map(lambda x: log_tailed(x, 0.001), plotted_data))

        ts_name = "Timestep {0}".format(
            vpic_reader.get_ts(tsidx),
        )

        violin_data = go.Violin(
            y=plotted_data,
            box_visible=False,
            meanline_visible=False,
            name=ts_name,
            side="positive",
            points=False,
            bandwidth=bw_value,
            scalemode="width",
            line=dict(width=4),
        )

        fig.add_trace(violin_data)
        label = "Tail Mass: <br />  {0:.1f}%".format(100 - percent_head_2)
        print(label)
        # gen_annotation(all_shapes, all_annotations, label, tsidx + 0.1, 4,
        #                max(plotted_data))

    fig.update_traces(width=1.8)
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Simulation Time (s)",
            ),
            ticktext=timestamps,
            tickvals=[x + 0.2 for x in range(timesteps)],
            color="#000",
            linecolor="#444",
            range=[-0.5, timesteps + 0.3],
            showgrid=True,
            gridcolor="#777",
        ),
        yaxis=dict(
            title=dict(
                text="Energy (eV)",
            ),
            tickmode="array",
            tickvals=list(range(0, 18, 2)),
            ticktext=[
                "{0:.0f}".format(log_tailed_reverse(x, 0.001)) for x in range(0, 18, 2)
            ],
            linecolor="#444",
            showgrid=True,
            gridcolor="#777",
            range=[-0.3, 8],
        ),
        # yaxis=dict(
        #     title=dict(
        #         text='Particle Energy (eV)',
        #     ),
        #     linecolor='#444',
        #     showgrid=True,
        #     gridcolor='#777'
        # ),
        plot_bgcolor="#fff",
        shapes=all_shapes,
        annotations=all_annotations,
        showlegend=False,
        legend=dict(x=0.87, y=1),
        font=dict(size=18),
    )

    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)

    fig.show()
    # fig.write_image(
    #     '../vis/poster/vpic32.distrib.violin.alt.{0}.pdf'.format(bw_value))
    # fig.write_image(fig_path)
    # pio.write_image(fig, fig_path, width=2048, height=1536)


def gen_skew_runtime_util(fig, data, num_ranks, col_hash_val, nodes=None, percent=None):
    col_x = data["skewpct"].to_numpy()

    col_xrev = col_x[::-1]
    col_mean = data["runtime"]["mean"].to_numpy()
    col_std = data["runtime"]["std"].to_numpy()
    col_min = col_mean - col_std
    col_max = col_mean + col_std
    # col_min = data['runtime']['min'].to_numpy()
    # col_max = data['runtime']['max'].to_numpy()
    col_min = col_min[::-1]

    col_hash = lambda x: ((x + 3) * (x - 1)) ^ (x % 13) % 239
    col_hash = lambda x: int(hashlib.sha1(str(x).encode("utf-8")).hexdigest(), 16) % (
        10**8
    )
    color_idx = col_hash(col_hash_val) % (len(px.colors.qualitative.Dark2))
    print(col_hash_val, color_idx)

    line_color = px.colors.qualitative.Dark2[color_idx]
    band_color = px.colors.qualitative.Pastel2[color_idx]
    band_color = "rgba" + band_color[3:-1] + ",0.5)"
    print(line_color, band_color)

    if percent:
        main_line = go.Scatter(
            x=col_x,
            y=col_mean,
            mode="lines",
            line=dict(color=line_color),
            name="{0} ranks ({1}%)".format(num_ranks, percent),
        )
        main_band = go.Scatter(
            x=np.concatenate([col_x, col_xrev]),
            y=np.concatenate([col_max, col_min]),
            fill="tozerox",
            fillcolor=band_color,
            line=dict(color="rgba(255,255,255,0)"),
            name="{0} ranks ({1}%)".format(num_ranks, percent),
            showlegend=False,
        )
    elif nodes:
        main_line = go.Scatter(
            x=col_x,
            y=col_mean,
            mode="lines",
            line=dict(color=line_color),
            name="{0} ranks ({1} nodes)".format(num_ranks, nodes),
        )
        main_band = go.Scatter(
            x=np.concatenate([col_x, col_xrev]),
            y=np.concatenate([col_max, col_min]),
            fill="tozerox",
            fillcolor=band_color,
            line=dict(color="rgba(255,255,255,0)"),
            name="{0} ranks ({1} nodes)".format(num_ranks, nodes),
            showlegend=False,
        )

    # fig.add_traces([main_line])
    return main_line, main_band


def gen_skew_runtime():
    csv_file = "../vis/poster/skewruntimejun25.csv"

    data = pd.read_csv(csv_file)

    fig = go.Figure()
    all_lines = []
    all_bands = []

    for skewnodecnt in data["skewnodecnt"].unique():
        print(skewnodecnt)

        plot_data = data[data["skewnodecnt"] == skewnodecnt]
        skewnodepct = plot_data["skewnodepct"].iloc[0]
        print(plot_data)

        aggr_data = plot_data.groupby("skewpct").agg(
            {
                "runtime": ["min", "max", "mean"],
            }
        )
        aggr_data["skewpct"] = aggr_data.index
        # print(aggr_data)

        # skewnodepct = 10 if 26 else 20

        line, band = gen_skew_runtime_util(
            fig, aggr_data, skewnodecnt, skewnodepct, percent=skewnodepct
        )
        all_lines.append(line)
        all_bands.append(band)

    # fig.add_traces(all_bands)
    fig.add_traces(all_lines)

    fig.update_layout(
        title_text="Change in runtime as workload skew is varied",
        yaxis=dict(range=[90, 130], title="Time (seconds)"),
        xaxis=dict(title="Additional workload for skewed ranks (%)"),
    )
    # fig.write_image('../vis/poster/runtimeskew.pdf')
    fig.show()
    return


def gen_skew_runtime_custom():
    csv_file = "../vis/runtime/runtime.ssd/runlog.vfstune.txt"

    data = pd.read_csv(csv_file, comment="#")
    data = data["runtime"]

    all_mean = []
    all_std = []
    all_labels = [
        "64G/32K/20G",
        "64G/32K/2G",
        "64G/32K/200M",
        "64G/32K/10%/r",
        "64G/32K/20G/r",
        "64G/32K/2G/r",
        "64G/32K/200M/r",
        "16G/32K/10%/r",
        "16G/32K/2G/r",
        "16G/32K/200M/r",
    ]

    for sidx in range(0, len(data), 9):
        # print(sidx, sidx+9)
        cur_data = data[sidx : sidx + 9]
        cur_mean = np.mean(cur_data)
        cur_std = np.std(cur_data)
        # print(cur_mean, cur_std)
        all_mean.append(cur_mean)
        all_std.append(cur_std)

    print(all_mean)
    print(all_std)

    fig = go.Figure()
    fig_bar = go.Bar(x=all_labels, y=all_mean, error_y=dict(type="data", array=all_std))
    fig.add_trace(fig_bar)
    fig.update_layout(
        title_text="Mean runtime for different VFS tuning parameters",
        yaxis=dict(title="Time (seconds)"),
    )
    # fig.show()
    fig.write_image("../vis/runtime/runtime.ssd/vfs_vs_runtime.pdf")


def gen_skew_runtime_phynode():
    csv_file = "../vis/runtime/runtime.batch.jul14/runlog.txt"

    data = pd.read_csv(csv_file)

    fig = go.Figure()
    all_lines = []
    all_bands = []

    for skewedphynodes in data["skewedphynodes"].unique():
        if skewedphynodes != 2:
            continue
        print(skewedphynodes)

        plot_data = data[data["skewedphynodes"] == skewedphynodes]
        skewnodepct = plot_data["skewedphynodes"].iloc[0]
        skewnodecnt = plot_data["skewnodecnt"].iloc[0]
        print(plot_data)

        aggr_data = plot_data.groupby("skewpct").agg(
            {
                "runtime": ["min", "max", "mean", "std"],
            }
        )
        aggr_data["skewpct"] = aggr_data.index
        aggr_data["skewpct"] = aggr_data["skewpct"].map(lambda x: x * 10 - 100)
        print(aggr_data)

        # skewnodepct =10 if 26 else 20

        line, band = gen_skew_runtime_util(
            fig, aggr_data, skewnodecnt, skewedphynodes, nodes=skewedphynodes
        )
        all_lines.append(line)
        all_bands.append(band)

    csv_file_2 = "../vis/runtime/runtime.ram16g.jul20/runlog.16g.txt"
    data_2 = pd.read_csv(csv_file_2)

    aggr_data_2 = data_2.groupby("skewpct").agg(
        {
            "runtime": ["min", "max", "mean", "std"],
        }
    )
    aggr_data_2["skewpct"] = aggr_data_2.index
    aggr_data_2["skewpct"] = aggr_data_2["skewpct"].map(lambda x: x * 10 - 100)
    print(aggr_data_2)
    line, band = gen_skew_runtime_util(fig, aggr_data_2, 32, 4, nodes=2)

    all_lines.append(line)

    all_lines[0].name = "mem=64g (32/2 overloaded)"
    all_lines[1].name = "mem=16g (32/2 overloaded)"
    all_bands.append(band)

    fig.add_traces(all_bands)
    fig.add_traces(all_lines)

    fig.update_layout(
        title_text="Change in runtime as workload skew is varied",
        yaxis=dict(range=[0, 145], title="Time (seconds)"),
        xaxis=dict(title="Additional workload for skewed ranks (%)"),
    )
    fig.write_image("../vis/runtime/runtime.ram16g.jul20/runtimeskew.pdf")
    # fig.show()
    return


if __name__ == "__main__":
    num_ranks = 32
    timesteps = 4
    data = "../data"
    #
    # # for bw_value in [0.0001, 0.001, 0.01, 0.1]:
    #
    bw_value = 0.03
    generate_distribution_violin_alt(data, num_ranks, bw_value)
    # generate_distribution_box(data, num_ranks, timesteps)
    # gen_skew_runtime_phynode()
    # gen_skew_runtime_custom()
