import glob
import os
import re
import numpy as np

from plotly import graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots


def get_key_id(key: str) -> int:
    mobj = re.findall(r"(\d+)_hist", key)
    key_id = int(mobj[0])
    print(key_id)
    return key_id


def get_tick_labels(keys: list[str]) -> list[str]:
    labels = list(map(lambda x: str(get_key_id(x) * 300), keys))
    return labels


def gen_data_from_hist(hist: np.ndarray, bins: np.ndarray, n: int) -> np.ndarray:
    cdf = np.cumsum(hist)
    print(cdf)

    values = np.random.rand(n)
    value_bins = np.searchsorted(cdf, values)

    blow = bins[value_bins]
    bhigh = bins[value_bins + 1]
    bsamples = np.random.uniform(blow, bhigh)

    print(bsamples)
    return bsamples


def read_data() -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    data_path = "/Users/schwifty/Repos/amr-data/20240326/blastwave01/hists"
    bins_path = f"{data_path}/bins.txt"
    bins = np.loadtxt(bins_path)

    glob_pat = f"{data_path}/*hist*txt"
    all_hist_files = glob.glob(glob_pat)

    all_hists = {}
    for f in all_hist_files:
        hist = np.loadtxt(f)
        fname = os.path.basename(f).replace(".txt", "")
        all_hists[fname] = hist

    all_keys = list(all_hists.keys())
    all_key_ids = list(
        map(lambda x: int(re.search(r"(\d+)_hist", x).group(1)), all_keys)
    )

    keys_sorted = [x for _, x in sorted(zip(all_key_ids, all_keys))]
    return (keys_sorted, bins, all_hists)


def get_hist_between(
    keys: list[str], hist: list[np.ndarray], bins: np.ndarray, low: float, high: float
) -> list[float]:
    all_hist_sums = []
    for k in keys:
        low_idx = np.searchsorted(bins, low)
        high_idx = np.searchsorted(bins, high)

        hist_sum = np.sum(hist[k][low_idx:high_idx])
        all_hist_sums.append(hist_sum)
    return all_hist_sums


def plot_add_bins(
    fig, keys: list[str], bins: np.ndarray, hist_dict: dict[str, np.ndarray]
) -> None:
    dx = list(range(len(keys)))
    hist_mid = get_hist_between(keys, hist_dict, bins, 0.1, 0.5)
    rightcolor = "rgba(0, 0, 100, 1.0)"
    rightcolor = "rgba(100, 0, 0, 0.9)"
    cutoffcolor = "rgba(100, 100, 100, 0.7)"
    annotcolor = "rgba(100, 100, 100, 0.9)"
    annottxtcolor = "rgba(50, 50, 50, 1.0)"

    trace = go.Scatter(
        x=dx,
        y=hist_mid,
        name="Mass (middle band)",
        line=dict(color=rightcolor, width=8),
        marker=dict(size=32, symbol="square"),
        mode="lines+markers",
        yaxis="y2",
    )
    fig.add_trace(trace, row=1, col=1, secondary_y=True)

    hist_tail = get_hist_between(keys, hist_dict, bins, 0.5, 3.0)
    trace = go.Scatter(
        x=dx,
        y=hist_tail,
        name="Mass (Tail band)",
        line=dict(color=rightcolor, width=8),
        marker=dict(size=32),
        mode="lines+markers",
        yaxis="y2",
    )
    fig.add_trace(trace, row=1, col=1, secondary_y=True)

    horizontal_x = [-1, len(keys)]

    trace = go.Scatter(
        x=horizontal_x,
        y=[0.1, 0.1],
        line=dict(color=cutoffcolor, width=5),
        showlegend=False,
    )
    fig.add_trace(trace, row=1, col=1, secondary_y=False)

    trace = go.Scatter(
        x=horizontal_x,
        y=[0.5, 0.5],
        line=dict(color=cutoffcolor, width=5),
        showlegend=False,
    )
    fig.add_trace(trace, row=1, col=1, secondary_y=False)

    fig.add_annotation(
        x=-0.8,
        y=0.5,
        text="<i>Cutoff <br />(Tail Band)</i>",
        font=dict(size=36, color=annottxtcolor),
        arrowcolor=annotcolor,
        arrowwidth=5,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        ax=200,
        ay=-320,
    )

    fig.add_annotation(
        x=-0.55,
        y=0.1,
        text="<i>Cutoff <br />(Mid Band)</i>",
        font=dict(size=36, color=annottxtcolor),
        arrowcolor=annotcolor,
        arrowwidth=5,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        ax=140,
        ay=-220,
    )

    pass


def plot(keys: list[str], bins: np.ndarray, hist_dict: dict[str, np.ndarray]):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for kidx, k in enumerate(keys):
        print(k)
        hist = hist_dict[k]
        data = gen_data_from_hist(hist, bins, 50000)

        leftcolor = "rgba(0, 100, 80, 1.0)"
        leftcolor_light = "rgba(0, 100, 80, 0.5)"
        rightcolor = "rgba(100, 0, 0, 0.9)"

        vdata = go.Violin(
            x0=kidx - 0.4,
            y=data,
            box_visible=False,
            meanline_visible=False,
            points=False,
            side="positive",
            bandwidth=0.15,
            scalemode="width",
            line=dict(width=8, color=leftcolor),
            yaxis="y",
            width=1.4,
            fillcolor=leftcolor_light,
            showlegend=False,
            pointpos=0,
        )

        fig.add_trace(vdata, row=1, col=1)

    plot_add_bins(fig, keys, bins, hist_dict)
    # fig.update_traces(width=1.8)

    fig.update_xaxes(
        title_text="Timestep",
        tickvals=np.arange(len(keys)),
        ticktext=get_tick_labels(keys),
        range=(-0.8, len(keys)),
    )

    fig.update_yaxes(
        title_text="Energy (dimensionless)",
        title_standoff=50,
        nticks=5,
        range=(0, 3),
        secondary_y=False,
        title_font=dict(color=leftcolor),
        tickfont=dict(color=leftcolor),
    )

    fig.update_yaxes(
        title_text="Mass %",
        nticks=5,
        range=(0, 0.15),
        tickformat=".0%",
        secondary_y=True,
        title_font=dict(color=rightcolor),
        tickfont=dict(color=rightcolor),
    )

    fig.update_layout(
        font=dict(size=52),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.63,
        ),
    )
    # fig.show()
    fig.write_image("violin_amr.pdf", width=2048, height=768)
    pass


def run():
    keys, bins, hist_dict = read_data()
    plot(keys, bins, hist_dict)


if __name__ == "__main__":
    run()
