import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_fontsz = 12
widths = [5, 20, 50, 100]
tags = ["carp", "comp"]
qlog_fmt = "{0}/querylog.{1}.{2}.csv"
base_fontsz = 12


def get_dfs() -> list[pd.DataFrame]:
    basedir = "/Users/schwifty/Repos/workloads/rundata/eval/runs.big.2/YCSB.eval"
    basedir = "/Users/schwifty/Repos/vpic-workloads/rundata/eval/runs.uniform/YCSB"

    all_dfs = []
    for tag in tags:
        dfs = []
        for width in widths:
            qlog_path = qlog_fmt.format(basedir, tag, width)
            df = pd.read_csv(qlog_path)
            df["width"] = width
            dfs.append(df)
        all_dfs.append(pd.concat(dfs))

    return all_dfs


def plot_query_ycsb_inner(axes: list[plt.Axes], all_dfs: list[pd.DataFrame]) -> None:
    epochs = [0, 4, 7, 11]
    epoch_labels = "200,7400,14600,19400".split(",")
    epoch_labels = dict(zip(epochs, epoch_labels))

    odf, mdf = all_dfs
    odf["qreadus"] /= 1e6
    odf["qsortus"] /= 1e6
    mdf["qreadus"] /= 1e6

    x = np.arange(len(widths))
    cmap = plt.get_cmap("Pastel2")
    bar_width = 0.35

    for epoch, ax in zip(epochs, axes):
        orig_df = odf[odf["epoch"] == epoch]
        merged_df = mdf[mdf["epoch"] == epoch]

        orig_df = orig_df.groupby(["width"], as_index=False).agg(
            {"qreadus": "sum", "qsortus": "sum"}
        )

        merged_df = merged_df.groupby(["width"], as_index=False).agg(
            {
                "qreadus": "sum",
            }
        )

        print(orig_df)
        print(merged_df)
        ax.bar(
            x - bar_width / 2,
            orig_df["qreadus"],
            bar_width,
            label="CARP/Read",
            color=cmap(0),
            edgecolor="#000",
        )
        ax.bar(
            x - bar_width / 2,
            orig_df["qsortus"],
            bar_width,
            bottom=orig_df["qreadus"],
            label="CARP/Sort",
            color=cmap(1),
            edgecolor="#000",
        )
        ax.bar(
            x + bar_width / 2,
            merged_df["qreadus"],
            bar_width,
            label="TritonSort/Read",
            color=cmap(2),
            edgecolor="#000",
        )
        ax.yaxis.grid(True, color="#bbb")
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in widths], fontsize=base_fontsz - 1)
        yticklabels = [0, 300, 600, 900, 1200]
        ax.set_yticks(yticklabels)
        ax.set_yticklabels(yticklabels, fontsize=base_fontsz - 1)
        ax.set_title("Epoch {0}".format(epoch_labels[epoch]), fontsize=base_fontsz)
    pass


def plot_query_ycsb() -> None:
    all_dfs = get_dfs()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes = [*axes[0], *axes[1]]

    plot_query_ycsb_inner(axes, all_dfs)
    plt.close(fig)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        ncol=3,
        bbox_to_anchor=(0.94, 0.93),
        fontsize=base_fontsz - 1,
        framealpha=0.8,
    )

    fig.supxlabel("Query Width (#SSTs)", x=0.55, y=0.04, fontsize=base_fontsz + 1)
    fig.supylabel(
        "Total Time Taken (seconds)", x=0.03, y=0.55, fontsize=base_fontsz + 1
    )
    fig.tight_layout()


def run():
    plot_query_ycsb()


if __name__ == "__main__":
    run()
