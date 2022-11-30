import matplotlib
import matplotlib.figure as pltfig
import matplotlib.pyplot as plt
import sys


def plot_init():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc(
        "font", size=SMALL_SIZE
    )  # controls default text sizes
    # plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('text', usetex=True)  # LaTeX


def plot_init_bigfont():
    SMALL_SIZE = 20
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 26

    plt.rc(
        "font", size=SMALL_SIZE
    )  # controls default text sizes
    # plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE - 4)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)  # fontsize of the figure title
    plt.rc('text', usetex=True)  # LaTeX


class PlotSaver:
    @staticmethod
    def save(fig: pltfig.Figure, fpath: str, fname: str):
        PlotSaver._save_to_fpath(fig, fpath, fname, ext="png", show=True)
        # PlotSaver._save_to_paper(fig, "eval", fname)
        # PlotSaver._save_to_project(fig, fname)
        # PlotSaver._save_unrolled(fig, fpath, fname)

    @staticmethod
    def _save_to_fpath(fig: pltfig.Figure, fpath: str, fname: str, ext="png",
                       show=True):
        full_path = f"{fpath}/{fname}.{ext}"
        if show:
            print(f"[PlotSaver] Displaying figure\n")
            fig.show()
        else:
            print(f"[PlotSaver] Writing to {full_path}\n")
            fig.savefig(full_path, dpi=300)

    @staticmethod
    def _save_to_paper(fig: pltfig.Figure, fig_type, fname: str):
        paper_root = "/Users/schwifty/Repos/carp/carp-paper/figures"
        if fig_type not in ["bg", "design", "impl", "eval"]:
            print("[PlotSaver] Error: unknown fig type: {fig_type}\n")
            sys.exit(-1)

        fpath = f"{paper_root}/{fig_type}"
        PlotSaver._save_to_fpath(fig, fpath, fname, ext="pdf", show=False)

    @staticmethod
    def _save_to_project(fig, fname: str):
        fpath = "/Users/schwifty/CMU/18911/documents/PDLR22/carptalk_plots"
        PlotSaver._save_to_fpath(fig, fpath, fname, ext="pdf", show=False)
        pass

    @staticmethod
    def _save_unrolled(fig: pltfig.Figure, fpath: str, fname: str):
        ax_all = fig.get_axes()
        ax = ax_all[0]
        handles, labels = ax.get_legend_handles_labels()
        handles[0].remove()
        PlotSaver._save_to_fpath(fig, fpath, fname, show=True)
