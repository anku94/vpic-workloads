import glob as glob
import matplotlib.pyplot as plt

class StagedBuildout:
    def __init__(self, ax: plt.Axes):
        self.ax = ax

        handles, labels = ax.get_legend_handles_labels()
        self.ohandles = handles
        self.olabels = labels
        self.olegvis = [True] * len(handles)

        self.oxlabels = ax.get_xticklabels()
        self.oxvis = [True] * len(self.oxlabels)

    def disable_legend_entry(self, index: int):
        self.olegvis[index] = False
        self.rebuild_legend()

    def enable_legend_entry(self, index: int):
        self.olegvis[index] = True
        self.rebuild_legend()

    def toggle_legend_entry(self, index: int):
        self.olegvis[index] = not self.olegvis[index]
        self.rebuild_legend()

    def disable_xticklabel(self, index: int):
        self.oxlabels[index].set_visible(False)
        self.oxvis[index] = False

    def enable_xticklabel(self, index: int):
        self.oxlabels[index].set_visible(True)
        self.oxvis[index] = True

    def rebuild_legend(self):
        handles = [h for h, v in zip(self.ohandles, self.olegvis) if v]
        labels = [l for l, v in zip(self.olabels, self.olegvis) if v]
        self.draw_legend(handles, labels)
        # self.ax.legend(handles, labels, ncol=1, markerscale=0.5, loc="lower left")

    def disable_artists(self, indices: list[int]):
        artists = self.ax.get_children()
        for i in indices:
            artists[i].set_visible(False)

    def enable_artists(self, indices: list[int]):
        artists = self.ax.get_children()
        for i in indices:
            artists[i].set_visible(True)

    def toggle_artists(self, indices: list[int]):
        artists = self.ax.get_children()
        for i in indices:
            artists[i].set_visible(not artists[i].get_visible())

    def disable_alx(
        self, artists: list[int], legend_items: list[int] = [], x_items: list[int] = []
    ):
        self.disable_artists(artists)

        for item in legend_items:
            print(f"Disabling legend entry {item}")
            self.disable_legend_entry(item)

        for item in x_items:
            print(f"Disabling xticklabel {item}")
            self.disable_xticklabel(item)

    def enable_alx(
        self, artists: list[int], legend_items: list[int] = [], x_items: list[int] = []
    ):
        self.enable_artists(artists)
        for item in legend_items:
            self.enable_legend_entry(item)

        for item in x_items:
            self.enable_xticklabel(item)

    def draw_legend(self, handles, labels):
        self.ax.legend(handles, labels)
