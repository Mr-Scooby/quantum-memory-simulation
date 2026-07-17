#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from radpattern.plotting.pattern_3d import plot_pattern_3d
from radpattern.plotting import load_data, THESIS_STYLE
from pathlib import Path
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplot2tikz

plt.style.use(THESIS_STYLE)

#plt.rcParams.update({
#    # Figure/export quality
#    "figure.figsize": (4.2, 3.6),
#    "figure.dpi": 150,
#    "savefig.dpi": 600,
#    "savefig.bbox": "tight",
#    "savefig.pad_inches": 0.03,
#
#    # Fonts
#    "font.family": "serif",
#    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
#    "mathtext.fontset": "stix",
#    "font.size": 17,
#
#    # Axes
#    "axes.labelsize": 9,
#    "axes.titlesize": 9,
#    "axes.linewidth": 0.8,
#
#    # Ticks
#    "xtick.labelsize": 8,
#    "ytick.labelsize": 8,
#    "xtick.direction": "in",
#    "ytick.direction": "in",
#    "xtick.major.size": 3,
#    "ytick.major.size": 3,
#    "xtick.major.width": 0.7,
#    "ytick.major.width": 0.7,
#
#    # Lines
#    "lines.linewidth": 1.2,
#    "lines.markersize": 4,
#
#    # Colormap
#    "image.nmap": "viridis",
#
#    # Vector export compatibility
#    "pdf.fonttype": 42,
#    "ps.fonttype": 42,
#    "svg.fonttype": "none",
#})
#

file = Path(input("File: "))

data, grid, exp, sim = load_data(file)

print(data.keys())
print(data["intensity"].shape)

for i in [0,50,99]: 
        #I  = data["intensity"][0] / data["intensity"][0]
        print(f"image time idx {i}")
        I = data["intensity"][i]
        I_ref = data["intensity"][0]


        plot_pattern_3d(
            grid,
            I,
            title=r"",
            I_ref= I_ref
        )

        fig = plt.gcf()
        fig.set_size_inches(4.2, 3.6)
        fig.patch.set_facecolor("white")
        fig.suptitle("")

        ax = None
        for candidate_ax in fig.axes:
            if hasattr(candidate_ax, "get_zlim"):
                ax = candidate_ax
                break

        if ax is None:
            ax = plt.gca()

        # Use clean physical labels: direction cosines on the observation sphere.
        ax.set_xlabel(r"$\hat{k}_x$", labelpad=3)
        ax.set_ylabel(r"$\hat{k}_y$", labelpad=3)
        ax.set_zlabel(r"$\hat{k}_z$", labelpad=3)

        # No title inside the plot for thesis/publication.
        ax.set_title("")

        # Symmetric unit-sphere limits.
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))

        # Fewer ticks.
        ticks = [-1, -0.5, 0, 0.5, 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.tick_params(axis="both", labelsize=8, pad=1)

        # Transparent panes.
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0))

        ax.xaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
        ax.yaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))
        ax.zaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))

        # Lighter 3D grid.
        ax.grid(True)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"]["linewidth"] = 0.35
            axis._axinfo["grid"]["linestyle"] = "-"
            axis._axinfo["grid"]["color"] = (0.70, 0.70, 0.70, 0.35)

            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.25

        # Clean colorbar axis if plot_pattern_3d created one.
        for candidate_ax in fig.axes:
            if candidate_ax is not ax:
                candidate_ax.set_ylabel(
                    r"Normalized intensity $log(I/I_{\max})$ [dB]",
                    fontsize=9,
                    labelpad=8,
                )
                candidate_ax.tick_params(
                    labelsize=8,
                    direction="in",
                    length=3,
                    width=0.7,
                )

                # Suitable for log plots normalized from -60 dB to 0 dB.
                candidate_ax.set_yticks([-60, -50, -40, -30, -20, -10, 0])

        # Keep enough room for the colorbar.
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.98)
        ax.view_init(elev=40, azim=-75)
        
        matplot2tikz.save(f"test_AF_t{i}.tex", figure = fig )
plt.show()


save_tex = input("Save as tiktex file? (y/n):").strip().upper() == "Y"
if save_tex: 
    file_name = input("File name (+ .tex) : ")
    #folder = r"D:\radek\Figures\manuscript_figures_push\Figures"
    full_path = Path( file_name + ".tex")
    #matplot2tikz.save(full_path)
