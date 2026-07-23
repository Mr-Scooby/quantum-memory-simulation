#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot three far-field patterns on unit spheres with one shared colorbar.

Matplotlib only.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from radpattern.plotting import load_data, THESIS_STYLE


plt.style.use(THESIS_STYLE)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

FRAME_INDICES = [0, 50, 99]
TIME_LABELS_US = [0, 10, 20]

DB_MIN = -60
DB_MAX = 0

CMAP = "viridis"

FIGURE_SIZE = (11.5, 3.4)

ELEVATION = 28
AZIMUTH = -65


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def extract_frame(intensity, frame_index):
    """Extract one intensity frame with shape (theta, phi)."""

    frame = np.squeeze(
        np.asarray(intensity[frame_index], dtype=float)
    )

    if frame.ndim != 2:
        raise ValueError(
            f"Frame {frame_index} has shape {frame.shape}. "
            "Expected shape (n_theta, n_phi)."
        )

    return frame


def close_phi(array):
    """
    Append the first azimuthal column to the end.

    This closes the surface at phi = 2*pi and removes the visible seam.
    """

    return np.concatenate(
        [array, array[:, :1]],
        axis=1,
    )


def convert_to_db(intensity, reference_intensity):
    """Convert intensity to dB using a common reference."""

    intensity = np.maximum(
        np.asarray(intensity, dtype=float),
        0.0,
    )

    intensity_db = 10 * np.log10(
        intensity / reference_intensity + 1e-12
    )

    return np.clip(
        intensity_db,
        DB_MIN,
        DB_MAX,
    )


def format_axis(ax, time_us):
    """Apply the same formatting to every sphere panel."""

    ax.set_title(
        rf"$t={time_us}\,\mu\mathrm{{s}}$",
        pad=5,
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")

    ax.view_init(
        elev=ELEVATION,
        azim=AZIMUTH,
    )

    ticks = [-1, 0, 1]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xlabel(
        r"$\hat{k}_x$",
        labelpad=0,
    )

    ax.set_ylabel(
        r"$\hat{k}_y$",
        labelpad=0,
    )

    ax.set_zlabel(
        r"$\hat{k}_z$",
        labelpad=0,
    )

    ax.tick_params(
        axis="both",
        labelsize=7,
        pad=-1,
        length=2.5,
        width=0.6,
    )

    # Remove pane backgrounds and grid lines.
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    ax.grid(False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # Load data.
    input_file = Path(
        input("Input NPZ file: ").strip()
    ).expanduser()

    data, grid, _, _ = load_data(input_file)

    if "intensity" not in data:
        raise KeyError(
            "The NPZ file does not contain an 'intensity' array."
        )

    intensity = np.asarray(
        data["intensity"]
    )

    print(f"Intensity shape: {intensity.shape}")

    if max(FRAME_INDICES) >= intensity.shape[0]:
        raise IndexError(
            f"Requested frame {max(FRAME_INDICES)}, but only "
            f"{intensity.shape[0]} frames are available."
        )

    # Use the first selected frame as the common intensity reference.
    first_frame = extract_frame(
        intensity,
        FRAME_INDICES[0],
    )

    reference_intensity = np.nanmax(
        first_frame
    )

    if reference_intensity <= 0 or not np.isfinite(reference_intensity):
        raise ValueError(
            "The first frame does not contain a valid positive intensity."
        )

    norm = Normalize(
        vmin=DB_MIN,
        vmax=DB_MAX,
    )

    cmap = plt.get_cmap(
        CMAP
    )

    # Close the azimuthal seam of the sphere coordinates.
    sphere_x = close_phi(grid.nx)
    sphere_y = close_phi(grid.ny)
    sphere_z = close_phi(grid.nz)

    # Create figure.
    fig = plt.figure(
        figsize=FIGURE_SIZE,
        dpi=150,
    )

    fig.patch.set_facecolor("white")

    # Columns:
    #   sphere 1 | sphere 2 | sphere 3 | spacer | colorbar
    grid_spec = fig.add_gridspec(
        nrows=1,
        ncols=5,
        width_ratios=(
            1.0,
            1.0,
            1.0,
            0.12,   # padding between third sphere and colorbar
            0.045,  # colorbar width
        ),
        left=0.015,
        right=0.965,
        bottom=0.02,
        top=0.98,
        wspace=0.0,
    )

    axes = [
        fig.add_subplot(
            grid_spec[0, column],
            projection="3d",
        )
        for column in range(3)
    ]

    # GridSpec column 3 is intentionally left empty.
    colorbar_axis = fig.add_subplot(
        grid_spec[0, 4]
    )

    # Plot the selected frames.
    for ax, frame_index, time_us in zip(
        axes,
        FRAME_INDICES,
        TIME_LABELS_US,
    ):

        frame = extract_frame(
            intensity,
            frame_index,
        )

        frame_db = convert_to_db(
            frame,
            reference_intensity,
        )

        frame_db = close_phi(
            frame_db
        )

        ax.plot_surface(
            sphere_x,
            sphere_y,
            sphere_z,
            rstride=1,
            cstride=1,
            facecolors=cmap(norm(frame_db)),
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        format_axis(
            ax,
            time_us,
        )

    # Shared colorbar.
    scalar_mappable = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar_mappable.set_array([])

    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        r"Relative intensity [dB]",
        fontsize=9,
        labelpad=7,
    )

    colorbar.set_ticks(
        [-60, -50, -40, -30, -20, -10, 0]
    )

    colorbar.ax.tick_params(
        labelsize=7,
        direction="in",
        length=3,
        width=0.7,
    )

    colorbar.outline.set_linewidth(0.6)

    plt.show()


if __name__ == "__main__":
    main()
