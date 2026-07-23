#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot three far-field intensity patterns side by side on unit spheres
with one shared colorbar.

Matplotlib only. No TikZ or matplot2tikz.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors

from radpattern.plotting import load_data, THESIS_STYLE


# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------

plt.style.use(THESIS_STYLE)

FRAME_INDICES = [0, 50, 99]

CMAP_NAME = "viridis"

DB_MIN = -60.0
DB_MAX = 0.0

FIGURE_SIZE = (10.8, 3.25)
OUTPUT_DPI = 600

ELEVATION = 28
AZIMUTH = -65


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def extract_frame(
    intensity: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    """
    Extract one two-dimensional angular intensity frame.

    Expected intensity shape:
        (time, theta, phi)

    Singleton dimensions are removed automatically.
    """

    frame = np.asarray(
        intensity[frame_index],
        dtype=float,
    )

    frame = np.squeeze(frame)

    if frame.ndim != 2:
        raise ValueError(
            f"Intensity frame {frame_index} has shape {frame.shape}. "
            "Expected a two-dimensional array with shape "
            "(n_theta, n_phi)."
        )

    return frame


def close_phi_seam(array: np.ndarray) -> np.ndarray:
    """
    Append the first azimuthal column to the end.

    This closes the surface at phi = 0 and phi = 2*pi and removes
    the visible dark seam produced by plot_surface.
    """

    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            f"close_phi_seam expects a 2D array, received {array.shape}."
        )

    return np.concatenate(
        [array, array[:, :1]],
        axis=1,
    )


def intensity_to_db(
    intensity: np.ndarray,
    reference_intensity: float,
) -> np.ndarray:
    """
    Convert intensity to dB using one shared reference intensity.
    """

    intensity = np.asarray(
        intensity,
        dtype=float,
    )

    # Avoid invalid logarithms caused by numerical negative values.
    intensity = np.maximum(
        intensity,
        0.0,
    )

    relative_intensity = intensity / reference_intensity

    intensity_db = 10.0 * np.log10(
        np.maximum(relative_intensity, 1.0e-12)
    )

    return np.clip(
        intensity_db,
        DB_MIN,
        DB_MAX,
    )


def get_panel_title(
    data: dict,
    frame_index: int,
    panel_label: str,
) -> str:
    """
    Use physical time when available, otherwise show the frame index.
    """

    if "times_us" in data:
        times_us = np.asarray(data["times_us"])

        if frame_index < times_us.size:
            return (
                rf"{panel_label} "
                rf"$t={times_us[frame_index]:.1f}\,\mu\mathrm{{s}}$"
            )

    return rf"{panel_label} frame ${frame_index}$"


def configure_axis(
    ax,
    title: str,
    show_axis_labels: bool,
) -> None:
    """
    Apply consistent formatting to one three-dimensional panel.
    """

    ax.set_title(
        title,
        pad=1,
        fontsize=10,
    )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    ax.set_box_aspect((1, 1, 1))

    # Orthographic projection avoids perspective distortion and makes
    # comparison between panels easier.
    ax.set_proj_type("ortho")

    ax.view_init(
        elev=ELEVATION,
        azim=AZIMUTH,
    )

    ticks = [-1.0, 0.0, 1.0]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=7,
        pad=-1,
        direction="in",
        length=2.5,
        width=0.6,
    )

    # Remove pane backgrounds.
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    # Remove the 3D grid for a cleaner publication figure.
    ax.grid(False)

    if show_axis_labels:
        ax.set_xlabel(
            r"$\hat{k}_x$",
            labelpad=-1,
        )

        ax.set_ylabel(
            r"$\hat{k}_y$",
            labelpad=-1,
        )

        ax.set_zlabel(
            r"$\hat{k}_z$",
            labelpad=-1,
        )

    else:
        # Keep the axes geometrically identical, but avoid repeating
        # labels and tick values in every panel.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])


# ---------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------

def main() -> None:

    # -----------------------------------------------------------------
    # Load simulation data
    # -----------------------------------------------------------------

    input_text = input(
        "Input NPZ file: "
    ).strip()

    if not input_text:
        raise ValueError(
            "No input file was provided."
        )

    input_file = Path(
        input_text
    ).expanduser()

    data, grid, exp, sim = load_data(
        input_file
    )

    if "intensity" not in data:
        raise KeyError(
            "The loaded NPZ file does not contain an "
            "'intensity' array."
        )

    intensity = np.asarray(
        data["intensity"]
    )

    print(
        f"Intensity array shape: {intensity.shape}"
    )

    if intensity.ndim < 3:
        raise ValueError(
            "The intensity array must contain time, theta, and phi "
            f"dimensions. Received shape {intensity.shape}."
        )

    maximum_requested_index = max(
        FRAME_INDICES
    )

    if maximum_requested_index >= intensity.shape[0]:
        raise IndexError(
            f"Requested frame {maximum_requested_index}, but the "
            f"intensity array contains only {intensity.shape[0]} "
            "time frames."
        )

    # -----------------------------------------------------------------
    # Shared intensity normalization
    # -----------------------------------------------------------------

    first_frame = extract_frame(
        intensity,
        FRAME_INDICES[0],
    )

    reference_intensity = float(
        np.nanmax(first_frame)
    )

    if (
        not np.isfinite(reference_intensity)
        or reference_intensity <= 0.0
    ):
        raise ValueError(
            "The first selected intensity frame does not contain a "
            "positive finite reference intensity."
        )

    norm = colors.Normalize(
        vmin=DB_MIN,
        vmax=DB_MAX,
    )

    cmap = plt.get_cmap(
        CMAP_NAME
    )

    # -----------------------------------------------------------------
    # Close the azimuthal seam in the angular grid
    # -----------------------------------------------------------------

    sphere_x = close_phi_seam(
        grid.nx
    )

    sphere_y = close_phi_seam(
        grid.ny
    )

    sphere_z = close_phi_seam(
        grid.nz
    )

    # -----------------------------------------------------------------
    # Create one figure with three panels and one colorbar
    # -----------------------------------------------------------------

    fig = plt.figure(
        figsize=FIGURE_SIZE,
        dpi=150,
    )

    fig.patch.set_facecolor(
        "white"
    )

    grid_spec = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=(
            1.0,
            1.0,
            1.0,
            0.045,
        ),
        left=0.015,
        right=0.965,
        bottom=0.02,
        top=0.98,
        wspace=-0.04,
    )

    axes = [
        fig.add_subplot(
            grid_spec[0, column],
            projection="3d",
        )
        for column in range(3)
    ]

    colorbar_axis = fig.add_subplot(
        grid_spec[0, 3]
    )

    panel_labels = [
        r"$\mathbf{(a)}$",
        r"$\mathbf{(b)}$",
        r"$\mathbf{(c)}$",
    ]

    # -----------------------------------------------------------------
    # Plot the three selected time frames
    # -----------------------------------------------------------------

    for panel_number, (
        ax,
        frame_index,
        panel_label,
    ) in enumerate(
        zip(
            axes,
            FRAME_INDICES,
            panel_labels,
        )
    ):

        frame = extract_frame(
            intensity,
            frame_index,
        )

        frame_db = intensity_to_db(
            frame,
            reference_intensity,
        )

        # Close the azimuthal seam in the color array.
        frame_db_closed = close_phi_seam(
            frame_db
        )

        facecolors = cmap(
            norm(frame_db_closed)
        )

        ax.plot_surface(
            sphere_x,
            sphere_y,
            sphere_z,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        title = get_panel_title(
            data=data,
            frame_index=frame_index,
            panel_label=panel_label,
        )

        configure_axis(
            ax=ax,
            title=title,
            show_axis_labels=(panel_number == 0),
        )

    # -----------------------------------------------------------------
    # Shared colorbar
    # -----------------------------------------------------------------

    scalar_mappable = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar_mappable.set_array(
        []
    )

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

    colorbar.outline.set_linewidth(
        0.6
    )


    plt.show()


if __name__ == "__main__":
    main()
