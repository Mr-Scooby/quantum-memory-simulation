from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from radpattern.plotting import THESIS_STYLE


plt.style.use(THESIS_STYLE)


# Symbolic links to the folders containing the .dat files
DATA_FOLDERS = [
    Path("CS_BeamChange"),
    #Path("RB_BeamChange"),
    #Path("path/to/symlink_folder_2"),
]


def decay_model(time_us, amplitude, tau_us, beta):
    """Stretched-exponential decay model."""
    return amplitude * np.exp(-(time_us / tau_us) ** beta)


def read_data(file_path):
    """Read a data file with named columns."""
    return np.genfromtxt(
        file_path,
        names=True,
        dtype=float,
        encoding="utf-8",
    )


def fit_decay(time_us, coupling):
    """Fit the coupling decay and return its 1/e lifetime."""
    time_us = np.asarray(time_us, dtype=float)
    coupling = np.asarray(coupling, dtype=float)

    valid = (
        np.isfinite(time_us)
        & np.isfinite(coupling)
        & (coupling >= 0)
    )

    time_us = time_us[valid]
    coupling = coupling[valid]

    if len(time_us) < 4:
        raise ValueError("Not enough valid points for the fit.")

    amplitude_guess = coupling[0]
    target = amplitude_guess / np.e

    below_target = np.where(coupling <= target)[0]

    if len(below_target) > 0:
        tau_guess = time_us[below_target[0]]
    else:
        tau_guess = 0.7 * time_us[-1]

    initial_guess = [
        amplitude_guess,
        max(tau_guess, 1.0),
        1.5,
    ]

    lower_bounds = [
        0.0,
        1.0e-12,
        0.2,
    ]

    upper_bounds = [
        np.inf,
        np.inf,
        5.0,
    ]

    parameters, covariance = curve_fit(
        decay_model,
        time_us,
        coupling,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50_000,
    )

    amplitude, tau_us, beta = parameters

    parameter_errors = np.sqrt(np.diag(covariance))
    tau_error_us = parameter_errors[1]

    return amplitude, tau_us, beta, parameter_errors


def extract_fixed_beam(file_path):
    """
    Read the fixed beam type and size from the filename.

    Examples
    --------
    Rb500CbeamVsSbeam.dat
        Control beam fixed at 500 um.
        Signal beam values are stored in the data columns.

    Rb300SbeamVsCbeam.dat
        Signal beam fixed at 300 um.
        Control beam values are stored in the data columns.
    """
    match = re.search(
        r"(\d+)([SC])beamVs([SC])beam",
        file_path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Cannot determine the scan type from filename: "
            f"{file_path.name}"
        )

    fixed_value = int(match.group(1))
    fixed_beam = match.group(2).upper()
    scanned_beam = match.group(3).upper()

    if fixed_beam == scanned_beam:
        raise ValueError(
            f"The fixed and scanned beams are identical in "
            f"{file_path.name}"
        )

    return fixed_beam, fixed_value, scanned_beam


def extract_scanned_beam(column_name):
    """
    Extract the scanned beam size from a column name.

    Example
    -------
    curve_300 -> 300
    """
    match = re.fullmatch(
        r"curve_(\d+)",
        column_name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Cannot determine the beam size from column: "
            f"{column_name}"
        )

    return int(match.group(1))


def find_data_files(data_folders):
    """Find all .dat files inside the supplied folders."""
    data_files = []

    for folder in data_folders:
        folder = folder.expanduser()

        if not folder.exists():
            print(f"Folder does not exist: {folder}")
            continue

        if not folder.is_dir():
            print(f"Path is not a directory: {folder}")
            continue

        data_files.extend(sorted(folder.glob("*.dat")))

    return data_files


def plot_file(file_path):
    def _read_data(file_path):
        data = np.genfromtxt(file_path, names=True)

        time_us = np.asarray(data["time_us"], dtype=float)

        curves = {
            column: np.asarray(data[column], dtype=float)
            for column in data.dtype.names
            if column != "time_us"
        }

        return time_us, curves

    time_us, curves = _read_data(file_path)

    fig, ax = plt.subplots()

    for column, coupling in curves.items():
        amplitude, tau_us, beta, parameter_errors = fit_decay(
            time_us,
            coupling,
        )

        tau_error_us = parameter_errors[1]

        signal_size = column.removeprefix("curve_")

        # Simulation data
        line, = ax.plot(
            time_us,
            coupling,
            linewidth=1.2,
            alpha=0.75,
        )

        # Smooth fitted curve
        fit_time = np.linspace(time_us.min(), time_us.max(), 1000)

        ax.plot(
            fit_time,
            decay_model(fit_time, amplitude, tau_us, beta),
            linestyle="--",
            linewidth=1.2,
            color=line.get_color(),
            label=(
                rf"$d_\mathrm{{s}}={signal_size}\,\mu\mathrm{{m}}$, "
                rf"$\tau={tau_us / 1000:.2f}"
                rf"\pm{tau_error_us / 1000:.2f}\,\mathrm{{ms}}$"
            ),
        )

        print(
            f"{file_path.name:30s} "
            f"{column:12s} "
            f"tau = {tau_us:8.1f} ± {tau_error_us:6.1f} us, "
            f"beta = {beta:.3f}"
        )

    ax.set_xlabel(r"Storage time $t$ ($\mu$s)")
    ax.set_ylabel("Coupling")
    ax.legend(frameon=False)

    fig.tight_layout()

    return fig



def build_tau_grid(data_files):
    """
    Fit all curves and construct a two-dimensional lifetime grid.

    Grid convention
    ---------------
    Rows:
        Control-beam size.

    Columns:
        Signal-beam size.

    Values:
        Fitted 1/e lifetime in milliseconds.
    """
    tau_values = defaultdict(list)
    tau_error_values = defaultdict(list)

    for file_path in data_files:
        try:
            data = read_data(file_path)

            fixed_beam, fixed_value, scanned_beam = (
                extract_fixed_beam(file_path)
            )

        except ValueError as error:
            print(f"Skipping {file_path.name}: {error}")
            continue

        column_names = data.dtype.names

        if column_names is None or "time_us" not in column_names:
            print(
                f"Skipping {file_path.name}: "
                f"no time_us column was found."
            )
            continue

        time_us = data["time_us"]

        for column_name in column_names:
            if column_name == "time_us":
                continue

            try:
                scanned_value = extract_scanned_beam(column_name)

                if fixed_beam == "C":
                    control_beam = fixed_value
                    signal_beam = scanned_value

                elif fixed_beam == "S":
                    signal_beam = fixed_value
                    control_beam = scanned_value

                else:
                    raise ValueError(
                        f"Unknown fixed beam type: {fixed_beam}"
                    )

                coupling = data[column_name]

                amplitude, tau_us, beta, parameter_errors = fit_decay(
                    time_us,
                    coupling,
                )
                
                tau_error_us = parameter_errors[1]

            except (ValueError, RuntimeError) as error:
                print(
                    f"Could not fit {file_path.name}, "
                    f"column {column_name}: { error}"
                )
                continue

            tau_ms = tau_us #/ 1000.0
            tau_error_ms = tau_error_us #/ 1000.0

            tau_values[(control_beam, signal_beam)].append(
                tau_ms
            )

            tau_error_values[(control_beam, signal_beam)].append(
                tau_error_ms
            )

            print(
                f"{file_path.name:30s}  "
                f"Cbeam = {control_beam:4d} um  "
                f"Sbeam = {signal_beam:4d} um  "
                f"tau = {tau_ms:7.3f} +/- "
                f"{tau_error_ms:7.3f} ms  "
                f"beta = {beta:.3f} ({parameter_errors[2]:.3f}) "
                f"amplitude = {amplitude:.3f} ({parameter_errors[0]:.3f})"
                )

    if not tau_values:
        raise RuntimeError("No valid lifetime values were obtained.")

    control_beams = sorted(
        {
            control_beam
            for control_beam, signal_beam in tau_values
        }
    )

    signal_beams = sorted(
        {
            signal_beam
            for control_beam, signal_beam in tau_values
        }
    )

    tau_grid = np.full(
        (len(control_beams), len(signal_beams)),
        np.nan,
    )
    tau_error_grid = np.full(
        (len(control_beams), len(signal_beams)),
        np.nan,
    )

    for row, control_beam in enumerate(control_beams):
        for column, signal_beam in enumerate(signal_beams):
            values = tau_values.get(
                (control_beam, signal_beam),
                [],
            )

    #        if values:
    #            # Average repeated points if the same combination appears
    #            # in both an Sbeam scan and a Cbeam scan.
    #            tau_grid[row, column] = np.mean(values)

            if values:
                tau_grid[row, column] = np.mean(values)

                errors = tau_error_values[
                    (control_beam, signal_beam)
                ]

                # Uncertainty of the mean for independent fitted values
                tau_error_grid[row, column] = (
                    np.sqrt(np.sum(np.asarray(errors) ** 2))
                    / len(errors)
                )
    return control_beams, signal_beams, tau_grid, tau_error_grid


def make_edges(values):
    values = np.asarray(values, dtype=float)
    mid = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], mid, [last]])


def plot_tau_grid(control_beams, signal_beams, tau_grid,tau_error_grid, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x_edges = np.arange(len(signal_beams) + 1)
    y_edges = np.arange(len(control_beams) + 1)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        tau_grid,
        shading="flat",
    )

    ax.set_xticks(np.arange(len(signal_beams)) + 0.5)
    ax.set_xticklabels(signal_beams)

    ax.set_yticks(np.arange(len(control_beams)) + 0.5)
    ax.set_yticklabels(control_beams)

    #ax.set_aspect("equal")
    ax.set_aspect("auto")

    ax.set_xlabel(r"Signal beam $w_{\mathrm{s}}$ ($\mu$m)")
    ax.set_ylabel(r"Control beam $w_{\mathrm{c}}$ ($\mu$m)")

    finite_values = tau_grid[np.isfinite(tau_grid)]
    colour_threshold = 0.5 * (finite_values.min() + finite_values.max())

    for i, cbeam in enumerate(control_beams):
        for j, sbeam in enumerate(signal_beams):
            value = tau_grid[i, j]
            if np.isfinite(value):
                #color = "white" if value < threshold else "black"

                text_colour = (
                    "white"
                    if value < colour_threshold
                    else "black"
                )
                error = tau_error_grid[i, j]

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    #rf"${value:.2f}\pm{error:.2f}$",
                    rf"${value:.3f}({error * 1000:.0f})$",
                    ha="center",
                    va="center",
                    color=text_colour,
                )

                #ax.text(
                #    j + 0.5,
                #    i + 0.5,
                #    f"{value:.2f}",
                #    ha="center",
                #    va="center",
                #    color=text_colour,
                #)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(r"$\tau$ ($\mu$s)")

    #fig.tight_layout()
    return fig, ax



def plot_tau_grid_V0(

    control_beams,
    signal_beams,
    tau_grid,
    ax=None,
):
    """
    Plot the lifetime grid.

    Horizontal axis:
        Signal-beam size.

    Vertical axis:
        Control-beam size.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    masked_tau_grid = np.ma.masked_invalid(tau_grid)

    image = ax.imshow(
        masked_tau_grid,
        origin="lower",
        aspect="auto",
        interpolation="none",
    )

    ax.set_xticks(np.arange(len(signal_beams)))
    ax.set_xticklabels(signal_beams)

    ax.set_yticks(np.arange(len(control_beams)))
    ax.set_yticklabels(control_beams)

    ax.set_xlabel(
        r"Signal Beam $w_{\mathrm{s}}$ ($\mu$m)"
    )
    ax.set_ylabel(
        r"Control beam $w_{\mathrm{c}}$ ($\mu$m)"
    )

    finite_values = tau_grid[np.isfinite(tau_grid)]

    if finite_values.size > 0:
        colour_threshold = (
            finite_values.min() + finite_values.max()
        ) / 2.0
    else:
        colour_threshold = 0.0

    for row in range(len(control_beams)):
        for column in range(len(signal_beams)):
            tau_ms = tau_grid[row, column]

            if not np.isfinite(tau_ms):
                continue

            text_colour = (
                "white"
                if tau_ms < colour_threshold
                else "black"
            )

            ax.text(
                column,
                row,
                rf"{tau_ms:.2f}",
                ha="center",
                va="center",
                color=text_colour,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.03,
    )

    colorbar.set_label(
        r"$\tau$ ($m s$)"
    )

    fig.tight_layout()

    return fig, ax


def main():
    data_files = find_data_files(DATA_FOLDERS)

    if not data_files:
        raise FileNotFoundError(
            "No .dat files were found in DATA_FOLDERS."
        )

    #control_beams, signal_beams, tau_grid = build_tau_grid(
    #    data_files
    #)
    (
        control_beams,
        signal_beams,
        tau_grid,
        tau_error_grid,
    ) = build_tau_grid(data_files)

    plot_tau_grid(
        control_beams,
        signal_beams,
        tau_grid,
        tau_error_grid,
    )

    #plt.show()
#
#    for file_path in data_files:
#        pass
#        plot_file(file_path)
#
#    plt.show()


if __name__ == "__main__":
    main()
