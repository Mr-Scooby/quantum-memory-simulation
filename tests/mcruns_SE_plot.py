#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo convergence plot for the fibre-coupled retrieval curve.

The script loads all saved Monte Carlo realizations from one simulation
folder and forms nested averages using the first N runs, for example

    N_MC = 10, 30, 100.

For each sample count, the mean normalized retrieval curve is plotted
together with its standard-error band,

    SE(t) = s(t) / sqrt(N_MC),

where s(t) is the sample standard deviation across the individual
Monte Carlo realizations at each time point.

The output contains one publication-style Matplotlib axis.

No TikZ or matplot2tikz is used.

Expected structure
------------------

Either

result_sim/
    cs133_ABC.npz
    cs133_ABC/
        mc_0000.npz
        mc_0001.npz
        ...

or

result_sim/
    cs133_ABC.npz
    cs133_ABC_mc_runs/
        cs133_ABC.npz
        mc_0000.npz
        mc_0001.npz
        ...

Each MC file must contain

    AF2

and may optionally contain

    times_code
"""

from __future__ import annotations

import argparse
import re
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radpattern.geometry.grids import AngleGrid
from radpattern.helpers.helpers import single_dipole_E
from radpattern.physics.coupling import gaussian_fiber_mode_on_sphere
from radpattern.physics.experimetal_setup import ExperimentalParams
from radpattern.physics.setup_params import SimParams
from radpattern.plotting import THESIS_STYLE


plt.style.use(THESIS_STYLE)


DEFAULT_SAMPLE_COUNTS = (10, 30, 100)
DEFAULT_MAX_TIME_US = 200.0
LOG_FLOOR = 1.0e-12


# ---------------------------------------------------------------------
# Metadata reconstruction
# ---------------------------------------------------------------------

def default_from_type(name, typ):
    """
    Choose a backward-compatible placeholder value for missing metadata.
    """

    if typ is float:
        return 999.9

    if typ is int:
        return 999

    if typ is str:
        return "None"

    if typ is bool:
        return False

    if typ is tuple:
        return (-1, -1, -1)

    if typ is list:
        return [-1, -1, -1]

    if typ is dict:
        return {"None": "None"}

    return None


def dataclass_kwargs(cls, data):
    """
    Keep only metadata keys accepted by a dataclass constructor.
    """

    kwargs = {}

    for field_info in fields(cls):
        if not field_info.init:
            continue

        if field_info.name in data:
            kwargs[field_info.name] = data[field_info.name]
        else:
            kwargs[field_info.name] = default_from_type(
                field_info.name,
                field_info.type,
            )

    return kwargs


def load_metadata(parent_npz_path: Path) -> dict:
    """
    Read the metadata dictionary stored in the parent result file.
    """

    with np.load(parent_npz_path, allow_pickle=True) as parent:
        return parent["metadata"].item()


def build_grid_from_metadata(metadata: dict) -> AngleGrid:
    """
    Reconstruct the angular grid used by the simulation.
    """

    sim_meta = metadata["sim"]

    return AngleGrid(
        n_theta=sim_meta["n_theta"],
        n_phi=sim_meta["n_phi"],
        theta_max=sim_meta["theta_max"],
    )


def build_exp_from_metadata(
    metadata: dict,
) -> ExperimentalParams:
    """
    Reconstruct the experimental-parameter object.
    """

    exp_meta = metadata.get(
        "experiment",
        metadata.get("regime"),
    )

    if exp_meta is None:
        raise KeyError(
            "Metadata contains neither 'experiment' nor 'regime'."
        )

    return ExperimentalParams(
        **dataclass_kwargs(
            ExperimentalParams,
            exp_meta,
        )
    )


def build_sim_from_metadata(
    metadata: dict,
) -> SimParams:
    """
    Reconstruct the simulation-parameter object.
    """

    return SimParams(
        **dataclass_kwargs(
            SimParams,
            metadata["sim"],
        )
    )


# ---------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------

def natural_mc_key(path: Path):
    """
    Sort MC files by their trailing integer.

    This ensures that mc_2 is placed before mc_10 when the filenames
    are not zero padded.
    """

    match = re.search(
        r"(\d+)(?=\.npz$)",
        path.name,
    )

    if match:
        return int(match.group(1))

    return path.name


def find_mc_files(
    mc_folder: Path,
) -> list[Path]:
    """
    Locate and naturally sort the saved Monte Carlo result files.
    """

    files = list(
        mc_folder.glob("mc_*.npz")
    )

    if not files:
        files = list(
            mc_folder.glob("mc_run*.npz")
        )

    if not files:
        raise FileNotFoundError(
            "No Monte Carlo NPZ files were found in:\n"
            f"  {mc_folder}"
        )

    return sorted(
        files,
        key=natural_mc_key,
    )


def resolve_result_paths(
    input_path: Path,
) -> tuple[Path, Path]:
    """
    Resolve the parent result file and the MC-run folder.

    Parameters
    ----------
    input_path
        Either the parent NPZ file or the directory containing the
        individual Monte Carlo files.

    Returns
    -------
    parent_npz_path
        Main result file containing the metadata.

    mc_folder
        Directory containing the individual mc_*.npz files.
    """

    input_path = input_path.expanduser().resolve()

    if input_path.is_dir():
        mc_folder = input_path

        parent_stem = mc_folder.name

        if parent_stem.endswith("_mc_runs"):
            parent_stem = parent_stem.removesuffix(
                "_mc_runs"
            )

        parent_candidates = [
            mc_folder / f"{parent_stem}.npz",
            mc_folder.parent / f"{parent_stem}.npz",
        ]

        for candidate in parent_candidates:
            if candidate.exists():
                return candidate, mc_folder

        raise FileNotFoundError(
            "Could not locate the parent metadata NPZ file.\n"
            "Tried:\n  "
            + "\n  ".join(
                str(path)
                for path in parent_candidates
            )
        )

    if (
        input_path.is_file()
        and input_path.suffix == ".npz"
    ):
        parent_npz_path = input_path
        root = parent_npz_path.parent
        stem = parent_npz_path.stem

        folder_candidates = [
            root / stem,
            root / f"{stem}_mc_runs",
        ]

        for candidate in folder_candidates:
            if candidate.is_dir():
                return parent_npz_path, candidate

        raise FileNotFoundError(
            "Could not locate the Monte Carlo folder.\n"
            "Tried:\n  "
            + "\n  ".join(
                str(path)
                for path in folder_candidates
            )
        )

    raise FileNotFoundError(
        "Input path does not exist or is unsupported:\n"
        f"  {input_path}"
    )


# ---------------------------------------------------------------------
# Fibre-coupling calculation
# ---------------------------------------------------------------------

def coupling_from_AF2(
    AF2_t: np.ndarray,
    grid: AngleGrid,
    dipole: np.ndarray,
    E_fib: np.ndarray,
    theta0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate fibre power, total power, and instantaneous coupling.

    Parameters
    ----------
    AF2_t
        Far-field array with shape

            (n_times, n_theta, n_phi).

    grid
        Angular integration grid.

    dipole
        Single-dipole angular emission pattern.

    E_fib
        Fibre-mode intensity profile on the angular grid.

    theta0
        Maximum angular integration aperture.

    Returns
    -------
    P_fib_t
        Fibre-coupled power at every time.

    P_tot_t
        Total power inside the angular integration aperture.

    eta_t
        Instantaneous ratio P_fib(t) / P_tot(t).
    """

    AF2_t = np.asarray(AF2_t)
    AF2_t = np.squeeze(AF2_t)

    if AF2_t.ndim != 3:
        raise ValueError(
            "AF2 must have shape (time, theta, phi); "
            f"received {AF2_t.shape}."
        )

    n_times = AF2_t.shape[0]

    P_fib_t = np.zeros(
        n_times,
        dtype=float,
    )

    P_tot_t = np.zeros(
        n_times,
        dtype=float,
    )

    eta_t = np.zeros(
        n_times,
        dtype=float,
    )

    mask = grid.TH <= theta0

    theta = grid.TH[:, 0]
    phi = grid.PH[0, :]
    sin_theta = np.sin(grid.TH)

    E_fib_masked = np.where(
        mask,
        E_fib,
        0.0,
    )

    for time_index in range(n_times):
        intensity = (
            AF2_t[time_index]
            * dipole
        )

        intensity_masked = np.where(
            mask,
            intensity,
            0.0,
        )

        P_fib = np.trapezoid(
            np.trapezoid(
                intensity_masked
                * E_fib_masked
                * sin_theta,
                phi,
                axis=1,
            ),
            theta,
            axis=0,
        )

        P_tot = np.trapezoid(
            np.trapezoid(
                intensity_masked
                * sin_theta,
                phi,
                axis=1,
            ),
            theta,
            axis=0,
        )

        P_fib_t[time_index] = P_fib
        P_tot_t[time_index] = P_tot

        eta_t[time_index] = (
            P_fib
            / (P_tot + 1.0e-30)
        )

    return P_fib_t, P_tot_t, eta_t


def load_times_us(
    mc_data,
    parent_npz_path: Path,
    sim: SimParams,
) -> np.ndarray:
    """
    Load the simulation time and convert it to microseconds.
    """

    if "times_code" in mc_data:
        times_code = np.asarray(
            mc_data["times_code"],
            dtype=float,
        )
    else:
        with np.load(
            parent_npz_path,
            allow_pickle=True,
        ) as parent:
            times_code = np.asarray(
                parent["times_code"],
                dtype=float,
            )

    return (
        times_code
        * sim.char_time
        * 1.0e6
    )


def load_mc_curves(
    parent_npz_path: Path,
    mc_folder: Path,
    exp: ExperimentalParams,
    sim: SimParams,
    grid: AngleGrid,
    required_runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the individual MC runs and calculate P_fib(t) / P_tot(0).

    The initial total power P_tot(0) is calculated separately for each
    realization. This removes run-to-run differences in absolute initial
    power and tests convergence of the normalized retrieval curve.

    Returns
    -------
    times_us
        One-dimensional simulation-time array in microseconds.

    curves
        Array with shape

            (required_runs, n_times).

        Each row contains one normalized Monte Carlo retrieval curve.
    """

    mc_files = find_mc_files(
        mc_folder
    )

    if len(mc_files) < required_runs:
        raise ValueError(
            f"The convergence test requests {required_runs} runs, "
            f"but only {len(mc_files)} MC files were found in:\n"
            f"  {mc_folder}"
        )

    mc_files = mc_files[:required_runs]

    dipole = single_dipole_E(
        grid.nx,
        grid.ny,
        grid.nz,
        np.array(
            [1.0, 0.0, 0.0]
        ),
    )

    theta0 = (
        12.0
        / (
            exp.atom.k_signal
            * exp.w0_signal
        )
    )

    E_fib = np.abs(
        gaussian_fiber_mode_on_sphere(
            grid,
            theta0,
        )
    ) ** 2

    curves = []
    reference_times_us = None

    for run_index, mc_file in enumerate(
        mc_files,
        start=1,
    ):
        with np.load(
            mc_file,
            allow_pickle=True,
        ) as data:
            AF2_t = data["AF2"]

            times_us = load_times_us(
                mc_data=data,
                parent_npz_path=parent_npz_path,
                sim=sim,
            )

        P_fib, P_tot, _ = coupling_from_AF2(
            AF2_t=AF2_t,
            grid=grid,
            dipole=dipole,
            E_fib=E_fib,
            theta0=theta0,
        )

        curve = (
            P_fib
            / (P_tot[0] + 1.0e-30)
        )

        if reference_times_us is None:
            reference_times_us = times_us
        else:
            if (
                times_us.shape
                != reference_times_us.shape
            ):
                raise ValueError(
                    "Time-array shape mismatch in "
                    f"{mc_file.name}."
                )

            if not np.allclose(
                times_us,
                reference_times_us,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "Time-array values differ in "
                    f"{mc_file.name}."
                )

        curves.append(curve)

        print(
            f"\rCalculated run "
            f"{run_index}/{required_runs}",
            end="",
            flush=True,
        )

    print()

    return (
        reference_times_us,
        np.asarray(curves),
    )


# ---------------------------------------------------------------------
# Monte Carlo statistics
# ---------------------------------------------------------------------

def calculate_mc_statistics(
    curves: np.ndarray,
    sample_counts: tuple[int, ...],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    """
    Calculate nested means, sample standard deviations, and standard errors.

    For each requested number of realizations N, the first N rows of
    ``curves`` are used.

    The standard error at each time is

        SE(t) = s(t) / sqrt(N),

    where s(t) is the sample standard deviation calculated with
    ``ddof=1``.

    Parameters
    ----------
    curves
        Array with shape (n_runs, n_times).

    sample_counts
        Nested Monte Carlo sample counts.

    Returns
    -------
    means
        Dictionary mapping N_MC to its mean curve.

    standard_deviations
        Dictionary mapping N_MC to its sample-standard-deviation curve.

    standard_errors
        Dictionary mapping N_MC to its standard-error curve.
    """

    means = {}
    standard_deviations = {}
    standard_errors = {}

    for count in sample_counts:
        subset = curves[:count]

        mean_curve = np.mean(
            subset,
            axis=0,
        )

        sample_std = np.std(
            subset,
            axis=0,
            ddof=1,
        )

        standard_error = (
            sample_std
            / np.sqrt(count)
        )

        means[count] = mean_curve
        standard_deviations[count] = sample_std
        standard_errors[count] = standard_error

    return (
        means,
        standard_deviations,
        standard_errors,
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_mc_convergence(
    times_us: np.ndarray,
    curves: np.ndarray,
    sample_counts: tuple[int, ...],
    max_time_us: float | None,
    output_path: Path | None,
    use_log_y: bool = False,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the Monte Carlo mean curves and standard-error bands.

    The figure contains only one axis. For each requested value of
    N_MC, the mean normalized retrieval curve is plotted together with

        mean(t) - SE(t)

    and

        mean(t) + SE(t).

    Parameters
    ----------
    times_us
        One-dimensional array of simulation times in microseconds.

    curves
        Array with shape (n_runs, n_times).

    sample_counts
        Monte Carlo sample counts to compare.

    max_time_us
        Maximum plotted time in microseconds. Use None to plot the
        complete simulation interval.

    output_path
        Optional output file. The extension determines the Matplotlib
        output format.

    use_log_y
        Use a logarithmic vertical axis when True.

    title
        Optional figure title.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """

    times_us = np.asarray(
        times_us,
        dtype=float,
    )

    curves = np.asarray(
        curves,
        dtype=float,
    )

    if times_us.ndim != 1:
        raise ValueError(
            "times_us must be one-dimensional."
        )

    if curves.ndim != 2:
        raise ValueError(
            "curves must have shape "
            "(n_runs, n_times); "
            f"received {curves.shape}."
        )

    if curves.shape[1] != times_us.size:
        raise ValueError(
            "The number of time points in curves "
            "does not match times_us."
        )

    sample_counts = tuple(
        sorted(
            set(
                int(count)
                for count in sample_counts
            )
        )
    )

    if not sample_counts:
        raise ValueError(
            "At least one sample count is required."
        )

    if any(
        count < 2
        for count in sample_counts
    ):
        raise ValueError(
            "Every sample count must be at least 2 "
            "to calculate a sample standard error."
        )

    if max(sample_counts) > curves.shape[0]:
        raise ValueError(
            f"N_MC={max(sample_counts)} was requested, "
            f"but only {curves.shape[0]} curves are available."
        )

    if max_time_us is None:
        plot_mask = np.ones(
            times_us.shape,
            dtype=bool,
        )
    else:
        plot_mask = (
            times_us
            <= max_time_us
        )

        if not np.any(plot_mask):
            raise ValueError(
                "No simulation times lie below "
                f"max_time_us={max_time_us}."
            )

    t_plot = times_us[plot_mask]

    (
        means,
        _,
        standard_errors,
    ) = calculate_mc_statistics(
        curves=curves,
        sample_counts=sample_counts,
    )

    fig, ax = plt.subplots(
        figsize=(6.4, 3.9),
    )

    line_styles = (
        ":",
        "--",
        "-",
        "-.",
    )

    largest_count = max(sample_counts)

    for style_index, count in enumerate(
        sample_counts
    ):
        mean_curve = (
            means[count][plot_mask]
        )

        se_curve = (
            standard_errors[count][plot_mask]
        )

        line_style = line_styles[
            min(
                style_index,
                len(line_styles) - 1,
            )
        ]

        line, = ax.plot(
            t_plot,
            mean_curve,
            linestyle=line_style,
            linewidth=(
                2.0
                if count == largest_count
                else 1.6
            ),
            label=(
                rf"$N_{{\mathrm{{MC}}}}={count}$"
            ),
            zorder=3,
        )

        lower_bound = (
            mean_curve
            - se_curve
        )

        upper_bound = (
            mean_curve
            + se_curve
        )

        if use_log_y:
            lower_bound = np.maximum(
                lower_bound,
                LOG_FLOOR,
            )

            upper_bound = np.maximum(
                upper_bound,
                LOG_FLOOR,
            )
        else:
            lower_bound = np.maximum(
                lower_bound,
                0.0,
            )

        ax.fill_between(
            t_plot,
            lower_bound,
            upper_bound,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

        relative_se = (
            se_curve
            / (
                np.abs(mean_curve)
                + 1.0e-30
            )
        )

        mean_relative_se = np.mean(
            relative_se
        )

        maximum_relative_se = np.max(
            relative_se
        )

        print(
            f"N_MC={count:>3d}: "
            f"mean relative SE = "
            f"{mean_relative_se:.6e}, "
            f"maximum relative SE = "
            f"{maximum_relative_se:.6e}"
        )

    if use_log_y:
        ax.set_yscale("log")

    ax.set_xlabel(
        r"Time [$\mu$s]"
    )

    ax.set_ylabel(
        r"$\left\langle "
        r"P_{\mathrm{fib}}(t)"
        r"/P_{\mathrm{tot}}(0)"
        r"\right\rangle_{\mathrm{MC}}$"
    )

    if title:
        ax.set_title(title)

    ax.set_xlim(
        t_plot[0],
        t_plot[-1],
    )

    ax.grid(
        True,
        which=(
            "both"
            if use_log_y
            else "major"
        ),
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="best",
        ncol=len(sample_counts),
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = (
            output_path
            .expanduser()
            .resolve()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )

        print(
            "Saved Monte Carlo convergence figure to:\n"
            f"  {output_path}"
        )

    return fig, ax


# ---------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------

def run_convergence_plot(
    input_path: Path,
    sample_counts: tuple[int, ...] = DEFAULT_SAMPLE_COUNTS,
    max_time_us: float | None = DEFAULT_MAX_TIME_US,
    output_path: Path | None = None,
    use_log_y: bool = False,
    title: str | None = None,
):
    """
    Load the MC data, calculate the statistics, and create the plot.
    """

    sample_counts = tuple(
        sorted(
            set(
                int(value)
                for value in sample_counts
            )
        )
    )

    if not sample_counts:
        raise ValueError(
            "At least one sample count is required."
        )

    parent_npz_path, mc_folder = resolve_result_paths(
        input_path
    )

    metadata = load_metadata(
        parent_npz_path
    )

    exp = build_exp_from_metadata(
        metadata
    )

    sim = build_sim_from_metadata(
        metadata
    )

    grid = build_grid_from_metadata(
        metadata
    )

    print(
        f"Parent result: {parent_npz_path}"
    )

    print(
        f"MC folder:     {mc_folder}"
    )

    print(
        f"Sample counts: {sample_counts}"
    )

    times_us, curves = load_mc_curves(
        parent_npz_path=parent_npz_path,
        mc_folder=mc_folder,
        exp=exp,
        sim=sim,
        grid=grid,
        required_runs=max(sample_counts),
    )

    if output_path is None:
        output_path = mc_folder / (
            parent_npz_path.stem
            + "_mc_mean_with_standard_error.pdf"
        )

    return plot_mc_convergence(
        times_us=times_us,
        curves=curves,
        sample_counts=sample_counts,
        max_time_us=max_time_us,
        output_path=output_path,
        use_log_y=use_log_y,
        title=title,
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot Monte Carlo mean retrieval curves with "
            "standard-error bands."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "Parent NPZ file or directory containing the saved "
            "Monte Carlo runs. When omitted, the path is requested "
            "interactively."
        ),
    )

    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SAMPLE_COUNTS
        ),
        help=(
            "Nested Monte Carlo sample counts. "
            "Default: --counts 10 30 100"
        ),
    )

    parser.add_argument(
        "--max-time",
        type=float,
        default=DEFAULT_MAX_TIME_US,
        help=(
            "Largest plotted time in microseconds. "
            "Use a negative value to plot the complete time range."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output filename. The extension determines the "
            "Matplotlib format, for example PDF, PNG, or SVG."
        ),
    )

    parser.add_argument(
        "--log-y",
        action="store_true",
        help=(
            "Use a logarithmic vertical axis. "
            "The default vertical axis is linear."
        ),
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help=(
            "Optional title shown above the plot. "
            "By default no title is drawn."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_args()

    input_path = args.input

    if input_path is None:
        input_text = input(
            "Parent NPZ file or MC folder: "
        ).strip()

        if not input_text:
            raise ValueError(
                "No input path was provided."
            )

        input_path = Path(
            input_text
        )

    max_time_us = (
        None
        if args.max_time < 0
        else args.max_time
    )

    run_convergence_plot(
        input_path=input_path,
        sample_counts=tuple(
            args.counts
        ),
        max_time_us=max_time_us,
        output_path=args.output,
        use_log_y=args.log_y,
        title=args.title,
    )

    plt.show()


if __name__ == "__main__":
    main()
