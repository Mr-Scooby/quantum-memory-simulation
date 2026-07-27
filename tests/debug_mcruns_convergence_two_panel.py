#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo convergence plot for the fibre-coupled retrieval curve.

This script loads all saved MC runs from one simulation folder and forms
nested averages using the first N runs.

It produces one publication-style Matplotlib figure containing

    (a) mean retrieval curves with standard-error bands for
        N_MC = 10, 30, 100 by default;
    (b) one convergence curve epsilon_RMS(N_MC) versus
        N_MC = 10, 20, 30, 50, 70, 100 by default.

The convergence curve is estimated from bootstrap-resampled subsets drawn
with replacement from the available MC runs and compared against the grand
mean of all loaded runs over the plotted time window.

No TikZ or matplot2tikz is used.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from radpattern.physics.experimetal_setup import ExperimentalParams
from radpattern.physics.setup_params import SimParams
from radpattern.geometry.grids import AngleGrid
from radpattern.helpers.helpers import single_dipole_E
from radpattern.physics.coupling import gaussian_fiber_mode_on_sphere
from radpattern.plotting import THESIS_STYLE


plt.style.use(THESIS_STYLE)


DEFAULT_SAMPLE_COUNTS = (10, 30, 100)
DEFAULT_RMS_COUNTS = (10, 20, 30, 50, 70, 100)
DEFAULT_BOOTSTRAP_SUBSETS = 250
DEFAULT_RANDOM_SEED = 20260727
DEFAULT_MAX_TIME_US = 200.0
RESIDUAL_FLOOR = 1.0e-12


# ---------------------------------------------------------------------
# Metadata reconstruction
# ---------------------------------------------------------------------

def default_from_type(name, typ):
    """Choose an obvious backward-compatible placeholder value."""

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
    """Keep only keys accepted by a dataclass constructor."""

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
    """Read the metadata dictionary stored in the parent result file."""

    with np.load(parent_npz_path, allow_pickle=True) as parent:
        return parent["metadata"].item()


def build_grid_from_metadata(metadata: dict) -> AngleGrid:
    """Reconstruct the angular grid used by the simulation."""

    sim_meta = metadata["sim"]

    return AngleGrid(
        n_theta=sim_meta["n_theta"],
        n_phi=sim_meta["n_phi"],
        theta_max=sim_meta["theta_max"],
    )


def build_exp_from_metadata(metadata: dict) -> ExperimentalParams:
    """Reconstruct the experimental-parameter object."""

    exp_meta = metadata.get(
        "experiment",
        metadata.get("regime"),
    )

    if exp_meta is None:
        raise KeyError(
            "Metadata contains neither 'experiment' nor 'regime'."
        )

    return ExperimentalParams(
        **dataclass_kwargs(ExperimentalParams, exp_meta)
    )


def build_sim_from_metadata(metadata: dict) -> SimParams:
    """Reconstruct the simulation-parameter object."""

    return SimParams(
        **dataclass_kwargs(SimParams, metadata["sim"])
    )


# ---------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------

def natural_mc_key(path: Path):
    """
    Sort MC files by their trailing integer rather than lexicographically.
    """

    match = re.search(r"(\d+)(?=\.npz$)", path.name)

    if match:
        return int(match.group(1))

    return path.name


def find_mc_files(mc_folder: Path) -> list[Path]:
    """Locate and naturally sort the saved MC result files."""

    files = list(mc_folder.glob("mc_*.npz"))

    if not files:
        files = list(mc_folder.glob("mc_run*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No MC NPZ files were found in:\n  {mc_folder}"
        )

    return sorted(files, key=natural_mc_key)


def resolve_result_paths(input_path: Path) -> tuple[Path, Path]:
    """
    Resolve the parent metadata file and the folder containing MC files.
    """

    input_path = input_path.expanduser().resolve()

    if input_path.is_dir():
        mc_folder = input_path

        parent_stem = mc_folder.name
        if parent_stem.endswith("_mc_runs"):
            parent_stem = parent_stem.removesuffix("_mc_runs")

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
            + "\n  ".join(str(path) for path in parent_candidates)
        )

    if input_path.is_file() and input_path.suffix == ".npz":
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
            "Could not locate the MC folder.\n"
            "Tried:\n  "
            + "\n  ".join(str(path) for path in folder_candidates)
        )

    raise FileNotFoundError(
        f"Input path does not exist or is unsupported:\n  {input_path}"
    )


# ---------------------------------------------------------------------
# Coupling calculation
# ---------------------------------------------------------------------

def coupling_from_AF2(
    AF2_t: np.ndarray,
    grid: AngleGrid,
    dipole: np.ndarray,
    E_fib: np.ndarray,
    theta0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate fibre power, total power, and instantaneous coupling."""

    AF2_t = np.asarray(AF2_t)
    AF2_t = np.squeeze(AF2_t)

    if AF2_t.ndim != 3:
        raise ValueError(
            f"AF2 must have shape (time, theta, phi); got {AF2_t.shape}."
        )

    n_times = AF2_t.shape[0]

    P_fib_t = np.zeros(n_times, dtype=float)
    P_tot_t = np.zeros(n_times, dtype=float)
    eta_t = np.zeros(n_times, dtype=float)

    mask = grid.TH <= theta0

    theta = grid.TH[:, 0]
    phi = grid.PH[0, :]
    sin_theta = np.sin(grid.TH)

    E_fib_masked = np.where(mask, E_fib, 0.0)

    for time_index in range(n_times):
        intensity = AF2_t[time_index] * dipole
        intensity_masked = np.where(mask, intensity, 0.0)

        P_fib = np.trapezoid(
            np.trapezoid(
                intensity_masked * E_fib_masked * sin_theta,
                phi,
                axis=1,
            ),
            theta,
            axis=0,
        )

        P_tot = np.trapezoid(
            np.trapezoid(
                intensity_masked * sin_theta,
                phi,
                axis=1,
            ),
            theta,
            axis=0,
        )

        P_fib_t[time_index] = P_fib
        P_tot_t[time_index] = P_tot
        eta_t[time_index] = P_fib / (P_tot + 1.0e-30)

    return P_fib_t, P_tot_t, eta_t


def load_times_us(mc_data, parent_npz_path: Path, sim: SimParams) -> np.ndarray:
    """Load simulation time and convert it to microseconds."""

    if "times_code" in mc_data:
        times_code = np.asarray(mc_data["times_code"], dtype=float)
    else:
        with np.load(parent_npz_path, allow_pickle=True) as parent:
            times_code = np.asarray(parent["times_code"], dtype=float)

    return times_code * sim.char_time * 1.0e6


def load_mc_curves(
    parent_npz_path: Path,
    mc_folder: Path,
    exp: ExperimentalParams,
    sim: SimParams,
    grid: AngleGrid,
    required_runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load each MC run and calculate P_fib(t) / P_tot(0).
    """

    mc_files = find_mc_files(mc_folder)

    if len(mc_files) < required_runs:
        raise ValueError(
            f"The convergence test requests {required_runs} runs, but only "
            f"{len(mc_files)} MC files were found in:\n  {mc_folder}"
        )

    mc_files = mc_files[:required_runs]

    dipole = single_dipole_E(
        grid.nx,
        grid.ny,
        grid.nz,
        np.array([1.0, 0.0, 0.0]),
    )

    theta0 = 12.0 / (exp.atom.k_signal * exp.w0_signal)
    E_fib = np.abs(gaussian_fiber_mode_on_sphere(grid, theta0)) ** 2

    curves = []
    reference_times_us = None

    for run_index, mc_file in enumerate(mc_files, start=1):
        with np.load(mc_file, allow_pickle=True) as data:
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

        curve = P_fib / (P_tot[0] + 1.0e-30)

        if reference_times_us is None:
            reference_times_us = times_us
        else:
            if times_us.shape != reference_times_us.shape:
                raise ValueError(f"Time-array shape mismatch in {mc_file.name}.")

            if not np.allclose(
                times_us,
                reference_times_us,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(f"Time-array values differ in {mc_file.name}.")

        curves.append(curve)

        print(
            f"\rCalculated run {run_index}/{required_runs}",
            end="",
            flush=True,
        )

    print()

    return reference_times_us, np.asarray(curves)


# ---------------------------------------------------------------------
# Convergence statistics
# ---------------------------------------------------------------------

def cumulative_statistics(
    curves: np.ndarray,
    sample_counts: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Calculate nested means and standard errors using the first N runs."""

    means = {}
    standard_errors = {}

    for count in sample_counts:
        subset = curves[:count]

        means[count] = np.mean(subset, axis=0)

        if count > 1:
            standard_errors[count] = np.std(subset, axis=0, ddof=1) / np.sqrt(count)
        else:
            standard_errors[count] = np.zeros_like(means[count])

    return means, standard_errors


def bootstrap_rms_convergence(
    curves: np.ndarray,
    sample_counts: tuple[int, ...],
    plot_mask: np.ndarray,
    n_subsets: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimate RMS sampling error from bootstrap subsets."""

    counts = np.asarray(sorted(set(int(value) for value in sample_counts)), dtype=int)

    if counts.size == 0:
        raise ValueError("At least one RMS sample count is required.")
    if np.any(counts <= 0):
        raise ValueError("Every RMS sample count must be positive.")
    if n_subsets <= 0:
        raise ValueError("The number of random subsets must be positive.")
    if curves.ndim != 2:
        raise ValueError(f"curves must have shape (runs, time); got {curves.shape}.")
    if plot_mask.shape != (curves.shape[1],):
        raise ValueError("plot_mask does not match the curve time axis.")
    if not np.any(plot_mask):
        raise ValueError("The RMS-error time window is empty.")

    rng = np.random.default_rng(random_seed)
    reference = np.mean(curves[:, plot_mask], axis=0)
    denominator = abs(reference[0]) + 1.0e-30

    all_errors = np.empty((counts.size, n_subsets), dtype=float)

    for count_index, count in enumerate(counts):
        for subset_index in range(n_subsets):
            indices = rng.integers(low=0, high=curves.shape[0], size=count)
            subset_mean = np.mean(curves[indices][:, plot_mask], axis=0)
            all_errors[count_index, subset_index] = (
                np.sqrt(np.mean((subset_mean - reference) ** 2)) / denominator
            )

    mean_errors = np.mean(all_errors, axis=1)
    std_errors = np.std(all_errors, axis=1, ddof=1)

    positive = mean_errors > 0.0

    if np.count_nonzero(positive) >= 2:
        fitted_slope, _ = np.polyfit(
            np.log(counts[positive]),
            np.log(mean_errors[positive]),
            deg=1,
        )
    else:
        fitted_slope = np.nan

    if np.any(positive):
        log_amplitude = np.mean(
            np.log(mean_errors[positive]) + 0.5 * np.log(counts[positive])
        )
        inverse_sqrt_guide = np.exp(log_amplitude) * counts.astype(float) ** (-0.5)
    else:
        inverse_sqrt_guide = np.full(counts.shape, np.nan, dtype=float)

    return counts, mean_errors, std_errors, inverse_sqrt_guide, float(fitted_slope)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_convergence(
    times_us: np.ndarray,
    curves: np.ndarray,
    sample_counts: tuple[int, ...],
    rms_counts: tuple[int, ...],
    bootstrap_subsets: int,
    random_seed: int,
    max_time_us: float | None,
    title: str,
    output_path: Path | None,
    use_log_main_axis: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create the requested two-panel convergence figure."""

    sample_counts = tuple(sorted(set(int(value) for value in sample_counts)))
    rms_counts = tuple(sorted(set(int(value) for value in rms_counts)))

    if any(value <= 0 for value in sample_counts):
        raise ValueError("Every sample count must be positive.")
    if any(value <= 0 for value in rms_counts):
        raise ValueError("Every RMS sample count must be positive.")
    if max(sample_counts) > curves.shape[0]:
        raise ValueError(
            f"Requested N_MC={max(sample_counts)}, but only {curves.shape[0]} curves are available."
        )

    means, standard_errors = cumulative_statistics(curves=curves, sample_counts=sample_counts)

    if max_time_us is None:
        plot_mask = np.ones_like(times_us, dtype=bool)
    else:
        plot_mask = times_us <= max_time_us
        if not np.any(plot_mask):
            raise ValueError(f"No time points lie below max_time_us={max_time_us}.")

    t_plot = times_us[plot_mask]

    rms_count_array, rms_mean, rms_std, inverse_sqrt_guide, fitted_slope = bootstrap_rms_convergence(
        curves=curves,
        sample_counts=rms_counts,
        plot_mask=plot_mask,
        n_subsets=bootstrap_subsets,
        random_seed=random_seed,
    )

    fig = plt.figure(figsize=(8.0, 3.8), layout="constrained")
    grid_spec = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(2.2, 1.15))

    ax_main = fig.add_subplot(grid_spec[0, 0])
    ax_rms = fig.add_subplot(grid_spec[0, 1])

    line_styles = (":", "--", "-")

    for style_index, count in enumerate(sample_counts):
        mean_curve = means[count][plot_mask]
        sem_curve = standard_errors[count][plot_mask]
        line_style = line_styles[min(style_index, len(line_styles) - 1)]

        line, = ax_main.plot(
            t_plot,
            mean_curve,
            linestyle=line_style,
            linewidth=1.7 if count == max(sample_counts) else 1.5,
            label=rf"$N_{{\mathrm{{MC}}}}={count}$",
        )

        ax_main.fill_between(
            t_plot,
            np.maximum(mean_curve - sem_curve, RESIDUAL_FLOOR),
            mean_curve + sem_curve,
            alpha=0.13,
            color=line.get_color(),
            linewidth=0,
        )

    if use_log_main_axis:
        ax_main.set_yscale("log")

    ax_main.set_xlabel(r"Time [$\mu$s]")
    ax_main.set_ylabel(r"$P_{\mathrm{fib}}(t)/P_{\mathrm{tot}}(0)$")
    ax_main.set_title(title)
    ax_main.grid(True, which="both", alpha=0.25)
    ax_main.legend(frameon=False, loc="best")
    ax_main.text(
        0.015,
        0.95,
        r"$\mathbf{(a)}$",
        transform=ax_main.transAxes,
        ha="left",
        va="top",
    )

    lower_error = np.minimum(rms_std, 0.95 * rms_mean)
    asymmetric_error = np.vstack((lower_error, rms_std))

    ax_rms.errorbar(
        rms_count_array,
        rms_mean,
        yerr=asymmetric_error,
        marker="o",
        linestyle="-",
        linewidth=1.4,
        markersize=4.0,
        capsize=2.5,
        label=r"bootstrap mean $\pm$ SD",
    )

    ax_rms.plot(
        rms_count_array,
        inverse_sqrt_guide,
        linestyle="--",
        linewidth=1.2,
        label=rf"$\propto N_{{\mathrm{{MC}}}}^{{-1/2}}$",
    )

    ax_rms.set_xscale("log")
    ax_rms.set_yscale("log")
    ax_rms.set_xticks(rms_count_array)
    ax_rms.set_xticklabels([str(value) for value in rms_count_array], rotation=30, ha="right")
    ax_rms.set_xlabel(r"$N_{\mathrm{MC}}$")
    ax_rms.set_ylabel(r"$\epsilon_{\mathrm{RMS}}(N_{\mathrm{MC}})$")
    ax_rms.grid(True, which="both", alpha=0.25)
    ax_rms.legend(frameon=False, loc="lower left", fontsize="x-small")
    ax_rms.text(
        0.015,
        0.95,
        r"$\mathbf{(b)}$",
        transform=ax_rms.transAxes,
        ha="left",
        va="top",
    )

    slope_text = (
        rf"fit $\sim N_{{\mathrm{{MC}}}}^{{{fitted_slope:.2f}}}$"
        if np.isfinite(fitted_slope)
        else "fit unavailable"
    )
    ax_rms.text(
        0.97,
        0.08,
        slope_text,
        transform=ax_rms.transAxes,
        ha="right",
        va="bottom",
        fontsize="x-small",
    )

    print(
        "Bootstrap RMS convergence: "
        f"{bootstrap_subsets} subsets per N, seed={random_seed}."
    )
    for count, mean_error, std_error in zip(rms_count_array, rms_mean, rms_std):
        print(
            f"N_MC={count:>4d}: mean RMS error = {mean_error:.6e} "
            f"(subset SD = {std_error:.6e})"
        )
    print(f"Fitted log-log slope = {fitted_slope:.4f}")

    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
        print(f"Saved convergence figure to:\n  {output_path}")

    return fig, (ax_main, ax_rms)


# ---------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------

def run_convergence_plot(
    input_path: Path,
    sample_counts: tuple[int, ...] = DEFAULT_SAMPLE_COUNTS,
    rms_counts: tuple[int, ...] = DEFAULT_RMS_COUNTS,
    bootstrap_subsets: int = DEFAULT_BOOTSTRAP_SUBSETS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_time_us: float | None = DEFAULT_MAX_TIME_US,
    output_path: Path | None = None,
    use_log_main_axis: bool = True,
):
    """Load the MC data, calculate convergence statistics, and plot them."""

    sample_counts = tuple(sorted(set(int(value) for value in sample_counts)))
    rms_counts = tuple(sorted(set(int(value) for value in rms_counts)))

    parent_npz_path, mc_folder = resolve_result_paths(input_path)
    metadata = load_metadata(parent_npz_path)
    exp = build_exp_from_metadata(metadata)
    sim = build_sim_from_metadata(metadata)
    grid = build_grid_from_metadata(metadata)

    print(f"Parent result: {parent_npz_path}")
    print(f"MC folder:     {mc_folder}")
    print(f"Sample counts: {sample_counts}")
    print(f"RMS counts:    {rms_counts}")

    required_runs = max(max(sample_counts), max(rms_counts))

    times_us, curves = load_mc_curves(
        parent_npz_path=parent_npz_path,
        mc_folder=mc_folder,
        exp=exp,
        sim=sim,
        grid=grid,
        required_runs=required_runs,
    )

    if output_path is None:
        output_path = mc_folder / (parent_npz_path.stem + "_mc_convergence.pdf")

    return plot_convergence(
        times_us=times_us,
        curves=curves,
        sample_counts=sample_counts,
        rms_counts=rms_counts,
        bootstrap_subsets=bootstrap_subsets,
        random_seed=random_seed,
        max_time_us=max_time_us,
        title=parent_npz_path.stem,
        output_path=output_path,
        use_log_main_axis=use_log_main_axis,
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Monte Carlo convergence using mean decay curves and "
            "one RMS-error convergence curve."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "Parent NPZ file or directory containing the saved MC runs. "
            "When omitted, the path is requested interactively."
        ),
    )

    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_COUNTS),
        help="MC sample counts for panel (a), for example: --counts 10 30 100",
    )

    parser.add_argument(
        "--rms-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_RMS_COUNTS),
        help=(
            "Sample counts used in panel (b), for example: "
            "--rms-counts 10 20 30 50 70 100"
        ),
    )

    parser.add_argument(
        "--bootstrap-subsets",
        type=int,
        default=DEFAULT_BOOTSTRAP_SUBSETS,
        help="Number of bootstrap subsets averaged at each RMS sample count.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for the bootstrap subsets.",
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
            "Output filename. The extension determines the Matplotlib "
            "format, for example PDF, PNG, or SVG."
        ),
    )

    parser.add_argument(
        "--linear-main",
        action="store_true",
        help="Use a linear rather than logarithmic y-axis in panel (a).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    if input_path is None:
        input_text = input("Parent NPZ file or MC folder: ").strip()
        if not input_text:
            raise ValueError("No input path was provided.")
        input_path = Path(input_text)

    max_time_us = None if args.max_time < 0 else args.max_time

    run_convergence_plot(
        input_path=input_path,
        sample_counts=tuple(args.counts),
        rms_counts=tuple(args.rms_counts),
        bootstrap_subsets=args.bootstrap_subsets,
        random_seed=args.seed,
        max_time_us=max_time_us,
        output_path=args.output,
        use_log_main_axis=not args.linear_main,
    )

    plt.show()


if __name__ == "__main__":
    main()
