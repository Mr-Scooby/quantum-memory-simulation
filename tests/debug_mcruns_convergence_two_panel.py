#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo convergence plot for the fibre-coupled retrieval curve.

The script loads all saved MC runs from one simulation folder and produces
one publication-style Matplotlib figure containing

    (a) mean retrieval curves with standard-error bands for
        N_MC = 10, 30, and 100;
    (b) one RMS convergence curve epsilon_RMS(N_MC) for
        N_MC = 10, 20, 30, 50, 70, and 100.

The convergence error is estimated by bootstrap resampling the available MC
curves. For each N_MC, many samples of N_MC runs are drawn with replacement,
their mean decay curve is compared with the full 100-run mean, and the RMS
deviation is averaged over resamples and plotted times.

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from radpattern.physics.experimetal_setup import ExperimentalParams
from radpattern.physics.setup_params import SimParams
from radpattern.geometry.grids import AngleGrid
from radpattern.helpers.helpers import single_dipole_E
from radpattern.physics.coupling import gaussian_fiber_mode_on_sphere
from radpattern.plotting import THESIS_STYLE


plt.style.use(THESIS_STYLE)


DEFAULT_PANEL_COUNTS = (10, 30, 100)
DEFAULT_CONVERGENCE_COUNTS = (10, 20, 30, 50, 70, 100)
DEFAULT_BOOTSTRAP_RESAMPLES = 200
DEFAULT_RANDOM_SEED = 12345
DEFAULT_MAX_TIME_US = 200.0
PLOT_FLOOR = 1.0e-12


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

    This ensures mc_2 comes before mc_10 when filenames are not padded.
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

    Returns
    -------
    parent_npz_path
        Main result file containing metadata.
    mc_folder
        Directory containing mc_*.npz files.
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
    """
    Calculate fibre power, total power, and instantaneous coupling.

    Parameters
    ----------
    AF2_t
        Array with shape (time, n_theta, n_phi).

    Returns
    -------
    P_fib_t
        Fibre-coupled power.
    P_tot_t
        Total power inside the angular integration aperture.
    eta_t
        Instantaneous ratio P_fib(t) / P_tot(t).
    """

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


def load_times_us(
    mc_data,
    parent_npz_path: Path,
    sim: SimParams,
) -> np.ndarray:
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

    The normalization P_tot(0) is calculated separately for every run.
    This suppresses run-to-run differences in the absolute initial power
    and tests convergence of the normalized retrieval curve.
    """

    mc_files = find_mc_files(mc_folder)

    if len(mc_files) < required_runs:
        raise ValueError(
            f"The convergence test requests {required_runs} runs, but only "
            f"{len(mc_files)} MC files were found in:\n  {mc_folder}"
        )

    # Only the largest requested number is needed.
    mc_files = mc_files[:required_runs]

    dipole = single_dipole_E(
        grid.nx,
        grid.ny,
        grid.nz,
        np.array([1.0, 0.0, 0.0]),
    )

    theta0 = 12.0 / (
        exp.atom.k_signal * exp.w0_signal
    )

    E_fib = np.abs(
        gaussian_fiber_mode_on_sphere(grid, theta0)
    ) ** 2

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
                raise ValueError(
                    f"Time-array shape mismatch in {mc_file.name}."
                )

            if not np.allclose(
                times_us,
                reference_times_us,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"Time-array values differ in {mc_file.name}."
                )

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
    """
    Calculate nested means and standard errors using the first N runs.
    """

    means = {}
    standard_errors = {}

    for count in sample_counts:
        subset = curves[:count]

        means[count] = np.mean(
            subset,
            axis=0,
        )

        if count > 1:
            standard_errors[count] = np.std(
                subset,
                axis=0,
                ddof=1,
            ) / np.sqrt(count)
        else:
            standard_errors[count] = np.zeros_like(
                means[count]
            )

    return means, standard_errors


def bootstrap_rms_errors(
    curves: np.ndarray,
    sample_counts: tuple[int, ...],
    reference_curve: np.ndarray,
    plot_mask: np.ndarray,
    n_resamples: int,
    random_seed: int,
) -> dict[int, float]:
    """
    Estimate epsilon_RMS(N_MC) by bootstrap resampling.

    For every requested sample count, ``n_resamples`` bootstrap samples are
    drawn from the available MC curves. Each bootstrap mean is compared with
    the full-run reference mean. The reported value is

        sqrt(mean[(mean_bootstrap(t) - mean_reference(t))**2])
        ------------------------------------------------------,
                      abs(mean_reference(0))

    where the mean is taken over bootstrap resamples and plotted times.
    """

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")

    n_runs = curves.shape[0]
    rng = np.random.default_rng(random_seed)
    denominator = abs(reference_curve[0]) + 1.0e-30
    reference_plot = reference_curve[plot_mask]

    errors = {}

    for count in sample_counts:
        bootstrap_indices = rng.integers(
            0,
            n_runs,
            size=(n_resamples, count),
        )

        bootstrap_means = np.mean(
            curves[bootstrap_indices],
            axis=1,
        )

        deviations = (
            bootstrap_means[:, plot_mask]
            - reference_plot[None, :]
        )

        errors[count] = (
            np.sqrt(np.mean(deviations ** 2))
            / denominator
        )

    return errors


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_convergence(
    times_us: np.ndarray,
    curves: np.ndarray,
    panel_counts: tuple[int, ...],
    convergence_counts: tuple[int, ...],
    max_time_us: float | None,
    title: str,
    output_path: Path | None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    use_log_main_axis: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Create the two-panel Monte Carlo convergence figure.

    Panel (a) contains the mean decay curves for ``panel_counts`` with
    standard-error bands. Panel (b) contains one bootstrap RMS convergence
    curve evaluated at ``convergence_counts``.
    """

    panel_counts = tuple(
        sorted(set(int(value) for value in panel_counts))
    )
    convergence_counts = tuple(
        sorted(set(int(value) for value in convergence_counts))
    )

    all_counts = panel_counts + convergence_counts

    if not panel_counts or not convergence_counts:
        raise ValueError("Both count lists must contain at least one value.")

    if any(value <= 0 for value in all_counts):
        raise ValueError("Every MC sample count must be positive.")

    if max(all_counts) > curves.shape[0]:
        raise ValueError(
            f"Requested N_MC={max(all_counts)}, but only "
            f"{curves.shape[0]} curves are available."
        )

    panel_means, panel_standard_errors = cumulative_statistics(
        curves=curves,
        sample_counts=panel_counts,
    )

    reference_count = max(convergence_counts)
    reference_curve = np.mean(
        curves[:reference_count],
        axis=0,
    )

    if max_time_us is None:
        plot_mask = np.ones_like(times_us, dtype=bool)
    else:
        plot_mask = times_us <= max_time_us

        if not np.any(plot_mask):
            raise ValueError(
                f"No time points lie below max_time_us={max_time_us}."
            )

    t_plot = times_us[plot_mask]

    rms_errors = bootstrap_rms_errors(
        curves=curves[:reference_count],
        sample_counts=convergence_counts,
        reference_curve=reference_curve,
        plot_mask=plot_mask,
        n_resamples=bootstrap_resamples,
        random_seed=random_seed,
    )

    fig, (ax_main, ax_convergence) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6.4, 5.4),
        layout="constrained",
        gridspec_kw={
            "height_ratios": (2.15, 1.0),
        },
    )

    line_styles = (":", "--", "-")

    # Panel (a): mean decay curves and standard-error bands.
    for style_index, count in enumerate(panel_counts):
        mean_curve = panel_means[count][plot_mask]
        sem_curve = panel_standard_errors[count][plot_mask]

        line, = ax_main.plot(
            t_plot,
            mean_curve,
            linestyle=line_styles[
                min(style_index, len(line_styles) - 1)
            ],
            linewidth=1.5 if count != max(panel_counts) else 2.0,
            label=rf"$N_{{\mathrm{{MC}}}}={count}$",
            zorder=3 if count == max(panel_counts) else 2,
        )

        lower_band = mean_curve - sem_curve
        if use_log_main_axis:
            lower_band = np.maximum(lower_band, PLOT_FLOOR)

        ax_main.fill_between(
            t_plot,
            lower_band,
            mean_curve + sem_curve,
            alpha=0.13,
            color=line.get_color(),
            linewidth=0,
            zorder=1,
        )

    if use_log_main_axis:
        ax_main.set_yscale("log")

    ax_main.set_xlabel(r"Time [$\mu$s]")
    ax_main.set_ylabel(
        r"$P_{\mathrm{fib}}(t)/P_{\mathrm{tot}}(0)$"
    )
    ax_main.set_title(title)
    ax_main.grid(True, which="both", alpha=0.25)
    ax_main.legend(
        frameon=False,
        ncol=len(panel_counts),
        loc="best",
    )
    ax_main.text(
        0.015,
        0.95,
        r"$\mathbf{(a)}$",
        transform=ax_main.transAxes,
        ha="left",
        va="top",
    )

    # Panel (b): one RMS convergence curve.
    count_array = np.asarray(convergence_counts, dtype=float)
    error_array = np.asarray(
        [rms_errors[count] for count in convergence_counts],
        dtype=float,
    )

    ax_convergence.plot(
        count_array,
        error_array,
        marker="o",
        linewidth=1.6,
        markersize=4.5,
    )

    ax_convergence.set_xscale("log")
    ax_convergence.set_yscale("log")
    ax_convergence.set_xticks(convergence_counts)
    ax_convergence.xaxis.set_major_formatter(
        mticker.ScalarFormatter()
    )
    ax_convergence.xaxis.set_minor_formatter(
        mticker.NullFormatter()
    )
    ax_convergence.set_xlabel(r"$N_{\mathrm{MC}}$")
    ax_convergence.set_ylabel(
        r"$\epsilon_{\mathrm{RMS}}(N_{\mathrm{MC}})$"
    )
    ax_convergence.grid(True, which="both", alpha=0.25)
    ax_convergence.text(
        0.015,
        0.93,
        r"$\mathbf{(b)}$",
        transform=ax_convergence.transAxes,
        ha="left",
        va="top",
    )

    if np.all(error_array > 0.0):
        slope, _ = np.polyfit(
            np.log(count_array),
            np.log(error_array),
            deg=1,
        )
        ax_convergence.text(
            0.98,
            0.92,
            rf"log--log slope $={slope:.2f}$",
            transform=ax_convergence.transAxes,
            ha="right",
            va="top",
        )

    for count, rms_error in zip(convergence_counts, error_array):
        print(
            f"N_MC={count:>4d}: bootstrap normalized RMS error "
            f"= {rms_error:.6e}"
        )

    fig.align_ylabels([ax_main, ax_convergence])

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

    return fig, (ax_main, ax_convergence)


# ---------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------

def run_convergence_plot(
    input_path: Path,
    panel_counts: tuple[int, ...] = DEFAULT_PANEL_COUNTS,
    convergence_counts: tuple[int, ...] = DEFAULT_CONVERGENCE_COUNTS,
    max_time_us: float | None = DEFAULT_MAX_TIME_US,
    output_path: Path | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    use_log_main_axis: bool = True,
):
    """
    Load the MC data, calculate convergence statistics, and plot them.
    """

    panel_counts = tuple(
        sorted(set(int(value) for value in panel_counts))
    )
    convergence_counts = tuple(
        sorted(set(int(value) for value in convergence_counts))
    )

    required_runs = max(
        max(panel_counts),
        max(convergence_counts),
    )

    parent_npz_path, mc_folder = resolve_result_paths(input_path)
    metadata = load_metadata(parent_npz_path)
    exp = build_exp_from_metadata(metadata)
    sim = build_sim_from_metadata(metadata)
    grid = build_grid_from_metadata(metadata)

    print(f"Parent result:       {parent_npz_path}")
    print(f"MC folder:           {mc_folder}")
    print(f"Panel-(a) counts:    {panel_counts}")
    print(f"Panel-(b) counts:    {convergence_counts}")
    print(f"Bootstrap resamples: {bootstrap_resamples}")
    print(f"Random seed:         {random_seed}")

    times_us, curves = load_mc_curves(
        parent_npz_path=parent_npz_path,
        mc_folder=mc_folder,
        exp=exp,
        sim=sim,
        grid=grid,
        required_runs=required_runs,
    )

    if output_path is None:
        output_path = mc_folder / (
            parent_npz_path.stem
            + "_mc_convergence.pdf"
        )

    return plot_convergence(
        times_us=times_us,
        curves=curves,
        panel_counts=panel_counts,
        convergence_counts=convergence_counts,
        max_time_us=max_time_us,
        title=parent_npz_path.stem,
        output_path=output_path,
        bootstrap_resamples=bootstrap_resamples,
        random_seed=random_seed,
        use_log_main_axis=use_log_main_axis,
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean Monte Carlo decay curves and a bootstrap RMS "
            "convergence curve."
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
        "--panel-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_PANEL_COUNTS),
        help=(
            "MC counts shown as mean decay curves in panel (a). "
            "Default: 10 30 100"
        ),
    )

    parser.add_argument(
        "--convergence-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_CONVERGENCE_COUNTS),
        help=(
            "MC counts used for epsilon_RMS in panel (b). "
            "Default: 10 20 30 50 70 100"
        ),
    )

    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help=(
            "Number of bootstrap resamples used for each convergence point."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for bootstrap resampling.",
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
        help="Use a linear rather than logarithmic y-axis in the upper panel.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input

    if input_path is None:
        input_text = input(
            "Parent NPZ file or MC folder: "
        ).strip()

        if not input_text:
            raise ValueError("No input path was provided.")

        input_path = Path(input_text)

    max_time_us = (
        None
        if args.max_time < 0
        else args.max_time
    )

    run_convergence_plot(
        input_path=input_path,
        panel_counts=tuple(args.panel_counts),
        convergence_counts=tuple(args.convergence_counts),
        max_time_us=max_time_us,
        output_path=args.output,
        bootstrap_resamples=args.bootstrap_resamples,
        random_seed=args.seed,
        use_log_main_axis=not args.linear_main,
    )

    plt.show()


if __name__ == "__main__":
    main()
