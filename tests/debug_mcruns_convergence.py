#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo convergence plot for the fibre-coupled retrieval curve.

The script loads all saved MC runs from one simulation folder and forms
nested averages using the first N runs, for example

    N_MC = 12, 30, 100.

It produces one publication-style Matplotlib figure containing

    (a) the mean retrieval curves with standard-error bands;
    (b) the absolute deviation from the largest-N reference curve.

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

from radpattern.physics.experimetal_setup import ExperimentalParams
from radpattern.physics.setup_params import SimParams
from radpattern.geometry.grids import AngleGrid
from radpattern.helpers.helpers import single_dipole_E
from radpattern.physics.coupling import gaussian_fiber_mode_on_sphere
from radpattern.plotting import THESIS_STYLE


plt.style.use(THESIS_STYLE)


DEFAULT_SAMPLE_COUNTS = (12, 30, 100)
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


def normalized_rms_error(
    curve: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    RMS deviation normalized by the initial value of the reference curve.
    """

    denominator = abs(reference[0]) + 1.0e-30

    return (
        np.sqrt(
            np.mean(
                (curve - reference) ** 2
            )
        )
        / denominator
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_convergence(
    times_us: np.ndarray,
    curves: np.ndarray,
    sample_counts: tuple[int, ...],
    max_time_us: float | None,
    title: str,
    output_path: Path | None,
    use_log_main_axis: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Create the two-panel convergence figure.
    """

    sample_counts = tuple(
        sorted(set(int(value) for value in sample_counts))
    )

    if any(value <= 0 for value in sample_counts):
        raise ValueError("Every sample count must be positive.")

    if max(sample_counts) > curves.shape[0]:
        raise ValueError(
            f"Requested N_MC={max(sample_counts)}, but only "
            f"{curves.shape[0]} curves are available."
        )

    means, standard_errors = cumulative_statistics(
        curves=curves,
        sample_counts=sample_counts,
    )

    reference_count = max(sample_counts)
    reference_curve = means[reference_count]

    if max_time_us is None:
        plot_mask = np.ones_like(
            times_us,
            dtype=bool,
        )
    else:
        plot_mask = times_us <= max_time_us

        if not np.any(plot_mask):
            raise ValueError(
                f"No time points lie below max_time_us={max_time_us}."
            )

    t_plot = times_us[plot_mask]

    fig, (ax_main, ax_residual) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6.4, 5.2),
        sharex=True,
        gridspec_kw={
            "height_ratios": (2.15, 1.0),
            "hspace": 0.08,
        },
    )

    line_styles = (":", "--", "-.", "-")

    # Upper panel: cumulative mean curves and uncertainty bands.
    for style_index, count in enumerate(sample_counts):
        mean_curve = means[count][plot_mask]
        sem_curve = standard_errors[count][plot_mask]

        line_style = line_styles[
            min(style_index, len(line_styles) - 1)
        ]

        line, = ax_main.plot(
            t_plot,
            mean_curve,
            linestyle=line_style,
            linewidth=1.5 if count != reference_count else 2.0,
            label=rf"$N_{{\mathrm{{MC}}}}={count}$",
            zorder=3 if count == reference_count else 2,
        )

        ax_main.fill_between(
            t_plot,
            np.maximum(mean_curve - sem_curve, RESIDUAL_FLOOR),
            mean_curve + sem_curve,
            alpha=0.13,
            color=line.get_color(),
            linewidth=0,
            zorder=1,
        )

    if use_log_main_axis:
        ax_main.set_yscale("log")

    ax_main.set_ylabel(
        r"$P_{\mathrm{fib}}(t)/P_{\mathrm{tot}}(0)$"
    )

    ax_main.set_title(title)
    ax_main.grid(True, which="both", alpha=0.25)
    ax_main.legend(
        frameon=False,
        ncol=len(sample_counts),
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

    # Lower panel: deviation from the largest-N mean.
    reference_initial = abs(reference_curve[0]) + 1.0e-30

    for style_index, count in enumerate(sample_counts[:-1]):
        residual = (
            np.abs(
                means[count] - reference_curve
            )
            / reference_initial
        )

        rms_error = normalized_rms_error(
            means[count][plot_mask],
            reference_curve[plot_mask],
        )

        ax_residual.plot(
            t_plot,
            np.maximum(
                residual[plot_mask],
                RESIDUAL_FLOOR,
            ),
            linestyle=line_styles[
                min(style_index, len(line_styles) - 1)
            ],
            linewidth=1.4,
            label=(
                rf"$N_{{\mathrm{{MC}}}}={count}$, "
                rf"$\epsilon_{{\mathrm{{RMS}}}}={rms_error:.2e}$"
            ),
        )

        print(
            f"N_MC={count:>4d}: normalized RMS deviation "
            f"from N_MC={reference_count} = {rms_error:.6e}"
        )

    ax_residual.set_yscale("log")

    ax_residual.set_xlabel(
        r"Time [$\mu$s]"
    )

    ax_residual.set_ylabel(
        rf"$|\bar{{P}}_N-\bar{{P}}_{{{reference_count}}}|"
        rf"/\bar{{P}}_{{{reference_count}}}(0)$"
    )

    ax_residual.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax_residual.legend(
        frameon=False,
        loc="best",
    )

    ax_residual.text(
        0.015,
        0.93,
        r"$\mathbf{(b)}$",
        transform=ax_residual.transAxes,
        ha="left",
        va="top",
    )

    fig.align_ylabels(
        [ax_main, ax_residual]
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = output_path.expanduser().resolve()
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

        print(f"Saved convergence figure to:\n  {output_path}")

    return fig, (ax_main, ax_residual)


# ---------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------

def run_convergence_plot(
    input_path: Path,
    sample_counts: tuple[int, ...] = DEFAULT_SAMPLE_COUNTS,
    max_time_us: float | None = DEFAULT_MAX_TIME_US,
    output_path: Path | None = None,
    use_log_main_axis: bool = True,
):
    """
    Load the MC data, calculate convergence statistics, and plot them.
    """

    sample_counts = tuple(
        sorted(set(int(value) for value in sample_counts))
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

    print(f"Parent result: {parent_npz_path}")
    print(f"MC folder:     {mc_folder}")
    print(f"Sample counts: {sample_counts}")

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
            + "_mc_convergence.pdf"
        )

    return plot_convergence(
        times_us=times_us,
        curves=curves,
        sample_counts=sample_counts,
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
            "Plot Monte Carlo convergence using cumulative means and "
            "residuals relative to the largest sample count."
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
        help="Nested MC sample counts, for example: --counts 12 30 100",
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
        sample_counts=tuple(args.counts),
        max_time_us=max_time_us,
        output_path=args.output,
        use_log_main_axis=not args.linear_main,
    )

    plt.show()


if __name__ == "__main__":
    main()
