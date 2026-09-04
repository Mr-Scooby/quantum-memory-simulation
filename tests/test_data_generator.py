#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic far-field sphere-pattern data compatible with

    from radpattern.plotting import load_data

and with plot_spherePattern_three_panels.py.

The generated intensity array has shape

    (time, theta, phi)

and contains a forward lobe that gradually moves, broadens, and decays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_angle_grid(
    n_theta: int,
    n_phi: int,
    theta_max: float = np.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the angular grid used by radpattern.geometry.grids.AngleGrid."""
    theta = np.linspace(0.0, theta_max, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")

    nx = np.sin(th) * np.cos(ph)
    ny = np.sin(th) * np.sin(ph)
    nz = np.cos(th)
    return th, ph, nx, ny, nz


def make_test_intensity(
    n_times: int,
    n_theta: int,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic time-dependent angular intensity pattern.

    The pattern is not intended as a physical simulation. It is designed
    to test the three-panel layout and shared color normalization.
    """
    _, _, nx, ny, nz = make_angle_grid(n_theta, n_phi)

    intensity = np.empty((n_times, n_theta, n_phi), dtype=np.float64)
    time_us = np.linspace(0.0, 20.0, n_times)

    for i, fraction in enumerate(np.linspace(0.0, 1.0, n_times)):
        # Move the main lobe slightly away from +z and rotate it in azimuth.
        tilt = np.deg2rad(18.0 * fraction)
        azimuth = np.deg2rad(75.0 * fraction)

        direction = np.array(
            [
                np.sin(tilt) * np.cos(azimuth),
                np.sin(tilt) * np.sin(azimuth),
                np.cos(tilt),
            ]
        )

        cos_gamma = (
            nx * direction[0]
            + ny * direction[1]
            + nz * direction[2]
        )
        gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))

        # The principal lobe broadens with time.
        sigma = np.deg2rad(8.0 + 13.0 * fraction)
        main_lobe = np.exp(-0.5 * (gamma / sigma) ** 2)

        # Add a weak angular ring and an even weaker backward lobe so that
        # low-intensity regions exercise the full -60 dB color scale.
        ring_radius = np.deg2rad(36.0 + 5.0 * fraction)
        ring_width = np.deg2rad(4.0)
        ring = 8.0e-4 * np.exp(
            -0.5 * ((gamma - ring_radius) / ring_width) ** 2
        )

        backward_gamma = np.pi - gamma
        backward_lobe = 8.0e-5 * np.exp(
            -0.5 * (backward_gamma / np.deg2rad(18.0)) ** 2
        )

        pattern = main_lobe + ring + backward_lobe + 1.0e-12
        pattern /= np.max(pattern)

        # Common-reference decay: the final maximum is about -20 dB
        # relative to the first frame.
        amplitude = 10.0 ** (-20.0 * fraction / 10.0)
        intensity[i] = amplitude * pattern

    return intensity, time_us


def build_metadata(
    n_times: int,
    n_theta: int,
    n_phi: int,
) -> dict:
    """
    Create metadata accepted by radpattern.plotting.load_data().

    Only the angular grid parameters are essential for plotting, but a
    complete experiment dictionary prevents placeholder values from being
    printed when ExperimentalParams is reconstructed.
    """
    char_time_s = 1.0e-6

    sim = {
        "n_mc": 1,
        "sim_time_us": 20.0,
        "char_time": char_time_s,
        "time_divisions": n_times,
        "time_spacing": "linspace",
        "n_theta": n_theta,
        "n_phi": n_phi,
        "theta_max": float(np.pi),
        "simulation_window_radius_w0_cutoff": 4.0,
        "sim_density": 1,
        "chunk_atoms": 1,
        "chunk_dirs": 1024,
        "normalize_each_time": False,
        "plane_restricted": False,
        "seed": 12345,
    }

    experiment = {
        "atoms": "Cs133",
        "lambda_control_m": 894.6e-9,
        "delta_f_hz": 9.192631770e9,
        "cell_length_m": 75.0e-3,
        "cell_diameter_m": 4.0e-3,
        "signal_fwhm_diameter_m": 120.0e-6,
        "signal_beam_direction": (0.0, 0.0, 1.0),
        "control_fwhm_diameter_m": 210.0e-6,
        "control_pulse_fwhm_ns": 25.0,
        "control_beam_direction": (0.0, 0.0, 1.0),
        "cell_geometry": "cylinder",
        "control_beam_AxisOffset_nm": (0.0, 0.0, 0.0),
        "g_g": -0.25,
        "m_g": 0.0,
        "g_s": 0.25,
        "m_s": 0.0,
        "density_cm3": 1.0e12,
        "temperature": 348.15,
        "buffer_gas": "N2",
        "buffer_pressure_Torr": 5.0,
        "diffusion_D0_cm2_s": 0.24,
        "diffusion_T0_K": 273.15,
        "diffusion_P0_Torr": 1.0,
        "B0_T": 0.0,
        "B_gradient": 0.0,
        "scalling": 100,
        "label": "synthetic sphere-pattern test data",
        "spin_destruction_cross_section_CsN2_m2": 2.9e-26,
        "spin_exchange_alpha_CsCs_m3_s": 6.5e-16,
    }

    return {
        "experiment": experiment,
        "sim": sim,
    }


def generate(output: Path, n_times: int, n_theta: int, n_phi: int) -> Path:
    if n_times < 100:
        raise ValueError(
            "n_times must be at least 100 because the plotting script "
            "uses indices [0, 50, 99]."
        )
    if n_theta < 2 or n_phi < 3:
        raise ValueError("The angular grid is too small.")

    intensity, time_us = make_test_intensity(
        n_times=n_times,
        n_theta=n_theta,
        n_phi=n_phi,
    )

    metadata = build_metadata(
        n_times=n_times,
        n_theta=n_theta,
        n_phi=n_phi,
    )

    # load_data converts times_code back to microseconds as
    # times_us = times_code * sim.char_time * 1e6.
    char_time_s = metadata["sim"]["char_time"]
    times_code = time_us * 1.0e-6 / char_time_s

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        intensity=intensity,
        times_code=times_code,
        metadata=metadata,
    )

    print(f"Saved test data: {output}")
    print(f"intensity shape: {intensity.shape}")
    print(
        "Selected-frame maxima relative to frame 0 [dB]:",
        [
            round(
                10.0
                * np.log10(
                    np.max(intensity[index]) / np.max(intensity[0])
                ),
                2,
            )
            for index in (0, 50, 99)
        ],
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sphere-pattern test data."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sphere_pattern_test_data.npz"),
        help="Output NPZ path.",
    )
    parser.add_argument("--n-times", type=int, default=100)
    parser.add_argument("--n-theta", type=int, default=91)
    parser.add_argument("--n-phi", type=int, default=181)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(
        output=args.output,
        n_times=args.n_times,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
    )


if __name__ == "__main__":
    main()
