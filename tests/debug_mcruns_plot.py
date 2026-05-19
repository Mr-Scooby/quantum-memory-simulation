#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot coupling vs time for each saved MC run.

Expected structure:

result_sim/
    cs133_ABC.npz
    cs133_ABC/
        mc_0000.npz
        mc_0001.npz
        ...
or:
result_sim/
    cs133_ABC.npz
    cs133_ABC_mc_runs/
        mc_0000.npz
        mc_0001.npz
        ...

Each MC file must contain:
    AF2
optionally:
    times_code
    eta
"""

from pathlib import Path
from dataclasses import fields
import numpy as np
import matplotlib.pyplot as plt

from radpattern.physics.experimetal_setup import ExperimentalParams
from radpattern.geometry.grids import AngleGrid
from radpattern.helpers.helpers import single_dipole_E
from coupling_calcualtion import (
    gaussian_fiber_mode_on_sphere,
    intensity_overlap_on_sphere,
)


def dataclass_kwargs(cls, data):
    """Keep only keys accepted by a dataclass constructor."""
    valid = {f.name for f in fields(cls) if f.init}
    return {k: v for k, v in data.items() if k in valid}


def load_metadata(parent_npz_path):
    parent = np.load(parent_npz_path, allow_pickle=True)
    metadata = parent["metadata"].item()
    return metadata


def build_grid_from_metadata(metadata):
    sim_meta = metadata["sim"]

    n_theta = sim_meta["n_theta"]
    n_phi = sim_meta["n_phi"]
    theta_max = sim_meta["theta_max"]

    return AngleGrid(
        n_theta=n_theta,
        n_phi=n_phi,
        theta_max=theta_max,
    )


def build_exp_from_metadata(metadata):
    exp_meta = metadata["experiment"]
    exp_kwargs = dataclass_kwargs(ExperimentalParams, exp_meta)
    return ExperimentalParams(**exp_kwargs)


def find_mc_folder(parent_npz_path):
    """
    Accept both:
        cs133_ABC/
    and:
        cs133_ABC_mc_runs/
    """
    parent_npz_path = Path(parent_npz_path)
    root = parent_npz_path.parent
    stem = parent_npz_path.stem

    candidates = [
        root / stem,
        root / f"{stem}_mc_runs",
    ]

    for folder in candidates:
        if folder.exists() and folder.is_dir():
            return folder

    raise FileNotFoundError(
        f"No MC folder found. Tried:\n"
        + "\n".join(str(c) for c in candidates)
    )


def find_mc_files(mc_folder):
    mc_folder = Path(mc_folder)

    files = sorted(mc_folder.glob("mc_*.npz"))

    if not files:
        files = sorted(mc_folder.glob("mc_run*.npz"))

    if not files:
        raise FileNotFoundError(f"No MC npz files found in {mc_folder}")

    return files


def coupling_from_AF2(AF2_t, grid, dipole, E_fib, theta0):
    """
    AF2_t shape:
        (T, n_theta, n_phi)

    returns:
        eta_t shape (T,)
    """
    T = AF2_t.shape[0]
    eta_t = np.zeros(T, dtype=float)

    for it in range(T):
        I = AF2_t[it] * dipole
        eta_t[it] = intensity_overlap_on_sphere(
            grid,
            I,
            E_fib,
            theta0,
        )

    return eta_t


def plot_mc_couplings(parent_npz_path, max_mc=None):
    parent_npz_path = Path(parent_npz_path)

    metadata = load_metadata(parent_npz_path)
    exp = build_exp_from_metadata(metadata)
    grid = build_grid_from_metadata(metadata)

    mc_folder = find_mc_folder(parent_npz_path)
    mc_files = find_mc_files(mc_folder)

    if max_mc is not None:
        mc_files = mc_files[:max_mc]

    print(f"Parent result: {parent_npz_path}")
    print(f"MC folder:     {mc_folder}")
    print(f"MC files:      {len(mc_files)}")

    dipole = single_dipole_E(
        grid.nx,
        grid.ny,
        grid.nz,
        np.array([1.0, 0.0, 0.0]),
    )

    theta0 = 12 / (exp.atom.k_signal * exp.w0_signal)
    E_fib = np.abs(gaussian_fiber_mode_on_sphere(grid, theta0)) ** 2

    all_eta = []

    plt.figure(figsize=(8, 5))

    for file_idx, mc_file in enumerate(mc_files):
        data = np.load(mc_file, allow_pickle=True)

        AF2_t = data["AF2"]

        if "times_code" in data:
            times_code = data["times_code"]
        else:
            parent = np.load(parent_npz_path, allow_pickle=True)
            times_code = parent["times_code"]

        eta_t = coupling_from_AF2(
            AF2_t=AF2_t,
            grid=grid,
            dipole=dipole,
            E_fib=E_fib,
            theta0=theta0,
        )

        all_eta.append(eta_t)

        plt.plot(
            times_code,
            eta_t,
            alpha=0.35,
            linewidth=1.0,
            label=mc_file.stem if file_idx < 10 else None,
        )

    all_eta = np.asarray(all_eta)
    eta_mean = all_eta.mean(axis=0)

    plt.plot(
        times_code,
        eta_mean,
        color="black",
        linewidth=3,
        label="MC mean",
    )

    plt.xlabel("time code")
    plt.ylabel("coupling eta")
    plt.title(parent_npz_path.stem)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return times_code, all_eta


if __name__ == "__main__":
    RESULT_FILE = Path(
        r"C:\Users\local_admin\radek\simulations\data\results_sims\cs133_ABC.npz"
    )

    times_code, eta_runs = plot_mc_couplings(
        RESULT_FILE,
        max_mc=None,
    )
