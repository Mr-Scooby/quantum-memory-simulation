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
from radpattern.physics.setup_params import SimParams
from radpattern.geometry.grids import AngleGrid
from radpattern.helpers.helpers import single_dipole_E
from coupling_calcualtion import (
    gaussian_fiber_mode_on_sphere,
    intensity_overlap_on_sphere,
)

import inspect

def default_from_type(name, typ):
    """
    Choose a backward-compatible value from the dataclass type.
    """
    # Generic type-based defaults
    if typ is float:
        return 999.0

    if typ is int:
        return 999

    if typ is str:
        return "None"

    if typ is bool:
        return False

    if typ is tuple:
        return (9,9,9)

    if typ is list:
        return [9,9,9]

    if typ is dict:
        return {"None": "None" }

    return None



def dataclass_kwargs(cls, data):
    """Keep only keys accepted by a dataclass constructor.
    adds None if not found (backwards compatability
    """
    kwargs = {}

    for f in fields(cls):
        if not f.init:
            continue

        if f.name in data:
            kwargs[f.name] = data[f.name]

        else:
            kwargs[f.name] = default_from_type(f.name, f.type)

#    valid = {f.name for f in fields(cls) if f.init}
#    return {
#        name: data[name] if name in data else default_from_type(f.name, f.type)
#        for name in valid
#        }

    return kwargs



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

def build_sim_from_metadata(metadata):
    sim_meta = metadata["sim"]
    return SimParams(**sim_meta)


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

    P_fib_t = np.zeros(T, dtype=float)
    P_tot_t = np.zeros(T, dtype=float)
    eta_t = np.zeros(T, dtype=float)

    mask = grid.TH <= theta0

    theta = grid.TH[:, 0]
    phi = grid.PH[0, :]
    sin_th = np.sin(grid.TH)

    E_fib_masked = np.where(mask, E_fib, 0.0)

    for it in range(T):
        I = AF2_t[it] * dipole
        I_masked = np.where(mask, I, 0.0)

        P_fib = np.trapezoid(
            np.trapezoid(I_masked * E_fib_masked * sin_th, phi, axis=1),
            theta,
            axis=0,
        )

        P_tot = np.trapezoid(
            np.trapezoid(I_masked * sin_th, phi, axis=1),
            theta,
            axis=0,
        )

        P_fib_t[it] = P_fib
        P_tot_t[it] = P_tot
        eta_t[it] = P_fib / (P_tot + 1e-30)

    return P_fib_t, P_tot_t, eta_t


def plot_mc_couplings(parent_npz_path, max_mc=None):
    parent_npz_path = Path(parent_npz_path)

    metadata = load_metadata(parent_npz_path)
    exp = build_exp_from_metadata(metadata)
    sim = build_sim_from_metadata(metadata)
    grid = build_grid_from_metadata(metadata)

    print(sim)
    print(exp)

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
    all_P_fib = []
    all_P_tot = []
    all_P_fib_over_Ptot0 = []

    fig_eta, ax_eta = plt.subplots(figsize=(8, 5))
    fig_power, ax_power = plt.subplots(figsize=(8, 5))

    for file_idx, mc_file in enumerate(mc_files):
        data = np.load(mc_file, allow_pickle=True)

        AF2_t = data["AF2"]

        if "times_code" in data:
            times_code = data["times_code"] * sim.char_time * 1e6
        else:
            parent = np.load(parent_npz_path, allow_pickle=True)
            times_code = parent["times_code"] * sim.char_time * 1e6

       P_fib, P_tot, eta_t = coupling_from_AF2(
            AF2_t=AF2_t,
            grid=grid,
            dipole=dipole,
            E_fib=E_fib,
            theta0=theta0,
        )

        all_eta.append(eta_t)
        all_P_fib.append(P_fib_t)
        all_P_tot.append(P_tot_t)
        all_P_fib_over_Ptot0.append(P_fib_over_Ptot0_t)
        
        label = mc_file.stem if file_idx < 10 else None

        ax_eta.plot(
            times_code,
            eta_t,
            alpha=0.35,
            linewidth=1.0,
            label=mc_file.stem if file_idx < 10 else None,
        )
        ax_power.plot(
            times_code,
            P_fib_over_Ptot0_t,
            alpha=0.35,
            linewidth=1.0,
            label=label,
        )

    all_eta = np.asarray(all_eta)
    all_P_fib = np.asarray(all_P_fib)
    all_P_tot = np.asarray(all_P_tot)
    all_P_fib_over_Ptot0 = np.asarray(all_P_fib_over_Ptot0)

    eta_mean = all_eta.mean(axis=0)
    P_fib_mean = all_P_fib.mean(axis=0)
    P_tot_mean = all_P_tot.mean(axis=0)
    P_fib_over_Ptot0_mean = all_P_fib_over_Ptot0.mean(axis=0)

    ax_eta.plot(
        times_code,
        eta_mean,
        color="black",
        linewidth=3,
        label="MC mean",
    )

    ax_eta.set_xlabel("time [us]")
    ax_eta.set_ylabel("coupling eta = P_fib / P_tot")
    ax_eta.set_title(parent_npz_path.stem)
    ax_eta.grid(True, alpha=0.3)
    ax_eta.legend()
    fig_eta.tight_layout()

    # mean fiber power
    ax_power.plot(
        times_us,
        P_fib_over_Ptot0_mean,
        color="black",
        linewidth=3,
        label="MC mean",
    )

    ax_power.set_xlabel("time [µs]")
    ax_power.set_ylabel("P_fib / P_tot[0]")
    ax_power.set_title(parent_npz_path.stem + " — fiber intensity")
    ax_power.grid(True, alpha=0.3)
    ax_power.legend()
    fig_power.tight_layout()

    plt.show()

    return times_code, all_eta


if __name__ == "__main__":
    
    running = True
    while running == True: 
        file = input("File : ")

        if file.upper() == "END": 
            running = False 
            break

        RESULT_FILE = Path(
            rf"{file}"
        )

        times_code, eta_runs = plot_mc_couplings(
            RESULT_FILE,
            max_mc=None,
        )
