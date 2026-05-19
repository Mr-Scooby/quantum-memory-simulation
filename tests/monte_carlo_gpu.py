#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# radpattern/simulation/monte_carlo_gpu.py

import numpy as np

from radpattern.helpers.helpers import single_dipole_E
from radpattern.physics.rpattern_gpu import prepare_gpu_grid
from coupling_calcualtion import gaussian_fiber_mode_on_sphere

from pathlib import Path
from single_sim_run_gpu import run_single_mc_gpu

def run_monte_carlo_gpu(
    *, # So all consecuent variables are keyword variables
    objs,
    save_full_mc=False,
    mc_dir=None,
    ):

    print(f"SAVE_FULL_MC = {save_full_mc}")
    exp = objs.exp
    sim = objs.sim

    grid = sim.create_grid()
    times_code = sim.time_array()

    T = sim.time_divisions
    nt, nphi = grid.shape

    dipole = single_dipole_E(
        grid.nx,
        grid.ny,
        grid.nz,
        np.array([1, 0, 0]),
    )

    theta0 = 12 / (exp.atom.k_signal * exp.w0_signal)
    E_fib = np.abs(gaussian_fiber_mode_on_sphere(grid, theta0)) ** 2

    n_hat_gpu = prepare_gpu_grid(grid.n_hat_flat)

    if save_full_mc:
        if mc_dir is None:
            raise ValueError("mc_dir must be given when save_full_mc=True")

        mc_dir = Path(mc_dir)
        mc_dir.mkdir(parents=True, exist_ok=True)

    eta_sum = np.zeros(T, dtype=np.float64)
    AF_sum = np.zeros((T, nt, nphi), dtype=np.complex128)
    AF2_sum = np.zeros((T, nt, nphi), dtype=np.float64)
    I_sum = np.zeros((T, nt, nphi), dtype=np.float64)

    for mc in range(sim.n_mc):
        eta_t, AF_t, AF2_t, I_t = run_single_mc_gpu(
            objs=objs,
            grid=grid,
            times_code=times_code,
            dipole=dipole,
            E_fib=E_fib,
            n_hat_gpu=n_hat_gpu,
            theta0=theta0,
            mc=mc,
        )

        # running mean sums
        eta_sum += eta_t
        AF_sum += AF_t
        AF2_sum += AF2_t
        I_sum += I_t

        # diagnostic: save full data for this MC only
        if save_full_mc:
            mc_path = mc_dir / f"mc_{mc:04d}.npz"

            np.savez_compressed(
                mc_path,
                mc_index=np.array(mc, dtype=np.int32),
                times_code=times_code.astype(np.float32),
                eta=eta_t.astype(np.float32),
                AF=AF_t.astype(np.complex64),
                AF2=AF2_t.astype(np.float32),
            )

        # delete for memory. 
        del eta_t, AF_t, AF2_t, I_t

    return {
        "times_code": times_code,
        "eta_mean": eta_sum / sim.n_mc,
        "AF_mean": (AF_sum / sim.n_mc).astype(np.complex64),
        "AF2_mean": (AF2_sum / sim.n_mc).astype(np.float32),
        "I_mean": (I_sum / sim.n_mc).astype(np.float32),
    }
