#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# radpattern/simulation/monte_carlo_gpu.py

import numpy as np

from radpattern.helpers.helpers import single_dipole_E
from radpattern.physics.rpattern_gpu import prepare_gpu_grid
from coupling_calcualtion import gaussian_fiber_mode_on_sphere

from single_sim_run_gpu import run_single_mc_gpu


def run_monte_carlo_gpu(objs):
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

        eta_sum += eta_t
        AF_sum += AF_t
        AF2_sum += AF2_t
        I_sum += I_t

        del eta_t, AF_t, AF2_t, I_t

    return {
        "times_code": times_code,
        "eta_mean": eta_sum / sim.n_mc,
        "AF_mean": (AF_sum / sim.n_mc).astype(np.complex64),
        "AF2_mean": (AF2_sum / sim.n_mc).astype(np.float32),
        "I_mean": (I_sum / sim.n_mc).astype(np.float32),
    }
