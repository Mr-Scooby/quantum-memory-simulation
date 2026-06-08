#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Single Simulation run. Uses gpu AF calculation function """

import time
import numpy as np
from copy import deepcopy

from radpattern.physics.rpattern_gpu import array_factor_general_gpu
from radpattern.physics.coupling import intensity_overlap_on_sphere


def run_single_mc_gpu(
    *, # So all next variables must be called by name.
    objs,
    grid,
    times_code,
    dipole,
    E_fib,
    n_hat_gpu, # n_hat copied in gpu 
    theta0,
    mc: int,
):
    """
    One Monte Carlo realization:
    - generate one random cloud
    - evolve it in time
    - compute AF on GPU at each time
    - return time traces for this one run
    """

    exp = objs.exp
    sim = objs.sim
    cloud = deepcopy(objs.cloud) 
    control_beam = deepcopy(objs.Cbeam)
    signal_beam = deepcopy(objs.Sbeam)

    T = sim.time_divisions
    nt, nphi = grid.shape

    print(f"Run {mc + 1}/{sim.n_mc}")
    t0_mc = time.perf_counter()

    # Independent RNG per MC run
    seed = 1000 if sim.seed is None else int(sim.seed)
    rng = np.random.default_rng(seed + mc)

    # Generate one cloud realization
    cloud.generate_cloud(rng=rng)
    cloud.generate_velocity_distribution(rng = rng)
    
    control_beam.generate_weights(cloud.r_xyz)
    signal_beam.generate_weights(cloud.r_xyz)
    cloud.generate_S_profile(signal_beam, control_beam)
    cloud.r0_xyz = cloud.r_xyz.copy()

    # One-run arrays only
    eta_t = np.zeros(T, dtype=np.float64)
    AF_t = np.zeros((T, nt, nphi), dtype=np.complex64)
    AF2_t = np.zeros((T, nt, nphi), dtype=np.float32)
    I_t = np.zeros((T, nt, nphi), dtype=np.float32)

    Diff_coef = exp.diffusion_coeff_code

    for it, t in enumerate(times_code):
        print(
            f"[MC {mc + 1}/{sim.n_mc}] time step {it + 1}/{T}",
            flush=True,
        )

        dt = 0.0 if it == 0 else times_code[it] - times_code[it - 1]

        # Move atoms
        cloud.update_position_diffusive(dt, Diff_coef, rng=rng)

        # Update spatial beam weights
        control_beam.generate_weights(cloud.r_xyz)

        # Spin-wave phase evolution
        dt_s = dt * sim.char_time
        motion_phase = cloud.update_motion_phase(
            dt_s=dt_s,
            B0_T=exp.B0_T,
            B_gradient_z_T_per_code=exp.B_gradient * exp.ref_length,
        )

        weights =  cloud.S * control_beam.w * motion_phase


        # GPU array factor
        AF = array_factor_general_gpu(
            n_hat_flat=n_hat_gpu,
            grid_shape=grid.shape,
            k_out=exp.atom.k_signal,
            r_xyz=cloud.r_xyz,
            w=weights,
            chunk_atoms=sim.chunk_atoms,
            chunk_dirs = sim.chunk_dirs, 
        )
        AF2 = np.abs(AF) ** 2
        I = AF2 * dipole

        eta_t[it] = intensity_overlap_on_sphere(
            grid,
            I,
            E_fib,
            theta0,
        )

        AF_t[it] = AF.astype(np.complex64)
        AF2_t[it] = AF2.astype(np.float32)
        I_t[it] = I.astype(np.float32)

    dt_mc = time.perf_counter() - t0_mc
    print(f"MC {mc + 1}/{sim.n_mc} runtime: {dt_mc:.2f} s", flush=True)
    del cloud, control_beam, rng 

    return eta_t, AF_t, AF2_t, I_t
