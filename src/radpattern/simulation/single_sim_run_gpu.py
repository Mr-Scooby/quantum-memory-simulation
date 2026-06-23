#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Single Simulation run. Uses gpu AF calculation function """

import time
import numpy as np
from copy import deepcopy

from radpattern.physics.rpattern_gpu import array_factor_general_gpu
from radpattern.physics.coupling import intensity_overlap_on_sphere

import logging 
log = logging.getLogger(__name__) 


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
    log.info("simulating Run %d / %d", mc + 1 , sim.n_mc)
    # Perfomace timing of the whole time simulation
    t0_mc = time.perf_counter()

    # Independent RNG per MC run
    seed = 1000 if sim.seed is None else int(sim.seed)
    log.debug("Seed = %d", seed)

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

    max_step_fraction = 0.1
    # Running time step simulation
    for it, t in enumerate(times_code):
        print(
            f"[MC {mc + 1}/{sim.n_mc}] time step {it + 1}/{T}",
            flush=True,
        )

        dt = 0.0 if it == 0 else times_code[it] - times_code[it - 1]
        # dt_max is the max time step possible so diffussion step is not larger than char szie of the ensemble 
        dt_max = (max_step_fraction * cloud.char_size) ** 2 / (2.0 *Diff_coef) 

        # If current dt is bigger subdivide accordingly
        n_sub = max(1, int(np.ceil( dt / dt_max) ))
        if n_sub > 1:
            log.warning(
                "Diffussion substep.  MC=%d step=%d/%d: dt=%.3e > dt_max=%.3e. "
                "Splitting into %d internal steps to keep rms diffusion step below %.2f * char_size.",
                mc,
                it + 1,
                T,
                dt,
                dt_max,
                n_sub,
                max_step_fraction,
            )
        dt /= n_sub
        dt_s = dt * sim.char_time

        for _ in range(n_sub):

            # Move atoms
            cloud.update_position_diffusive(dt, Diff_coef, rng=rng)

            # Spin-wave phase evolution
            motion_phase = cloud.update_motion_phase(
                dt_s=dt_s,
                B0_T=exp.B0_T,
                B_gradient_z_T_per_code=exp.B_gradient * exp.ref_length,
            )

        # Update spatial beam weights
        control_beam.generate_weights(cloud.r_xyz)

        

        # Generating weights
        weights =  cloud.S * control_beam.w * motion_phase

        # Warning control if weights goes to zero. 
        if np.max(np.abs(weights)) < 1e-14:
            log.warning(
                "MC %d timestep %d: all retrieval weights are near zero. "
                "Coupling/AF result may be meaningless.",
                mc,
                it,
            )


        # GPU array factor
        ## timing control of the AF calculation
        t0_af = time.perf_counter() 
        AF = array_factor_general_gpu(
            n_hat_flat=n_hat_gpu,
            grid_shape=grid.shape,
            k_out=exp.atom.k_signal,
            r_xyz=cloud.r_xyz,
            w=weights,
            chunk_atoms=sim.chunk_atoms,
            chunk_dirs = sim.chunk_dirs, 
        )
        log.debug("AF timestep %d calculation runtime %.5f", it, time.perf_counter() - t0_af) 
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
    log.info(f"MC %d/%d  runtime: %.2f s || %.3f min", (mc + 1), sim.n_mc, dt_mc, dt_mc/60 ) 
    del cloud, control_beam, rng 

    return eta_t, AF_t, AF2_t, I_t
