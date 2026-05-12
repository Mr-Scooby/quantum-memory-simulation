#!/usr/bin/env mathD
# -*- coding: utf-8 -*-

import numpy as np
import os

from radpattern.physics.setup_params import PhysicalRegime, SimParams, SetupParams
from radpattern.geometry.cloud_model import CloudModel
from radpattern.physics.beam import BeamModel
from radpattern.geometry.grids import AngleGrid

from radpattern.physics.rpattern_gpu import array_factor_general_gpu, prepare_gpu_grid
from radpattern.helpers.helpers import single_dipole_E, intensity_from_field
from radpattern.helpers.io import save_simulation_npz

from radpattern.physics.mcpattern import mc_static, static_AF_calculation

from radpattern.plotting.rplotting import plot_atoms
import matplotlib.pyplot as plt

from coupling_calcualtion import intensity_overlap_on_sphere, gaussian_fiber_mode_on_sphere

from dataclasses import asdict 

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import copy
import time 

from config_object import build_run_objects 

import logging
logging.basicConfig(level = logging.INFO) 
log = logging.getLogger(__name__)

from radpattern.physics.experimetal_setup import ExperimentalParams




def run_one_config(config_path, output_dir):
    ### Experimenatal parameters.
    objs = build_run_objects("config_path")

    exp = objs.exp
    sim = objs.sim
    cloud = objs.cloud
    beam = objs.beam

    setp = sim.sim_metadataSetUp(exp, beam)
    # Grid formation
    grid = sim.create_grid()
    cloud.log_info()

    # Dipole generation for intensity
    dipole = single_dipole_E(grid.nx, grid.ny, grid.nz, np.array([1,0,0]))

    # Sim times array
    times_code = sim.time_array() 

    # Array dimensions
    T = sim.time_divisions
    nt, np_ = grid.shape

    # Prepare array storage
    AF_t = np.zeros((T, nt, np_), dtype=np.complex128)
    AF2_mean_t = np.zeros((T, nt, np_), dtype= np.complex128 )
    I_t  = np.zeros((T, nt, np_), dtype=float)
    n_inside_t = np.zeros(T)
    n_beam_t = np.zeros(T)
        # Efficiencies
    eta_all = np.zeros((sim.n_mc, T))
    eta_t = np.zeros(sim.time_divisions)


    # Diffusion coefficients.
    Diff_coef = exp.diffusion_coeff_code

    # Coupling E field for calculating eta

    theta0 = 12 / ( exp.atom.k_signal * exp.w0_signal)
    E_fib = np.abs(gaussian_fiber_mode_on_sphere(grid, theta0))**2

    # Moves n_hat to gpu, to reduce timing
    In_hat_gpu = prepare_gpu_grid(grid.n_hat_flat)
    # MC runs. 
    def mc_single_run(mc): 
        print(f"Run {mc}/{ sim.n_mc}")
        t0_mc = time.perf_counter() 

        # Random generator for each mc instance. 
        rng = np.random.default_rng(1000 + mc )

        # Generate cloud
        cloud.generate_cloud(rng=rng)
        # Generate velocity field
        cloud.generate_velocity_distribution()
        # Generate spinWave profile
        cloud.generate_S_profile(exp.w0_signal)
        # Store original position for time advancement
        cloud.r0_xyz = cloud.r_xyz.copy()

        eta_t = np.zeros(T)
        AF_t = np.zeros((T, nt, np_), dtype=np.complex64)
        AF2_t = np.zeros((T, nt, np_), dtype=np.float32)
        I_t = np.zeros((T, nt, np_), dtype=np.float32)

        # Time evolution runs
        for it, t in enumerate(times_code):
            # getting Delta t. 
            dt = 0.0 if it == 0 else times_code[it] - times_code[it-1]
            print(f"[MC {mc+1}/{sim.n_mc}] time step {it+1}/{T}", flush=True)

            # Evolve and compute new weights
            cloud.update_position_diffusive(dt, Diff_coef, rng=rng)
            beam.generate_weights(cloud.r_xyz)
            motion_phase = cloud.update_motion_phase()

            weights = cloud.S * beam.w * motion_phase

            # Compute far field emission. 
            AF = array_factor_general_gpu(
                n_hat_flat=In_hat_gpu,
                grid_shape=grid.shape,
                k_out=exp.atom.k_signal,
                r_xyz=cloud.r_xyz,
                w=weights,
                chunk_atoms = sim.chunk_atoms,
            )

            # Intensity
            AF2 = np.abs(AF)**2
            I = AF2 * dipole

            # Coupling for time t
            eta_t[it] = intensity_overlap_on_sphere(grid,I,E_fib, theta0 )  
            AF_t[it] = AF.astype(np.complex64)
            AF2_t[it] = AF2.astype(np.float32)
            I_t[it] = I.astype(np.float32)

        dt_mc = time.perf_counter() - t0_mc
        print(f"MC {mc+1}/{sim.n_mc} runtime: {dt_mc:.2f} s", flush=True)
        return eta_t,AF_t, AF2_t, I_t


    n_jobs = 1   # start with 4, then test 6, 8, etc.
    t0_total = time.perf_counter()

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(mc_single_run)(mc)
        for mc in range(sim.n_mc)
    )


    eta_all = np.array([r[0] for r in results])
    AF_all  = np.array([r[1] for r in results])
    AF2_all = np.array([r[2] for r in results])
    I_all   = np.array([r[3] for r in results])

    eta_mean = eta_all.mean(axis=0)
    AF_mean_t = AF_all.mean(axis=0)
    AF2_mean_t = AF2_all.mean(axis=0)
    I_mean_t = I_all.mean(axis=0)



    total_seconds = time.perf_counter() - t0_total
    print(f"Total runtime: {total_seconds:.2f} s")
    print(f"Total runtime: {total_seconds/60:.2f} min")

    path = os.path.join(
            os.path.expanduser("~"),
            "radek",
            "simulations",
            "data",
            "results_sims",
            setp.run_name,
        )


    save_simulation_npz(
        path,
        metadata=asdict(setp),
        times_code=times_code,
        AF = AF_mean_t,
        AF2 = AF2_mean_t,
        intensity = I_mean_t, 
        eta_all=eta_all,
        )
##

