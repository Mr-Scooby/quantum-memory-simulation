#!/usr/bin/env math
# -*- coding: utf-8 -*-

import numpy as np

from radpattern.physics.setup_params import PhysicalRegime, SimParams, SetupParams
from radpattern.geometry.cloud_model import CloudModel
from radpattern.physics.beam import BeamModel
from radpattern.geometry.grids import AngleGrid

from radpattern.physics.rpattern_gpu import array_factor_general_gpu
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

import logging
logging.basicConfig(level = logging.INFO) 
log = logging.getLogger(__name__)

from radpattern.physics.experimetal_setup import ExperimentalParams


### Experimenatal parameters.
# Martins data

exp = ExperimentalParams(
        atoms = "Cs133", 
        lambda_control_m = 895e-9,
        delta_f_hz = 9.12e9, 
        cell_length_m = 75e-3, 
        cell_diameter_m = 4e-3, 
        signal_fwhm_diameter_m = 120e-6, 
        control_fwhm_diameter_m = 300e-6, 
        density_cm3 = 1e13, 
        scalling = 10000,
        temperature = 75 + 273.15, 
        label = "Parallelization Testing, sim density low ",
        buffer_gas = "N2",
        buffer_pressure_Torr = 10, 
        diffusion_D0_cm2_s = 0.240 , # From liteture. Phd LuisaEsguerra 
        diffusion_T0_K = 273.15, 
        diffusion_P0_Torr = 760  # 1 atm. 1Torr = 1/760 atm 
        ) 
print(exp)


# Sim params
sim = SimParams(n_theta = 100, n_phi = 100,
                theta_max = 10 * exp.forwardlobe_angular_width,
                sim_time_us = 50, #microseconds
                time_divisions = 10, 
                char_time = exp.char_time, 
                sim_density = 1e6,
                chunk_atoms = 1000,
                n_mc =4 ) 

cloud = CloudModel( geometry = "cylinder", 
                   distribution = "random", 
                   atoms = exp.atom, 
                   Lz = exp.Lz,
                   R = 3 * exp.w0_control, 
                   sim_density = sim.sim_density, 
                   )

beam = BeamModel(
    beam_type="gaussian_pulse",
    w0= exp.w0_control,
    sigma_long = 3,
    k_in_hat=np.array([0, 0, 1]),
    k_in=exp.atom.k_control,
    box_size=cloud.box_size,
    pcenter_at_origin = True,
)

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
I_t  = np.zeros((T, nt, np_), dtype=float)
n_inside_t = np.zeros(T)
n_beam_t = np.zeros(T)
    # Efficiencies
eta_all = np.zeros((sim.n_mc, T))
eta_t = np.zeros(sim.time_divisions)


# Diffusion coefficients.
Diff_coef = exp.diffusion_coeff_code

# Coupling E field for calculating eta
E_fib = np.abs(gaussian_fiber_mode_on_sphere(grid, exp.forwardlobe_angular_width))**2

# MC runs. 
def mc_single_run(mc): 
    print(f"Run {mc}/{ sim.n_mc}")
    
    # Random generator for each mc instance. 
    rng = np.random.default_rng()

    # Generate cloud
    cloud.generate_cloud(rng=rng)
    # Generate velocity field
    cloud.generate_velocity_distribution()
    # Generate spinWave profile
    cloud.generate_S_profile(exp.w0_signal)
    # Store original position for time advancement
    cloud.r0_xyz = cloud.r_xyz.copy()

    eta_t = np.zeros(T)

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
            n_hat_flat=grid.n_hat_flat,
            grid_shape=grid.shape,
            k_out=exp.atom.k_signal,
            r_xyz=cloud.r_xyz,
            w=weights,
            chunk_atoms = sim.chunk_atoms,
        )

        # Intensity
        I = np.abs(AF)**2 * dipole

        # Coupling for time t
        eta_t[it] = intensity_overlap_on_sphere(grid,I,E_fib, exp.forwardlobe_angular_width)  

    return eta_t

if __name__ == "__main__":

    n_jobs = 4   # start with 4, then test 6, 8, etc.

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(mc_single_run)(mc)
        for mc in range(sim.n_mc)
    )

    eta_all = np.array(results)

    path = "../data/results_sims/parallel_testCS133"

    save_simulation_npz(
        path + setp.run_name,
        metadata=asdict(setp),
        times_code=times_code,
        eta_all=eta_all,
        )
##

