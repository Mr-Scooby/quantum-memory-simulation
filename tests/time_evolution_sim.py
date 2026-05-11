#!/usr/bin/env math
# -*- coding: utf-8 -*-

import numpy as np

from radpattern.physics.setup_params import PhysicalRegime, SimParams, SetupParams
from radpattern.geometry.cloud_model import CloudModel
from radpattern.physics.beam import BeamModel
from radpattern.geometry.grids import AngleGrid

from radpattern.physics.rpattern import array_factor_general
from radpattern.helpers.helpers import single_dipole_E, intensity_from_field
from radpattern.helpers.io import save_simulation_npz

from radpattern.physics.mcpattern import mc_static, static_AF_calculation

from radpattern.plotting.beam_test import check_beam_window, plot_atom_distribution,plot_weight_distribution
from radpattern.plotting.rplotting import plot_atoms
import matplotlib.pyplot as plt

from coupling_calcualtion import intensity_overlap_on_sphere, gaussian_fiber_mode_on_sphere

from dataclasses import asdict 

import logging
logging.basicConfig(level = logging.INFO) 
log = logging.getLogger(__name__)

from radpattern.physics.experimetal_setup import ExperimentalParams


exp = ExperimentalParams(
        atoms = "Cs133", 
        lambda_control_m = 895e-9,
        delta_f_hz = 9.12e9, 
        cell_length_m = 75e-3, 
        cell_diameter_m = 4e-3, 
        signal_fwhm_diameter_m = 2 * 120e-6, 
        control_fwhm_diameter_m = 300e-6, 
        density_cm3 = 1e13, 
        scalling = 10000,
        temperature = 50 + 273.15, 
        label = "2/1 ratios of signal/control beam. 5Torr bufferPresure. GeomTimeSpacing",
        buffer_gas = "N2",
        buffer_pressure_Torr = 10, 
        diffusion_D0_cm2_s = 0.240 , # From liteture. Phd LuisaEsguerra 
        diffusion_T0_K = 273.15, 
        diffusion_P0_Torr = 760  # 1 atm. 1Torr = 1/760 atm 
        ) 
print(exp)
sim = SimParams(n_theta = 100, n_phi = 100,
                theta_max = 10 * exp.forwardlobe_angular_width,
                sim_time_us = 50, #microseconds
                time_divisions = 10, 
                char_time = exp.char_time, 
                sim_density = 1e4,
                n_mc =1 ) 

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


grid = sim.create_grid()
cloud.log_info()

dipole = single_dipole_E(grid.nx, grid.ny, grid.nz, np.array([1,0,0]))

times_code = sim.time_array() 

T = sim.time_divisions
nt, np_ = grid.shape
AF_t = np.zeros((T, nt, np_), dtype=np.complex128)
I_t  = np.zeros((T, nt, np_), dtype=float)
n_inside_t = np.zeros(T)
n_beam_t = np.zeros(T)


Diff_coef = exp.diffusion_coeff_code

eta_all = np.zeros((sim.n_mc, T))
E_fib = np.abs(gaussian_fiber_mode_on_sphere(grid, exp.forwardlobe_angular_width))**2

for mc in range(sim.n_mc):
    print(f"Run {mc}/{ sim.n_mc}")
    rng = np.random.default_rng(1000 + mc)

    cloud.generate_cloud(rng=rng)
    cloud.generate_velocity_distribution()
    cloud.generate_S_profile(exp.w0_signal)
    cloud.r0_xyz = cloud.r_xyz.copy()

    eta_t = np.zeros(T)

    for it, t in enumerate(times_code):
        dt = 0.0 if it == 0 else times_code[it] - times_code[it-1]

        cloud.update_position_diffusive(dt, Diff_coef, rng=rng)

        beam.generate_weights(cloud.r_xyz)
        motion_phase = cloud.update_motion_phase()
        weights = cloud.S * beam.w * motion_phase

        inside = cloud.cylinder_mask()

        AF = array_factor_general(
            n_hat_flat=grid.n_hat_flat,
            grid_shape=grid.shape,
            k_out=exp.atom.k_signal,
            r_xyz=cloud.r_xyz[inside],
            w=weights[inside],
        )

        I = np.abs(AF)**2 * dipole

        eta_t[it] = intensity_overlap_on_sphere(grid,I,E_fib, exp.forwardlobe_angular_width)  # replace with your coupling function

    eta_all[mc] = eta_t

eta_mean = eta_all.mean(axis=0)
eta_std = eta_all.std(axis=0)
eta_sem = eta_std / np.sqrt(sim.n_mc)

path = "../data/results_sims/DiffusiveBufferN2_3Torr_simDens1e6_exp_data_s_wave_reduce_cone_50ustimeSim_GeomSpace"
save_simulation_npz(path + setp.run_name,
    metadata=asdict(setp),intensity = I_t,atom_pos = cloud.r_xyz,w = weights,AF = AF, times_code = times_code, speed_distribution = cloud.v_xyz, eta = eta_all
                )
##

