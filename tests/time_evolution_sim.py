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
        signal_fwhm_diameter_m = 120e-6, 
        control_fwhm_diameter_m = 300e-6, 
        density = 1e13, 
        scalling = 10000,
        temperature = 200 + 273.15 
        ) 
print(exp)

cloud = CloudModel( "cylinder", 
                   "random", 
                   exp.atom, 
                   Lz = exp.Lz,
                   R = 3 * exp.w0_control, 
                   density = exp.density_rescalled, 
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

theta0 = 1 / (exp.atom.k_signal * exp.w0_signal)

sim = SimParams(n_theta = 100, n_phi = 100, theta_max = 10 * theta0, n_mc = 1) 

setp = sim.sim_metadataSetUp(exp, beam)

grid = sim.create_grid()

cloud.log_info()


cloud.generate_cloud()
cloud.generate_velocity_distribution()
cloud.generate_S_profile(exp.w0_signal) 

dipole = single_dipole_E(grid.nx, grid.ny, grid.nz, np.array([1,0,0]))

times_si = np.linspace(0, 50e-6, 10)
times_code = times_si / exp.char_time
T = len(times_code)
nt, np_ = grid.shape
AF_t = np.zeros((T, nt, np_), dtype=np.complex128)
I_t  = np.zeros((T, nt, np_), dtype=float)
n_inside_t = np.zeros(T)
n_beam_t = np.zeros(T)

beam.generate_weights(cloud.r_xyz)

cloud.r0_xyz = cloud.r_xyz.copy()
Diff_coef = exp.diffusion_coeff_code
for it, t in enumerate(times_code):
    dt = 0.0 if it == 0 else times_code[it] - times_code[it-1]

    cloud.update_position_diffusive(dt,Diff_coef )
    #cloud.update_position(dt)

    beam.generate_weights(cloud.r_xyz)
    motion_phase = cloud.update_motion_phase()
    weights = cloud.S * beam.w * motion_phase

    print("it, dt =", it, dt)
    print("mean displacement =", np.mean(np.linalg.norm(cloud.r_xyz - cloud.r0_xyz, axis=1)))
    print("std phase motion =", np.std(np.angle(motion_phase)))
    print("same r?", np.allclose(cloud.r_xyz, cloud.r0_xyz))

    inside = cloud.cylinder_mask()
    mask_beam = beam.beam_mask(cloud.r_xyz, radius_factor=2.0)
    n_beam_t[it] = np.count_nonzero(mask_beam)
    
    n_inside_t[it] = np.count_nonzero(inside)

    AF = array_factor_general(
        n_hat_flat=grid.n_hat_flat,
        grid_shape=grid.shape,
        k_out=exp.atom.k_signal,
        r_xyz=cloud.r_xyz[inside],
        w= weights[inside],
    )

    AF_t[it] = AF
    I_t[it] = np.abs(AF)**2 * dipole

path = "../data/results_sims/Diffusive_exp_data_s_wave_reduce_cone_50ustimeSim"
save_simulation_npz(path + setp.run_name,
    metadata=asdict(setp),intensity = I_t,atom_pos = cloud.r_xyz,w = weights,AF = AF_t, times_code = times_code, speed_distribution = cloud.v_xyz, n_inside = n_inside_t, n_beam = n_beam_t
                )
####
#
