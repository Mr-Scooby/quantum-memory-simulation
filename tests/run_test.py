#!/usr/bin/env python3
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
        scalling = 10000
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
    w0=exp.w0_control,
    sigma_long = 2,
    k_in_hat=np.array([0, 0, 1]),
    k_in=exp.atom.k_control,
    box_size=cloud.box_size,
    pcenter_at_origin = True,
)

cloud.log_info()
cloud.generate_cloud()
cloud.generate_S_profile(exp.w0_signal) 
#
beam.generate_weights(cloud.r_xyz)
weights = cloud.S * beam.w 

theta0 = 1 / (exp.atom.k_signal * exp.w0_signal)
sim = SimParams(n_theta = 45, n_phi = 91, n_mc = 1 ) #20 * theta0, n_mc = 1) 
setp = sim.sim_metadataSetUp(exp, beam)

grid = sim.create_grid()

###Compute AF
#AF = array_factor_general(
#    n_hat_flat=grid.n_hat_flat,
#    grid_shape=grid.shape,
#    k_out=exp.atom.k_control,
#    r_xyz=cloud.r_xyz,
#    w=weights,
#)

# after building retrieval polarization weights
phase_expected = np.exp(-1j * exp.atom.k_signal * cloud.r_xyz[:,2])

print("phase expected overlap : ", np.abs(np.vdot(weights, phase_expected)) /
      np.sqrt(np.vdot(weights, weights).real * np.vdot(phase_expected, phase_expected).real))

AF_mean, AF2_mean = mc_static(cloud, beam, exp, grid, sim.n_mc)
#
dipole = single_dipole_E(grid.nx, grid.ny, grid.nz, np.array([1,0,0]))
I = np.abs(AF2_mean)**2 * dipole
#
path = "../data/results_sims/exp_data_s_wave_reduce_cone_test"
save_simulation_npz(path + setp.run_name,
    metadata=asdict(setp),intensity = I,atom_pos = cloud.r_xyz,w = weights,AF = AF_mean, AF2_mean = AF2_mean)
##
#
