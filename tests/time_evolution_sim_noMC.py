#!/usr/bin/env math
# -*- coding: utf-8 -*-

import numpy as np

from radpattern.physics.setup_params import PhysicalRegime, SimParams, SetupParams
from radpattern.geometry.cloud_model import CloudModel, AtomSpeciment
from radpattern.physics.beam import BeamModel
from radpattern.geometry.grids import AngleGrid

from radpattern.physics.rpattern import array_factor_general
from radpattern.helpers.helpers import single_dipole_E, intensity_from_field
from radpattern.helpers.io import save_simulation_npz

from radpattern.physics.mcpattern import mc_static, static_AF_calculation

from coupling_calcualtion import intensity_overlap_on_sphere, gaussian_fiber_mode_on_sphere

from dataclasses import asdict 

import logging
logging.basicConfig(level = logging.INFO) 
log = logging.getLogger(__name__)

from radpattern.physics.experimetal_setup import ExperimentalParams


def main(Cdiameterfactor): 

    exp = ExperimentalParams(
            atoms = "Rb87", 
            lambda_control_m = 795e-9,
            delta_f_hz = 6.834682610e9, 

            # |g> = |F=1, mF=+1>
            g_g = -0.5018,
            m_g = +1     ,

            # |s> = |F=2, mF=+1>
            g_s = +0.4998,
            m_s = +1     ,

            cell_length_m = 75e-3, 
            cell_diameter_m = 4e-3, 
            signal_fwhm_diameter_m = 1.5 * 235.2* 1e-6, 
            control_fwhm_diameter_m = Cdiameterfactor * 588.7 * 1e-6, 
            density_cm3 = 1e8, 
            scalling = 10000,
            temperature =  10e-6, 
            label = "RB87. Varying Control. NO buffer gas. Ballistic motion. ",
            buffer_gas = "NONE",
            buffer_pressure_Torr = 1, # 10, 
            diffusion_D0_cm2_s = 1  , #0.240 , # From liteture. Phd LuisaEsguerra 
            diffusion_T0_K = 1      , #273.15, 
            diffusion_P0_Torr = 1   , # 760  # 1 atm. 1Torr = 1/760 atm 
            ) 
    print(exp)
    sim = SimParams(n_theta = 100, n_phi = 100,
                    theta_max = 10 * exp.forwardlobe_angular_width,
                    sim_time_us = 200_000, #microseconds
                    time_divisions = 15, 
                    char_time = exp.char_time, 
                    sim_density = 1e5,
                    n_mc =1 ) 

    cloud = CloudModel( geometry = "cylinder", 
                       distribution = "gaussian", 
                       atoms = exp.atom,  # Type 
                       Lz = exp.Lz,
                       R = 3 * 0.100559, 
                       sim_density = sim.sim_density, 
                        
                       sigma_x=1e-3 / exp.ref_length,# 1 mm in code units. 
                       sigma_y=1e-3 / exp.ref_length,
                       sigma_z=1e-3 / exp.ref_length,
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

    # Dipole generator for intensity
    dipole = single_dipole_E(grid.nx, grid.ny, grid.nz, np.array([1,0,0]))

    # Sim times array 
    times_code = sim.time_array("linspace") 

    # Grid dimensions. 
    nt, np_ = grid.shape

    # Prepare array storage
    AF_t = np.zeros((sim.time_divisions, nt, np_), dtype=np.complex128)
    I_t  = np.zeros((sim.time_divisions, nt, np_), dtype=float)
    # Efficiencie
    eta_all = np.zeros((sim.n_mc, sim.time_divisions))
    eta_t = np.zeros(sim.time_divisions)


    # Diagnostics 
    # Count of atoms inside beam. 
    T = sim.time_divisions
    diag = {
        "mean_disp": np.zeros(T),
        "rms_disp": np.zeros(T),
        "rms_disp_theory": np.zeros(T),
        "mean_dz": np.zeros(T),
        "std_dz": np.zeros(T),
        "std_dz_theory": np.zeros(T),
        "phase_coherence": np.zeros(T),
        "phase_std": np.zeros(T),
        "phase_coherence_theory": np.zeros(T),
        "mean_beam_abs": np.zeros(T),
        "source_norm": np.zeros(T),
        "n_inside_dynamic": np.zeros(T),
        "n_inside_fixed": np.zeros(T),
        "S_norm": np.zeros(T) 
    }


    # E_field for coupling and calculating Eta
    E_fib = np.abs(gaussian_fiber_mode_on_sphere(grid, exp.forwardlobe_angular_width))**2

    # Random generator.
    rng = np.random.default_rng(1000 + sim.n_mc)

    # Generate simulated system. 
    cloud.generate_cloud(rng=rng)
    cloud.generate_velocity_distribution()

    cloud.generate_S_profile(
        exp.w0_signal,
        z_span_mode="percentile",
        z_percentiles=(0.5, 99.5),
        profile="sqrt_1_minus_z2",
    )

    beam.generate_weights(cloud.r_xyz)
    weights0 = cloud.S * beam.w
    norm0 = np.sqrt(np.sum(np.abs(weights0)**2))

    cloud.r0_xyz = cloud.r_xyz.copy()

    # Set difffusion coef for brownian motion. 
    Diff_coef = exp.diffusion_coeff_code

    # Run time evolution simulations 
    for it, t in enumerate(times_code):
        dt = 0.0 if it == 0 else times_code[it] - times_code[it-1]

        # Update atom positions.
        #cloud.update_position_diffusive(dt, Diff_coef, rng=rng)
        cloud.update_position(dt)
        # Mask to not count atoms outside of >3sigma of control beam. 
        # Basically no activation.
        r = cloud.r_xyz
        dr = r - cloud.r0_xyz
        dz = dr[:, 2]
        disp2 = np.sum(dr**2, axis=1)

        # diffusion theory in your code units:
        # each axis: std = sqrt(2 D t)
        # 3D: <r^2> = 6 D t
        t_abs = times_code[it]

        diag["mean_disp"][it] = np.mean(np.sqrt(disp2))
        diag["rms_disp"][it] = np.sqrt(np.mean(disp2))
        diag["rms_disp_theory"][it] = np.sqrt(6 * Diff_coef * t_abs)

        diag["mean_dz"][it] = np.mean(dz)
        diag["std_dz"][it] = np.std(dz)
        diag["std_dz_theory"][it] = np.sqrt(2 * Diff_coef * t_abs)


        # Recalculate the Weights.
        beam.generate_weights(cloud.r_xyz)
        # Update motion phase of SW.
        motion_phase = cloud.update_motion_phase()

        # update Magnetic phas#e
        #b_phase = cloud.update_motion_phase(dt * exp.char_time, exp.B_gradient_z_T_per_code)
        b_phase = 1 
        phase = np.angle(motion_phase)

        diag["phase_coherence"][it] = np.abs(np.mean(motion_phase))
        diag["phase_std"][it] = np.std(np.unwrap(phase))
        diag["phase_coherence_theory"][it] = np.exp(-Diff_coef * exp.atom.k_sw**2 * t_abs)
        diag["S_norm"][it] = np.sum(np.abs(cloud.S)**2)
        weights = cloud.S * beam.w * motion_phase * b_phase

        diag["mean_beam_abs"][it] = np.mean(np.abs(beam.w))
        diag["source_norm"][it] = np.sum(np.abs(cloud.S * beam.w * motion_phase)**2)

        diag["n_inside_dynamic"][it] = np.count_nonzero(cloud.cylinder_mask())


        # Compute AF. 
        AF = array_factor_general(
            n_hat_flat=grid.n_hat_flat,
            grid_shape=grid.shape,
            k_out=exp.atom.k_signal,
            r_xyz=cloud.r_xyz,
            w=weights, 
        )
       
        # Intensity. 
        I = np.abs(AF)**2 * dipole
        AF_t[it]  = AF
        I_t[it] = I

        eta_t[it] = intensity_overlap_on_sphere(grid,I,E_fib, exp.forwardlobe_angular_width)  # replace with your coupling function

    path = f"../data/results_sims/ColdAtomRB87_simDens{int(sim.sim_density // 1e6)}e6_ExpData_Swave_reduce_cone_{int(sim.sim_time_us)}ustimeSim_GeomSpace_{Cdiameterfactor}ControlBeamfactor"
    save_simulation_npz(path + setp.run_name,
        metadata=asdict(setp),intensity = I_t,atom_pos = cloud.r_xyz,w = weights,AF = AF_t, times_code = times_code, speed_distribution = cloud.v_xyz, eta = eta_all, 
                        **diag,
                    )
##


if __name__ == "__main__":

    for Cdiameterfactor in [0.7,0.9,1,1.2, 1.5, 1.7, 2, 2.2 ] : 
        #print(Sdiamter)
        main(Cdiameterfactor)


