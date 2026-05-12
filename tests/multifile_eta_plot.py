#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coupling_calcualtion as cp
from radpattern.physics.setup_params import ExperimentalParams, SimParams 
from radpattern.physics.beam import BeamModel 
from radpattern.geometry.cloud_model import CloudModel 
import matplotlib.pyplot as plt 
import numpy as np
import re 


PATH = "../data/results_sims/"
#files = [
#         "DiffusiveBufferN2_2Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_c2e6fcd3",
#         "DiffusiveBufferN2_3Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_260f4b5d",
#         "DiffusiveBufferN2_4Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_88032039",
#         "DiffusiveBufferN2_5Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_1a93c62c",
#         "DiffusiveBufferN2_6Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_14e191de",
#         "DiffusiveBufferN2_7Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_7c19ae1e",
#         "DiffusiveBufferN2_8Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_3c2c60c5",
#         "DiffusiveBufferN2_9Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_1c3f4742",
#         "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_582bdb8b",
#         "DiffusiveBufferN2_11Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_6e487271",
#         "DiffusiveBufferN2_12Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_286e4962",
#         ]
#
#files = [
#        "DiffusiveBufferN2_1Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_b943d55c",
#        "DiffusiveBufferN2_2Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_4f4c8533",
#        "DiffusiveBufferN2_3Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_2dfc3fc2",
#        "DiffusiveBufferN2_4Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_a50b2201",
#        "DiffusiveBufferN2_5Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_6c059ef2",
#        ]
#
#files = [
#            "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_300Cdiamter_simT50us_nt16_582bdb8b",
#            "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_400Cdiamter_simT50us_nt16_db008155",
#            "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_500Cdiamter_simT50us_nt16_04f05c8e",
#            "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_600Cdiamter_simT50us_nt16_28abe6c2",
#            "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_700Cdiamter_simT50us_nt16_110992e2",
#            "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_800Cdiamter_simT50us_nt16_cec61e39",
#            "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_900Cdiamter_simT50us_nt16_287b99bb",
#]

# CS. Changing signal 
files = [
"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_120Sdiamter_simT50us_nt16_f1118cef",
"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_150Sdiamter_simT50us_nt16_72859d29",
"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_180Sdiamter_simT50us_nt16_deede175",
"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_210Sdiamter_simT50us_nt16_db316c0c",
"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_240Sdiamter_simT50us_nt16_30183bee",
"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_270Sdiamter_simT50us_nt16_6c0a1957",
]

## Beam Ratios change 
#files =[
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_1ratioCS_simT50us_nt16_f1118cef",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_1.3ratioCS_simT50us_nt16_5fa362c1",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_1.5ratioCS_simT50us_nt16_0a558c77",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_1.7ratioCS_simT50us_nt16_ae75dd25",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_2ratioCS_simT50us_nt16_90252e96",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_2.3ratioCS_simT50us_nt16_3e6d918a",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_2.5ratioCS_simT50us_nt16_bcc75753",
#]
#
#
## Cs133 Sim Changing Control beam. 
#files =[
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_300Cdiam_fixedR_simT50us_nt16_1636c86f",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_400Cdiam_fixedR_simT50us_nt16_50205c0c",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_500Cdiam_fixedR_simT50us_nt16_fceee68b",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_600Cdiam_fixedR_simT50us_nt16_a30e91bb",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_700Cdiam_fixedR_simT50us_nt16_29da55fe",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_800Cdiam_fixedR_simT50us_nt16_f01e48b9",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_900Cdiam_fixedR_simT50us_nt16_537225a9",
#]
#
# RB87 sim with changing control beam. The signal is 100 um
#files = [
##"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_0.7ControlBeamfactor_simT200000us_nt15_0bd1783f",
##"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_0.9ControlBeamfactor_simT200000us_nt15_4596d951",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1ControlBeamfactor_simT200000us_nt15_05856623",
##"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.2ControlBeamfactor_simT200000us_nt15_8fec7ff7",
##"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.5ControlBeamfactor_simT200000us_nt15_ef27abec",
##"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.7ControlBeamfactor_simT200000us_nt15_c51d58be",
##"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_2ControlBeamfactor_simT200000us_nt15_78ade96a",
        # ]
#
#files = ["ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.5ControlBeamfactor_simT200000us_nt15_ef27abec"]


# RB87 Sim with changing control beam.  A  signal beam of 17 um
#files = [
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_0.7ControlBeamfactor_simT200000us_nt15_5b43f4cc",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_0.9ControlBeamfactor_simT200000us_nt15_4240ffe8",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1ControlBeamfactor_simT200000us_nt15_74619d5f",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.2ControlBeamfactor_simT200000us_nt15_4adb2fd3",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.5ControlBeamfactor_simT200000us_nt15_ee05c512",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.7ControlBeamfactor_simT200000us_nt15_a971ba17",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_2ControlBeamfactor_simT200000us_nt15_0b537299",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_2.2ControlBeamfactor_simT200000us_nt15_372b7454",
#]


files = [
        "Cs133_simT50us_nt30_f5923077", 
        "Cs133_simT50us_nt30_9ae4db70",
        "270Sdiam_Cs133_simT50us_nt30_2f9f2cd4",
"270Sdiam_Cs133_simT50us_nt30_50062153",
"120Sdiam_Cs133_simT50us_nt30_668eec6f",
        ]

files =[ 
"120Sdiam_100McCs133_simT50us_nt30_0ca5af1d",
"120Sdiam_10McCs133_simT50us_nt30_f5923077",
]
#
# plot title and legend title
title =  "Coupling decay. Rb87. T= 1mk.\n ControlBeam varying, signal 170um"
legend_title = "Control Diam" 
timeScale = "ms" 
# To match from file name for labels 
regex_pattern =r'(\d+)Sdiamter'

#Plot labels 
labels = np.zeros(len(files))
beamRatios = np.zeros(len(files)) # Control/signal ratio 


Time_division = 30

etas = np.zeros((len(files), Time_division)) 

P_fiber = np.zeros((len(files), Time_division)) 
P_total = np.zeros((len(files), Time_division)) 
eta_i = np.zeros((len(files), Time_division)) 
I = np.zeros((len(files), Time_division))

Diffusion_cte = np.zeros(len(files))


def fiber_coupling_vs_time(I_t, grid, theta_f):
    """
    I_t: shape (T, ntheta, nphi), already |AF|^2 * dipole
    theta_f: Gaussian fiber intensity radius in radians

    Returns
    -------
    eta : np.ndarray
        Fiber coupling efficiency versus time, with shape (T,). Defined as
        P_fiber / P_total.

    P_fiber : np.ndarray
        Fiber-mode-weighted angular power versus time, with shape (T,).

    P_total : np.ndarray
        Total angularly integrated power versus time, with shape (T,).
    """
    theta = grid.TH

    Gfiber = np.exp(-(grid.TH / theta_f)**2)

    dtheta = grid.theta[1] - grid.theta[0]
    dphi = grid.phi[1] - grid.phi[0]
    dOmega = np.sin(theta) * dtheta * dphi

    P_fiber = np.sum(I_t * Gfiber[None, :, :] * dOmega[None, :, :], axis=(1, 2))
    P_total = np.sum(I_t * dOmega[None, :, :], axis=(1, 2))

    eta = P_fiber / P_total

    return eta, P_fiber, P_total





### Data extraction and formation of new objects to get the properties values. 
for file_idx, file in enumerate(files): 
    print(f"showing file = {PATH+file}")
    npz = np.load(PATH+file+'.npz', allow_pickle=True)
    try: 
        meta = npz["metadata"].item()

        exp = ExperimentalParams(
                atoms = meta['regime']['atoms'], 
                lambda_control_m = meta['regime']['lambda_control_m'],
                delta_f_hz = meta['regime']['delta_f_hz'], 
                cell_length_m = meta['regime']['cell_length_m'], 
                cell_diameter_m = meta['regime']['cell_diameter_m'], 
                signal_fwhm_diameter_m = meta['regime']['signal_fwhm_diameter_m'], 
                control_fwhm_diameter_m = meta['regime']['control_fwhm_diameter_m'], 
                density_cm3 = meta['regime']['density_cm3'], 
                scalling = meta['regime']['scalling'],
                temperature = meta['regime']['temperature'],
                label = meta['regime']['label'],
                buffer_pressure_Torr = meta['regime']['buffer_pressure_Torr'],
                buffer_gas = meta['regime']['buffer_gas'],
                diffusion_D0_cm2_s =meta['regime']['diffusion_D0_cm2_s'], #0.240 , # From liteture. Phd LuisaEsguerra 
                diffusion_T0_K = meta['regime']['diffusion_T0_K'], #273.15, 
                diffusion_P0_Torr =meta['regime']['diffusion_P0_Torr'], # 760  # 1 atm. 1Torr = 1/760 atm 
) 
        print(f"label: {exp.label.upper()}")
        print(exp)
        cloud = CloudModel( "cylinder", 
                           "random", 
                           exp.atom, 
                           Lz = exp.Lz,
                           R = 3 * exp.w0_control, 
                           sim_density = 1e6, 
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

        sim = SimParams(n_theta = meta["sim"]["n_theta"],
                    n_phi = meta["sim"]["n_phi"],
                    theta_max = meta["sim"]["theta_max"],
                    sim_time_us = meta["sim"]["sim_time_us"], #microseconds
                    time_divisions = meta["sim"]["time_divisions"],
                    char_time =meta["sim"]["char_time"],
                    sim_density =meta["sim"]["sim_density"],
                    n_mc =meta["sim"]["n_mc"]
                        ) 

        cloud.log_info()

        # convert code time -> SI -> microseconds
        times_code = npz["times_code"]
        char_time = exp.char_time          # [s] = ref_length / ref_velocity
        times_si = times_code * char_time  # [s]
        times_us = times_si * 1e6          # [µs]

        print(f"char_rime {char_time}")
        print(times_us)

        grid = sim.create_grid()


        AF = npz["AF"]
        Intensity  = npz["intensity"]
        
        # stripping labels: 
        match = re.search(regex_pattern, file)
        value = int(match.group(1)) if match else None
        labels[file_idx] = value

        beamRatios[file_idx] = exp.control_to_signal_waist_ratio 
    except KeyError: 
        print(" unable to generate objet from metadata... keyerror") 



    ### Calculating Gaussian mode. 
    ### Coupling to gaussian mode calculation.
    theta0 = 10 / ( exp.atom.k_signal * exp.w0_signal)
    print(f"theta0 = {theta0}, forwardLobe = {exp.forwardlobe_angular_width}, equal? {theta0 == exp.forwardlobe_angular_width}") 

    E_fib = cp.gaussian_fiber_mode_on_sphere(grid, theta0)#* np.exp(1j * np.angle(AF))
    I_fib = np.abs(E_fib)**2

    eta_t = np.zeros(AF.shape[0])
    eta_abs_t =np.zeros(AF.shape[0])

    print(f"Shape Inetensity {Intensity.shape}")
    I_t = np.zeros(AF.shape[0])
    eta_i_   =np.zeros(AF.shape[0]) 
    P_fiber_ =np.zeros(AF.shape[0]) 
    P_total_ =np.zeros(AF.shape[0]) 
    for it in range(AF.shape[0]):
        try: 
            E_field = AF[it]
            eta, amp = cp.overlap_on_sphere(grid, E_field, E_fib)
            print(f"eta = {eta}, amp ={amp}")

            eta_test = np.sum(np.abs(AF[it])**2 * np.abs(E_fib)**2) / np.sum(np.abs(AF[it])**2)

            #eta_abs, _ = cp.overlap_on_sphere(grid, np.abs(AF[it]),np.abs( E_fib))
            eta_abs = cp.intensity_overlap_on_sphere(grid,np.abs(E_field)**2 , I_fib, theta_max = theta0)
#            eta_i_[it], P_fiber_[it], P_total_[it] = fiber_coupling_vs_time(np.abs(E_field)**2, grid, theta0)

            I_t[it] = np.mean(np.abs(E_field[0,:])**2)

            eta_t[it] = eta
            eta_abs_t[it] = eta_abs
            

        except KeyError: 
            eta = np.nan
            amp = np.nan

   # eta_abs_t
    etas[file_idx, : ] = eta_abs_t
    I[file_idx,:] = I_t 
    eta_i[file_idx,:] = eta_i_
    P_fiber[file_idx,:] = P_fiber_
    P_total[file_idx,:] = P_total_

print(f"Intensity +z {I}")
print(f"Pfiber +z {P_fiber}")
print(etas)


########################################

plt.rcParams.update({
    'font.size': 12,          # Default text size
    'axes.titlesize': 30,     # Plot title size
    'axes.labelsize': 27,     # X/Y axis label size
    'xtick.labelsize': 18,    # X-axis tick label size
    'ytick.labelsize': 18,    # Y-axis tick label size
    'legend.fontsize': 19,     # Legend text size
    'legend.title_fontsize': 19 
})


# Setting time sclae [us or ms]
if timeScale.upper() == "MS": 
    times_us /= 1e3    # Convet us to ms

labels = [120, 180, 270, 270_1, 120_1]

##### coupling / dephasing plot ---
fig, ax = plt.subplots(figsize=(7, 4.8))

for idx, file in enumerate(files[:7]): 
    ax.plot(times_us, etas[idx, : ], "o-", label=labels[idx])
for idx, file in enumerate(files[7:],7): 
    ax.plot(times_us, etas[idx, : ], "*--", label=labels[idx])

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"coupling $\eta$")
ax.set_title(title)

ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)

plt.show()
print(beamRatios)


# Plots forwards intensity vs Time.
fig, ax = plt.subplots(figsize = (7, 4.8) )
for idx, file in enumerate(files[:7]): 
    ax.plot(times_us, I[idx,:], "o-", label=labels[idx])

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"Intensity [AF^2] $")
ax.set_title(title)

ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)
plt.show()


# Plots fiber-Coupled power vs Time 
fig, ax = plt.subplots(figsize=(7, 4.8))
for idx, file in enumerate(files[:7]): 
    ax.plot(times_us, P_fiber[idx, : ], "o-", label=labels[idx])

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"P_fiber $")
ax.set_title(title)
ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)
plt.show()


# Plots total emitted power vs time
fig, ax = plt.subplots(figsize=(7, 4.8))
for idx, file in enumerate(files[:7]): 
    ax.plot(times_us, P_total[idx, : ], "o-", label=labels[idx])

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"P_total  $")
ax.set_title(title)
ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)
plt.show()


# Plot fiber couploing efficiency vs Time eta= P_in / P_t 
fig, ax = plt.subplots(figsize=(7, 4.8))
for idx, file in enumerate(files[:7]): 
    ax.plot(times_us, eta_i[idx, : ], "o-", label=labels[idx])

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"P_fiber/P_total $")
ax.set_title(title)
ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)

# Compare normalize P_fiber and P_total for the first data set 
fig = plt.figure()
plt.plot(times_us, P_fiber[0,:] / P_fiber[0,0], label="P_fiber norm")
plt.plot(times_us, P_total[0,:] / P_total[0,0], label="P_total norm")
#plt.plot(times_us, I[0,:] / I[0,0], label="I(+z) norm")
plt.legend()
plt.grid(True)
plt.show()
plt.show()

