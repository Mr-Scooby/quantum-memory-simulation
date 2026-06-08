#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import radpattern.physics.coupling as cp
from radpattern.physics.setup_params import ExperimentalParams, SimParams 
from radpattern.physics.beam import BeamModel 
from radpattern.geometry.cloud_model import CloudModel 
from radpattern.helpers.helpers import single_dipole_E
import matplotlib.pyplot as plt 
import numpy as np
import re 
from pathlib import Path

from radpattern.plotting import load_data

from debug_mcruns_plot import  coupling_from_AF2

here = Path.cwd()
PATH = (Path.cwd() / ".." / "data" / "results_sims" ).resolve()
#PATH = (Path.cwd() / ".." / "data" / "test").resolve()

files = [
         "DiffusiveBufferN2_2Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_c2e6fcd3",
         #"DiffusiveBufferN2_3Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_260f4b5d.npz",
         #"DiffusiveBufferN2_4Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_88032039.npz",
#         "DiffusiveBufferN2_5Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_1a93c62c",
#         #"DiffusiveBufferN2_6Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_14e191de.npz",
#         "DiffusiveBufferN2_7Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_7c19ae1e",
#         #"DiffusiveBufferN2_8Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_3c2c60c5.npz",
#         #"DiffusiveBufferN2_9Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_1c3f4742.npz",
#         "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_582bdb8b",
#         #"DiffusiveBufferN2_11Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_6e487271.npz",
#         "DiffusiveBufferN2_12Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_286e4962",
         ]
#
##files = [
##        "DiffusiveBufferN2_1Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_b943d55c.npz",
##        "DiffusiveBufferN2_2Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_4f4c8533.npz",
##        "DiffusiveBufferN2_3Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_2dfc3fc2.npz",
##        "DiffusiveBufferN2_4Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_a50b2201.npz",
##        "DiffusiveBufferN2_5Torr_simDens1e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_NormalizeWeights_simT50us_nt16_6c059ef2.npz",
##        ]
##
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
#files = [
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_120Sdiamter_simT50us_nt16_f1118cef",
##"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_150Sdiamter_simT50us_nt16_72859d29",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_180Sdiamter_simT50us_nt16_deede175",
##"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_210Sdiamter_simT50us_nt16_db316c0c",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_240Sdiamter_simT50us_nt16_30183bee",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_270Sdiamter_simT50us_nt16_6c0a1957",
#]
#
### Beam Ratios change 
#files =[
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_1ratioCS_simT50us_nt16_f1118cef",
#        #"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_1.3ratioCS_simT50us_nt16_5fa362c1",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_1.5ratioCS_simT50us_nt16_0a558c77",
#        #"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_1.7ratioCS_simT50us_nt16_ae75dd25",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_2ratioCS_simT50us_nt16_90252e96",
#        #"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_2.3ratioCS_simT50us_nt16_3e6d918a",
#        "DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_2.5ratioCS_simT50us_nt16_bcc75753",
#]
##
##
### Cs133 Sim Changing Control beam. 
#files =[
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_300Cdiam_fixedR_simT50us_nt16_1636c86f",
##"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_400Cdiam_fixedR_simT50us_nt16_50205c0c",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_500Cdiam_fixedR_simT50us_nt16_fceee68b",
##"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_600Cdiam_fixedR_simT50us_nt16_a30e91bb",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_700Cdiam_fixedR_simT50us_nt16_29da55fe",
##"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_800Cdiam_fixedR_simT50us_nt16_f01e48b9",
#"DiffusiveBufferN2_10Torr_simDens0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_900Cdiam_fixedR_simT50us_nt16_537225a9",
#]
#
# RB87 sim with changing control beam. The signal is 100 um
#files = [
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_0.7ControlBeamfactor_simT200000us_nt15_0bd1783f",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_0.9ControlBeamfactor_simT200000us_nt15_4596d951",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1ControlBeamfactor_simT200000us_nt15_05856623",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.2ControlBeamfactor_simT200000us_nt15_8fec7ff7",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.5ControlBeamfactor_simT200000us_nt15_ef27abec",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.7ControlBeamfactor_simT200000us_nt15_c51d58be",
#"ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_2ControlBeamfactor_simT200000us_nt15_78ade96a",
#         ]
#
#files = ["ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace_1.5ControlBeamfactor_simT200000us_nt15_ef27abec"]


# RB87 Sim with changing control beam.  A  signal beam of 17 um
#files = [
#"Cs133_10Torr120SDia_300Cdia_simT100us_nt30_100runs_e418e57e"
#]

#
#files =[ 
#"Cs133_2Torr120SDia_300Cdia_simT50us_nt16_50runs_ad9b5931",
#"Cs133_2Torr120SDia_300Cdia_simT50us_nt16_10runs_468ef87d",
#"Cs133_2Torr120SDia_300Cdia_simT50us_nt16_1runs_9b0fea05",
#"Cs133_2Torr120SDia_300Cdia_simT50us_nt16_100runs_942014b1"
#]

#files = [
#"Cs133_10Torr120SDia_300Cdia_simT75.0us_nt100_100runs_759fd982.npz ",
#"Cs133_20Torr120SDia_300Cdia_simT75.0us_nt100_20runs_283b5245.npz",
#"Cs133_0Torr120SDia_300Cdia_simT75.0us_nt100_100runs_8400b548.npz",
#"Cs133_5Torr120SDia_300Cdia_simT75.0us_nt100_20runs_fee5a36e.npz",
#]



#
title = "Cs133. N2 @ 10 Torr. Varying Control diameter" 
# plot title and legend title
legend_title = r"Diameter [$\mu$m]" 
timeScale = "us" 
timeScale = r"$\mu$s" 
# To match from file name for labels 
regex_pattern =r'_(\d+)ControlBeamfactor'

#Plot labels 
#labels = np.zeros(len(files))
labels = [None] * len(files)
beamRatios = np.zeros(len(files)) # Control/signal ratio 


Time_division = 16

etas = np.zeros((len(files), Time_division)) 

P_fiber = np.zeros((len(files), Time_division)) 
P_total = np.zeros((len(files), Time_division)) 
P_OverTotal0 = np.zeros((len(files), Time_division)) 
eta_i = np.zeros((len(files), Time_division)) 
I = np.zeros((len(files), Time_division))

Diffusion_cte = np.zeros(len(files))
seed = np.zeros(len(files))


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
    
    data, grid, exp, sim  = load_data(PATH/file)
    AF = np.abs(data["AF"])**2
    #print(data.files) 
    Intensity =data["intensity"]

    #if "times_code" in data:
    #    times_us = data["times_code"] * sim.char_time * 1e6
    #else:
    #    parent = np.load(parent_npz_path, allow_pickle=True)
    #    times_us = parent["times_code"] * sim.char_time * 1e6
    times_us = data["times_us"]

    ### Calculating Gaussian mode. 
    ### Coupling to gaussian mode calculation.
    theta0 = 12 / (exp.atom.k_signal * exp.w0_signal)
    E_fib = np.abs(cp.gaussian_fiber_mode_on_sphere(grid, theta0)) ** 2
    print(f"theta0 = {theta0}, forwardLobe = {exp.forwardlobe_angular_width}, equal? {theta0 == exp.forwardlobe_angular_width}") 
    dipole = single_dipole_E(
            grid.nx,
            grid.ny,
            grid.nz,
            np.array([1.0, 0.0, 0.0]),
        )
    eta_t = np.zeros(AF.shape[0])
    eta_abs_t =np.zeros(AF.shape[0])

    print(f"Shape Inetensity {Intensity.shape}")
    I_t = np.zeros(AF.shape[0])
    eta_i_   =np.zeros(AF.shape[0]) 
    P_fiber_ =np.zeros(AF.shape[0]) 
    P_total_ =np.zeros(AF.shape[0]) 

    P_fib, P_tot, eta_t = coupling_from_AF2(
            AF2_t=AF,
            grid=grid,
            dipole=dipole,
            E_fib=E_fib,
            theta0=theta0,
            )

    P_fib_over_Ptot0_t = P_fib / (P_tot[0] + 1e-30)



   # eta_abs_t
    etas[file_idx, : ] = eta_t
    #I[file_idx,:] = I_t 
    #eta_i[file_idx,:] = eta_i_
    P_fiber[file_idx,:] = P_fib
    P_total[file_idx,:] = P_tot
    P_OverTotal0[file_idx,:] = P_fib_over_Ptot0_t

    match = re.search(regex_pattern, str(file))

    if match:
        value = match.group(1)      # string, e.g. "120"
        labels[file_idx] = value 
    else:
        labels[file_idx] = None


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

# plot title and legend title
# To match from file name for labels 
#regex_pattern =r'_(\d+)runs#'

#Plot labels 
#labels = [2,5,7,10,12] 
beamRatios = np.zeros(len(files)) # Control/signal ratio 

#labels = [0, 0.1, 1, 10]


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

# Plots total emitted power vs time
fig, ax = plt.subplots(figsize=(7, 4.8))
for idx, file in enumerate(files[:7]): 
    ax.plot(times_us[:14], P_OverTotal0[idx,:14 ], "o-", label=labels[idx])

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"Coupling $\eta$")
ax.set_title(title)
ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)
plt.show()
