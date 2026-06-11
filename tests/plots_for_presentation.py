        #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import radpattern.physics.coupling as cp
from radpattern.physics.setup_params import ExperimentalParams, SimParams 
from radpattern.physics.beam import BeamModel 
from radpattern.geometry.cloud_model import CloudModel 
from radpattern.helpers.helpers import single_dipole_E

from radpattern.plotting.pattern_3d import plot_pattern_3d
from plotting_atomsSystem import plotting_cloud_from_json
import matplotlib.pyplot as plt 

import numpy as np
import re 
from pathlib import Path


from radpattern.plotting import load_data

from debug_mcruns_plot import  coupling_from_AF2

here = Path.cwd()
PATH = (Path.cwd() / ".." / "data" / "results_sims" ).resolve()
#PATH = (Path.cwd() / ".." / "data" / "test").resolve()

PATH = Path(input("folder path "))

try:
        files = [file.name for file in PATH.iterdir() if file.is_file() and file.suffix==".npz"]
except NotADirectoryError as e: 
        print(e)
        files =[PATH.name]
        PATH = PATH.parent

#
title = "Rb87 changing Cbeam Angle "
# plot title and legend title
#legend_title = r"Diameter [$\mu$m]" 
legend_title = r"Cbeam angle [mrad]"
timeScale = "ms" 
#timeScale = r"$\mu$s" 

# To match from file name for labels 
regex_pattern =r'_(\d+)ControlBeamfactor'

#Plot labels 
#labels = np.zeros(len(files))
labels = [None] * len(files)
beamRatios = np.zeros(len(files)) # Control/signal ratio 


Time_division = 100

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
    AF = np.abs(data["AF2"])
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

    _label= exp.label 
    match = re.search(re.compile(r"angle\s*=\s*([\d.]+)"), _label)
    labels[file_idx] = float(match.group(1))

#    labels[file_idx] = exp.B_gradient
        
#    match = re.search(regex_pattern, str(file))
#
#    if match:
#        value = match.group(1)      # string, e.g. "120"
#        labels[file_idx] = value 
#    else:
#        labels[file_idx] = None
#
#
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

arg_sorted = np.argsort(labels)


# plot title and legend title
# To match from file name for labels 
#regex_pattern =r'_(\d+)runs#'

#Plot labels 
#labels = [2,5,7,10,12] 
beamRatios = np.zeros(len(files)) # Control/signal ratio 

#labels = [0, 0.1, 1, 10]
#times_us= times_us * 1e-3
sorted_files = np.array(files)[arg_sorted]
labels = np.array(labels)[arg_sorted] 
etas = etas[arg_sorted]
P_OverTotal0 = P_OverTotal0[arg_sorted]

##### coupling / dephasing plot ---
fig, ax = plt.subplots(figsize=(7, 4.8))

for idx, file in enumerate(sorted_files[:7]): 
    ax.plot(times_us, etas[idx, :], "o-", label=labels[idx])
for idx, file in enumerate(sorted_files[7:],7): 
    ax.plot(times_us[20], etas[idx, :20 ], "*--", label=labels[idx])

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"coupling $\eta$")
ax.set_title(title)

ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)

print(beamRatios)

# Plots total emitted power vs time
fig, ax = plt.subplots(figsize=(7, 4.8))
for idx, file in enumerate(sorted_files[:7]): 
    ax.plot(times_us[:20], P_OverTotal0[idx,:20 ], "o-", label=labels[idx])

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"Coupling $\eta$")
ax.set_title(title)
ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)


# Plotting the emission pattern.
plot_pattern_3d(grid, data["intensity"][0], title= exp.label )


#
## Plotting one of the Cloud setups
### Searching for the .json file
#try:
#        json_file= None
#        for file in PATH.iterdir():
#            if file.is_file() and file.suffix == ".json":
#                json_file = file 
#        if json_file is not None: 
#            fig1, ax1, fig2, ax2 = plotting_cloud_from_json(file)
#except NotADirectoryError as e: 
#        print(" No json file found.") 
#
#
#
plt.show()
