        #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import radpattern.physics.coupling as cp
from radpattern.physics.setup_params import ExperimentalParams, SimParams 
from radpattern.physics.beam import BeamModel 
from radpattern.geometry.cloud_model import CloudModel 
from radpattern.helpers.helpers import single_dipole_E
from radpattern.helpers.io import parse_run_filename


from radpattern.plotting.pattern_3d import plot_pattern_3d
from plotting_atomsSystem import plotting_cloud_from_json

import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

import numpy as np
import re 
from pathlib import Path


from radpattern.plotting import load_data

from debug_mcruns_plot import  coupling_from_AF2

import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

here = Path.cwd()
PATH = (Path.cwd() / ".." / "data" / "results_sims" ).resolve()
#PATH = (Path.cwd() / ".." / "data" / "test").resolve()

PATH = Path(input("folder path "))
log.info("Reading simulation results from: %s", PATH)

try:
        files = sorted(file.name for file in PATH.iterdir() if file.is_file() and file.suffix==".npz") 
        log.info("Found %d .npz files in folder", len(files))
        default_title = Path(files[0]).stem
        title = input(f"Plot title? [default: {default_title}]: ").strip() or default_title
        legend_title = input("legend title? (Default: TBD )").strip() or "TBD" 


except NotADirectoryError as e: 
        log.warning("Input path is not a directory, treating it as a single file: %s", PATH)
        log.debug("NotADirectoryError details", exc_info=e)

        files = [PATH.name]
        title = files[0]
        legend_title = "TBD" 
        PATH = PATH.parent
        log.info("Using parent folder: %s", PATH)
        print(e)

# Parse metadata from filenames
run_info = []
for file in files:
    try:
        info = parse_run_filename(file)
        run_info.append(info)

        log.debug("Parsed filename %s -> %s", file, info)

    except ValueError as e:
        log.warning("%s", e)
        raise

atoms_set = {info["atoms"] for info in run_info}
sim_time_set = {info["sim_time_us"] for info in run_info}
time_division_set = {info["time_divisions"] for info in run_info}
n_mc_set = {info["n_mc"] for info in run_info}
file_hashes = np.array([info["hash"] for info in run_info ])

# Checking all files are same system and sim runtime
if len(atoms_set) > 1:
    log.warning("Files contain different atoms: %s", sorted(atoms_set))

if len(sim_time_set) > 1:
    log.warning("Files contain different simulation times [us]: %s", sorted(sim_time_set))

if len(time_division_set) > 1:
    log.warning("Files contain different time divisions: %s", sorted(time_division_set))

if len(n_mc_set) > 1:
    log.info("Files contain different Monte Carlo run counts: %s", sorted(n_mc_set))

# Assigning automatic time scale to plot. 
if len(atoms_set) == 1:
    atom_name = next(iter(atoms_set)).lower()

    if atom_name == "cs133":
        timeScale = "us"
    elif atom_name == "rb87":
        timeScale = "ms"
    else:
        timeScale = "us"
        log.warning("Unknown atom type %s. Using time scale: us", atom_name)

else:
    timeScale = "us"
    log.warning("Mixed atom species found. Using time scale: us")

log.info("Using time scale: %s", timeScale)

max_time_raw = input(f"Max time to plot [{timeScale}]?. (Default: {max(sim_time_set)})").strip()

# To match from file name for labels 
regex_pattern =r'_(\d+)ControlBeamfactor'

#Plot labels 
#labels = np.zeros(len(files))
labels = [None] * len(files)
beamRatios = np.zeros(len(files)) # Control/signal ratio 

# Allocating arrays from filenameMetadata
Time_division = max(info["time_divisions"] for info in run_info)
log.info("Allocating arrays with Time_division = %d", Time_division)

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
 article   I_t: shape (T, ntheta, nphi), already |AF|^2 * dipole
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


def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def fit_exp_decay(t, y):
    """
    Fit y = A exp(-t/tau).

    Returns
    -------
    A_fit, tau_fit, corr
    """
    valid = np.isfinite(t) & np.isfinite(y) & (y > 0)

    if np.sum(valid) < 3:
        raise RuntimeError("Not enough valid positive points for exponential fit")

    t_fit_data = t[valid]
    y_fit_data = y[valid]

    p0 = [y_fit_data[0], (t_fit_data[-1] - t_fit_data[0]) / 2]

    popt, _ = curve_fit(
        exp_decay,
        t_fit_data,
        y_fit_data,
        p0=p0,
        maxfev=10000,
    )

    A_fit, tau_fit = popt

    y_model = exp_decay(t_fit_data, A_fit, tau_fit)

    if np.std(y_fit_data) == 0 or np.std(y_model) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(y_fit_data, y_model)[0, 1]

    return A_fit, tau_fit, corr

def fit_exp_decay_loglinear(t, y):
    """
    Fit y = A exp(-t/tau) using log-linear regression.
    Returns
    -------
    A_fit : float
    tau_fit : float
    corr : float
        Pearson correlation between y and fitted y.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y) & (y > 0)

    if np.sum(valid) < 3:
        raise RuntimeError("Not enough valid positive points for exponential fit")

    t_valid = t[valid]
    y_valid = y[valid]

    # Shift time for numerical stability.
    # This does not change tau.
    t0 = t_valid[0]
    t_shift = t_valid - t0

    log_y = np.log(y_valid)

    # log(y) = log(A0) - t_shift/tau
    slope, intercept = np.polyfit(t_shift, log_y, 1)

    if slope >= 0:
        raise RuntimeError(
            f"Fit slope is positive: slope={slope:.6g}. Data is not decaying in selected window."
        )

    tau_fit = -1.0 / slope
    A0_fit = np.exp(intercept)

    y_model = A0_fit * np.exp(-t_shift / tau_fit)

    if np.std(y_valid) == 0 or np.std(y_model) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(y_valid, y_model)[0, 1]

    return A0_fit, tau_fit, corr, t0

### Data extraction and formation of new objects to get the properties values. 
for file_idx, file in enumerate(files): 
    
    log.info("Processing file %d/%d: %s", file_idx + 1, len(files), file)
    data, grid, exp, sim  = load_data(PATH/file)

    log.debug("Loaded keys from npz file: %s", list(data.keys()))
    log.debug("AF2 shape: %s", data["AF2"].shape)
    log.debug("Intensity shape: %s", data["intensity"].shape)

    AF = np.abs(data["AF2"])
    Intensity =data["intensity"]
    times_us = data["times_us"]

    log.debug("Time array shape: %s", times_us.shape)
    log.debug("Experiment label: %s", exp.label)

    ### Calculating Gaussian mode. 
    ### Coupling to gaussian mode calculation.
    theta0 = 12 / (exp.atom.k_signal * exp.w0_signal)
    E_fib = np.abs(cp.gaussian_fiber_mode_on_sphere(grid, theta0)) ** 2
    log.info(
        "theta0 = %.6e rad, forward lobe = %.6e rad, match = %s",
        theta0,
        exp.forwardlobe_angular_width,
        np.isclose(theta0, exp.forwardlobe_angular_width),
    )

    log.debug("Building single-dipole radiation pattern")
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

    log.info("Computing fiber coupling for file: %s", file)
    P_fib, P_tot, eta_t = coupling_from_AF2(
            AF2_t=AF,
            grid=grid,
            dipole=dipole,
            E_fib=E_fib,
            theta0=theta0,
            )

    if np.any(P_tot <= 0):
        log.warning("Some total-power values are zero or negative in file: %s", file)

    P_fib_over_Ptot0_t = P_fib / (P_tot[0] + 1e-30)



    etas[file_idx, : ] = eta_t
    P_fiber[file_idx,:] = P_fib
    P_total[file_idx,:] = P_tot
    P_OverTotal0[file_idx,:] = P_fib_over_Ptot0_t

    _label= exp.label 
    match = re.search(re.compile(r"angle\s*=\s*([\d.]+)"), _label)
    if match is None:
        log.warning("Could not extract label")
        labels[file_idx] = np.nan
    else:
        labels[file_idx] = float(match.group(1))


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

if np.any(np.isnan(labels)):
    log.warning("Some labels are NaN. Sorting and plotting may be incorrect.")
else:
    log.info("Sorting files by labels")

# Max time to plot
if max_time_raw == "":
     idx_max = len(times_us)
     max_time_plot = times_us[-1]

else:
        max_time_plot = float(max_time_raw)
        max_time_plot = min(max_time_plot, times_us[-1])
        log.info("max time plot: %s", max_time_plot)
        #idx_max = np.searchsorted(sorted(times_us), max_time_raw, side="right")
        idx_max = np.abs(times_us - max_time_plot).argmin() +1 

log.info("plotting up to %.6g %s using %d/%d time points", max_time_plot, timeScale, idx_max, len(times_us))


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
file_hashes = file_hashes[arg_sorted]

##### coupling / dephasing plot ---
log.info("Creating coupling vs time plot")
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
log.info("Creating normalized emitted power plot")
fig, ax = plt.subplots(figsize=(7, 4.8))

tau_fits = np.full(len(sorted_files), np.nan)
corr_fits = np.full(len(sorted_files), np.nan)
A_fits = np.full(len(sorted_files), np.nan)

for idx, file in enumerate(sorted_files[:7]): 
    
    x = times_us[:idx_max]
    y = P_OverTotal0[idx, :idx_max]

    ax.plot(x,y, "o-", label=labels[idx])
    try:
        A_fit, tau_fit, corr, t0_fit = fit_exp_decay_loglinear(x, y)

        A_fits[idx] = A_fit
        tau_fits[idx] = tau_fit
        corr_fits[idx] = corr

        x_fit = np.linspace(x[0], x[-1], 300)
        y_fit = A_fit * np.exp(-(x_fit - t0_fit) / tau_fit)

        ax.plot(
            x_fit,
            y_fit,
            "--",
            color="red",
            alpha=0.8,
            linewidth=1.5,
        )

        log.info(
            "FIT | hash=%s | label=%s | file=%s | tau=%.6g %s | corr=%.6f | A0=%.6g | t0=%.6g %s",
            file_hashes[idx],
            labels[idx],
            file,
            tau_fit,
            timeScale,
            corr,
            A_fit,
            t0_fit,
            timeScale,
        )

        print(
            f"FIT | hash={file_hashes[idx]} | label={labels[idx]} | "
            f"tau={tau_fit:.6g} {timeScale} | corr={corr:.6f}"
        )

    except RuntimeError as e:
        log.warning(
            "FIT FAILED | hash=%s | label=%s | file=%s | reason=%s",
            file_hashes[idx],
            labels[idx],
            file,
            e,
        )

ax.set_xlabel(f"time [{timeScale}]")
ax.set_ylabel(r"Coupling $\eta$")
ax.set_title(title)
ax.legend(title = legend_title)
ax.grid(True, alpha=0.25)


# Plotting the emission pattern.
log.info("Plotting 3D emission pattern for last loaded file: %s", sorted_files[-1])
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
