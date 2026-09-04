#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from radpattern.plotting.pattern_3d import plot_pattern_3d
from radpattern.plotting import rplotting
from radpattern.plotting.beam_test import plot_atom_distribution, plot_weight_distribution
from radpattern.plotting import plotanimation
from radpattern.helpers import helpers as hps
from radpattern.geometry import grids
from matplotlib import colors 


from radpattern.physics.setup_params import ExperimentalParams 
from radpattern.physics.beam import BeamModel 
from radpattern.geometry.cloud_model import CloudModel 

from scipy.optimize import curve_fit

import coupling_calcualtion as cp

import matplotlib.pyplot as plt 
import numpy as np


PATH = "../data/results_sims/"
#PATH = ""
#test2DslabN100000_mc1_nt20_k001_a2cc34fd
        #DiffusiveBufferN2_10Torr_simDens0.0e6_ExpData_Swave_reduce_cone_50ustimeSim_GeomSpace_simT50us_nt11_717e6837
FILE = "ColdAtomRB87_simDens0e6_ExpData_Swave_reduce_cone_200000ustimeSim_GeomSpace__simT200000us_nt15_7c957ee9"
print(f"showing file = {PATH+FILE}")

npz = np.load(PATH+FILE+'.npz', allow_pickle=True)

# Extreact data from file
pos  = npz['atom_pos']
w = npz['w']
I = npz['intensity']
meta = npz["metadata"].item()
try: 
    AF = npz['AF']
except KeyError:
    print("no AF found") 

try: 
    times_code = npz["times_code"]
except KeyError: 
    print("No times_code provided")


print("\n=== metadata ===")
for keys, values in meta.items():
    print(f"\n=== {keys} ===")
    for key, value in meta[keys].items():
        print(f"{key}: {value}")
print("========")

nt = meta["sim"]["n_theta"]
np_ = meta["sim"]["n_phi"]

# crewates the grid 
try: 
    grid = grids.AngleGrid(n_theta = nt, n_phi = np_, theta_max = meta["sim"]["theta_max"]) 
except KeyError: 
    grid = grids.AngleGrid(n_theta = nt, n_phi = np_, theta_max = np.pi) 


# Figures from atoms and weight distribution. 

fig, ax = rplotting.plot_atoms(pos, w=w)
fig2, ax2 = plot_atom_distribution(pos) 
fig3, ax3 = plot_weight_distribution(pos, w)
plt.show()

#normalize relative to the global max over all entries
I_max = np.max(I)
print( f" Max intensity {I_max}") 

imax = np.unravel_index(np.argmax(I), I.shape)

## Report of the simulation. 
try: 
    exp = ExperimentalParams(
            atoms = meta['regime']['atoms'], 
            lambda_control_m = meta['regime']['lambda_control_m'],
            delta_f_hz = meta['regime']['delta_f_hz'], 
            cell_length_m = meta['regime']['cell_length_m'], 
            cell_diameter_m = meta['regime']['cell_diameter_m'], 
            temperature = meta['regime']['temperature'],
            signal_fwhm_diameter_m = meta['regime']['signal_fwhm_diameter_m'], 
            control_fwhm_diameter_m = meta['regime']['control_fwhm_diameter_m'], 
            density_cm3 = meta['regime']['density_cm3'], 
            scalling = meta['regime']['scalling'],
            buffer_pressure_Torr = meta['regime']['buffer_pressure_Torr'],
            label = meta['regime']['label']
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

    cloud.log_info()

    # convert code time -> SI -> microseconds
    char_time = exp.char_time          # [s] = ref_length / ref_velocity
    times_si = times_code * char_time  # [s]
    times_us = times_si * 1e6          # [µs]
    print(f"CHAR_TIMEm{char_time}")
except KeyError: 
    print(" unable to generate objet from metadata... keyerror") 


try: 
    K = np.round(meta['beam']['k_in_hat'], 3)
    aspect = meta['regime']['aspect_ratio']
    Lz  = meta['regime'].get('optical_size_z')
    spacing = meta['regime']['optical_spacing']
    illum = meta['regime']['illumination_ratio']
    fill  = meta['regime']['filling_factor']
    rho = 1 / ( spacing **3)
    rho = round(rho) 

    Lxy = Lz / aspect

    # beam waist used by gaussian runs in your lambda=1 units
    w0 = illum * Lxy

    # key nondimensional ratios
    beam_to_box = w0 / Lxy
    # Fresnel number.
    F = w0**2 / Lz if Lz != 0 else np.nan
    sigma_long = fill * Lz
    long_fill = sigma_long / Lz if Lz != 0 else np.nan
except KeyError:
    pass 
try: 
    K = exp.atom.k_control
    aspect = exp.Lz / (3 * exp.w0_control)
    Lz  = exp.Lz
    spacing = exp.a_spacing_reescaled
    illum = 3   
    fill  = 2
    rho = 1 / ( spacing **3)
    rho = round(rho) 

    Lxy = Lz / aspect

    # beam waist used by gaussian runs in your lambda=1 units
    w0 = exp.w0_control

    # key nondimensional ratios
    beam_to_box = w0 / Lxy
    # Fresnel number.
    F = w0**2 / Lz if Lz != 0 else np.nan
    sigma_long = fill * Lz
    long_fill = sigma_long / Lz if Lz != 0 else np.nan
except KeyError:
    print("Could not fill variables") 


####################
######### Speed distribution #########
try: 
    v = npz["speed_distribution"]
    speed = np.linalg.norm(v, axis=1) * exp.probable_speed

    fig, ax = plt.subplots(figsize=(7, 4.8))

    ax.hist(speed, bins=80, density=True, alpha=0.85)

    ax.set_xlabel(r"Speed $|\mathbf{v}|$ [m/s]")
    ax.set_ylabel(r"Probability density")
    ax.set_title("Atom speed distribution")

    mean_v = np.mean(speed)* exp.probable_speed
    std_v = np.std(speed)* exp.probable_speed
    median_v = np.median(speed)* exp.probable_speed
    vmax = np.max(speed)* exp.probable_speed

    info = (
        f"file: {FILE}\n"
        f"N = {speed.size:,}\n"
        rf"$\langle |v| \rangle$ = {mean_v:.3g} m/s" "\n"
        rf"$\sigma_{{|v|}}$ = {std_v:.3g} m/s" "\n"
        rf"median = {median_v:.3g} m/s" "\n"
        rf"max = {vmax:.3g} m/s"
    )

    ax.text(
        0.92, 0.92, info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.grid(True, alpha=0.25)
    fig.tight_layout()

except KeyError:
    print("No speed array found")


##############################
### Coupling to gaussian mode calculation.
theta0 = (2) / ( exp.atom.k_signal * exp.w0_signal)
print(f"theta0 = {theta0}, forwardLobe = {exp.forwardlobe_angular_width}, equal? {theta0 == exp.forwardlobe_angular_width}") 

E_fib = cp.gaussian_fiber_mode_on_sphere(grid, theta0)#* np.exp(1j * np.angle(AF))

eta_t = np.zeros(AF.shape[0])
eta_abs_t =np.zeros(AF.shape[0])
phase_std_t = np.zeros(AF.shape[0])


for it in range(AF.shape[0]):
    try: 
        E_field = AF[it]
        #theta0 = (2* np.sqrt(29)) / (5 * exp.atom.k_signal * exp.w0_signal)
                # overlap
        eta, amp = cp.overlap_on_sphere(grid, E_field, E_fib)
        print(f"eta = {eta}, amp ={amp}")

        eta_test = np.sum(np.abs(AF[it])**2 * np.abs(E_fib)**2) / np.sum(np.abs(AF[it])**2)
        print("eta_test ", eta_test)
#        phase = np.angle(AF[it])

        I = np.abs(AF[it])**2
        mask = I > 0.01* I.max()
#
        phase = np.unwrap(np.angle(AF[it][mask]))
        print("AF phase std =", np.std(phase))
#
        E_fib = cp.gaussian_fiber_mode_on_sphere(grid, theta0)
#
        eta_coh, _ = cp.overlap_on_sphere(grid, AF[it], E_fib)
        eta_abs, _ = cp.overlap_on_sphere(grid, np.abs(AF[it]), np.abs(E_fib))
#
        print("coherent eta =", eta_coh)
        print("phase-erased eta =", eta_abs)
#
##        I = np.abs(AF[it])**2
##        I /= I.max()

        eta_t[it] = eta
        eta_abs_t[it] = eta_abs
        phase_std_t[it] = np.std(np.unwrap(np.angle(AF[it][mask])))

    except KeyError: 
        eta = np.nan
        amp = np.nan

### 
## fitting of coupling. 
def exp_decay(t, A, tau, C):
    return A * np.exp(-t/ tau) + C

# fits
#p0 = [0.9, 2.0, 0.1]
#popt_erased, pcov_erased = curve_fit(exp_decay, times_us, eta_abs_t, p0=p0, maxfev=10000)
#A_e, tau_e, C_e = popt_erased
## smooth curves
##t_fit = np.linspace(times_us.min(), times_us.max(), 400)

########################################
##### coupling / dephasing plot ---
fig, ax = plt.subplots(figsize=(7, 4.8))

ax.plot(times_us, eta_t, "o-", label="coherent")
ax.plot(times_us, eta_abs_t, "o-", label="phase-erased")
#ax.plot(t_fit, exp_decay(t_fit, *popt_erased), ":",color="k", label="phase-erased fit")

ax.set_xlabel("time [$\mu$s]")
ax.set_ylabel(r"coupling $\eta$")
ax.set_title("Coupling decay: coherent vs phase-erased")

info = (
    f"file: {FILE}\n"
)

text = (
    r"Fit: $\eta(t)=A e^{-t/\tau}+C$" "\n"
    #rf"phase-erased: $A={A_e:.3f}$, $\tau={tau_e:.2f}\,\mu s$, $C={C_e:.3f}$"
)

plt.text(
    0.03, 0.05, text,
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
)
ax.text(
    0.93, 0.03, info,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

ax.legend()
ax.grid(True, alpha=0.25)
plt.show()

# --- atoms inside volume ---
try:
    n_inside = npz["n_inside"]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(times_us, n_inside, "o-", label="atoms inside volume")

    ax.set_xlabel(r"time [$\mu$s]")
    ax.set_ylabel("number of atoms")
    ax.set_title("Atoms remaining inside simulation volume")

    info = (
        f"file: {FILE}\n"
        f"N0 = {n_inside[0]:,}\n"
        f"Nfinal = {n_inside[-1]:,}\n"
        f"fraction final = {n_inside[-1] / n_inside[0]:.3f}"
    )

    ax.text(
        0.98, 0.98, info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()

except KeyError:
    print("No n_inside...")


# --- atoms inside beam volume ---
try:
    n_beam = npz["n_beam"]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(times_us, n_beam, "o-", label=r"atoms inside beam volume at $2\sigma$")

    ax.set_xlabel(r"time [$\mu$s]")
    ax.set_ylabel("number of atoms")
    ax.set_title(r"Atoms inside beam volume at $2\sigma$")

    info = (
        f"file: {FILE}\n"
        f"N0 = {n_beam[0]:,}\n"
        f"Nfinal = {n_beam[-1]:,}\n"
        f"fraction final = {n_beam[-1] / n_beam[0]:.3f}"
    )

    ax.text(
        0.98, 0.98, info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()

except KeyError:
    print("No n_beam...")

print("coupling eta =", eta)
print("overlap amplitude =", amp)

try: 
    left = [
        f"Lxy/{exp.scalling}λ      = {Lxy:.2f}",
        f"Lz/{exp.scalling}λ       = {Lz:.2f}",
        f"spacing/{exp.scalling}λ  = {spacing:.2f}",
        fr"coupling $\eta$ = {eta_t[0]:.3f}", 
        f"file       = {FILE}",
    ]

    right = [
        f"w0/{exp.scalling}λ       = {w0:.2f}",
        f"k_in       = {K}",
        f"rho          = {rho}",
        "",
        "",
    ]

    info = "\n".join(
        f"{l:<22} {r}" for l, r in zip(left, right)
    )

    title = (
        rf"rho={rho} $\lambda$^-3, k_in={K}, "
        rf"Lz/Lxy={aspect:.2f}, w0/Lxy={beam_to_box:.3f}, "
        rf"$\sigma$/Lz={long_fill:.2f}, F={F:.2f}"
    )


except NameError: 
    title = "Intentisity. No metadata retrieve" 
    info = " No metada retrieved"

#title=rf"atoms:{N}. Cube geometry array. k_in={K}, L/ $\lambda$: {osize}, a/$\lambda$: {ospacing}"

I = np.abs(AF[0])**2
fig,ax = plot_pattern_3d(grid,I,
                         title = title, 
                         stride = 1,
                         info_text=info,
                         log_plot = False,
                         sphere_map = True, 
                         )


plt.show()

