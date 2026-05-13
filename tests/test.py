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


import coupling_calcualtion as cp

import matplotlib.pyplot as plt 
import numpy as np

PATH = "../data/results_sims/"
#PATH = ""
#test2DslabN100000_mc1_nt20_k001_a2cc34fd
FILE = "exp_data_s_wave_reduce_cone_testrho10000000000000_mc1_nt100_k001_c1cafbca"
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



def azimuthal_average_vs_theta(I, grid, normalize=True):
    """
    Average intensity over phi for each theta.
    """
    I = np.asarray(I, dtype=float)

    if I.shape != grid.shape:
        raise ValueError(f"I must have shape {grid.shape}, got {I.shape}")

    # Simple average over phi.
    I_theta = np.mean(I, axis=1)

    if normalize and np.max(I_theta) > 0:
        I_theta = I_theta / np.max(I_theta)

    return grid.theta, I_theta
def report_theta_peak(theta, I_theta):
    """
    Print the peak direction of the azimuthally averaged curve.
    """
    i_max = int(np.argmax(I_theta))
    th_max = theta[i_max]

    print("\n=== Azimuthally averaged pattern ===")
    print(f"Peak theta index = {i_max}")
    print(f"Peak theta [rad] = {th_max:.6f}")
    print(f"Peak theta [deg] = {np.rad2deg(th_max):.6f}")
    print(f"Peak averaged intensity = {I_theta[i_max]:.6g}")


theta, I_theta = azimuthal_average_vs_theta(I, grid, normalize=True)
report_theta_peak(theta, I_theta)


print(f"Max intensity = {I_max}")
print(f"Grid index of max = {imax}")

print("Direction of max:")
print(f"nx = {grid.nx[imax]}")
print(f"ny = {grid.ny[imax]}")
print(f"nz = {grid.nz[imax]}")
I_max_vec = np.round([grid.nx[imax], grid.ny[imax], grid.nz[imax]])
print(f"normalize max intensity vector = {I_max_vec}")




# normalize each frame by its own max
frame_max = np.max(I, axis=tuple(range(1, I.ndim)), keepdims=True)
frame_max[frame_max == 0] = 1.0
I = I / frame_max

#k_inhat =np.round( meta.get('k_in_hat', 'missing'), 3)


# Report of the simulation. 
try: 
    exp = ExperimentalParams(
            atoms = meta['regime']['atoms'], 
            lambda_control_m = meta['regime']['lambda_control_m'],
            delta_f_hz = meta['regime']['delta_f_hz'], 
            cell_length_m = meta['regime']['cell_length_m'], 
            cell_diameter_m = meta['regime']['cell_diameter_m'], 
            signal_fwhm_diameter_m = meta['regime']['signal_fwhm_diameter_m'], 
            control_fwhm_diameter_m = meta['regime']['control_fwhm_diameter_m'], 
            density = meta['regime']['density'], 
            scalling = meta['regime']['scalling']
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


#### Coupling to gaussian mode calculation.
try: 
    E_field = AF
    #theta0 = (2* np.sqrt(29)) / (5 * exp.atom.k_signal * exp.w0_signal)
    theta0 = (2) / ( exp.atom.k_signal * exp.w0_signal)
    E_fib = cp.gaussian_fiber_mode_on_sphere(grid, theta0)#* np.exp(1j * np.angle(AF))
    # overlap
    eta, amp = cp.overlap_on_sphere(grid, E_field, E_fib)
    print(f"eta = {eta}, amp ={amp}")

    eta_test = np.sum(np.abs(AF)**2 * np.abs(E_fib)**2) / np.sum(np.abs(AF)**2)
    print("eta just amplitude ", eta_test)

except KeyError: 
    eta = np.nan
    amp = np.nan

print("coupling eta =", eta)
print("overlap amplitude =", amp)


try: 
    left = [
        f"Lxy/{exp.scalling}λ      = {Lxy:.2f}",
        f"Lz/{exp.scalling}λ       = {Lz:.2f}",
        f"spacing/{exp.scalling}λ  = {spacing:.2f}",
        fr"coupling $\eta$ = {eta:.3f}", 
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

fig,ax = plot_pattern_3d(grid,I,
                         title = title, 
                         stride = 1,
                         info_text=info,
                         log_plot = False,
                         sphere_map = True, 
                         )


plt.show()

