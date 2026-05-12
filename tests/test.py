#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from radpattern.plotting import pattern_3d as prt
from radpattern.plotting import rplotting
from radpattern.plotting import plotanimation
from radpattern.helpers import helpers as hps
from radpattern.geometry import grids
from matplotlib import colors 

import matplotlib.pyplot as plt 
import numpy as np

#N1771561_mc1_nt1_k110_0976c3c8
#N1771561_mc1_nt1_k011_3596ce19.npz


PATH = "../data/results_sims/"
#test2DslabN100000_mc1_nt20_k001_a2cc34fd
FILE = "_simT50us_nt10_b9636ccc"
print(f"showing file = {PATH+FILE}")

npz = np.load(PATH+FILE+'.npz', allow_pickle=True)

# Extreact data from file
pos  = npz['atom_pos']
w = npz['w']
I = npz['intensity']
meta = npz["metadata"].item()


print("\n=== metadata ===")
for keys, values in meta.items():
    print(f"\n=== {keys} ===")
    for key, value in meta[keys].items():
        print(f"{key}: {value}")
print("========")

nt = meta["sim"]["n_theta"]
np_ = meta["sim"]["n_phi"]
# crewates the grid 
grid = grids.AngleGrid(n_theta = nt, n_phi = np_) 
#theta, phi, nx, ny, nz = hps.make_angle_grid(n_theta=nt, n_phi=np_)

rplotting.plot_atoms(pos, w=w)
plt.show()
#normalize relative to the global max over all entries
I_max = np.max(I)
print( f" Max intensity {I_max}") 

imax = np.unravel_index(np.argmax(I), I.shape)

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


K = np.round(meta['phys']['k_in_hat'])
osize = meta['regime']['optical_size']
ospacing = meta['regime']['optical_spacing']
N = meta['sim']['n_atoms']

K = np.round(meta['phys']['k_in_hat'], 3)

Lxy = meta['regime']['optical_size']
Lz  = meta['regime'].get('optical_size_z', Lxy)
spacing = meta['regime']['optical_spacing']
illum = meta['regime']['illumination_ratio']
fill  = meta['regime']['filling_factor']
N = meta['sim']['n_atoms']

# beam waist used by gaussian runs in your lambda=1 units
w0 = illum * Lxy

# key nondimensional ratios
aspect = Lz / Lxy
beam_to_box = w0 / Lxy
F = w0**2 / Lz if Lz != 0 else np.nan
sigma_long = fill * Lz
long_fill = sigma_long / Lz if Lz != 0 else np.nan

#info = (
#    f"file = {FILE}\n"
#    f"N = {N}, spacing/λ = {spacing:.2f}\n"
#    rf"I_max = {I_max:.3e}, $\hat n$_max = {I_max_vec}\n"
#    f"Lxy/λ = {Lxy:.2f}, Lz/λ = {Lz:.2f}\n"
#    f"Lz/Lxy = {aspect:.2f}\n"
#    f"w0/λ = {w0:.2f}, w0/Lxy = {beam_to_box:.3f}\n"
#    fr"F = $w0^2$/Lz = {F:.3f}\n"
#    fr"$\sigma$z/Lz = {long_fill:.2f}\n"
#    f"k_in = {K}"
#)


left = [
    f"Lxy/λ      = {Lxy:.2f}",
    f"Lz/λ       = {Lz:.2f}",
    f"spacing/λ  = {spacing:.2f}",
    f"file       = {FILE}",
]

right = [
    f"w0/λ       = {w0:.2f}",
    f"k_in       = {K}",
    f"N          = {N}",
    "",
]

info = "\n".join(
    f"{l:<22} {r}" for l, r in zip(left, right)
)

title = (
    rf"N={N}, k_in={K}, "
    rf"Lz/Lxy={aspect:.2f}, w0/Lxy={beam_to_box:.3f}, "
    rf"$\sigma$/Lz={long_fill:.2f}, F={F:.2f}"
)




#title=rf"atoms:{N}. Cube geometry array. k_in={K}, L/ $\lambda$: {osize}, a/$\lambda$: {ospacing}"

fig,ax = prt.plot_pattern_3d(grid,I,
                             title = title, 
                             stride = 1,
                             info_text=info)


plt.show()

