#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from radpattern.plotting import rplotting
from radpattern.geometry import grids
import matplotlib.pyplot as plt 
import numpy as np



PATH = "../data/results_sims/"
FILE = "N1000_mc1_nt100_k-111_3e9d9d4c"
print(f"showing file = {PATH+FILE}")

npz = np.load(PATH+FILE+'.npz', allow_pickle=True)
print(npz.files)

# Extreact data from file
pos  = npz['atom_pos']
w = npz['w']
I = npz['intensity']
meta = npz["metadata"].item()

print("Done") 
print(meta) 
nt = meta["sim"]["n_theta"]
np_ = meta["sim"]["n_phi"]
# crewates the grid 
grid = grids.AngleGrid(n_theta = nt, n_phi = np_, theta_max = np.pi) 


fig, ax = rplotting.plot_atoms(pos.reshape(-1, 3))

#fig, ax = rplotting.plot_pattern_3d(grid.nx, grid.ny, grid.nz, I ) 
plt.show() 
