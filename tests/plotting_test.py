#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 

Plotting the analytical functions 


"""



from radpattern.plotting import pattern_3d as pt
from radpattern.physics import setup_params as stp
from radpattern.physics import analytical_patterns as anp
from radpattern.helpers import helpers as hps
from radpattern.geometry import grids

import matplotlib.pyplot as plt 
import numpy as np
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)






# Creates the system to plot. Basically we want the k_in direction. 
sys = stp.PhysicalParams(k_in_hat = [0,-4,7])

# Sim params to create the resolution of the grid. 
sim = stp.SimParams(
        n_theta=2*60,#241,#91
        n_phi=2*120,#481,#181
    )

grid  = grids.AngleGrid(sim.n_theta, sim.n_phi)

# Dipole orienrantion. 
dipole = hps.single_dipole_E(grid.nx,grid.ny,grid.nz, [1,0,0])

# Creates observation vector q = k_in - k_out
q_vec =  sys.k_in_hat[None, None, :] - grid.n_hat

AF_sphere = anp.sphere_af(q_vec, 10) 

AF_cube = anp.box_af(q_vec, 20,20,20)

#AF_slab = anp.slab_2d_gaussian_af(q_vec, 20)
#AF_slab = anp.slab_2d_af(q_vec, 20)

I = np.abs(AF_cube)**2 * dipole

#I = hps.intensity_from_field(AF_cube, dipole) 
print(I.max(), I.min())


k_in = np.round(sys.k_in_hat)


pt.plot_pattern_3d(grid, I, title=f"Analitical result Cube.  bk  = 10, k_in [0,-4,7]", alpha=0.5) 
#rpt.plot_analytic_pattern_3d(anp.sphere_af, sys.k_in_hat, {'kR':10})



log.info("test")
plt.show()



