#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import helpers 
import mcpattern
import rpattern
import rplotting
import time
import logging
import numpy as np
import matplotlib.pyplot as plt 
import beam
import setup_params as stp 
from dataclasses import asdict 

# ---------------------------------------------------------------------
# Logging
def get_logger(name="run_sim", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 

    if logger.handlers:
        return logger

    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

#log = get_logger()





def main():
    """ run simualtion"""
    regime = stp.PhysicalRegime(
        optical_size=100.0,
        optical_spacing=1.5,
        illumination_ratio=0.7,
        filling_factor=0.4,
        pulse_transit=1.1,
    )
    
    phys = stp.PhysicalParams(
        regime=regime,
        wavelength=1,
        v_front=1.0,
        v_thermal=0.00,
        k_in_hat = [0,1,1],
        p_hat = [1,0,0],
        beam_r0 = 20,
       pcenter_atOrigin = False
    )
    
    sim = stp.SimParams(
        n_atoms=1000,
        n_mc= 1,
        t_max_factor = 1,
        t_char= phys.t_char,
        n_times=100,
        n_theta=91,#241,#91
        n_phi=181,#481,#181
        seed=1 ,
    )

    setup = stp.SetupParams(regime, phys, sim) 
   
   # Angle grid
    theta, phi, nx, ny, nz = helpers.make_angle_grid(n_theta= sim.n_theta, n_phi=sim.n_phi)
    
    # weights 
    #w_fn = beam.make_weight_fn_gaussian_pulse(phys.beam_waist,
    #                                          phys.sigma_long,
    #                                          phys.k_in_hat,
    #                                          phys.k0,
    #                                          v_front = phys.v_front, 
    #                                          box_size= phys.box_size, 
    #                                          pulse_center_t0= phys.beam_r0,
    #                                          pcenter_at_origin = phys.pcenter_atOrigin,
    #                                          margin = 0
    #                                              )

    w_fn =beam.make_weight_fn_plane_wave(k_in_hat=phys.k_in_hat, k_in=phys.k0)
    rng =  np.random.default_rng(0)
#    r_xyz, v_xyz = mcpattern.sample_realization(sim.n_atoms, rng, v_std= phys.v_thermal) 
#    position, weights, pulse_center = helpers.atom_weights_sim(sim.times,
#                                                      r_xyz,
#                                                      v_xyz,
#                                                      w_fn)
#
    I, position, weights, pulse_center = mcpattern.mc_sim(
        nx=nx, ny=ny, nz=nz,
        grid_shape=sim.grid_shape,
        k_out=phys.k0,
        p_hat=phys.p_hat,
        times=sim.times,
        n_mc=sim.n_mc,
        n_atoms=sim.n_atoms,
        w_fn=w_fn,
        chunk_atoms=sim.chunk_atoms,
        seed=sim.seed,
        normalize_each_time=sim.normalize_each_time,
        plane_restricted = sim.plane_restricted,
        box_size=phys.box_size,
        v_std=phys.v_thermal,
        center= [0,0,0],
        pcenter_at_origin= phys.pcenter_atOrigin
    )

    log.info("sim times window t_min = %s, t_max = %s", min(sim.times), max(sim.times))
    # Saving
    #helpers.save_simulation_npz("test_0001", atom_pos = position, w = weights, pcenter = pulse_center,times=setup.sim.times) 
    helpers.save_simulation_npz("sims_runs/"+setup.run_name,
    metadata=asdict(setup), intensity = I, atom_pos = position, w = weights, pcenter = pulse_center,times=setup.sim.times) 

#    #plt.show()
    log.info("plotting") 
#
#    #rplotting.plot_pattern_3d(nx, ny, nz, I_plot0, title=f"MC mean pattern, t={times[0]:.3g}", alpha=1.0, stride=2)
#

if __name__ == "__main__":

    log.info("Starting run_sim. TIME DEPENDENC MC SIMULATION")
    main()
    



