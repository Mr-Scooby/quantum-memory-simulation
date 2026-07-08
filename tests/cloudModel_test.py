#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from radpattern.config.builder import build_default_object
from radpattern.simulation.preflight_check import pre_simulation_warnings
#from radpattern.plotting.rplotting import plot_atoms  

import numpy as np
#import matplotlib.pyplot as plt 
import logging
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.FileHandler("cloudModel_test.log", mode='w'),          # Writes to file
        logging.StreamHandler()        # Writes to console
    ]
)
log = logging.getLogger(__name__)
log.warning(" Cloud model test file. Testing functionality of CloudModel and ExpParams withouit AF calculation") 


# Building sim system
log.info(" Building objs from defualt file for testing") 
ref = build_default_object("Cs133")
# Pre sim check
pre_simulation_warnings(ref,12*ref.exp.forwardlobe_angular_width )

# Copying objects
cloud = ref.cloud
log.info(cloud) 
control_beam = ref.Cbeam
signal_beam = ref.Sbeam
exp = ref.exp 
log.info(exp)
sim = ref.sim

## Independent RNG per MC run
seed = 1000
rng = np.random.default_rng(seed)

# Grid and time array formations
grid = ref.sim.create_grid()
times_code = sim.time_array() 

# Cloud generation 
cloud.generate_cloud(rng=rng)
cloud.generate_velocity_distribution(rng = rng )

#Weights generation
signal_beam.generate_weights(cloud.r_xyz) 
control_beam.generate_weights(cloud.r_xyz)
cloud.generate_S_profile(signal_beam, control_beam)
# Copy r_0 
log.debug("Copy r0_xyz")
cloud.r0_xyz = cloud.r_xyz.copy()

# Diffusion constant
Diff_coef = exp.diffusion_coeff_code

# Time Sim 
log.warning("Simulating only 3 steps of the sim.timearray for testing purposes") 
for it, t in enumerate(times_code[:3 ]): # Reduce to only the first 3 steps for the test.. 
    dt = 0.0 if it == 0 else times_code[it] - times_code[it - 1]
    log.debug("iteration : %d, dt = %.5f", it, dt ) 

    # Move atoms
    cloud.update_position_diffusive(dt ,Diff_coef, rng=rng)

    control_beam.generate_weights(cloud.r_xyz) 

    # Spin-wave phase evolution
    dt_s = dt * sim.char_time
    motion_phase = cloud.update_motion_phase(
        dt_s=dt_s,
        B0_T=exp.B0_T,
        B_gradient_z_T_per_code=exp.B_gradient * exp.ref_length,
    )

    # Generating weights
    weights =  cloud.S * control_beam.w * motion_phase
    # Warning control if weights goes to zero. 
    if np.max(np.abs(weights)) < 1e-14:
        log.warning(
            "MC %d timestep %d: all retrieval weights are near zero. "
            "Coupling/AF result may be meaningless.",
            mc,
            it,
        )

    ## AF CALCULATION

    log.info("\n")
    time.sleep(1) 
##
#
#
