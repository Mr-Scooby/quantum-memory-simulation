#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import traceback


import time 
import numpy as np 
from config_object import build_run_objects

#from jsonSim_parallelizationGpu_MC_TimeEvolution import run_one_config



config_path = "config_file_template.json" 
config_path = "fallBack_test.json" 

objs = build_run_objects(str(config_path))
print(objs.exp)
print(objs.sim)
print(objs.cloud)
print(objs.beam)

exp = objs.exp
sim = objs.sim
beam = objs.beam
cloud = objs.cloud
setp = sim.sim_metadataSetUp(exp, beam)

grid = sim.create_grid()
times_code = sim.time_array()

T = sim.time_divisions
nt, nphi = grid.shape


# Independent RNG per MC run
seed = 1000 if sim.seed is None else int(sim.seed)
rng = np.random.default_rng(seed)

# Generate one cloud realization
cloud.generate_cloud(rng=rng)
cloud.generate_velocity_distribution(rng = rng)
cloud.generate_S_profile(exp.w0_signal)
cloud.r0_xyz = cloud.r_xyz.copy()

Diff_coef = exp.diffusion_coeff_code

dt = 1.0 

# Move atoms
cloud.update_position_diffusive(dt, Diff_coef, rng=rng)

# Update spatial beam weights
beam.generate_weights(cloud.r_xyz)

# Spin-wave phase evolution
dt_s = dt * sim.char_time
motion_phase = cloud.update_motion_phase(
    dt_s=dt_s,
    B0_T=exp.B0_T,
    B_gradient_z_T_per_code=exp.B_gradient * exp.ref_length,
)

weights =  cloud.S * beam.w * motion_phase



