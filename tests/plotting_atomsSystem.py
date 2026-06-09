#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from old_builder import build_run_objects
from radpattern.config.builder import build_run_objects
from radpattern.plotting.rplotting import plot_atoms, plot_cloud_slices
import matplotlib.pyplot as plt
from pathlib import Path  




path = Path(input("Path to file:"))

objs = build_run_objects(path)

cloud = objs.cloud 
cloud.n_sim_atoms= 1e6
control_beam = objs.Cbeam
signal_beam = objs.Sbeam


cloud.generate_cloud()
control_beam.generate_weights(cloud.r_xyz)
signal_beam.generate_weights(cloud.r_xyz)
cloud.generate_S_profile(control_beam, signal_beam)

plot_atoms(cloud.r_xyz, w = cloud.S)
plot_cloud_slices(cloud.r_xyz, w = cloud.S)
plt.show()
