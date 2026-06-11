#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from old_builder import build_run_objects
from radpattern.config.builder import build_run_objects
from radpattern.plotting.rplotting import plot_atoms, plot_cloud_slices




def plotting_cloud_from_json(path):
        objs = build_run_objects(path)

        cloud = objs.cloud 
        cloud.n_sim_atoms= objs.exp.density / 1e3
        print(f"density = {objs.exp.density}")
        control_beam = objs.Cbeam
        signal_beam = objs.Sbeam

        cloud.generate_cloud()
        control_beam.generate_weights(cloud.r_xyz)
        signal_beam.generate_weights(cloud.r_xyz)
        cloud.generate_S_profile(control_beam, signal_beam)

        fig1, ax1 = plot_atoms(cloud.r_xyz, w = cloud.S)
        fig2, ax2 = plot_cloud_slices(cloud.r_xyz, w = cloud.S)
        return fig1, ax1, fig2, ax2 

if __name__ == "__main__": 

        import matplotlib.pyplot as plt
        from pathlib import Path  


        path = Path(input("Path to file:"))
        fig1, ax1, fig2, ax2 = plotting_cloud_from_json(path)
        plt.show()
