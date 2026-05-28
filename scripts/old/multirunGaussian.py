#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

from radpattern.helpers import helpers
from radpattern.physics import AF_parallel, beam
from radpattern.physics import setup_params as stp
from radpattern.geometry import grids

log = logging.getLogger(__name__)





def generating_sim(regime, sys, sim): 

    log.info("Generating sim world...") 

    # Sim set up for naming 
    setup = stp.SetupParams(regime, sys, sim)
    # angular grid
    grid = grids.AngleGrid(n_theta=sim.n_theta,n_phi=sim.n_phi)
    
    log.info("Generating atoms geometry...")
    Nx = int(round(sys.Lx / sys.spacing)) + 1
    Ny = int(round(sys.Ly / sys.spacing)) + 1
    Nz = int(round(sys.Lz / sys.spacing)) + 1
    # Static regular cube of atoms
    log.warning("Atoms number Nx = %s, Ny= %s, Nz=%s, N_total = %s", Nx, Ny, Nz, Nx*Ny*Nz)
    r_xyz = helpers.atom_grid( Nx,Ny,Nz,
        dx=sys.spacing,
        dy=sys.spacing,
        dz=sys.spacing,
    )

    log.info("updating atom count...")
    sim.n_atoms = r_xyz.shape[0]
    
    # Plane-wave weights
    log.info("Generating atoms geometrys...")
    w_fn = beam.make_weight_fn_gaussian_pulse(
    w0=sys.beam_waist,
    sigma_long=sys.sigma_long,
    k_in_hat=sys.k_in_hat,
    k_in=sys.k0,
    v_front=0.0,
    box_size=sys.box_size,
    center=[0.0, 0.0, 0.0],
    pulse_center_t0=0.0,
    pcenter_at_origin=True,
    margin=0.0,
    )
   
    w = w_fn(r_xyz, t=0.0)
    
    # Array factor + intensity
    AF = AF_parallel.array_factor_general_parallel(
        n_hat_flat=grid.n_hat_flat,
        grid_shape=sim.grid_shape,
        k_out=sys.k0,
        r_xyz=r_xyz,
        w=w,
        chunk_atoms=sim.chunk_atoms,
        n_workers = 2
    )
    
    dipole = helpers.single_dipole_E(grid.nx,grid.ny, grid.nz, sys.p_hat)
    I = helpers.intensity_from_field(AF, dipole=dipole)
    
    # Save
    helpers.save_simulation_npz(
        "../data/results_sims/multirun" + setup.run_name,
        metadata=asdict(setup),
        atom_pos=r_xyz,
        w=w,
        intensity= I,
        grid_shape=grid.shape,
    )
    
