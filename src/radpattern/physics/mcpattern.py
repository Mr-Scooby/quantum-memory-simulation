#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo simulation for a time-dependent array factor with moving atoms.
"""

import time
import logging
import numpy as np


from radpattern.geometry.cloud_model import CloudModel
from radpattern.geometry.grids import AngleGrid
from radpattern.physics.beam import BeamModel
from radpattern.physics.rpattern import array_factor_general



from radpattern.helpers.helpers import (
    make_angle_grid,
    gaussian_weights,
    intensity_from_field,
    random_position,
    random_velocity_thermal,
    single_dipole_E,
    filter_kwargs,
    atom_weights_sim
)

# ---------------------------------------------------------------------
# Logging
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
#def compute_realization_af_series_ballistic(
#    cloud, 
#    beam, 
#    grid,
#    times: np.ndarray,
#    chunk_atoms: int = 2000,
#    **kwargs
#) -> np.ndarray:
#    """
#    Compute AF(t, theta, phi) for one realization using ballistic motion:
#        r_j(t) = r0_j + v_j t
#
#    We precompute, per chunk:
#        phi_r0 = k_out * (n_hat · r0)
#        phi_v  = k_out * (n_hat · v)
#
#    Then at each time:
#        phase(t) = phi_r0 + t * phi_v
#
#    Notes
#    -----
#    - w_fn must be atom-local, since it is called chunk by chunk.
#    - Returns complex AF series with shape (T, nt, np_).
#    """
#    times = np.asarray(times, dtype=float)
#    nt, np_ = grid.shape
#    T = times.size
#    n_atoms = r0_xyz.shape[0]
#    
#    log.info("computing AF ... ") 
#    AF_series = np.zeros((T, nt * np_), dtype=np.complex128)
#
#    for a0 in range(0,n_atoms, chunk_atoms):
#        a1 = min(a0 + chunk_atoms, n_atoms)
#
#        r0_chunk = np.asarray(r0_xyz[a0:a1], dtype=float)   # (C, 3)
#        v_chunk  = np.asarray(v_xyz[a0:a1], dtype=float)    # (C, 3)
#
#        # Precompute ballistic phase ingredients for this chunk
#        phi_r0 = k_out * (grid.n_hat_flat @ r0_chunk.T)          # (M, C)
#        phi_v  = k_out * (grid.n_hat_flat @ v_chunk.T)           # (M, C)
#
#        for it, t in enumerate(times):
#            rt_chunk = r0_chunk + t * v_chunk              # (C, 3)
#            wt_chunk = np.asarray(w_fn(rt_chunk, t, return_pulse_center = kwargs.get('return_pulse_center') ), dtype=np.complex128)  # (C,)
#
#            AF_series[it] += np.exp(1j * (phi_r0 + t * phi_v)) @ wt_chunk
#
#    
#    return AF_series.reshape(T, nt, np_)
#

def compute_realization_intensity_series(grid, dipole:np.ndarray,  k_out: float,
                                        p_hat: np.ndarray, times: np.ndarray,
                                         n_atoms:int,rng:np.random, w_fn,
                                         chunk_atoms: int = 20000,
                                         **kwargs
                                        ) -> np.ndarray:
    """
    Compute the intensity time series for one realization of moving atoms.
    basically a single run simulation. deterministic physics. 

    Parameters
    ----------
    n_hat_flat : np.ndarray
        stack array of the irection cosine grids for the observation directions.
    grid_shape: (nt, np_) 
        for shape later 
    k_out : float
        Output wave number.
    p_hat : np.ndarray
        Dipole polarization / orientation unit vector.

    n_atoms: int 
        number of atoms
    times : np.ndarray
        Times at which the intensity is evaluated.
    rng: np.random 
        Radom generator object
    w_fn : callable or None, optional
        Weight function w_fn(r_t, t) for this realization. If None, unit weights are used.
    chunk_atoms : int, optional
        Number of atoms per chunk in the array factor calculation.

    Returns
    -------
    np.ndarray
        Intensity array with shape (T, n_theta, n_phi).
    """

    # time array
    times = np.asarray(times, dtype=float)
    T = times.size
    nt, np_ = grid.shape

    # Sample radom atoms and velocity
    r0_xyz, v_xyz = sample_realization(n_atoms,rng, **kwargs)
    
    # memory assignation
    I_series = np.zeros((T, nt, np_), dtype=float)

    log.info(
        "compute_realization_intensity_series : N=%d, T=%d, grid=(%d,%d), chunk_atoms=%d ",
        len(r0_xyz), T, nt, np_, chunk_atoms, 
    )

    AF_series = compute_realization_af_series_ballistic(
        grid ,
        k_out=k_out,
        times=times,
        r0_xyz=r0_xyz,
        v_xyz=v_xyz,
        w_fn=w_fn,
        chunk_atoms=chunk_atoms,
        **kwargs
    )

    log.info("Computing intensities...") 
    for it in range(T):
        I_series[it] = intensity_from_field(
            AF_series[it],
            dipole=dipole,
        )


    return I_series



def static_AF_calculation(cloud, beam, exp, grid, rng = None): 
    """ Single shot computation of the cloud and beam """ 

    cloud.generate_cloud(rng)
    cloud.generate_S_profile(exp.w0_signal) 

    beam.generate_weights(cloud.r_xyz)
    weights = cloud.S * beam.w 

    return array_factor_general(
        n_hat_flat=grid.n_hat_flat,
        grid_shape=grid.shape,
        k_out= cloud.atoms.k_signal,
        r_xyz=cloud.r_xyz,
        w=weights,
        )

def mc_static(cloud, beam,exp, grid, runs, seed = 0): 

    log.info("mc_sim: runs=%d",runs )

    AF2_acc = np.zeros(grid.shape, dtype=float)
    AF_acc = np.zeros(grid.shape, dtype=complex)

    log.info("Array factor calcualtion with P = w0 * S" )
    # Random generator object, helps to reproduce runs. 
    rng_master = np.random.default_rng(seed)
    # time to measure how long it takes. 
    t_start = time.time()

    # Runs of the simulation
    log.info("Starting mc runs ... 1 /%d", runs)
    for mc in range(runs ):
        # each simulation time keeper
        t_mc = time.time()
        # Independent RNG for this realization.
        rng = np.random.default_rng(rng_master.integers(0, 2**32))

        AF_mc = static_AF_calculation(cloud, beam,exp, grid, rng= rng)
        # Intensity acumalation. 
        AF_acc += AF_mc
        AF2_acc += np.abs(AF_mc)**2

        log.info(
            "mc run %d/%d done in %.2fs",
            mc + 1, runs, time.time() - t_mc
        )
    AF2_mean = AF2_acc / float(runs)
    AF_mean = AF_acc / float(runs)

    log.info(
        "mc_intensity_time_series end: total=%.2fs, avg_per_mc=%.2fs",
        time.time() - t_start,
        (time.time() - t_start) / max(float(runs), 1.0),
    )

    return AF_mean, AF2_mean


def mc_sim( grid,
    k_out: float, p_hat: np.ndarray,
    times: np.ndarray,
    n_mc: int,
    n_atoms: int, 
    w_fn,
    chunk_atoms: int = 20000, seed: int = 0,
    **kwargs
) -> np.ndarray:
    """
    Monte Carlo average of the intensity time series for moving atoms.

    Each realization samples a new atomic configuration internally and
    computes its intensity time series. The result returned here is the
    average over all realizations.

    Parameters
    ----------
    grid: grid obejct
    k_out : float
        Output wave number.
    p_hat : np.ndarray
        Dipole orientation / polarization vector.
    times : np.ndarray
        Times at which the intensity is evaluated.
    n_mc : int
        Number of Monte Carlo realizations.
    n_atoms : int
        Number of atoms in each realization.
    chunk_atoms : int, optional
        Chunk size used in the array factor calculation.
    seed : int, optional
        Master seed for reproducible Monte Carlo sampling.
        w_fn_factory : callable or None, optional
        Callable with signature w_fn_factory(rng) -> w_fn(r_t, t).
        If None, unit weights are used.

    Returns
    -------
    np.ndarray
        Monte Carlo averaged intensity series with shape (T, n_theta, n_phi).
    """
 
    times = np.asarray(times, dtype=float)
    T = times.size
    nt, np_ = grid_shape 

    log.info(
        "mc_sim: n_mc=%d, n_atoms=%d, T=%d, grid=(%d,%d), "
        "chunk_atoms=%d ",
        n_mc, n_atoms, T, nt, np_, chunk_atoms, 
    )

    # Flatten directions: n_hat_flat (M,3), M=nt*np
    dipole = single_dipole_E(nx,ny,nz, p_hat) 

    I_accum = np.zeros((T, nt, np_), dtype=float)
    # Random generator object
    # helps to reproduce runs. 
    rng_master = np.random.default_rng(seed)
    # time to measure how long it takes. 
    t_start = time.time()

    # Runs of the simulation
    log.info("Starting mc runs ... 1 /%d", n_mc)
    for mc in range(n_mc):
        # each simulation time keeper
        t_mc = time.time()

        # Independent RNG for this realization.
        rng = np.random.default_rng(rng_master.integers(0, 2**32))

        # Simulation
        I_mc = compute_realization_intensity_series(n_hat_flat= grid.n_hat_flat,
                grid_shape = grid_shape, 
                dipole= dipole, 
                k_out=k_out,
                p_hat=p_hat,
                times=times,
                n_atoms = n_atoms,
                rng=rng,
                chunk_atoms=chunk_atoms,
                w_fn=w_fn, **kwargs
        )

        # Intensity acumalation. 
        I_accum += I_mc

        log.info(
            "mc run %d/%d done in %.2fs",
            mc + 1, n_mc, time.time() - t_mc
        )

    I_mean = I_accum / float(n_mc)

    log.info(
        "mc_intensity_time_series end: total=%.2fs, avg_per_mc=%.2fs",
        time.time() - t_start,
        (time.time() - t_start) / max(float(n_mc), 1.0),
    )

    log.info("Simulating atom and beam movement")
    r_xyz, v_xyz = sample_realization(n_atoms, rng, **kwargs) 
    pos, weight, pulscenter = atom_weights_sim(times, r_xyz, v_xyz, w_fn)

    return I_mean, pos, weight, pulscenter
