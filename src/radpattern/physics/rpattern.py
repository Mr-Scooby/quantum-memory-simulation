#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from radpattern.helpers.helpers import (
    make_angle_grid,
    atom_grid,
    gaussian_weights,
    intensity_from_field,
    random_position, 
    random_velocity_thermal,
    single_dipole_E
    )

import logging

log = logging.getLogger(__name__)

#################################################################################
# Array factor
# Flatten directions: n_hat_flat (M,3), M=nt*np
#n_hat_flat = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)

def array_factor_general(n_hat_flat, grid_shape, k_out, r_xyz, w=None, chunk_atoms=2000 ,chunk_dirs = 8192):

    """
    Compute the array factor on CPU using NumPy.

    The calculation is chunked over both atoms and observation
    directions to limit temporary memory usage.

    AF(n_hat) = sum_j w_j * exp(i * k_out * n_hat · r_j)

    Parameters
    ----------
    n_hat_flat : np.ndarray
        Observation directions with shape (M, 3).

    grid_shape : tuple[int, int]
        Final angular-grid shape, normally (n_theta, n_phi).

    k_out : float
        Output wave number.

    r_xyz : np.ndarray
        Atomic positions with shape (N, 3).

    w : np.ndarray | None
        Complex atomic weights with shape (N,). If None, all weights
        are set to one.

    chunk_atoms : int
        Maximum number of atoms processed at once.

    chunk_dirs : int
        Maximum number of observation directions processed at once.

    Returns
    -------
    np.ndarray
        Complex array factor with shape ``grid_shape``.
    """
    n_hat_flat = np.asarray(n_hat_flat, dtype=np.float64)
    r_xyz = np.asarray(r_xyz, dtype=np.float64)

    n_atoms = r_xyz.shape[0]
    n_dirs = n_hat_flat.shape[0]

    if n_hat_flat.ndim != 2 or n_hat_flat.shape[1] != 3:
        raise ValueError(
            f"n_hat_flat must have shape (M, 3), got {n_hat_flat.shape}"
        )

    if r_xyz.ndim != 2 or r_xyz.shape[1] != 3:
        raise ValueError(
            f"r_xyz must have shape (N, 3), got {r_xyz.shape}"
        )

    if np.prod(grid_shape) != n_dirs:
        raise ValueError(
            f"grid_shape={grid_shape} contains {np.prod(grid_shape)} "
            f"directions, but n_hat_flat contains {n_dirs}"
        )

    if chunk_atoms <= 0:
        raise ValueError("chunk_atoms must be positive")

    if chunk_dirs <= 0:
        raise ValueError("chunk_dirs must be positive")

    if w is None:
        weights = np.ones(n_atoms, dtype=np.complex128)
    else:
        weights = np.asarray(w, dtype=np.complex128)

        if weights.shape != (n_atoms,):
            raise ValueError(
                f"w must have shape ({n_atoms},), got {weights.shape}"
            )

    af_flat = np.zeros(n_dirs, dtype=np.complex128)

    n_direction_chunks = (
        n_dirs + chunk_dirs - 1
    ) // chunk_dirs

    n_atom_chunks = (
        n_atoms + chunk_atoms - 1
    ) // chunk_atoms

    for d0 in range(0, n_dirs, chunk_dirs):
        d1 = min(d0 + chunk_dirs, n_dirs)
        direction_chunk_index = d0 // chunk_dirs + 1

        log.info(
            "AF direction chunk %d/%d",
            direction_chunk_index,
            n_direction_chunks,
        )

        directions = n_hat_flat[d0:d1]
        af_block = np.zeros(d1 - d0, dtype=np.complex128)

        for a0 in range(0, n_atoms, chunk_atoms):
            a1 = min(a0 + chunk_atoms, n_atoms)
            atom_chunk_index = a0 // chunk_atoms + 1

            log.debug(
                "AF direction chunk %d/%d, atom chunk %d/%d",
                direction_chunk_index,
                n_direction_chunks,
                atom_chunk_index,
                n_atom_chunks,
            )

            positions = r_xyz[a0:a1]
            atom_weights = weights[a0:a1]

            # Shape:
            # (direction chunk, atom chunk)
            phase = k_out * (directions @ positions.T)

            af_block += np.exp(1j * phase) @ atom_weights

        af_flat[d0:d1] = af_block

    log.info("CPU array-factor calculation finished")

    return af_flat.reshape(grid_shape)

# Array factor (separable lattice)
# ---------------------------
def centered_indices(N):
    return np.arange(N) - (N - 1)/2

def array_factor_separable(nx, ny, nz, k, dx, dy, dz, Nx, Ny, Nz):
    log.info("AF separable")
    mx = centered_indices(Nx)[:, None, None]
    my = centered_indices(Ny)[:, None, None]
    mz = centered_indices(Nz)[:, None, None]

    ux = k * dx * nx[None, :, :]
    uy = k * dy * ny[None, :, :]
    uz = k * dz * nz[None, :, :]

    Sx = np.sum(np.exp(1j * mx * ux), axis=0)
    Sy = np.sum(np.exp(1j * my * uy), axis=0)
    Sz = np.sum(np.exp(1j * mz * uz), axis=0)
    return Sx * Sy * Sz


# ---- sanity checks for x-dipole with 1 atom ----
def get_I_at(th0, ph0):
    i = np.argmin(np.abs(theta - th0))
    dphi = np.angle(np.exp(1j*(phi - ph0)))
    j = np.argmin(np.abs(dphi))
    return float(I[i, j])

def sanity_printing():
    print("Sanity :")
    print("  I(+x):", get_I_at(np.pi/2, 0.0))
    print("  I(+y):", get_I_at(np.pi/2, np.pi/2))
    print("  I(+z):", get_I_at(0.0, 0.0))
