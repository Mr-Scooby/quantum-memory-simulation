#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" AF calculations using GPU 
Should be susefull as we have large matrix 
"""

import numpy as np
import cupy as cp

def prepare_gpu_grid(n_hat_flat):
    """
    Move fixed angular grid to GPU once.
    Call this once before the time / MC loop.
    """
    return cp.asarray(n_hat_flat, dtype=cp.float32)


def array_factor_general_gpu(
    n_hat_flat,
    grid_shape,
    k_out,
    r_xyz,
    w=None,
    chunk_atoms=1000,
):
    """
    GPU version of array_factor_general using CuPy.

    Computes:
        AF(n_hat) = sum_j w_j * exp(i k_out n_hat · r_j)

    Parameters
    ----------
    n_hat_flat : np.ndarray
        Shape (M, 3), flattened observation directions.
    grid_shape : tuple
        Shape (n_theta, n_phi), used to reshape output.
    k_out : float
        Output wave number.
    r_xyz : np.ndarray
        Shape (N, 3), atom positions.
    w : np.ndarray or None
        Shape (N,), complex atom weights.
    chunk_atoms : int
        Number of atoms per GPU chunk.

    Returns
    -------
    AF : np.ndarray
        Complex array factor with shape grid_shape.
    """
    nt, np_ = grid_shape
    N = r_xyz.shape[0]
    M = n_hat_flat.shape[0]

    # Move fixed arrays to GPU
    r_gpu = cp.asarray(r_xyz, dtype=cp.float64)

    if w is None:
        w_gpu = cp.ones(N, dtype=cp.complex128)
    else:
        w_gpu = cp.asarray(w, dtype=cp.complex128)

    AF_gpu = cp.zeros(M, dtype=cp.complex128)
    k_out_gpu = cp.float32(k_out)

    for a0 in range(0, N, chunk_atoms):
        a1 = min(a0 + chunk_atoms, N)

        r_chunk = r_gpu[a0:a1]      # shape (chunk, 3)
        w_chunk = w_gpu[a0:a1]      # shape (chunk,)

        # shape: (M, chunk)
        phase = k_out_gpu * (n_hat_flat @ r_chunk.T)

        # sum over atoms in this chunk
        AF_gpu += cp.exp(1j * phase).astype(cp.complex64) @ w_chunk

        # Optional: free temporary memory between chunks
        del phase

    AF = cp.asnumpy(AF_gpu).reshape(nt, np_)

    return AF
