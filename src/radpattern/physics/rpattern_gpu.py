#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" AF calculations using GPU 
Should be susefull as we have large matrix 
"""

import logging 
log = logging.getLogger(__name__)
import numpy as np

try:
    import cupy as cp
except ImportError:
    log.warning("Cupy moduled could not be imported. Assuming  no GPU available") 
    cp = None

## If gpu not available fallback to gpu 
from .rpattern import array_factor_general


## Define if gpu available
def _cuda_available() -> bool:
    if cp is None:
        return False

    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

CUDA_AVAILABLE = _cuda_available()

def prepare_gpu_grid(n_hat_flat):
    """
    Move fixed angular grid to GPU once.
    Call this once before the time / MC loop.
    """
    if CUDA_AVAILABLE:
        return cp.asarray(n_hat_flat, dtype=cp.float32)

    return np.asarray(n_hat_flat, dtype=np.float64)


def array_factor_general_gpu(
    n_hat_flat,
    grid_shape,
    k_out,
    r_xyz,
    w=None,
    chunk_atoms=1000,
    chunk_dirs = 8192, 
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
    ## Temp code, fallback to cpu AF calcualtion to not having to recode much. 
    if not CUDA_AVAILABLE:
        log.warning("CUDA unavailable. Using NumPy CPU backend.")
        log.warning("Chunks atoms and dirs capped at 1000, 1024 for safety. HARDCODED. ")

        return array_factor_general(
            n_hat_flat=n_hat_flat,
            grid_shape=grid_shape,
            k_out=k_out,
            r_xyz=r_xyz,
            w=w,
            chunk_atoms=chunk_atoms,
            chunk_dirs = chunk_dirs
        )

    ### GPU AF CALCULATION 
    nt, np_ = grid_shape
    N = r_xyz.shape[0]
    M = n_hat_flat.shape[0]

    # Move fixed arrays to GPU
    r_gpu = cp.asarray(r_xyz, dtype=cp.float32)

    if w is None:
        w_gpu = cp.ones(N, dtype=cp.complex64)
    else:
        w_gpu = cp.asarray(w, dtype=cp.complex64)

    AF_gpu = cp.zeros(M, dtype=cp.complex128)
    k_out_gpu = cp.float32(k_out)
    
    # Loop over angular direction chunks
    for d0 in range(0, M, chunk_dirs):
        d1 = min(d0 + chunk_dirs, M)

        n_block = n_hat_flat[d0:d1]          # shape: (chunk_dirs, 3)
        AF_block = cp.zeros(d1 - d0, dtype=cp.complex64)

        for a0 in range(0, N, chunk_atoms):
            a1 = min(a0 + chunk_atoms, N)

            r_chunk = r_gpu[a0:a1]      # shape (chunk, 3)
            w_chunk = w_gpu[a0:a1]      # shape (chunk,)

            # shape: (M, chunk)
            phase = k_out_gpu * (n_block @ r_chunk.T)

            # sum over atoms in this chunk
            AF_block += cp.exp(1j * phase).astype(cp.complex64) @ w_chunk

            # Optional: free temporary memory between chunks
            del phase

        AF_gpu[d0:d1] = AF_block

    AF = cp.asnumpy(AF_gpu).reshape(nt, np_)

    return AF
