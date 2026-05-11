#/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from radpattern.helpers import helpers
from radpattern.physics import rpattern
from radpattern.plotting.pattern_3d import plot_pattern_3d
from radpattern.geometry import grids

def gaussian_fiber_mode_on_sphere(grid, theta_f=0.1 ):
    """ gaussian profile """
    return np.exp(-(grid.TH / theta_f)**2).astype(np.complex128)

def overlap_on_sphere(grid, E_emit, E_mode, theta_max = 0.1):
    """
    Normalized mode overlap on a spherical angular grid.
    E_emit, E_mode : (Nt, Np) complex fields
    """
    E_emit = np.asarray(E_emit, np.complex128)
    E_mode = np.asarray(E_mode, np.complex128)

    # deifining the angular space for the coupling. 
    if theta_max > 2*np.pi: 
        theta_max % 2*np.pi
        print(f"Theta_max > 2 pi. refactoring to theta_max %2*pi. = {theta_max}")
    mask = grid.TH <= theta_max

    theta = grid.TH[:, 0]   # 1D theta, shape (n_theta,)
    phi   = grid.PH[0, :]   # 1D phi,   shape (n_phi,)
    sin_th = np.sin(grid.TH)   # 2D, same shape as E_emit

    num = np.trapezoid(
        np.trapezoid(E_emit * np.conj(E_mode) * sin_th, phi, axis=1),
        theta,
        axis=0
    )

    den_emit = np.trapezoid(
        np.trapezoid(np.abs(E_emit)**2 * sin_th, phi, axis=1),
        theta,
        axis=0
    )

    den_mode = np.trapezoid(
        np.trapezoid(np.abs(E_mode)**2 * sin_th, phi, axis=1),
        theta,
        axis=0
    )

    eta = np.abs(num)**2 / (den_emit * den_mode + 1e-30)
    return eta, num

def intensity_overlap_on_sphere(grid, I_emit, I_mode, theta_max=0.1):
    mask = grid.TH <= theta_max

    theta = grid.TH[:, 0]
    phi = grid.PH[0, :]
    sin_th = np.sin(grid.TH)

    I_emit = np.asarray(I_emit, float)
    I_mode = np.asarray(I_mode, float)

    I_emit = np.where(mask, I_emit, 0.0)
    I_mode = np.where(mask, I_mode, 0.0)

    num = np.trapezoid(
        np.trapezoid(I_emit * I_mode * sin_th, phi, axis=1),
        theta,
        axis=0,
    )

    den_emit = np.trapezoid(
        np.trapezoid(I_emit * sin_th, phi, axis=1),
        theta,
        axis=0,
    )

    den_mode = np.trapezoid(
        np.trapezoid(I_mode * sin_th, phi, axis=1),
        theta,
        axis=0,
    )

    return num / (den_emit+ 1e-30)




def analytic_eta_gaussian_width_mismatch(theta1, theta2):
    """
    Analytic paraxial overlap efficiency between two centered
    axisymmetric Gaussian angular modes:

        E_i(theta) = exp(-(theta/theta_i)^2)

    Result:
        eta = 4 theta1^2 theta2^2 / (theta1^2 + theta2^2)^2
    """
    theta1 = float(theta1)
    theta2 = float(theta2)
    return 4.0 * theta1**2 * theta2**2 / (theta1**2 + theta2**2)**2

def run_overlap_test_protocol(
    grid,
    overlap_fn,
    fiber_mode_fn,
    theta_max = 0.1,
    seed = np.random.rand(),
    verbose=True,
):
    """
    Basic validation protocol for spherical-overlap code.

    Parameters
    ----------
    grid : object
        Must provide TH, PH meshgrids.
    overlap_fn : callable
        Function like:
            eta, a = overlap_fn(grid, E_emit, E_mode)
    fiber_mode_fn : callable
        Function like:
            E = fiber_mode_fn(grid, n_fiber_hat=[0,0,1], theta_f=...)
    theta_ref : float
        Reference angular width.
    theta_wide : float
        Second width for mismatch test.
    phase_shift : float
        Global phase for invariance test.
    atol : float
        Absolute tolerance for pass/fail.
    verbose : bool
        Print results.

    Returns
    -------
    results : dict
        Dictionary with measured and expected values.
    """
    np.random.seed(seed)

    theta_ref   = theta_max * np.random.rand()
    theta_wide  = theta_max * np.random.rand()
    phase_shift = 2*np.pi * np.random.rand() 

    atol=1e-5,
    results = {}

    # Build reference mode
    E_ref = fiber_mode_fn(grid, theta_f=theta_ref)

    # Test 1: self-overlap
    eta_self, a_self = overlap_fn(grid, E_ref, E_ref)
    pass_self = abs(eta_self - 1.0) < atol

    results["self_overlap"] = {
        "eta": float(eta_self),
        "expected": 1.0,
        "pass": bool(pass_self),
        "amplitude": a_self,
    }

    # Test 2: global phase invariance
    E_phase = np.exp(1j * phase_shift) * E_ref
    eta_phase, a_phase = overlap_fn(grid, E_phase, E_ref)
    pass_phase = abs(eta_phase - 1.0) < atol

    results["global_phase"] = {
        "eta": float(eta_phase),
        "expected": 1.0,
        "pass": bool(pass_phase),
        "amplitude": a_phase,
    }

    # Test 3: width mismatch against analytic result
    for i in range(4):
        theta_wide  = theta_max * np.random.rand()
        E_wide = fiber_mode_fn(grid, theta_f=theta_wide)
        eta_mismatch, a_mismatch = overlap_fn(grid, E_ref, E_wide)
        eta_expected = analytic_eta_gaussian_width_mismatch(theta_ref, theta_wide)
        pass_mismatch = abs(eta_mismatch - eta_expected) < atol
        print(theta_ref, theta_wide) 
        results[f"width_mismatch_{i}"] = {
            "eta": float(eta_mismatch),
            "expected": float(eta_expected),
            "pass": bool(pass_mismatch),
            "amplitude": a_mismatch,
            "theta_ref": float(theta_ref),
            "theta_wide": float(theta_wide),
        }

    results["all_pass"] = all( test["pass"] for key, test in results.items() if isinstance(test, dict) and "pass" in test)

    if verbose:
        print("\n=== Overlap test ===")
        print(f"grid shape: TH={grid.TH.shape}, PH={grid.PH.shape}")

        print("\n[1] self overlap")
        print(f"  eta       = {results['self_overlap']['eta']:.8f}")
        print(f"  expected  = {results['self_overlap']['expected']:.8f}")
        print(f"  pass      = {results['self_overlap']['pass']}")

        print("\n[2] global phase invariance")
        print(f"  eta       = {results['global_phase']['eta']:.8f}")
        print(f"  expected  = {results['global_phase']['expected']:.8f}")
        print(f"  pass      = {results['global_phase']['pass']}")

        print("\n[3] width mismatch")
        for i in range(4): 
            print(f"  theta_ref, tehta_wide = {results['width_mismatch_'+str(i)]['theta_ref']:.3f},{results['width_mismatch_'+str(i)]['theta_wide']:.3f}")
            print(f"  eta       = {results[f'width_mismatch_'+str(i)]['eta']:.8f}")
            print(f"  expected  = {results[f'width_mismatch_'+str(i)]['expected']:.8f}")
            print(f"  pass      = {results[f'width_mismatch_'+str(i)]['pass']}")
            if not results[f'width_mismatch_'+str(i)]['pass']:
                error = results['width_mismatch_'+str(i)]['eta'] - results['width_mismatch_'+str(i)]['expected']
                print(f"error  = {error}")

        print("\nOverall:", results["all_pass"])

    return results

