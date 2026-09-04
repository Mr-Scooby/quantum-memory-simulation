#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataclasses import dataclass, fields
import numpy as np 
from pathlib import Path 



@dataclass
class RunAnalisis: 
    """ Stores analisis data from sim runs """

    file : Path
    file_hash: str
    plot_label: str
    exp_label: str 
    times_us : np.ndarray = field(repr= False)
    D_cte: float  
    eta: np.ndarray 
    p_fib: np.ndarray 
    p_tot: np.ndarray 
    

    @property
    def P_fib_over_Ptot0(self):
        return self.P_fib / (self.P_tot[0] + 1e-30)

    def compute_couplin_curves(self, I_fib, I_emit): 
        """Computes the coupling calculation for the given I_fib (intensity mode from the fiber  to couple) 
        and the I_emmited from the ensemble (emission intensity). 
        computes:  eta, p_fib, p_tot"""

        data, grid, exp, sim  = load_data(PATH/file)

        log.debug("Loaded keys from npz file: %s", list(data.keys()))
        log.debug("AF2 shape: %s", data["AF2"].shape)
        log.debug("Intensity shape: %s", data["intensity"].shape)

        AF = np.abs(data["AF2"])
        Intensity =data["intensity"]
        times_us = data["times_us"]

        log.debug("Time array shape: %s", times_us.shape)
        log.debug("Experiment label: %s", exp.label)

        ### Calculating Gaussian mode. 
        ### Coupling to gaussian mode calculation.
        theta0 = 12 / (exp.atom.k_signal * exp.w0_signal)
        E_fib = np.abs(cp.gaussian_fiber_mode_on_sphere(grid, theta0)) ** 2
        log.info(
            "theta0 = %.6e rad, forward lobe = %.6e rad, match = %s",
            theta0,
            exp.forwardlobe_angular_width,
            np.isclose(theta0, exp.forwardlobe_angular_width),
        )

        log.debug("Building single-dipole radiation pattern")
        dipole = single_dipole_E(
                grid.nx,
                grid.ny,
                grid.nz,
                np.array([1.0, 0.0, 0.0]),
            )
        eta_t = np.zeros(AF.shape[0])
        eta_abs_t =np.zeros(AF.shape[0])

        print(f"Shape Inetensity {Intensity.shape}")
        I_t = np.zeros(AF.shape[0])
        eta_i_   =np.zeros(AF.shape[0]) 
        P_fiber_ =np.zeros(AF.shape[0]) 
        P_total_ =np.zeros(AF.shape[0]) 

        log.info("Computing fiber coupling for file: %s", file)
        P_fib, P_tot, eta_t = coupling_from_AF2(
                AF2_t=AF,
                grid=grid,
                dipole=dipole,
                E_fib=E_fib,
                theta0=theta0,
                )

        if np.any(P_tot <= 0):
            log.warning("Some total-power values are zero or negative in file: %s", file)

