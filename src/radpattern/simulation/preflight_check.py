#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from copy import deepcopy
import logging 
log = logging.getLogger(__name__)

def pre_simulation_warnings(objs, theta0):
    """ Pre sim check to raise warnings and errors before expensive computation"""

    log.info("Running pre-simulation physical checks")

    objs = deepcopy(objs) 
    exp = objs.exp
    sim = objs.sim
    cloud = objs.cloud
    signal_beam = objs.Sbeam
    control_beam = objs.Cbeam

    grid = sim.create_grid() 
    times_code = sim.time_array()


    rng = np.random.default_rng(1234)

    # triggers boundary/no-buffer warnings
    _ = exp.diffusion_coeff_code
    if hasattr(exp, "should_apply_boundary_conditions"):
        exp.should_apply_boundary_conditions(sim.simulation_window_radius_w0_cutoff)


    # 1. Angular grid vs fiber mode
    log.info(
        "Precheck fiber mode: theta0=%.3e, theta_max=%.3e, ratio=%.2f",
        theta0,
        grid.theta_max,
        theta0 / grid.theta_max,
    )
    if theta0 > grid.theta_max:
        log.warning(
            "Fiber mode angular width theta0=%.3e is larger than grid theta_max=%.3e. "
            "Coupling integral may be truncated.",
            theta0,
            grid.theta_max,
        )

    # 2. Angular grid vs forward lobe
    log.info(
        "Precheck angular grid: theta_max=%.3e, forward_lobe=%.3e, ratio=%.2f",
        sim.theta_max,
        exp.forwardlobe_angular_width,
        sim.theta_max / exp.forwardlobe_angular_width,
    )
    if sim.theta_max < 3.0 * exp.forwardlobe_angular_width:
        log.warning(
            "theta_max may be too small: theta_max=%.3e, forward_lobe≈%.3e. "
            "Forward emission lobe may be truncated.",
            sim.theta_max,
            exp.forwardlobe_angular_width,
        )

    # 3. Diffusion step size, before running MC
    D_code = exp.diffusion_coeff_code
    dt_max = np.max(np.diff(times_code)) if len(times_code) > 1 else 0.0
    step_std = np.sqrt(2.0 * D_code * dt_max)

    char_size = np.min(objs.cloud.box_size)

    log.info(
        "Precheck diffusion step: step_std=%.3e, limit=%.3e, ratio=%.3f",
        step_std,
        0.1 * char_size,
        step_std / char_size,
    )

    if step_std > 0.1 * char_size:
        log.warning(
            "Large maximum diffusion step before simulation: step_std=%.3e, char_size=%.3e. "
            "Boundary reflection may be inaccurate; reduce dt/time spacing.",
            step_std,
            char_size,
        )

    ## Warning number of substeps for large steps in geomspacing
    # The code set a dt_max limit of dt_max = 0.1 * min( exp.char_size) **2 / (2.0 * Diff_coef) 
    # Then for dt bigger it divides into smaller steps. 
    max_step_fraction = 0.1
    dt_max_lim = (max_step_fraction * cloud.char_size) ** 2 / (2.0 * exp.diffusion_coeff_code) 
    n_sub = max(1, int(np.ceil( dt_max / dt_max_lim) ))
    if n_sub >= 2_000: 
        log.warning("max_n_sub= %d. This run is effectively very expensive. Consider more time points or shorter sim_time.") 

    # 4. Boundary condition check, before cloud evolution
    if hasattr(exp, "should_apply_boundary_conditions"):
        exp.should_apply_boundary_conditions(sim.simulation_window_radius_w0_cutoff)

    # 5. Magnetic phase estimate
    z_max = 0.5 * getattr(objs.cloud, "box_size", np.array([0, 0, 0]))[2]
    B_max = abs(exp.B0_T) + abs(exp.B_gradient_z_T_per_code) * z_max
    max_phase_step = abs(exp.atom.magnetic_sensitivity_rad_s_T) * B_max * dt_max * sim.char_time

    log.info(
        "Precheck magnetic phase: max_dphi=%.3e rad, limit=%.3e rad",
        max_phase_step,
        np.pi,
    )

    if max_phase_step > np.pi:
        log.warning(
            "Large estimated magnetic phase step before simulation: max Δphi≈%.3g rad. "
            "Time step may undersample dephasing.",
            max_phase_step,
        )
    log.info("Checks doned")
