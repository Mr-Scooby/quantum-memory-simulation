#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Run one simulation configuration and save the Monte Carlo averaged results.
"""

from dataclasses import asdict
from pathlib import Path

from radpattern.helpers.io import save_simulation_npz
from .monte_carlo_gpu import run_monte_carlo_gpu

from .preflight_check import pre_simulation_warnings

def run_one_config(objs, output_dir, **kwargs):
    """
    Run one config, save its mean MC results, and optionally save full MC runs.
    """
    pre_simulation_warnings(objs, 10*ref.exp.forwardlobe_angular_width )

    setp = objs.sim.sim_metadataSetUp(objs.exp, objs.Cbeam)

    output_dir = Path(output_dir)
    path = output_dir/ setp.run_name
    mc_dir = output_dir / f"{setp.run_name}_mc_runs"

    result = run_monte_carlo_gpu(
        objs=objs,
        save_full_mc=kwargs.get("save_full_mc", False),
        mc_dir=mc_dir,
    )

    print(f"File save in {path}")
    save_simulation_npz(
        mc_dir/ setp.run_name,
        metadata=asdict(setp),
        times_code=result["times_code"],
        AF=result["AF_mean"],
        AF2=result["AF2_mean"],
        intensity=result["I_mean"],
        eta_mean=result["eta_mean"],
    )
    return mc_dir
