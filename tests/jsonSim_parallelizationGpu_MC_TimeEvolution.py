#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
from dataclasses import asdict
from pathlib import Path

from radpattern.helpers.io import save_simulation_npz
from monte_carlo_gpu import run_monte_carlo_gpu


def run_one_config(objs, output_dir, **kwargs):
    exp = objs.exp
    sim = objs.sim
    beam = objs.beam

    setp = sim.sim_metadataSetUp(exp, beam)

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
        path,
        metadata=asdict(setp),
        times_code=result["times_code"],
        AF=result["AF_mean"],
        AF2=result["AF2_mean"],
        intensity=result["I_mean"],
        eta_mean=result["eta_mean"],
    )
