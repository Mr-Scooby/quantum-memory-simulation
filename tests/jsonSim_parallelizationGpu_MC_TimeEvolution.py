#!/usr/bin/env mathD
# -*- coding: utf-8 -*-
from dataclasses import asdict
from pathlib import Path

from radpattern.helpers.io import save_simulation_npz
from radpattern.simulation.monte_carlo_gpu import run_monte_carlo_gpu


def run_one_config(objs, output_dir):
    exp = objs.exp
    sim = objs.sim
    beam = objs.beam

    setp = sim.sim_metadataSetUp(exp, beam)

    result = run_monte_carlo_gpu(objs)

    #path = Path(output_dir) / setp.run_name
    path = os.path.join(
                os.path.expanduser("~"),
                "radek",
                "simulations",
                "data",
                "results_sims",
                setp.run_name,
            )


    save_simulation_npz(
        path,
        metadata=asdict(setp),
        times_code=result["times_code"],
        AF=result["AF_mean"],
        AF2=result["AF2_mean"],
        intensity=result["I_mean"],
        eta_mean=result["eta_mean"],
    )
