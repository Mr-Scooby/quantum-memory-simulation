#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from dataclasses import asdict

import numpy as np

#import helpers
#import rpattern
#import beam
from radpattern.physics import setup_params as stp
import multirun_static as mp


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    """
    Run a batch of static cube + plane-wave simulations.
    No MC, no time dependence, no Gaussian profile.
    """

    cases = [
        {
            "name": "baseline_z",
            "optical_size": 2.0,
            "optical_size_z":6.0,
            "optical_spacing": 0.01,
            "k_in_hat": [0.0, 0.0, 1.0],
            "p_hat": [1.0, 0.0, 0.0],
        },
        {
            "name": "larger_box_z",
            "optical_size": 10.0,
            "optical_size_z":6.0,
            "optical_spacing": 0.1,
            "k_in_hat": [0.0, 0.0, 1.0],
            "p_hat": [1.0, 0.0, 0.0],
        },
        {
            "name": "largerrr_box_z",
            "optical_size": 30.0,
            "optical_size_z":6.0,
            "optical_spacing": 0.1,
            "k_in_hat": [0.0, 0.0, 1.0],
            "p_hat": [1.0, 0.0, 0.0],
        },
        ## Too dense. too heavy 
#        {
#            "name": "denser_larger_box_z",
#            "optical_size": 10.0,
#            "optical_size_z":6.0,
#            "optical_spacing": 0.01,
#            "k_in_hat": [0.0, 0.0, 1.0],
#            "p_hat": [1.0, 0.0, 0.0],
#        },
#        {
#            "name": "denser_taller_box_z",
#            "optical_size": 2.0,
#            "optical_size_z":6.0,
#            "optical_spacing": 0.01,
#            "k_in_hat": [0.0, 0.0, 1.0],
#            "p_hat": [1.0, 0.0, 0.0],
#        },

# {
       #     "name": "denser_box_z",
       #     "optical_size": 150.0,
       #     "optical_spacing": 0.20,
       #     "k_in_hat": [0.0, 0.0, 1.0],
       #     "p_hat": [1.0, 0.0, 0.0],
       # },
#        {
#            "name": "test1",
#            "optical_size": 50.0,
#            "optical_size_z":6.0,
#            "optical_spacing": 1,
#            "k_in_hat": [0.0, 1.0, 1.0],
#            "p_hat": [1.0, 0.0, 0.0],
#        },
#        {
#            "name": "test2",
#            "optical_size":2.0,
#            "optical_size_z":6.0,
#            "optical_spacing": 1,
#            "k_in_hat": [1.0, 1.0, 0.0],
#            "p_hat": [0.0, 0.0, 1.0],
#        },
    ]

    for i_case, case in enumerate(cases, start=1):
        log.info("===================================================")
        log.info("Running case %d / %d : %s", i_case, len(cases), case["name"])

        print( "generatiion ")
        regime = stp.PhysicalRegime(
            optical_size=case["optical_size"],
            optical_size_z=case["optical_size_z"],
            optical_spacing=case["optical_spacing"],
            illumination_ratio=1.0,
            filling_factor=1.0,
            pulse_transit=1.0,
        )

        print( "generatiion ")
        phys = stp.PhysicalParams(
            regime=regime,
            wavelength=1.0,
            v_front=1.0,
            v_thermal=0.0,
            k_in_hat=case["k_in_hat"],
            p_hat=case["p_hat"],
            beam_r0=0.0,
            pcenter_atOrigin=True,
        )

        print( "generatiion ")
        sim = stp.SimParams(
            n_atoms=1,          # placeholder, corrected after grid is built
            n_mc=1,
            t_max_factor=1.0,
            t_char=1.0,
            n_times=1,
            n_theta=120,
            n_phi=240,
            seed=1,
            chunk_atoms=250,
        )

        print( "generatiion ")
        mp.generating_sim(regime,phys, sim ) 


        log.info("All static batch runs finished.")


if __name__ == "__main__":
    log.info("Starting static batch run: cube + plane wave only")
    main()
