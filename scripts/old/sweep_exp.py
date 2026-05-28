#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from radpattern.physics import setup_params as stp
import multirunGaussian as mp

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    """
    Static sweep for the cell question, using a rectangular box.
    lambda = 1 units.
    We preserve the important ratios, not the SI numbers themselves.
    """

    # Reduced box:
    # long cell in z, narrow transverse x,y
    # chosen so it stays computable
    Lxy = 12.0
    Lz = 240.0

    # Effective beam radii in the reduced model
    # signal narrower than control
    w_signal = 2.0
    w_control = 4.0

    cases = [
        #{
        #    "name": "plane_wave_ref",
        #    "optical_size": Lxy,
        #    "optical_size_z": Lz,
        #    "optical_spacing": 3.0,
        #    "illumination_ratio": 1.0,   # ignored if plane-wave weights
        #    "k_in_hat": [0.0, 0.0, 1.0],
        #    "p_hat": [1.0, 0.0, 0.0],
        #    "weight_mode": "plane",
        #},
        {
            "name": "signal_like",
            "optical_size": Lxy,
            "optical_size_z": Lz,
            "optical_spacing": 0.1,
            "illumination_ratio": w_signal / Lxy,
            "k_in_hat": [0.0, 0.0, 1.0],
            "p_hat": [1.0, 0.0, 0.0],
            "weight_mode": "gaussian",
        },
        {
            "name": "control_like",
            "optical_size": Lxy,
            "optical_size_z": Lz,
            "optical_spacing": 0.1,
            "illumination_ratio": w_control / Lxy,
            "k_in_hat": [0.0, 0.0, 1.0],
            "p_hat": [1.0, 0.0, 0.0],
            "weight_mode": "gaussian",
        },
    ]

    for i_case, case in enumerate(cases, start=1):
        log.info("===================================================")
        log.info("Running case %d / %d : %s", i_case, len(cases), case["name"])

        regime = stp.PhysicalRegime(
            optical_size=case["optical_size"],
            optical_size_z=case["optical_size_z"],
            optical_spacing=case["optical_spacing"],
            illumination_ratio=case["illumination_ratio"],
            filling_factor=1.0,
            pulse_transit=1.0,
        )

        phys = stp.PhysicalParams(
            regime=regime,
            wavelength=1.0,
            v_front=0.0,
            v_thermal=0.0,
            k_in_hat=case["k_in_hat"],
            p_hat=case["p_hat"],
            beam_r0=0.0,
            pcenter_atOrigin=True,
        )

        sim = stp.SimParams(
            n_atoms=1,
            n_mc=1,
            t_max_factor=1.0,
            t_char=1.0,
            n_times=1,
            n_theta=120,
            n_phi=240,
            seed=1,
            chunk_atoms=2000,
        )

        mp.generating_sim(
            regime,
            phys,
            sim
        )

    log.info("All static batch runs finished.")


if __name__ == "__main__":
    log.info("Starting static batch run for beam-limited vs geometry-limited cases")
    main()
