#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from radpattern.physics import setup_params as stp
import multirunGaussian as gauss_run
import multirun_static as plane_run

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_case(name, runner, Lxy, Lz, spacing, illum, fill=0.20):
    return {
        "name": name,
        "runner": runner,
        "optical_size": Lxy,
        "optical_size_z": Lz,
        "optical_spacing": spacing,
        "illumination_ratio": illum,
        "filling_factor": fill,
        "k_in_hat": [0.0, 0.0, 1.0],
        "p_hat": [1.0, 0.0, 0.0],
    }


def main():
    # ---- Small screening set ----
    # Preserves:
    #   Lz/Lxy = 18.75
    #   control/signal = 2
    #   Fs ~ 0.096, Fc ~ 0.384
    spacing_C = 1.0
    spacing_D = 0.5

    Lxy_B = 100.0
    Lz_B = 18 * Lxy_B
    spacing_B = 1.5
    a_signal_B = 6.0
    a_control_B = 12.0

    cases = [
           # larger check: smaller a/Lxy, same aspect and same F
        build_case("B_signal",  "gaussian", Lxy_B, Lz_B, spacing_B, a_signal_B / Lxy_B),
        build_case("B_control", "gaussian", Lxy_B, Lz_B, spacing_B, a_control_B / Lxy_B),

#        build_case("C_signal",  "gaussian", Lxy_B, Lz_B, spacing_C, a_signal_B / Lxy_B),
#        build_case("C_control", "gaussian", Lxy_B, Lz_B, spacing_C, a_control_B / Lxy_B),
#
#        build_case("D_signal",  "gaussian", Lxy_B, Lz_B, spacing_D, a_signal_B / Lxy_B),
#        build_case("D_control", "gaussian", Lxy_B, Lz_B, spacing_D, a_control_B / Lxy_B),
    ]
    for i, case in enumerate(cases, start=1):
        log.info("===================================================")
        log.info("Running case %d / %d : %s", i, len(cases), case["name"])

        regime = stp.PhysicalRegime(
            optical_size=case["optical_size"],
            optical_size_z=case["optical_size_z"],
            optical_spacing=case["optical_spacing"],
            illumination_ratio=case["illumination_ratio"],
            filling_factor=case["filling_factor"],
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
            n_atoms=1,   # overwritten inside runner after grid is built
            n_mc=1,
            t_max_factor=1.0,
            t_char=1.0,
            n_times=1,
            n_theta=121,
            n_phi=241,
            seed=1,
            chunk_atoms=2000,
        )

        if case["runner"] == "plane":
            plane_run.generating_sim(regime, phys, sim)
        else:
            gauss_run.generating_sim(regime, phys, sim)

    log.info("All sweep runs finished.")


if __name__ == "__main__":
    main()
