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



# A set
Lxy_A = 5.0
Lz_A = 93.75
spacing_A = 1.5
a_signal_A = 3.0
a_control_A = 6.0

# B set
Lxy_B = 10.0
Lz_B = 187.5
spacing_B = 2.0
a_signal_B = 3.0
a_control_B = 6.0


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



cases = [
    build_case("A_plane",   "plane",    Lxy_A, Lz_A, spacing_A, 1.0),
    build_case("A_signal",  "gaussian", Lxy_A, Lz_A, spacing_A, a_signal_A / Lxy_A),
    build_case("A_control", "gaussian", Lxy_A, Lz_A, spacing_A, a_control_A / Lxy_A),

    build_case("B_plane",   "plane",    Lxy_B, Lz_B, spacing_B, 1.0),
    build_case("B_signal",  "gaussian", Lxy_B, Lz_B, spacing_B, a_signal_B / Lxy_B),
    build_case("B_control", "gaussian", Lxy_B, Lz_B, spacing_B, a_control_B / Lxy_B),
]


def print_case_ratios(case):
    Lxy = case["Lxy"]
    Lz  = case["Lz"]
    a_over_Lxy = case["a_over_Lxy"]

    print(f"\n=== {case['name']} ===")
    print(f"Lz / Lxy        = {Lz / Lxy:.5f}")
    print(f"a / Lxy         = {a_over_Lxy:.5f}")


def compare_signal_control(signal_case, control_case):
    ratio = control_case["a_over_Lxy"] / signal_case["a_over_Lxy"]
    print(f"\n=== {signal_case['name']} vs {control_case['name']} ===")
    print(f"a_control / a_signal = {ratio:.5f}")


# ---- run ----
for c in cases:
    print_case_ratios(c)

# pair comparisons
compare_signal_control(cases[1], cases[2])  # A_signal vs A_control
compare_signal_control(cases[4], cases[5])  # B_signal vs B_control
