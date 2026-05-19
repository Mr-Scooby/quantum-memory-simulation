#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from copy import deepcopy


#JSON-EQUIVALENT CONFIG
base_config = {
    "exp": {
        "atoms": "Cs133",
        "lambda_control_m": 8.95e-07,
        "delta_f_hz": 9192631770.0,
        "cell_length_m": 0.075,
        "cell_diameter_m": 0.004,
        "signal_fwhm_diameter_m": 0.00012,
        "signal_beam_direction": ( 0,0,1), 
        "control_fwhm_diameter_m": 0.00030,
        "control_pulse_fwhm_ns": 25 , 
        "control_beam_direction": (0,0,1),
        'Control_beam_AxisOffset_nm':0, 
        "cell_geometry": "cylinder",
        "density_cm3": 1000000000000.0,
        "temperature": 348.15,
        "buffer_gas": "N2",
        "buffer_pressure_Torr": 10.0,
        "diffusion_D0_cm2_s": 0.24,
        "diffusion_T0_K": 273.15,
        "diffusion_P0_Torr": 1.0,
        "B0_T": 0.0,
        "B_gradient": 0,
        "scalling": 10000,
        "label": "cs133_jutisz_75mm_4mm_120um_300um_75C_10TorrN2_0TpMGradient",
        'g_g':  -0.5018, 
        'm_g': +1 ,
        'g_s':+0.4998,
        'm_s':+1 ,
        'spin_destruction_cross_section_CsN2_m2': 2.9e-26,
        'spin_exchange_alpha_CsCs_m3_s':6.5e-16
    },

    "sim": {
        "n_mc": 100,
        "sim_time_us": 75.0,
        "time_divisions": 100,
        "time_spacing": "linspace",
        "n_theta": 91,
        "n_phi": 181,
        "simulation_window_radius_w0_cutoff": 3.0,
        "sim_density": 1_000_000,
        "chunk_atoms": 2000,
        "normalize_each_time": False,
        "plane_restricted": False,
        "seed": None,
    },
}


def save_config(config, filename, folder=".", preview=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / filename

    if preview:
        print("\nWriting JSON file:")
        print(path)
        print(json.dumps(config, indent=2))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved: {path}")
    return path


def make_single_file():
    config = deepcopy(base_config)
    filename = "config_file_template.json"
    save_config(config, filename)


def loop_over_gradient():
    gradients = [0, 1e-8, 1e-7, 1e-6]

    for grad in gradients:
        config = deepcopy(base_config)

        # Change only this value
        config["exp"]["B_gradient"] = grad

        # Also update label so you know what file is what
        config["exp"]["label"] = (
            f"cs133_jutisz_75mm_4mm_120um_300um_75C_10TorrN2_"
            f"{grad:g}TpMGradient"
        )

        filename = f"cs133_gradient_{grad:g}TpM.json"
        filename = filename.replace("+", "")

        save_config(config, filename, preview=False)


def loop_over_many_variables():
    gradients = [0, 1e-8, 1e-7]
    sim_densities = [1_000_000, 10_000_000]
    n_mcs = [20, 50, 100]

    for grad in gradients:
        for sim_density in sim_densities:
            for n_mc in n_mcs:

                config = deepcopy(base_config)

                # Values changed for this run
                config["exp"]["B_gradient"] = grad
                config["sim"]["sim_density"] = sim_density
                config["sim"]["n_mc"] = n_mc

                config["exp"]["label"] = (
                    f"cs133_jutisz_"
                    f"gradient{grad:g}_"
                    f"simD{sim_density:.0e}_"
                    f"MC{n_mc}"
                ).replace("+", "")

                filename = (
                    f"cs133_"
                    f"gradient{grad:g}_"
                    f"simD{sim_density:.0e}_"
                    f"MC{n_mc}.json"
                ).replace("+", "")

                save_config(config, filename, preview=False)


# 
if __name__ == "__main__":

    # Use only ONE of these at a time:
    make_single_file()

    # loop_over_gradient()
    # loop_over_many_variables()
