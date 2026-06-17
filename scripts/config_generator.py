#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generates json files for the simulations. Reads the default json files in radpattern.config.default. 

allows for variable rewrite and looping over values to genarete various files. 
"""

import json
from importlib.resources import files # For files import from package. 
from pathlib import Path
from copy import deepcopy


DEFAULT_PACKAGE = "radpattern.config.defaults"

DEFAULT_FILES = {
    "cs133": "cs133_default.json",
    "ncs133": "N_Cs133_default.json",
    "rb87": "rb87_default.json",
}


def load_default_config(system):
    system = system.lower()

    if system not in DEFAULT_FILES:
        raise ValueError(
            f"Unknown system {system!r}. Available: {list(DEFAULT_FILES.keys())}"
        )

    path = DEFAULT_FILES[system]

    path = files(DEFAULT_PACKAGE).joinpath(DEFAULT_FILES[system])

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, filename, folder, preview=True):
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

def make_single_file(folder, system, filename = None):
    base_config = load_default_config(system )
    config = deepcopy(base_config)
    if filename is None: 
        filename = "config_file_template.json"
    save_config(config, filename, folder)


def dir_from_mrad_x(theta_mrad):
    """ Converts mrad into vector direction array. reference signal (0,0,1)"""
    theta = theta_mrad * 1e-3  # mrad -> rad
    return np.array([
        np.sin(theta),
        0.0,
        np.cos(theta)
    ])

def loop_over_variable(folder, system):
    base_config = load_default_config(system) 

    N_bounces_power  = [0,1,3,5,7, 10]

    for mrad in N_bounces_power:
        config = deepcopy(base_config)

        # Change only this value
        config["exp"]["coating_N_bounces"] = 10**mrad
        config["exp"]["coating_max_temp_C"] = 999
        config["exp"]["coating_label"] = "Test Coating. Artificial"


        # Also update label so you know what file is what
        config["exp"]["label"] = (
            f"{system} changing coating bouces. keeping temp same at 75C. not real coating is just testing. Coating bounces 10**{mrad}"
        )

        filename = f"{system}_Nounces1e{mrad}.json"
        filename = filename.replace("+", "")

        save_config(config, filename, folder, preview=False)


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

    folder = r"C:\Users\local_admin\radek\simulations\tests\locals_runs\queue"
    system = "NCS133" # or rb87
    # Use only ONE of these at a time:
    #make_single_file(folder, system, filename = f"{system}test_default_config.json" )
    
    loop_over_variable(folder, system)
    
    # loop_over_many_variables()
