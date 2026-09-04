#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re 
from pathlib import Path
from radpattern.plotting import load_data


def find_result_files(path):
    path = Path(path)

    if path.is_file():
        return path.parent, [path.name]

    files = sorted(
        file.name
        for file in path.iterdir()
        if file.is_file() and file.suffix == ".npz"
    )

    return path, files

def parse_run_files(files):
    run_info = []

    for file in files:
        info = parse_run_filename(file)
        run_info.append(info)

    file_hashes = np.array([info["hash"] for info in run_info])

    return run_info, file_hashes

def choose_time_scale(run_info):
    atoms_set = {info["atoms"] for info in run_info}

    if len(atoms_set) != 1:
        return "us"

    atom_name = next(iter(atoms_set)).lower()

    if atom_name == "rb87":
        return "ms"

    return "us"

def load_one_result(path, file):
    data, grid, exp, sim = load_data(path / file)

    return data, grid, exp, sim

def check_run_consistency(run_info):
    atoms_set = {info["atoms"] for info in run_info}
    sim_time_set = {info["sim_time_us"] for info in run_info}
    time_division_set = {info["time_divisions"] for info in run_info}
    n_mc_set = {info["n_mc"] for info in run_info}

    if len(atoms_set) > 1:
        log.warning("Files contain different atoms: %s", sorted(atoms_set))

    if len(sim_time_set) > 1:
        log.warning("Files contain different simulation times [us]: %s", sorted(sim_time_set))

    if len(time_division_set) > 1:
        log.warning("Files contain different time divisions: %s", sorted(time_division_set))

    if len(n_mc_set) > 1:
        log.info("Files contain different MC run counts: %s", sorted(n_mc_set))

    return {
        "atoms": atoms_set,
        "sim_time_us": sim_time_set,
        "time_divisions": time_division_set,
        "n_mc": n_mc_set,
        }
