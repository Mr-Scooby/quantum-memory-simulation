#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" data loader for plotting. 

load npz data from main file. Extracts metadata to generate exp, sim objects 
to print the info. ( fills in the missing params with absurb placeholders) 

- convert time array to proper experiimental time 
- generates grid array for plotting
"""

from pathlib import Path
from dataclasses import fields
import inspect
import numpy as np
from copy import deepcopy

from radpattern.physics.experimetal_setup import ExperimentalParams
from radpattern.physics.setup_params import SimParams


# Generates Absurb default for eassy spotting. 
# Backaward compatibility with missing variables in file. 
def default_from_type(name, typ):
    """
    Choose a backward-compatible value from the dataclass type.
    """
    # Generic type-based defaults
    if typ is float:
        return 999.9

    if typ is int:
        return 999

    if typ is str:
        return "None"

    if typ is bool:
        return False

    if typ is tuple:
        return (-1,-1,-1)

    if typ is list:
        return [-1,-1,-1]

    if typ is dict:
        return {"None": "None" }

    return None

def dataclass_kwargs(cls, data):
    """Keep only keys accepted by a dataclass constructor.
    adds None if not found (backwards compatability
    """
    kwargs = {}

    for f in fields(cls):
        if not f.init:
            continue

        if f.name in data:
            kwargs[f.name] = data[f.name]

        else:
            kwargs[f.name] = default_from_type(f.name, f.type)

#    valid = {f.name for f in fields(cls) if f.init}
#    return {
#        name: data[name] if name in data else default_from_type(f.name, f.type)
#        for name in valid
#        }

    return kwargs

def load_metadata(parent_npz_path):
    parent = np.load(parent_npz_path, allow_pickle=True)
    metadata = parent["metadata"].item()
    return metadata

def build_exp_from_metadata(metadata):
    """ reads metadata and recreates the Exp object"""
 
    try:
        exp_meta = metadata["experiment"]

    except KeyError: 
        # backward compatibility.
        exp_meta = metadata["regime"]

    # Fills missing parameters with absurbd defaults
    exp_kwargs = dataclass_kwargs(ExperimentalParams, exp_meta)
    return ExperimentalParams(**exp_kwargs)

def build_sim_from_metadata(metadata):
    sim_meta = metadata["sim"]

    # Fills missing parameters with absurbd defaults
    sim_kwargs = dataclass_kwargs(SimParams, sim_meta) 
    return SimParams(**sim_kwargs)


def load_data(path): 
    """
    loads data from path (searches for path and path.npz ). 
    Regenerates the Exp and Sim objects. 
    Prints values. 
    creates grid object for plotting. 
    updates time array to be real experimental time
    returns:
    - data dict loaded. 
    - grid object
    - exp object 
    - sim object
    """

    print(f"loading file = {path}")
    
    # Checks if file has extension .npz
    if path.suffix == ".npz": 
        # Removes extension if present
        path = path.with_suffix("") 

    # Loads data
    try:
        print(f"reading {path}") 
        metadata = load_metadata(path )
    except FileNotFoundError as e: 
        print(e) 
        # If file not found without extension adds it back and retries. 
        path = path.with_suffix(path.suffix + ".npz")
        metadata = load_metadata(path )

    #Builds exp,sim, and grid bjects. 
    exp = build_exp_from_metadata(metadata)
    sim = build_sim_from_metadata(metadata)
    grid = sim.create_grid()

    print(sim)
    print(exp)

    npz = np.load(path, allow_pickle=True)

    # Convert npz to normal mutable dict
    data = {key: npz[key] for key in npz.files}

    # Updating times to have real time values. 
    if "times_code" in data:
        data["times_us"] = data["times_code"] * sim.char_time * 1e6

    return data, grid, exp, sim
        

