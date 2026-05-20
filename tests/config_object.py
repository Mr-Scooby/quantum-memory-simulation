#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create simulation objects from a JSON config.

Input
-----
config_path : str
    Path to a JSON file with exp, sim.

Output
------
RunObjects
    Container with cfg, exp, sim, cloud, beam, grid, and setup.
"""

import json
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import numpy as np

from radpattern.physics.experimetal_setup import ExperimentalParams
from radpattern.physics.setup_params import SimParams
from radpattern.geometry.cloud_model import CloudModel
from radpattern.physics.beam import BeamModel


@dataclass
class RunObjects:
    """
    Built objects for one simulation run.

    Attributes
    ----------
    cfg : dict
        Original JSON config.
    exp : ExperimentalParams
        Experimental parameters in SI and code units.
    sim : SimParams
        Numerical simulation parameters.
    cloud : CloudModel
        Atomic cloud geometry and distribution.
    beam : BeamModel
        Input/control beam model.
    grid : AngleGrid
        Angular grid with shape (n_theta, n_phi).
    setup : SetupParams
        Metadata and run-name helper.
    """

    cfg: Dict[str, Any]
    exp: ExperimentalParams
    sim: SimParams
    cloud: CloudModel
    beam: BeamModel


def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON file.

    Input: path : str JSON file path.
    Output: dict Parsed config.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataclass_kwargs(cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only valid dataclass constructor keys.

    Input
    -----
    cls : dataclass type
        Target dataclass.
    data : dict
        Candidate keyword values.

    Output: dict  Constructor-safe keyword values.
    """

    valid = set(f.name for f in fields(cls) if f.init)
    unknown = sorted(set(data) - valid)

    if unknown:
        raise ValueError("%s got unknown keys: %s" % (cls.__name__, unknown))

    return dict(data)


def vector3(value: Any, name: str) -> np.ndarray:
    """
    Convert a length-3 list into a float array.

    Input
    -----
    value : list
        Vector values with shape (3,).
    name : str
        Name used in error messages.

    Output
    ------
    np.ndarray
        Float vector with shape (3,).
    """

    arr = np.asarray(value, dtype=float)

    if arr.shape != (3,):
        raise ValueError("%s must have shape (3,), got %s" % (name, arr.shape))

    return arr


def build_exp(cfg: Dict[str, Any]) -> ExperimentalParams:
    """
    Build ExperimentalParams from cfg['exp'].

    Input
    -----
    cfg : dict
        Full config with section exp.

    Output
    ------
    ExperimentalParams
        Experiment object with derived code units.
    """

    exp_cfg = dataclass_kwargs(ExperimentalParams, cfg["exp"])
    return ExperimentalParams(**exp_cfg)


def build_sim(cfg: Dict[str, Any], exp: ExperimentalParams) -> SimParams:
    """
    Build SimParams and inject derived timing/grid defaults.

    Input
    -----
    cfg : dict
        Full config with section sim.
    exp : ExperimentalParams
        Experiment object used for char_time and theta_max.

    Output
    ------
    SimParams
        Simulation object.
    """

    sim_cfg = dict(cfg.get("sim", {}))

    sim_cfg.setdefault("char_time", exp.char_time)
    sim_cfg.setdefault("theta_max", 10.0 * exp.forwardlobe_angular_width)

    sim_cfg = dataclass_kwargs(SimParams, sim_cfg)
    return SimParams(**sim_cfg)


def build_cloud(sim: SimParams, exp: ExperimentalParams) -> CloudModel:
    """
    Build CloudModel from cfg['cloud'].

    Input
    -----
    sim : simParams: Sim parameters
    exp : ExperimentalParams:Experiment object used for default length scales.

    Output: CloudModel
    The physical dimensions come from exp.
    The simulated atom number comes from sim.sim_density.
    """

    geometry = exp.cell_geometry
    distribution = "random"

    R = sim.simulation_window_radius_w0_cutoff * exp.w0_control
    Lz = exp.Lz

    defaults = {
        "geometry": geometry,
        "atoms" : exp.atom,
        "distribution": distribution,
        "Lz": Lz,
        "R": R,
        "sim_density": sim.sim_density,
    }

    cloud_kwargs = dataclass_kwargs(CloudModel, defaults)
    return CloudModel(**cloud_kwargs)
    

def build_beam(
    exp: ExperimentalParams,
    cloud: CloudModel,
) -> BeamModel:
    """
    Build BeamModel from ExperimentalParams and CloudModel.
    The beam is derived from the experimental setup.
    """

    defaults = {
        "beam_type": "gaussian_pulse",
        "w0": exp.w0_control,
        "sigma_long": exp.control_sigma_long,
        "k_in_hat": exp.control_beam_direction, 
        "k_in": exp.atom.k_control,
        "box_size": cloud.box_size,
        "pcenter_at_origin": True,
        "margin": [0,0,0],
        "pulse_center_t0": True,
        "v_front": 0, 
        "center": (0,0,0) 
    }


    beam_kwargs = dataclass_kwargs(BeamModel, defaults)
    return BeamModel(**beam_kwargs)



def build_run_objects(config_path: str) -> RunObjects:
    """
    Build all run objects from one JSON file.

    Input: config_path : str Path to the run config.
    Output : RunObjects: Ready-to-use objects for the runner.
    """

    cfg = load_json(config_path)
    print(cfg)

    exp = build_exp(cfg)
    sim = build_sim(cfg, exp)
    cloud = build_cloud(sim, exp)
    beam = build_beam(exp, cloud)

    return RunObjects(
        cfg=cfg,
        exp=exp,
        sim=sim,
        cloud=cloud,
        beam=beam,
    )



# TESTING FUNCTIONALITY. 
if __name__ == "__main__": 

    from pathlib import Path
    # test file
    file = Path("config_file_template.json")
    objs = build_run_objects(str(file)) 

    print(objs.exp)
    print(objs.sim)
    print(objs.cloud)
    print(objs.beam)
