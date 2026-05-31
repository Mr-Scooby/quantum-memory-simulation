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
from importlib.resources import files # For files import from package.
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Union

import numpy as np

from radpattern.physics.setup_params import SimParams
from radpattern.physics.WarmVaporExperimentalSetup import WarmVaporExp
from radpattern.physics.BECExpSetUp import BECExpParams
from radpattern.geometry.WarmVaporCloud import WarmVaporCloud
from radpattern.geometry.BECModel import BECModel

from radpattern.physics.BaseExpSetUp import ExpBaseParams
from radpattern.geometry.base_cloud import BaseCloud

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
    exp: ExpBaseParams
    sim: SimParams
    cloud: BaseCloud
    Cbeam: BeamModel
    Sbeam: BeamModel


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


def build_exp(cfg: Dict[str, Any]): # -> ExperimentalParams:
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

    atom = cfg["exp"].get("atoms", "").lower()

    if atom in {"cs133", "cs", "cesium", "cesium133"}:
        exp_cfg = dataclass_kwargs(WarmVaporExp, cfg["exp"])
        return WarmVaporExp(**exp_cfg)

    if atom in {"rb87", "rb", "rubidium", "rubidium87"}:
        exp_cfg = dataclass_kwargs(BECExpParams, cfg["exp"])
        return BECExpParams(**exp_cfg)

    raise ValueError(
        f"Unknown atom type {cfg['exp'].get('atoms')!r}. "
        "Expected 'cs133' or 'rb87'."
    )
    



def build_sim(cfg: Dict[str, Any], exp: ExpBaseParams) -> SimParams:
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


def build_cloud(
    sim: SimParams,
    exp: Union[WarmVaporExp, BECExpParams],
) -> Union[WarmVaporCloud, BECModel]:
    """
    Build the correct cloud model from the experiment type.

    Cs133 / WarmVaporExp -> WarmVaporCloud
    Rb87  / BECExpParams    -> BECModel
    """

    atom = getattr(exp, "atoms", "").lower()

    # Warm vapor Cs cloud
    if isinstance(exp, WarmVaporExp) or atom in {"cs133", "cs", "cesium", "cesium133"}:

        defaults = {
            "atoms": exp.atom,
            "sim_density": sim.sim_density,
            "Lz": exp.Lz,
            "R": sim.simulation_window_radius_w0_cutoff * exp.w0_control,
        }

        cloud_kwargs = dataclass_kwargs(WarmVaporCloud, defaults)
        return WarmVaporCloud(**cloud_kwargs)

    # BEC / cold Rb cloud
    if isinstance(exp, BECExpParams) or atom in {"rb87", "rb", "rubidium", "rubidium87"}:
        defaults = {
            "atoms": exp.atom,
            "n_sim_atoms": sim.sim_density,   # for BEC this means N_sim, not density
            "sigmas": exp.sigmas_code,
        }

        cloud_kwargs = dataclass_kwargs(BECModel, defaults)
        return BECModel(**cloud_kwargs)

    raise ValueError(
        f"Unknown cloud type for atom {getattr(exp, 'atoms', None)!r}. "
        "Expected Cs133/WarmVaporExp or Rb87/BECExpParams."
    )


def build_control_beam(
    exp: ExpBaseParams,
    cloud: BaseCloud,
) -> BeamModel:
    """
    Build BeamModel from ExperimentalParams and CloudModel.
    The beam is derived from the experimental setup.
    """

    control_offset_code = (
        np.asarray(exp.control_beam_AxisOffset_nm, dtype=float)
        * 1e-9 # To m
        / exp.ref_length # to code length 
    )
    defaults = {
        "beam_type": "gaussian_pulse",
        "w0": exp.w0_control,
        "sigma_long": exp.control_sigma_long,
        "k_in_hat": exp.control_beam_direction, 
        "k_in": exp.atom.k_control,
        "box_size": cloud.box_size,
        "pcenter_at_origin": True,
        "margin": 0,
        "pulse_center_t0":0.0 ,
        "v_front": 0, 
        "center": control_offset_code, 
        "label": "Control Beam" 
    }


    beam_kwargs = dataclass_kwargs(BeamModel, defaults)
    return BeamModel(**beam_kwargs)

def build_signal_beam(
    exp: ExpBaseParams,
    cloud: BaseCloud,
) -> BeamModel:
    """
    Build Sigbal Beam. 
    Build BeamModel from ExperimentalParams and CloudModel.
    The beam is derived from the experimental setup.
    """

    defaults = {
        "beam_type": "gaussian_pulse",
        "w0": exp.w0_signal,
        "sigma_long": exp.control_sigma_long,
        "k_in_hat": exp.signal_beam_direction, 
        "k_in": exp.atom.k_signal,
        "box_size": cloud.box_size,
        "pcenter_at_origin": True,
        "margin": 0,
        "pulse_center_t0":0.0 ,
        "v_front": 0, 
        "center": (0,0,0), # Signal is reference center 
        "label": "Signal Beam" 
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
    Cbeam = build_control_beam(exp, cloud)
    Sbeam = build_signal_beam(exp, cloud)

    return RunObjects(
        cfg=cfg,
        exp=exp,
        sim=sim,
        cloud=cloud,
        Cbeam=Cbeam,
        Sbeam=Sbeam
    )

def default_configPath(system):
    """ Provides defaults json path from defaults"""

    DEFAULT_PACKAGE = "radpattern.config.defaults"
    DEFAULT_FILES = {
        "cs133": "cs133_default.json",
        "rb87": "rb87_default.json",
    }
    system = system.lower()

    if system not in DEFAULT_FILES:
        raise ValueError(
            f"Unknown system {system!r}. Available: {list(DEFAULT_FILES.keys())}"
        )

    path = DEFAULT_FILES[system]
    return files(DEFAULT_PACKAGE).joinpath(DEFAULT_FILES[system])




def build_default_object(system = "cs133"):
    """ Generates pbjects from default file"""
    path = default_configPath(system) 
    return build_run_objects(path) 


# TESTING FUNCTIONALITY. 
if __name__ == "__main__": 

    from pathlib import Path
    # test file
    objs = build_default_object(system = "rb87")

    print(objs.exp)
    print(objs.sim)
    print(objs.cloud)
    print(objs.Cbeam)
    print(objs.Sbeam)
