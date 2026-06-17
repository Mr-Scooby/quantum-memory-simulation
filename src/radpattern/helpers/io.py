#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import inspect
import logging
from dataclasses import fields
#from radpattern.geometry.grids import AngleGrid
#from radpattern.physics.experimetal_setup import ExperimentalParams
log = logging.getLogger(__name__)


def save_simulation_npz(path, **data):
    np.savez(path, **data)
    log.info("Saving simulation run. FileName = %s", path) 


def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def fmt_attr(value):
    if hasattr(value, "shape"):
        return f"shape={value.shape}"
    return str(value)


def log_attrs(logger, obj, names, prefix=""):
    msg = " | ".join(f"{name}={fmt_attr(getattr(obj, name))}" for name in names)
    logger.info("%s%s", prefix, msg)


def dataclass_kwargs(cls, data):
    """Keep only keys accepted by a dataclass constructor."""
    valid = {f.name for f in fields(cls) if f.init}
    return {k: v for k, v in data.items() if k in valid}


def load_metadata(parent_npz_path):
    parent = np.load(parent_npz_path, allow_pickle=True)
    metadata = parent["metadata"].item()
    return metadata


def build_grid_from_metadata(metadata):
    sim_meta = metadata["sim"]

    n_theta = sim_meta["n_theta"]
    n_phi = sim_meta["n_phi"]
    theta_max = sim_meta["theta_max"]

    return AngleGrid(
        n_theta=n_theta,
        n_phi=n_phi,
        theta_max=theta_max,
    )


def build_exp_from_metadata(metadata):
    exp_meta = metadata["experiment"]
    exp_kwargs = dataclass_kwargs(ExperimentalParams, exp_meta)
    return ExperimentalParams(**exp_kwargs)


RUN_NAME_RE = re.compile(
    r"^"
    r"(?P<atoms>[A-Za-z]+\d+)"
    r"(?P<signal_dia_um>\d+)SDia_"
    r"(?P<control_dia_um>\d+)Cdia"
    r"_simT(?P<sim_time_us>[\d.]+)us"
    r"_nt(?P<time_divisions>\d+)"
    r"_(?P<n_mc>\d+)runs"
    r"(?:_(?P<buffer_pressure_Torr>[\d.]+)Torr)?"
    r"_(?P<hash>[0-9a-fA-F]{8})"
    r"$"
)


def parse_run_filename(file_name: str) -> dict:
    """
    Parse filenames produced by SimMetadataSetUp.run_name().

    Expected examples:
        Rb87120SDia_210Cdia_simT1000us_nt100_50runs_ab12cd34.npz
        Cs133120SDia_210Cdia_simT10us_nt100_50runs_5Torr_ab12cd34.npz
    """
    stem = Path(file_name).stem
    match = RUN_NAME_RE.match(stem)

    if match is None:
        raise ValueError(f"Could not parse simulation filename: {file_name}")

    info = match.groupdict()

    info["signal_dia_um"] = int(info["signal_dia_um"])
    info["control_dia_um"] = int(info["control_dia_um"])
    info["sim_time_us"] = float(info["sim_time_us"])
    info["time_divisions"] = int(info["time_divisions"])
    info["n_mc"] = int(info["n_mc"])

    if info["buffer_pressure_Torr"] is not None:
        info["buffer_pressure_Torr"] = float(info["buffer_pressure_Torr"])

    return info

