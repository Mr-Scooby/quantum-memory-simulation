#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import inspect
import logging
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


