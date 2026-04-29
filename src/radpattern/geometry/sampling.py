#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Build the atoms positions array """

import numpy as np
import logging 
from dataclasses import fields

log = logging.getLogger(__name__)

def sample_axis(rng, cloud, axis)  -> np.ndarray:
    """ Samples the atom pos for one dimensional axis. random or gaussian distribution"""

    half_length = {"x": cloud.Lx/2, "y": cloud.Ly/2, "z":cloud.Lz/2} 
    sigmas = {"x": cloud.sigma_x, "y": cloud.sigma_y, "z":cloud.sigma_z} 
    
    if cloud.distribution.strip() == "random" :
        return rng.uniform(-half_length[axis], half_length[axis], size=cloud.n_atoms)
    return rng.normal(0.0, sigmas[axis], size=cloud.n_atoms)


def generate_candidates_box(cloud, rng) -> np.ndarray:
    """ generates box with atoms position. size volumen  cloud.Lx * cloud.Ly * cloud.Lz  """

    if cloud.Lx is None or cloud.Ly is None or cloud.Lz is None:
        raise ValueError("Box requires Lx, Ly, Lz")

    x = sample_axis(rng,cloud, "x")
    y = sample_axis(rng,cloud, "y")
    z = sample_axis(rng,cloud, "z")

    return np.column_stack([x, y, z])


def mask_box(xyz: np.ndarray, cloud) -> np.ndarray:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return (
        (np.abs(x) <= cloud.Lx / 2.0) &
        (np.abs(y) <= cloud.Ly / 2.0) &
        (np.abs(z) <= cloud.Lz / 2.0)
    )


def mask_sphere(xyz: np.ndarray, cloud) -> np.ndarray:
    if cloud.R is None:
        raise ValueError("Sphere requires R")
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return x*x + y*y + z*z <= cloud.R**2


def mask_cylinder(xyz: np.ndarray, cloud) -> np.ndarray:
    if cloud.R is None or cloud.Lz is None:
        raise ValueError("Cylinder requires R and Lz")
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    return (x*x + y*y <= cloud.R**2) & (np.abs(z) <= cloud.Lz / 2.0)


def sample_with_mask(cloud, mask_fn, rng) -> np.ndarray:

    """ for complex geometryes i.e not box, we create a mask and keep only atoms lying inside the mask. repeat untill we have the number of atoms that match the desnity """

    parts = []
    n_kept = 0

    while n_kept < cloud.n_atoms:
        xyz_try = generate_candidates_box(cloud, rng)
        keep = mask_fn(xyz_try, cloud)
        xyz_keep = xyz_try[keep]

        if xyz_keep.size == 0:
            continue

        parts.append(xyz_keep)
        n_kept += xyz_keep.shape[0]

    xyz = np.vstack(parts)
    return xyz[:cloud.n_atoms]


def make_positions(cloud, rng=None) -> np.ndarray:

    rng = np.random.default_rng(rng)

    if cloud.distribution not in {"random", "gaussian"}:
        raise ValueError("distribution must be 'random' or 'gaussian'")

    if cloud.distribution == "gaussian" and not cloud.has_any_sigma:
        raise ValueError(
            "distribution='gaussian' requires at least one of "
            "sigma_x, sigma_y, sigma_z"
        )

    # If distribution is "random", sigmas can still be None on all axes,
    # which means fully uniform.
    init_kwargs = {f.name: getattr(cloud, f.name) for f in fields(cloud) if f.init}
    cloud_local = type(cloud)(**init_kwargs)

    if cloud.geometry == "box":
        return sample_with_mask(cloud,mask_box, rng)

    if cloud.geometry == "sphere":
        if cloud.R is None:
            raise ValueError("Sphere requires R")
        # bounding box for sphere
        cloud_local.Lx = 2.0 * cloud.R
        cloud_local.Ly = 2.0 * cloud.R
        cloud_local.Lz = 2.0 * cloud.R
        return sample_with_mask(cloud_local, mask_sphere, rng)

    if cloud.geometry == "cylinder":
        if cloud.R is None or cloud.Lz is None:
            raise ValueError("Cylinder requires R and Lz")
        cloud_local.Lx = 2.0 * cloud.R
        cloud_local.Ly = 2.0 * cloud.R
        return sample_with_mask(cloud_local, mask_cylinder, rng)

    raise ValueError(f"Unsupported geometry: {cloud.geometry!r}")
