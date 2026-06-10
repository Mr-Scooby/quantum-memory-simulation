#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from .AtomModel import AtomSpeciment
from .base_cloud import BaseCloud 
import numpy as np
import logging
from radpattern.helpers.timing import debug_timer

log = logging.getLogger(__name__)

@dataclass
class WarmVaporCloud(BaseCloud):
    """ Uniformally distributed atoms in a Cylinder cell. """

    Lz: float
    R: float 
    sim_density: int 
    boundary_condition_apply: bool

    @property
    def n_atoms(self): 
        return int(self.sim_density * self.volume)

    @property
    def volume(self):
        """ Cylinder cell vloume """
        return np.pi * self.R**2 * self.Lz

    @property
    def box_size(self):
        return np.array([2 * self.R, 2 * self.R, self.Lz])

    @debug_timer()  
    def _generate_cloud_impl(self, rng=None):
        log.debug("Generating cloud...") 
        if rng is None:
            rng = np.random.default_rng()

        # Uniform random in cylinder
        rho = self.R * np.sqrt(rng.random(self.n_atoms))
        phi = 2 * np.pi * rng.random(self.n_atoms)
        z = rng.uniform(-self.Lz / 2, self.Lz / 2, self.n_atoms)

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        self.r_xyz = np.column_stack([x, y, z])
        log.debug("Cloud points generated. Size %s", self.r_xyz.shape) 
        return self.r_xyz

    def _reflect_radial_boundary(self, max_iter=10):
        for _ in range(max_iter):
            x = self.r_xyz[:, 0]
            y = self.r_xyz[:, 1]

            rho = np.sqrt(x**2 + y**2)
            outside = rho > self.R

            if not np.any(outside):
                log.debug("Reflection on radial boundary condition done") 
                return self.r_xyz

            n_hat = self.r_xyz[outside, :2] / rho[outside, None]

            rho_reflected = 2.0 * self.R - rho[outside]

            self.r_xyz[outside, :2] = n_hat * rho_reflected[:, None]
        raise RuntimeError(
              "Boundary reflection did not converge. "
              "Your diffusive timestep is probably too large."
              )


    def _reflect_z_boundaries(self, max_iter=10):
        z_min = -0.5 * self.Lz
        z_max =  0.5 * self.Lz

        for _ in range(max_iter):
            z = self.r_xyz[:, 2]

            above = z > z_max
            below = z < z_min

            if not (np.any(above) or np.any(below)):
                log.debug("Reflection on z boundary condition done") 
                return self.r_xyz

            self.r_xyz[above, 2] = 2.0 * z_max - self.r_xyz[above, 2]
            self.r_xyz[below, 2] = 2.0 * z_min - self.r_xyz[below, 2]

        raise RuntimeError(
              "Boundary reflection did not converge. "
              "Your diffusive timestep is probably too large."
              )

    def apply_boundary_conditions(self):
        self._reflect_z_boundaries()
        self._reflect_radial_boundary()
        log.debug("Application of boundary condition met") 
        return self.r_xyz

    def update_position_diffusive(self, **kwargs):
        """ Diffusive motion. takes reflection into account if necessary """
        super().update_position_diffusive(**kwargs)
        if self.boundary_condition_apply:
            self.apply_boundary_conditions()

        return self.r_xyz



