#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from .AtomModel import AtomSpeciment
from .base_cloud import BaseCloud 
import numpy as np
import logging

log = logging.getLogger(__name__)

@dataclass
class WarmVaporCloud(BaseCloud):
    """ Uniformally distributed atoms in a Cylinder cell. """

    Lz: float
    R: float 

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

    def generate_cloud(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Uniform random in cylinder
        rho = self.R * np.sqrt(rng.random(self.n_atoms))
        phi = 2 * np.pi * rng.random(self.n_atoms)
        z = rng.uniform(-self.Lz / 2, self.Lz / 2, self.n_atoms)

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        self.r_xyz = np.column_stack([x, y, z])
        return self.r_xyz

