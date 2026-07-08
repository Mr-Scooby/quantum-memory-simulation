#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" BEC model. Creates guassian profile BEC distribution of atoms"""

from dataclasses import dataclass
from .AtomModel import AtomSpeciment
from .base_cloud import BaseCloud 
import numpy as np
import logging

log = logging.getLogger(__name__)


@dataclass
class BECModel(BaseCloud):
    """
    Localized cold/BEC cloud with anisotropic Gaussian density.

    All lengths are in code units.
    """
    # Cloud dimensions. 
    sigmas: np.asarray([ float, float, float])
    n_sim_atoms: int # Number of simulated atoms. 

    @property 
    def n_atoms(self):
        """ Number of Atoms to form the cloud"""
        return int(self.n_sim_atoms)

    def _generate_cloud_impl(self, rng = None): 
        """ generates Gaussian shape atom cloud """

        if rng is None:
            rng = np.random.default_rng()

        self.r_xyz = rng.normal(
            loc=(0,0,0), #  self.center,
            scale= self.sigmas, 
            size=(self.n_atoms, 3),
        )

        return self.r_xyz

    @property
    def sigma_x(self):
        return self.sigmas[0]

    @property
    def sigma_y(self):
        return self.sigmas[1]

    @property
    def sigma_z(self):
        return self.sigmas[2]


    @property
    def box_size(self):
        """ Effective cloud box size. twice cutoff cloud dimension. we set for 3* sigmas as cutoff. thus size = 6 * sigmas """
        return 6 * self.sigmas
    


    def update_position_diffusive(self,dt_code, *args,  **kwargs):
        """ Balistic motion"""
        self.r_xyz = self.r_xyz + self.v_xyz * dt_code
        return self.r_xyz







