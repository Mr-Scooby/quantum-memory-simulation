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

    def generate_cloud(self, rng = None): 
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
    def box_size(self):
        """ Effective cloud box size. twice cutoff cloud dimension. we set for 3* sigmas as cutoff. thus size = 6 * sigmas """
        return 6 * self.sigmas
