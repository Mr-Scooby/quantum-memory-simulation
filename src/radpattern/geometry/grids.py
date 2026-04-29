#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import numpy as np
from radpattern.helpers import helpers
from radpattern.helpers import io
import logging

log = logging.getLogger(__name__)


@dataclass
class AngleGrid:
    n_theta: int = 91
    n_phi: int = 181
    theta_max: float = np.pi 

    TH: np.ndarray = field(init=False)
    PH: np.ndarray = field(init=False)
    nx: np.ndarray = field(init=False)
    ny: np.ndarray = field(init=False)
    nz: np.ndarray = field(init=False)
    n_hat: np.ndarray = field(init=False)
    n_hat_flat: np.ndarray = field(init=False)

    def __post_init__(self):
        self.TH, self.PH, self.nx, self.ny, self.nz = helpers.make_angle_grid(
            n_theta=self.n_theta,
            n_phi=self.n_phi,
            theta_max = self.theta_max, 
        )
        self.n_hat = np.stack([self.nx, self.ny, self.nz], axis=-1)
        self.n_hat_flat = self.n_hat.reshape(-1, 3)
        
        self.log_info()
        
        
    @property
    def shape(self):
        return self.nx.shape

    @property
    def theta(self):
        return self.TH[:, 0]

    @property
    def phi(self):
        return self.PH[0, :]

    def log_info (self): 
        log.info(" Grid construnction. shape (%s, %s)", self.n_theta, self.n_phi )



