#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, asdict
import math
import numpy as np
from typing import Dict, Any, Optional
import logging 
from radpattern.geometry.cloud_model import CloudModel, AtomSpeciment

from .BaseExpSetUp import ExpBaseParams

@dataclass 
class BECExpParams(ExpBaseParams): 

    n_atoms :int  = None
    sigma : (float, float, float) = (None, None, None)   # Cloud sigma dimension in um
    

    @staticmethod
    def um_to_m(value_um):
        return value_um * 1e-6

    @property
    def sigma_m(self):
        return np.array([
            self.um_to_m(self.sigma_x_um),
            self.um_to_m(self.sigma_y_um),
            self.um_to_m(self.sigma_z_um),
        ], dtype=float)

    @property
    def sigma_code(self):
        return self.sigma_m / self.ref_length

    @property
    def sigma_x(self):
        return self.sigma_code[0]

    @property
    def sigma_y(self):
        return self.sigma_code[1]

    @property
    def sigma_z(self):
        return self.sigma_code[2]
