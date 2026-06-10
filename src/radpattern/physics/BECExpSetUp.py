#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, asdict
import math
import numpy as np
from typing import Dict, Any, Optional
import logging 
from radpattern.geometry.AtomModel import  AtomSpeciment

from .BaseExpSetUp import ExpBaseParams

log = logging.getLogger(__name__)

@dataclass 
class BECExpParams(ExpBaseParams): 

    n_real_atoms :int # Experimental atoms in cloud ( real )
    sigmas : (float, float, float) # Cloud sigma dimension in um
    

    @property
    def density(self): 
        return self.n_real_atoms
    @staticmethod
    def um_to_m(value_um):
        return value_um * 1e-6

    @property
    def sigmas_m(self):
        return np.asarray( [ self.um_to_m(sigma) for sigma in self.sigmas ] ) 

    @property
    def sigmas_code(self):
        return np.asarray([ sigma_m / self.ref_length for sigma_m in self.sigmas_m]) 

    @property
    def sigma_x(self):
        return self.sigmas_code[0]

    @property
    def sigma_y(self):
        return self.sigmas_code[1]

    @property
    def sigma_z(self):
        return self.sigmas_code[2]

    @property 
    def peak_density(self): 
        return self.n_atoms / ((2*np.pi)**1.5 * self.sigma_x * self.sigma_y * self.sigma_z)

    @property
    def effective_density(self): 
        return self.n_atoms / ((4/3)*np.pi*(3*self.sigma_x)*(3*self.sigma_y)*(3*self.sigma_z))

    @property 
    def diffusion_coeff_SI(self): 
        """
        Buffer-gas diffusion estimate [m^2/s].
        """
        log.warning("No buffer gas. Diffusion coeff = 1/3 * mfp * v_average""") 
        return self.interparticle_distance * self.mean_speed /3 



    def __str__(self):
        base = super().__str__()
        base = base[:-1] if base.endswith(")") else base

        lines = [base]

        lines.append("  --- BEC / cold cloud ---")
        lines.append(f"  n_atoms                       = {self.n_real_atoms}")
        lines.append(f"  sigma_x_um                    = {self.sigmas[0]:.6g}")
        lines.append(f"  sigma_y_um                    = {self.sigmas[1]:.6g}")
        lines.append(f"  sigma_z_um                    = {self.sigmas[2]:.6g}")

        lines.append("  --- BEC cloud size ---")
        lines.append(f"  sigma_m                       = {self.sigmas_m}")
        lines.append(f"  sigma_x                       = {self.sigma_x:.6g} code units")
        lines.append(f"  sigma_y                       = {self.sigma_y:.6g} code units")
        lines.append(f"  sigma_z                       = {self.sigma_z:.6g} code units")

        if hasattr(self, "n_sigma_cutoff"):
            lines.append(f"  n_sigma_cutoff                = {self.n_sigma_cutoff:.6g}")
            lines.append(
                f"  effective_box_size_code       = "
                f"{2.0 * self.n_sigma_cutoff * self.sigma_code}"
            )

        lines.append(")")

        return "\n".join(lines)
