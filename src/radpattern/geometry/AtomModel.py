#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Defines and sets the characteristics of the atom specimen use in the 
simalation. 

its main frequencies and k-vectors both in SI unist and code units """

from dataclasses import dataclass

import numpy as np
import math
import logging 

log = logging.getLogger(__name__) 

C = 299_792_458.0  # m/s
MU_B = 9.2740100783e-24
HBAR = 1.054571817e-34

@dataclass 
class AtomSpeciment: 
    name : str
    lambda_control_m : float
    delta_f_hz : float
    k_sw_SI_vector: tuple
    ref_length: float 

    g_g: float 
    m_g: float 
    g_s: float 
    m_s: float 


    # basic optical quantities
    @property
    def f_control(self):
        return C / self.lambda_control_m

    @property
    def f_signal(self):
        """ signal beam frequency"""
        return self.f_control + self.delta_f_hz

    @property
    def lambda_signal_m(self):
        """ signal beam wavelength"""
        return C / self.f_signal

    @property
    def k_control_SI(self):
        return 2.0 * math.pi / self.lambda_control_m

    @property
    def k_signal_SI(self):
        return 2.0 * math.pi / self.lambda_signal_m

    @property
    def k_sw_SI(self):
        "Get the k value of sw "
        return np.linalg.norm(self.k_sw_SI_vector)

    @property
    def lambda_sw_SI(self) -> float:
        return 2.0 * math.pi / self.k_sw_SI

    @property 
    def f_sw (self): 
        return C/ self.lambda_sw_SI

    # wavevectors in chosen units
    @property
    def k_control(self) -> float:
        return self.k_control_SI * self.ref_length 

    @property
    def k_signal(self) -> float:
        return self.k_signal_SI * self.ref_length 

    @property
    def k_sw(self) -> float:
        "Convert |k| of sw to code units"  
        return np.linalg.norm(self.k_sw_SI) * self.ref_length
    
    @property 
    def k_sw_vector(self) -> np.array: 
        "Convert vect k_sw to code units" 
        return self.k_sw_SI_vector * self.ref_length

    @property
    def lambda_signal(self) -> float:
        return self.lambda_signal_m / self.ref_length

    @property
    def lambda_sw(self) -> float:
        return self.lambda_sw_SI / self.ref_length

    @property 
    def mass(self): 
        masses = {"Cs133":132.90, "Rb87":86.90 }
        return masses[self.name]

    @property 
    def magnetic_sensitivity_rad_s_T(self):
        try: 
            return MU_B / HBAR * (self.g_s * self.m_s - self.g_g * self.m_g)
        except ZeroDivisionError:
            return 0


