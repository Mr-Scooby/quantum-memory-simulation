#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .BaseExpSetUp import BaseExpSetUp
from dataclasses import dataclass, asdict
import math
import numpy as np
from typing import Dict, Any, Optional
import logging 

@dataclass
class WarmVaporExp(BaseExpSetUp): 
    cell_length_m: float             # cylinder length [m]
    cell_diameter_m: float           # cylinder diameter [m]

    cell_geometry: str #= "cylinder" 

    density_cm3 : float           # atoms/ cm^3 

    buffer_gas : str #= "N2"
    buffer_pressure_Torr : float #= 0.0          # Torr #= 1/760 atm #= 101325/760 Pa
    diffusion_D0_cm2_s: float #= 0.24
    diffusion_T0_K: float #= 273.15
    diffusion_P0_Torr: float #= 1.0

    spin_destruction_cross_section_CsN2_m2: float
    spin_exchange_alpha_CsCs_m3_s: float 


    @property
    def radius_m(self) -> float:
        return 0.5 * self.cell_diameter_m
    # geometry in chosen units
    @property
    def Lz(self):
        return self.cell_length_m / self.ref_length

    @property
    def D(self):
        return self.cell_diameter_m / self.ref_length

    @property
    def R(self) -> float:
        return self.radius_m / self.ref_length

    @property 
    def diffusion_coeff_SI(self): 
        """
        Buffer-gas diffusion estimate [m^2/s].

        Uses empirical scaling:
            D(T,P) = D0 * (P0/P) * sqrt(T/T0)

        D0 sets the buffer-gas reference value at T0, P0.
        
        Note. if buffer_gas == None => D = mfp * v_average / 3
        """
        if self.buffer_gas is None or float(self.buffer_pressure_Torr) <= 0.0: 
            log.info("No buffer gas. Diffusion coeff = 1/3 * mfp * v_average""") 
            return self.interparticle_distance * self.mean_speed /3 

        else: 
            # Computes Diffusion constante empirical. 
            # Avoids ZeroDivisionError. 
            P = max(float(self.buffer_pressure_Torr), 1e-30)
            D0 = self.diffusion_D0_cm2_s * 1e-4 # Converts cm^2/s -> m^2/s 
            return D0 * (self.diffusion_P0_Torr / P) * math.sqrt(
                self.temperature / self.diffusion_T0_K
            )

    @property
    def diffusion_coeff_code(self):
        """
        Convert D from SI to code units:
            D_code = D_si * time_unit / length_unit^2
        """
        return self.diffusion_coeff_SI * self.char_time / (self.ref_length ** 2)


    @property
    def cs_vapor_pressure_Torr(self):
        """Approx the  Cs atoms vapor pressure [Torr]. Check constants against Steck."""
        return 10 ** (2.881 + 4.165 - 3830.0 / self.temperature)
    
    @property
    def cs_density_m3(self):
        P_pa = self.cs_vapor_pressure_Torr * 101325.0 / 760.0
        return P_pa / (KB * self.temperature)
    
    @property
    def buffer_density_m3(self):
        P_pa = self.buffer_pressure_Torr * 101325.0 / 760.0
        return P_pa / (KB * self.temperature)
    
    def mean_relative_speed(self, mass2_amu):
        m1 = self.atom.mass * AMU
        m2 = mass2_amu * AMU
        mu = m1 * m2 / (m1 + m2)
        return math.sqrt(8 * KB * self.temperature / (math.pi * mu))
    
    @property
    def pressure_broadening_signal_Hz(self):
        gamma_MHz_per_Torr = 19.18
        return gamma_MHz_per_Torr * 1e6 * self.buffer_pressure_Torr
    
    @property
    def spin_exchange_rate_CsCs_Hz(self):
        alpha_cm3_s = 6.5e-10
        return alpha_cm3_s * (self.cs_density_m3 / 1e6)
    
    @property
    def spin_destruction_rate_CsN2_Hz(self):
        sigma = 2.9e-26  # m^2
        vrel = self.mean_relative_speed(mass2_amu=28.0)
        return self.buffer_density_m3 * sigma * vrel

     @property
    def spin_exchange_rate_CsCs_Hz(self):
        alpha_m3_s = 6.5e-16
        return alpha_m3_s * self.cs_density_m3

    @property
    def diffusive_transit_rate_Hz(self):
        """HWHM transit rate for diffusive motion through signal beam. Gamma"""
        return self.diffusion_coeff_SI / (self.w0_signal_m / math.log(2.0))**2

    @property
    def transit_time_rate_Hz(self):
        """Use slower escape mechanism as rough estimate."""
        return min(self.ballistic_transit_rate_Hz, self.diffusive_transit_rate_Hz)

    # useful ratios
    @property
    def aspect_ratio_full_cell(self):
        return self.Lz / self.D

    @property
    def signal_illumination_ratio_full_cell(self) -> float:
        return self.w0_signal / self.D

    @property
    def control_illumination_ratio_full_cell(self) -> float:
        return self.w0_control / self.D

    @property
    def kz_phase_accumulation(self) -> float:
        """Total spin-wave phase accumulation across full cell."""
        return abs(self.atom.k_sw) * self.Lz

    @property
    def sw_periods_across_cell(self) -> float:
        return self.kz_phase_accumulation / (2.0 * math.pi)


