#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .BaseExpSetUp import ExpBaseParams, KB, AMU
from dataclasses import dataclass, asdict
import math
import numpy as np
from typing import Dict, Any, Optional
import logging 
log = logging.getLogger(__name__)


@dataclass
class WarmVaporExp(ExpBaseParams): 
    cell_length_m: float             # cylinder length [m]
    cell_diameter_m: float           # cylinder diameter [m]

    cell_geometry: str 

    density_cm3 : float           # atoms/ cm^3 

    buffer_gas : str 
    buffer_pressure_Torr : float #= 0.0          # Torr #= 1/760 atm #= 101325/760 Pa
    diffusion_D0_cm2_s: float #= 0.24
    diffusion_T0_K: float #= 273.15
    diffusion_P0_Torr: float #= 1.0

    spin_destruction_cross_section_CsN2_m2: float
    spin_exchange_alpha_CsCs_m3_s: float 


    @property
    def density(self):
        """Convert density cm^-3 -> m^-3."""
        return self.density_cm3 * 1e6

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
    def spin_destruction_rate_CsN2_Hz(self):
        sigma = 2.9e-26  # m^2
        vrel = self.mean_relative_speed(mass2_amu=28.0)
        return self.buffer_density_m3 * sigma * vrel

    @property
    def spin_exchange_rate_CsCs_Hz(self):
        alpha_m3_s = self.spin_exchange_alpha_CsCs_m3_s
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

    def __str__(self):
        base = super().__str__()
        base = base[:-1] if base.endswith(")") else base

        lines = [base]

        lines.append("  --- warm vapor cell ---")
        lines.append(f"  cell_geometry                 = {self.cell_geometry}")
        lines.append(f"  cell_length_m                 = {self.cell_length_m:.6e}")
        lines.append(f"  cell_diameter_m               = {self.cell_diameter_m:.6e}")
        lines.append(f"  radius_m                      = {self.radius_m:.6e}")
        lines.append(f"  Lz                            = {self.Lz:.6g} code units")
        lines.append(f"  D                             = {self.D:.6g} code units")
        lines.append(f"  R                             = {self.R:.6g} code units")
        lines.append(f"  aspect_ratio_full_cell        = {self.aspect_ratio_full_cell:.6g}")

        lines.append("  --- warm vapor density ---")
        lines.append(f"  density_cm3                   = {self.density_cm3:.6e} cm^-3")
        lines.append(f"  density                       = {self.density:.6e} m^-3")
        lines.append(f"  cs_vapor_pressure_Torr        = {self.cs_vapor_pressure_Torr:.6e}")
        lines.append(f"  cs_density_m3                 = {self.cs_density_m3:.6e}")

        lines.append("  --- buffer gas / diffusion ---")
        lines.append(f"  buffer_gas                    = {self.buffer_gas}")
        lines.append(f"  buffer_pressure_Torr          = {self.buffer_pressure_Torr:.6g}")
        lines.append(f"  buffer_density_m3             = {self.buffer_density_m3:.6e}")
        lines.append(f"  diffusion_D0_cm2_s            = {self.diffusion_D0_cm2_s:.6e}")
        lines.append(f"  diffusion_T0_K                = {self.diffusion_T0_K:.6g}")
        lines.append(f"  diffusion_P0_Torr             = {self.diffusion_P0_Torr:.6g}")
        lines.append(f"  diffusion_coeff_SI            = {self.diffusion_coeff_SI:.6e} m^2/s")
        lines.append(f"  diffusion_coeff_code          = {self.diffusion_coeff_code:.6e}")

        lines.append("  --- warm vapor rates ---")
        lines.append(f"  pressure_broadening_signal_Hz = {self.pressure_broadening_signal_Hz:.6e}")
        lines.append(f"  spin_destruction_rate_CsN2_Hz = {self.spin_destruction_rate_CsN2_Hz:.6e}")
        lines.append(f"  spin_exchange_rate_CsCs_Hz    = {self.spin_exchange_rate_CsCs_Hz:.6e}")
        lines.append(f"  diffusive_transit_rate_Hz     = {self.diffusive_transit_rate_Hz:.6e}")
        lines.append(f"  transit_time_rate_Hz          = {self.transit_time_rate_Hz:.6e}")
        lines.append(f"  transit_time_s                = {self.transit_time_s:.6e}")

        lines.append("  --- illumination / spin wave ---")
        lines.append(f"  signal_illumination_ratio     = {self.signal_illumination_ratio_full_cell:.6g}")
        lines.append(f"  control_illumination_ratio    = {self.control_illumination_ratio_full_cell:.6g}")
        lines.append(f"  kz_phase_accumulation         = {self.kz_phase_accumulation:.6g} rad")
        lines.append(f"  sw_periods_across_cell        = {self.sw_periods_across_cell:.6g}")

        lines.append(")")

        return "\n".join(lines)

