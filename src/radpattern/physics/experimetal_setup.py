#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, asdict
import math
from typing import Dict, Any, Optional
import logging 
from radpattern.geometry.cloud_model import CloudModel, AtomSpeciment


log = logging.getLogger(__name__)



C = 299_792_458.0  # m/s
KB = 1.380649*10**(-23) # J / K
AMU = 1.66053906660 * 10**(-27) # Kg 


@dataclass
class ExperimentalParams:
    """
    Raw experimental inputs in SI units.

    Notes
    -----
    - beam diameters are assumed to be FWHM diameters unless otherwise stated
    - lambda_control is the optical wavelength used as the default physical reference
    """
    atoms : str 
    lambda_control_m: float          # control wavelength [m]
    delta_f_hz: float                # hyperfine / optical offset [Hz]

    cell_length_m: float             # cylinder length [m]
    cell_diameter_m: float           # cylinder diameter [m]

    signal_fwhm_diameter_m: float    # signal beam FWHM diameter [m]
    control_fwhm_diameter_m: float   # control beam FWHM diameter [m]

    density : float = 1e11
    temperature: float = 75+ 273.15          # Temperature in Kelvin 

    scalling: int = 1
    label: str = "experiment"

    def __post_init__(self): 
        self.atom = AtomSpeciment( self.atoms,
                                  self.lambda_control_m,
                                  self.delta_f_hz,
                                  self.ref_length, 
                                  ) 

    @property
    def radius_m(self) -> float:
        return 0.5 * self.cell_diameter_m

    # Gaussian beam conversion
    # code convention: field amp ~ exp(-r^2 / w0^2)
    # If diameter is FWHM diameter:
    #   w0 = d_FWHM / sqrt(2 ln 2)
    # ----------------------------
    @staticmethod
    def fwhm_diameter_to_w0(d_fwhm: float) -> float:
        return d_fwhm / math.sqrt(2.0 * math.log(2.0))

    @property
    def w0_signal_m(self):
        return self.fwhm_diameter_to_w0(self.signal_fwhm_diameter_m)

    @property
    def w0_control_m(self):
        return self.fwhm_diameter_to_w0(self.control_fwhm_diameter_m)

    # chosen simulation reference unit
    # 1 code unit = unit_scale_lambda * lambda_control
    @property
    def ref_length(self) -> float:
        return self.scalling * self.lambda_control_m

    @property
    def interparticle_distance(self): 
        return self.density**(-1/3)

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
    def probable_speed( self): 
        return math.sqrt(2* KB* self.temperature / ( self.atom.mass * AMU)) # Velocity in m/s
    @property
    def mean_speed (self): 
        return 2 * self.probable_speed / math.sqrt( math.pi) 

    @property 
    def diffusion_coeff_SI(self): 
        """
        Simple kinetic estimate:
            D = (1/3) lambda_mfp * v_mean [m2/ s]
        """
        # self.density**(-1.0/3.0)
        # temporarly set mean free path manually to check. 
        return (1.0 / 3.0) *self.density**(-1.0/3.0)   * self.mean_speed


    @property
    def diffusion_coeff_code(self):
        """
        Convert D from SI to code units:
            D_code = D_si * time_unit / length_unit^2
        """
        return self.diffusion_coeff_SI * self.char_time / (self.ref_length ** 2)
    @property 
    def char_time (self):
        return self.ref_length / self.mean_speed  # Char code time units. Taking as velocity reference the mean thermal velocity. 
        
    @property
    def density_rescalled(self): 
        return self.density * self.ref_length**3 
    
    @property
    def a_spacing_reescaled(self):
        return  self.density_rescalled**(-1/3)

    @property
    def w0_signal(self):
        return  self.w0_signal_m  / self.ref_length

    @property
    def w0_control(self):
        return self.w0_control_m / self.ref_length
    
    @property
    def forwardlobe_angular_width(self): 
        # Forward emission angular width from diffraction (FT of transverse mode):
        # tetha ~ 1 / (k_signal * w0_signal)
        return 1 / (exp.atom.k_signal * exp.w0_signal)

    # useful ratios
    @property
    def aspect_ratio_full_cell(self):
        return self.Lz / self.D

    @property
    def signal_to_control_waist_ratio(self) -> float:
        return self.w0_signal / self.w0_control

    @property
    def control_to_signal_waist_ratio(self) -> float:
        return self.w0_control / self.w0_signal

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

    def __str__(self) -> str:
        lines = []
        lines.append("==============================")
        lines.append("Experimental parameters")
        lines.append("==============================")
        lines.append("")
        lines.append(f"atoms type {self.atoms}")
        lines.append(f"density {self.density:.3E}")
        lines.append(f"geometry: cylinder")
        lines.append(f"dimensions: Lz {self.cell_length_m}[m], Diameter: {self.cell_diameter_m} [m]") 
        lines.append(f"Signal diameter_FWHM at center: {self.signal_fwhm_diameter_m:.3e} [m]")
        lines.append(f"control diameter_FWHM at center: {self.control_fwhm_diameter_m:.3e} [m]")
        lines.append(f" Temperature : {self.temperature} K. Most probable velocity: {self.probable_speed:.4f} m/s, mean speed: {self.mean_speed:.3f} m/s")
        lines.append(f"char time {self.char_time:.4e} [s]")
        lines.append(f"Diffusive Coeff {self.diffusion_coeff_SI:.4e} [m^2/s]")
        lines.append(f"lobe angular size 1/ ( k_s w_s0)  : {1 / (self.atom.k_signal_SI * self.signal_fwhm_diameter_m) :.4e} [rad]")
        lines.append("")
        lines.append("==============================")
        lines.append("Experimental -> computational scaling")
        lines.append("==============================")
        lines.append(f"label                       : {self.label}")
        lines.append(f"atoms                       : {self.atoms}")
        lines.append(f"1 code length unit          : {self.scalling:g} * lambda_control")
        lines.append(f"length reference [m]        : {self.ref_length:.6e}")
    
        lines.append("")
        lines.append("--- wavelengths ---")
        lines.append(f"lambda_control [m]          : {self.lambda_control_m:.6e}")
        lines.append(f"lambda_signal  [m]          : {self.atom.lambda_signal_m:.6e}")
        lines.append(f"lambda_sw      [m]          : {self.atom.lambda_sw_SI:.6e}")
    
        lines.append("")
        lines.append("--- wave numbers SI ---")
        lines.append(f"k_control [1/m]            : {self.atom.k_control_SI:.6e}")
        lines.append(f"k_signal  [1/m]            : {self.atom.k_signal_SI:.6e}")
        lines.append(f"k_sw      [1/m]            : {self.atom.k_sw_SI:.6e}")
    
        lines.append("")
        lines.append("--- geometry in code units ---")
        lines.append(f"Lz                          : {self.Lz:.6f}")
        lines.append(f"D                           : {self.D:.6f}")
        lines.append(f"R                           : {self.R:.6f}")
        lines.append(f"density                     : {self.density_rescalled:.3E}")
        lines.append(f"interspacing                 :{self.a_spacing_reescaled:.3E}")
    
        lines.append("")
        lines.append("--- beams in code units ---")
        lines.append(f"signal FWHM diameter        : {self.signal_fwhm_diameter_m / self.ref_length:.6f}")
        lines.append(f"control FWHM diameter       : {self.control_fwhm_diameter_m / self.ref_length:.6f}")
        lines.append(f"signal w0                   : {self.w0_signal:.6f}")
        lines.append(f"control w0                  : {self.w0_control:.6f}")
    
        lines.append("")
        lines.append("--- wave quantities in code units ---")
        lines.append(f"lambda_signal               : {self.atom.lambda_signal:.6f}")
        lines.append(f"lambda_sw                   : {self.atom.lambda_sw:.6f}")
        lines.append(f"k_control                   : {self.atom.k_control:.12f}")
        lines.append(f"k_signal                    : {self.atom.k_signal:.12f}")
        lines.append(f"k_sw                        : {self.atom.k_sw:.12e}")
    
        lines.append("")
        lines.append("--- ratios ---")
        lines.append(f"aspect ratio full cell      : {self.aspect_ratio_full_cell:.6f}")
        lines.append(f"signal/control waist        : {self.signal_to_control_waist_ratio:.6f}")
        lines.append(f"signal w0 / cell diameter   : {self.signal_illumination_ratio_full_cell:.6f}")
        lines.append(f"control w0 / cell diameter  : {self.control_illumination_ratio_full_cell:.6f}")
        lines.append(f"w0_control/a                : {self.w0_control / self.a_spacing_reescaled}") 
        lines.append(f"w0_signal/a                 : {self.w0_signal / self.a_spacing_reescaled}") 
        lines.append(f"Lz/a                        : {self.Lz/ self.a_spacing_reescaled}") 
        lines.append(f"t_perp = w_control / v_mean : {self.w0_control_m/ self.mean_speed :.4e} [s]") 
        lines.append(f"t_z = Lz/ V_mean            : {self.cell_length_m/ self.mean_speed :.4e} [s]") 

        lines.append("")
        lines.append("--- spin-wave across sample ---")
        lines.append(f"|k_sw| * Lz                 : {self.kz_phase_accumulation:.6f}")
        lines.append(f"spin-wave periods in Lz     : {self.sw_periods_across_cell:.6f}")
    
        lines.append("")
        lines.append("==============================")
        return "\n".join(lines)
