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

    Function:
    --------
    - Calculates physical parameters and variables of the experiment in SI units and translates it to sim_units. 
    """
    atoms : str 
    lambda_control_m: float          # control wavelength [m]
    delta_f_hz: float                # hyperfine / optical offset [Hz]

    cell_length_m: float             # cylinder length [m]
    cell_diameter_m: float           # cylinder diameter [m]

    signal_fwhm_diameter_m: float    # signal beam FWHM diameter [m]
    control_fwhm_diameter_m: float   # control beam FWHM diameter [m]

    density_cm3 : float = 1e11           # atoms/ cm^3 
    temperature: float = 75+ 273.15          # Temperature in Kelvin 

    buffer_gas : str = "N2"
    buffer_pressure_Torr : float = 5.0          # Torr = 1/760 atm = 101325/760 Pa
    diffusion_D0_cm2_s: float = 0.2
    diffusion_T0_K: float = 300.0
    diffusion_P0_Torr: float = 1.0

    scalling: int = 1
    label: str = "experiment"

    spin_destruction_cross_section_CsN2_m2: float = 2.9e-26  # verify
    spin_exchange_alpha_CsCs_m3_s: float = 6.5e-16

    def __post_init__(self): 
        self.atom = AtomSpeciment( self.atoms,
                                  self.lambda_control_m,
                                  self.delta_f_hz,
                                  self.ref_length, 
                                  ) 

    @property
    def density(self): 
        """ convert density cm^-3 -> m^-3""" 
        return self.density_cm3 * 1e6 # atoms/ m^3 
    @property
    def interparticle_distance(self): 
        """ mean interparticle distance """
        return self.density**(-1/3)

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
        Buffer-gas diffusion estimate [m^2/s].

        Uses empirical scaling:
            D(T,P) = D0 * (P0/P) * sqrt(T/T0)

        D0 sets the buffer-gas reference value at T0, P0.
        """
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
    def char_time (self):
        # Char code time units. Taking as velocity reference the mean thermal velocity. 
        return self.ref_length / self.mean_speed  
        
    @property
    def density_rescalled(self): 
        """ density in code units"""
        return self.density * self.ref_length**3 
    
    @property
    def a_spacing_reescaled(self):
        """ Mean interparticle distance in code units """ 
        return  self.density_rescalled**(-1/3)

    @property
    def w0_signal(self):
        # Waist of signal in code units
        return  self.w0_signal_m  / self.ref_length

    @property
    def w0_control(self):
        # Waist of control in code units
        return self.w0_control_m / self.ref_length
    
    @property
    def forwardlobe_angular_width(self): 
        # Forward emission angular width from diffraction (FT of transverse mode):
        # tetha ~ 1 / (k_signal * w0_signal)
        return 1 / (self.atom.k_signal * self.w0_signal)

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
        return self.buffer_density_m3 * sigma * vrel@property
  
    @property
    def spin_destruction_rate_CsN2_Hz(self):
        sigma = self.spin_destruction_cross_section_CsN2_m2
        return self.buffer_density_m3 * sigma * self.mean_relative_speed(28.0)

    @property
    def spin_exchange_rate_CsCs_Hz(self):
        alpha_m3_s = 6.5e-16
        return alpha_m3_s * self.cs_density_m3

    @property
    def ballistic_transit_rate_Hz(self):
        """HWHM transit rate for ballistic Cs atoms through signal beam."""
        return (self.mean_speed / self.w0_signal_m) * math.log(2.0)

    @property
    def diffusive_transit_rate_Hz(self):
        """HWHM transit rate for diffusive motion through signal beam. Gamma"""
        return self.diffusion_coeff_SI / (self.w0_signal_m / math.log(2.0))**2

    @property
    def transit_time_rate_Hz(self):
        """Use slower escape mechanism as rough estimate."""
        return min(self.ballistic_transit_rate_Hz, self.diffusive_transit_rate_Hz)

    @property
    def transit_time_s(self):
        return 1.0 / self.transit_time_rate_Hz


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
        lines.append(f"density {self.density:.3E} [atoms/ m^3]")
        lines.append(f"geometry: cylinder")
        lines.append(f"dimensions: Lz {self.cell_length_m}[m], Diameter: {self.cell_diameter_m} [m]") 
        lines.append(f"Signal diameter_FWHM at center: {self.signal_fwhm_diameter_m:.3e} [m]")
        lines.append(f"control diameter_FWHM at center: {self.control_fwhm_diameter_m:.3e} [m]")
        lines.append(f" Temperature : {self.temperature} K. Most probable velocity: {self.probable_speed:.4f} m/s, mean speed: {self.mean_speed:.3f} m/s")
        lines.append(f"char time {self.char_time:.4e} [s]")
        lines.append(f"lobe angular size 1/ ( k_s w_s0)  : {self.forwardlobe_angular_width:.4e} [rad]")
        lines.append("")
        lines.append("--- buffer gas / diffusion ---")
        lines.append(f"buffer gas                  : {self.buffer_gas}")
        lines.append(f"buffer pressure [Torr]      : {self.buffer_pressure_Torr:.6g}")º
        lines.append(f"cell temperature [C]        : {self.temperature - 273.15:.6g}")
        lines.append(f"diffusion D [m^2/s]         : {self.diffusion_coeff_SI:.6e}")
        lines.append(f"diffusion D0 [cm^2/s]       : {self.diffusion_D0_cm2_s:.6g}")
        lines.append(f"diffusion T0 [K]            : {self.diffusion_T0_K:.6g}")
        lines.append(f"diffusion P0 [Torr]         : {self.diffusion_P0_Torr:.6g}")
        lines.append(f"diffusion D code units      : {self.diffusion_coeff_code:.6e}")
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
        lines.append("--- vapor / buffer gas / intrinsic relaxation ---")
        lines.append(f"Cs vapor pressure [Torr]    : {self.cs_vapor_pressure_Torr:.6e}")
        lines.append(f"Cs density [m^-3]           : {self.cs_density_m3:.6e}")
        lines.append(f"{self.buffer_gas} density [m^-3]        : {self.buffer_density_m3:.6e}")
        lines.append(f"mean Cs speed [m/s]         : {self.mean_speed:.6f}")
        lines.append(f"Cs-{self.buffer_gas} v_rel [m/s]        : {self.mean_relative_speed(28.0):.6f}")
        lines.append(f"pressure broadening [Hz]    : {self.pressure_broadening_signal_Hz:.6e}")
        lines.append(f"pressure broadening [MHz]   : {self.pressure_broadening_signal_Hz / 1e6:.6f}")
        lines.append(f"Cs-Cs spin exchange [Hz]    : {self.spin_exchange_rate_CsCs_Hz:.6e}")
        lines.append(f"Cs-{self.buffer_gas} spin destruction [Hz]: {self.spin_destruction_rate_CsN2_Hz:.6e}")
        lines.append("")
        lines.append("--- transit-time broadening ---")
        lines.append(f"ballistic transit HWHM [Hz]  : {self.ballistic_transit_rate_Hz:.6e}")
        lines.append(f"ballistic transit HWHM [kHz] : {self.ballistic_transit_rate_Hz / 1e3:.6f}")
        lines.append(f"diffusive transit HWHM [Hz]  : {self.diffusive_transit_rate_Hz:.6e}")
        lines.append(f"diffusive transit HWHM [kHz] : {self.diffusive_transit_rate_Hz / 1e3:.6f}")
        lines.append(f"chosen transit HWHM [Hz]     : {self.transit_time_rate_Hz:.6e}")
        lines.append(f"transit dephasing time [us]  : {self.transit_time_s * 1e6:.6f}")
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
        lines.append(f"tau_beam = w_c0^2 / 4D       : {self.w0_control_m**2 / (4 * self.diffusion_coeff_SI ):.4e} [s]") 
        lines.append(f"sw_dephasing tau_sw = 1 / (D* k_sw^2) : {1 / (self.diffusion_coeff_SI * self.atom.k_sw_SI**2) :.4e} [s]") 

        lines.append("")
        lines.append("--- spin-wave across sample ---")
        lines.append(f"|k_sw| * Lz                 : {self.kz_phase_accumulation:.6f}")
        lines.append(f"spin-wave periods in Lz     : {self.sw_periods_across_cell:.6f}")
    
        lines.append("")
        lines.append("==============================")
        return "\n".join(lines)
