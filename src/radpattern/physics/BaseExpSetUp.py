#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, asdict
import math
import numpy as np
from typing import Dict, Any, Optional
import logging 
from radpattern.geometry.cloud_model import CloudModel, AtomSpeciment


log = logging.getLogger(__name__)



C = 299_792_458.0  # m/s
KB = 1.380649*10**(-23) # J / K
AMU = 1.66053906660 * 10**(-27) # Kg 

# Helper
def normalize_vector(v, name):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm == 0.0:
        raise ValueError(f"{name} cannot be the zero vector.")

    return v / norm


@dataclass
class ExpBaseParams:
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

    signal_fwhm_diameter_m: float    # signal beam FWHM diameter [m]
    signal_beam_direction: tuple           # Signal beam direction. 
    control_fwhm_diameter_m: float   # control beam FWHM diameter [m]
    control_pulse_fwhm_ns: float #= 25  # control intensity FWHM duration [ns] e-9
    control_beam_direction: tuple          # Control beam direction.  

    control_beam_AxisOffset_nm : tuple = (0.0,0.0,0.0)   # offset of the control beam relative to teh center of signal beam. Units nm (10^-9 m )
    
    # |g> #= |F#=1, mF#=+1>
    g_g:float=0 
    m_g:float =0 

    # |s> #= |F#=2, mF#=+1>
    g_s:float=0 
    m_s:float =0 

    temperature: float = 75+ 273.15          # Temperature in Kelvin 

    B0_T: float=0 
    B_gradient: float =0

    scalling: int =0
    label: str = "NON"

    def __post_init__(self): 

        self.control_beam_direction = normalize_vector(
            self.control_beam_direction,
            "control_beam_direction",
        )

        self.signal_beam_direction = normalize_vector(
            self.signal_beam_direction,
            "signal_beam_direction",
        )

        f_control = C / self.lambda_control_m
        f_signal = f_control + self.delta_f_hz
        lambda_signal_m = C / f_signal

        k_signal = 2.0 * np.pi / lambda_signal_m * self.signal_beam_direction
        k_control = 2.0 * np.pi / self.lambda_control_m * self.control_beam_direction


        self.atom = AtomSpeciment(name= self.atoms,
                                  lambda_control_m = self.lambda_control_m,
                                  delta_f_hz = self.delta_f_hz,
                                  k_sw_SI_vector = ( k_signal - k_control),
                                  ref_length = self.ref_length, 
                                  g_g = self.g_g , 
                                  m_g = self.m_g , 
                                  g_s = self.g_s , 
                                  m_s = self.m_s , 
                                  ) 


    @property
    def density(self): 
        raise NotImplementedError

    @property
    def interparticle_distance(self): 
        """ mean interparticle distance """
        return self.density**(-1/3)

    @staticmethod
    def fwhm_diameter_to_w0(d_fwhm: float) -> float:
        return d_fwhm / math.sqrt(2.0 * math.log(2.0))

    @property
    def w0_signal_m(self):
        return self.fwhm_diameter_to_w0(self.signal_fwhm_diameter_m)

    @property
    def w0_control_m(self):
        return self.fwhm_diameter_to_w0(self.control_fwhm_diameter_m)

    @property
    def control_pulse_fwhm_s(self):
        """ convert ns to s """ 
        return self.control_pulse_fwhm_ns * 1e-9

    @property
    def control_sigma_long_m(self) -> float:
        """
        Longitudinal Gaussian amplitude width used by BeamModel.

        BeamModel uses:
            env_long = exp(-(u_par**2) / sigma_long**2)

        If control_pulse_fwhm_s is the INTENSITY FWHM duration:
            sigma_long_m = c * tau_fwhm / sqrt(2 ln 2)
        """
        return C * self.control_pulse_fwhm_s / math.sqrt(2.0 * math.log(2.0))


    @property
    def control_sigma_long(self) -> float:
        """Longitudinal pulse width in code length units."""
        return self.control_sigma_long_m / self.ref_length

    # chosen simulation reference unit
    # 1 code unit = unit_scale_lambda * lambda_control
    @property
    def ref_length(self) -> float:
        return self.scalling * self.lambda_control_m

    @property
    def probable_speed( self): 
        return math.sqrt(2* KB* self.temperature / ( self.atom.mass * AMU)) # Velocity in m/s
    @property
    def mean_speed (self): 
        return 2 * self.probable_speed / math.sqrt( math.pi) 

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
    def ballistic_transit_rate_Hz(self):
        """HWHM transit rate for ballistic Cs atoms through signal beam."""
        return (self.mean_speed / self.w0_signal_m) * math.log(2.0)

    @property
    def transit_time_s(self):
        return 1.0 / self.transit_time_rate_Hz

    @property
    def signal_to_control_waist_ratio(self) -> float:
        return self.w0_signal / self.w0_control

    @property
    def control_to_signal_waist_ratio(self) -> float:
        return self.w0_control / self.w0_signal

    @property
    def B_gradient_z_T_per_code(self):
        """Magnetic-field gradient in code units: [T / code_length]."""
        return self.B_gradient * self.ref_length
