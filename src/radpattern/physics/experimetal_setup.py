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
    signal_beam_direction: tuple           # Signal beam direction. 
    control_fwhm_diameter_m: float   # control beam FWHM diameter [m]
    control_pulse_fwhm_ns: float #= 25  # control intensity FWHM duration [ns] e-9
    control_beam_direction: tuple          # Control beam direction.  


    cell_geometry: str #= "cylinder" 
    control_beam_AxisOffset_nm : tuple #= (0.0,0.0,0.0)   # offset of the control beam relative to teh center of signal beam. Units nm (10^-9 m )
    
    # |g> #= |F#=1, mF#=+1>
    g_g:float
    m_g:float 

    # |s> #= |F#=2, mF#=+1>
    g_s:float
    m_s:float 

    density_cm3 : float           # atoms/ cm^3 
    temperature: float #= 75+ 273.15          # Temperature in Kelvin 

    buffer_gas : str #= "N2"
    buffer_pressure_Torr : float #= 0.0          # Torr #= 1/760 atm #= 101325/760 Pa
    diffusion_D0_cm2_s: float #= 0.24
    diffusion_T0_K: float #= 273.15
    diffusion_P0_Torr: float #= 1.0

    B0_T: float 
    B_gradient: float 

    scalling: int 
    label: str 

    spin_destruction_cross_section_CsN2_m2: float
    spin_exchange_alpha_CsCs_m3_s: float 

    def __post_init__(self): 
        if self.buffer_gas is None: 
            self.buffer_pressure_Torr : float = 0.0         # Torr = 1/760 atm = 101325/760 Pa
            self.diffusion_D0_cm2_s: float    = 0.0
            self.diffusion_T0_K: float        = 0.0
            self.diffusion_P0_Torr: float     = 0.0

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
        return self.buffer_density_m3 * sigma * vrel

  
 #   @property
 #   def spin_destruction_rate_CsN2_Hz(self):
 #       sigma = self.spin_destruction_cross_section_CsN2_m2
 #       return self.buffer_density_m3 * sigma * self.mean_relative_speed(28.0)

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

    @property
    def B_gradient_z_T_per_code(self):
        """Magnetic-field gradient in code units: [T / code_length]."""
        return self.B_gradient * self.ref_length

    def __str__(self) -> str:
        """
        Human-readable report for ExperimentalParams.
        Layout:
        1. Raw input fields, exactly as stored in the dataclass.
        2. Main physical quantities in SI units.
        3. Relevant derived rates, times, and dimensionless ratios.
        4. Same geometry / beam / wave quantities in code units.
        """

        def fmt(x, precision: int = 6) -> str:
            """Format scalars, vectors, arrays, None, and strings consistently."""
            if x is None:
                return "None"

            if isinstance(x, np.ndarray):
                return np.array2string(
                    x,
                    precision=precision,
                    suppress_small=False,
                    separator=", ",
                )

            if isinstance(x, (tuple, list)):
                return "(" + ", ".join(fmt(v, precision=precision) for v in x) + ")"

            if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
                return f"{x:d}"

            if isinstance(x, (np.floating, float)):
                if not math.isfinite(float(x)):
                    return str(x)

                ax = abs(float(x))
                if ax == 0.0:
                    return "0"
                if ax < 1e-3 or ax >= 1e4:
                    return f"{x:.{precision}e}"
                return f"{x:.{precision}g}"

            return str(x)

        def safe(getter):
            """Evaluate derived quantities without letting __str__ crash."""
            try:
                return getter()
            except Exception as exc:
                return f"<error: {type(exc).__name__}: {exc}>"

        lines = []

        def title(text: str) -> None:
            lines.append("")
            lines.append("=" * 78)
            lines.append(text)
            lines.append("=" * 78)

        def section(text: str) -> None:
            lines.append("")
            lines.append(f"--- {text} ---")

        def add(name: str, value, unit: str = "", note: str = "") -> None:
            unit_text = f" [{unit}]" if unit else ""
            note_text = f"    # {note}" if note else ""
            lines.append(f"{name:<42s}: {fmt(value):>18s}{unit_text}{note_text}")

        def add_vec(name: str, value, unit: str = "", note: str = "") -> None:
            unit_text = f" [{unit}]" if unit else ""
            note_text = f"    # {note}" if note else ""
            lines.append(f"{name:<42s}: {fmt(value)}{unit_text}{note_text}")

        def div(a, b):
            try:
                return a / b
            except Exception:
                return np.nan

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        title("ExperimentalParams report")
        add("label", self.label)
        add("atoms", self.atoms)
        add("cell geometry", self.cell_geometry)

        # ------------------------------------------------------------------
        # Raw input fields
        # ------------------------------------------------------------------
        title("Raw input data")

        for field_name, field_info in self.__dataclass_fields__.items():
            if not getattr(field_info, "init", True):
                continue

            value = getattr(self, field_name)
            add_vec(field_name, value)

        # ------------------------------------------------------------------
        # SI inputs and directly converted inputs
        # ------------------------------------------------------------------
        title("Input data SI units")

        section("geometry")
        add("cell length", self.cell_length_m, "m")
        add("cell diameter", self.cell_diameter_m, "m")
        add("cell radius", self.radius_m, "m")

        section("beams")
        add("signal FWHM diameter", self.signal_fwhm_diameter_m, "m")
        add("control FWHM diameter", self.control_fwhm_diameter_m, "m")
        add("signal amplitude waist w0", self.w0_signal_m, "m")
        add("control amplitude waist w0", self.w0_control_m, "m")
        add("control pulse FWHM", self.control_pulse_fwhm_s, "s")
        add("control longitudinal sigma", self.control_sigma_long_m, "m")
        add_vec("signal beam direction", self.signal_beam_direction)
        add_vec("control beam direction", self.control_beam_direction)
        add_vec(
            "control beam axis offset",
            tuple(np.asarray(self.control_beam_AxisOffset_nm, dtype=float) * 1e-9),
            "m",
        )

        section("density and temperature")
        add("input density", self.density, "m^-3")
        add("input density", self.density_cm3, "cm^-3")
        add("mean interparticle distance", self.interparticle_distance, "m")
        add("temperature", self.temperature, "K")
        add("temperature", self.temperature - 273.15, "degC")

        section("buffer gas")
        add("buffer gas", self.buffer_gas)
        add(
            "buffer pressure",
            self.buffer_pressure_Torr * 101325.0 / 760.0,
            "Pa",
        )
        add("buffer pressure", self.buffer_pressure_Torr, "Torr")
        add("diffusion D0", self.diffusion_D0_cm2_s * 1e-4, "m^2/s")
        add("diffusion reference T0", self.diffusion_T0_K, "K")
        add(
            "diffusion reference P0",
            self.diffusion_P0_Torr * 101325.0 / 760.0,
            "Pa",
        )

        section("magnetic field")
        add("B0", self.B0_T, "T")
        add("B gradient", self.B_gradient, "T/m")

        section("Zeeman states")
        add("g_g", self.g_g)
        add("m_g", self.m_g)
        add("g_s", self.g_s)
        add("m_s", self.m_s)
        add("g_s*m_s - g_g*m_g", self.g_s * self.m_s - self.g_g * self.m_g)

        section("relaxation input constants")
        add(
            "spin destruction cross section Cs-N2",
            self.spin_destruction_cross_section_CsN2_m2,
            "m^2",
        )
        add(
            "spin exchange alpha Cs-Cs",
            self.spin_exchange_alpha_CsCs_m3_s,
            "m^3/s",
        )

        # ------------------------------------------------------------------
        # Derived SI quantities
        # ------------------------------------------------------------------
        title("Derived physical quantities in SI units")

        section("optical frequencies and wavelengths")
        add("control wavelength", self.lambda_control_m, "m")
        add("signal wavelength", safe(lambda: self.atom.lambda_signal_m), "m")
        add("spin-wave wavelength", safe(lambda: self.atom.lambda_sw_SI), "m")
        add("control frequency", safe(lambda: self.atom.f_control), "Hz")
        add("signal frequency", safe(lambda: self.atom.f_signal), "Hz")
        add("spin-wave frequency", safe(lambda: self.atom.f_sw), "Hz")
        add("input delta_f", self.delta_f_hz, "Hz")

        section("wave vectors")
        add("k_control magnitude", safe(lambda: self.atom.k_control_SI), "m^-1")
        add("k_signal magnitude", safe(lambda: self.atom.k_signal_SI), "m^-1")
        add("k_sw magnitude", safe(lambda: self.atom.k_sw_SI), "m^-1")
        add(
            "k_sw copropagating estimate",
            2.0 * np.pi * self.delta_f_hz / C,
            "m^-1",
        )
        add_vec("k_sw vector", safe(lambda: self.atom.k_sw_SI_vector), "m^-1")
        add_vec(
            "k_sw direction",
            safe(lambda: self.atom.k_sw_SI_vector / np.linalg.norm(self.atom.k_sw_SI_vector)),
        )

        section("atomic motion")
        add("atomic mass", safe(lambda: self.atom.mass), "amu")
        add("most probable speed", self.probable_speed, "m/s")
        add("mean thermal speed", self.mean_speed, "m/s")
        add("characteristic time", self.char_time, "s")

        section("diffusion and densities")
        add("diffusion coefficient D", self.diffusion_coeff_SI, "m^2/s")
        add("Cs vapor pressure", self.cs_vapor_pressure_Torr, "Torr")
        add("Cs vapor density", self.cs_density_m3, "m^-3")
        add(f"{self.buffer_gas} buffer density", self.buffer_density_m3, "m^-3")
        add(
            f"Cs-{self.buffer_gas} mean relative speed",
            self.mean_relative_speed(28.0),
            "m/s",
        )

        section("broadening / relaxation / dephasing")
        add("pressure broadening", self.pressure_broadening_signal_Hz, "Hz")
        add("pressure broadening", self.pressure_broadening_signal_Hz / 1e6, "MHz")
        add("Cs-Cs spin exchange rate", self.spin_exchange_rate_CsCs_Hz, "Hz")
        add(f"Cs-{self.buffer_gas} spin destruction rate", self.spin_destruction_rate_CsN2_Hz, "Hz")
        add("ballistic transit HWHM", self.ballistic_transit_rate_Hz, "Hz")
        add("diffusive transit HWHM", self.diffusive_transit_rate_Hz, "Hz")
        add("chosen transit HWHM", self.transit_time_rate_Hz, "Hz")
        add("transit dephasing time", self.transit_time_s, "s")
        add("transit dephasing time", self.transit_time_s * 1e6, "us")
        add(
            "diffusive beam time w_control^2/(4D)",
            safe(lambda: self.w0_control_m**2 / (4.0 * self.diffusion_coeff_SI)),
            "s",
        )
        add(
            "spin-wave diffusion dephasing time",
            safe(lambda: 1.0 / (self.diffusion_coeff_SI * self.atom.k_sw_SI**2)),
            "s",
        )

        section("spin-wave and emission geometry")
        add("forward lobe angular width", self.forwardlobe_angular_width, "rad")
        add("|k_sw| * Lz", self.kz_phase_accumulation, "rad")
        add("spin-wave periods across Lz", self.sw_periods_across_cell)

        # ------------------------------------------------------------------
        # Dimensionless ratios
        # ------------------------------------------------------------------
        title("Useful dimensionless ratios")

        add("cell aspect ratio Lz/D", self.aspect_ratio_full_cell)
        add("signal/control waist ratio", self.signal_to_control_waist_ratio)
        add("control/signal waist ratio", self.control_to_signal_waist_ratio)
        add("signal w0 / cell diameter", self.signal_illumination_ratio_full_cell)
        add("control w0 / cell diameter", self.control_illumination_ratio_full_cell)
        add("w0_signal / mean spacing", div(self.w0_signal_m, self.interparticle_distance))
        add("w0_control / mean spacing", div(self.w0_control_m, self.interparticle_distance))
        add("cell length / mean spacing", div(self.cell_length_m, self.interparticle_distance))
        add("ballistic transverse time w0_control/v_mean", div(self.w0_control_m, self.mean_speed), "s")
        add("ballistic axial time Lz/v_mean", div(self.cell_length_m, self.mean_speed), "s")

        # ------------------------------------------------------------------
        # Code units
        # ------------------------------------------------------------------
        title("Computational scaling and code units")

        section("base units")
        add("scalling", self.scalling)
        add("1 code length", self.ref_length, "m")
        add("1 code time", self.char_time, "s")
        add("1 code velocity", self.ref_length / self.char_time, "m/s")
        add("1 code diffusion", self.ref_length**2 / self.char_time, "m^2/s")

        section("geometry in code length units")
        add("Lz", self.Lz)
        add("D", self.D)
        add("R", self.R)

        section("beams in code length units")
        add("signal FWHM diameter", self.signal_fwhm_diameter_m / self.ref_length)
        add("control FWHM diameter", self.control_fwhm_diameter_m / self.ref_length)
        add("signal w0", self.w0_signal)
        add("control w0", self.w0_control)
        add("control longitudinal sigma", self.control_sigma_long)
        add_vec(
            "control beam axis offset",
            tuple(np.asarray(self.control_beam_AxisOffset_nm, dtype=float) * 1e-9 / self.ref_length),
        )

        section("density and spacing in code units")
        add("density", self.density_rescalled, "code_length^-3")
        add("mean interparticle spacing", self.a_spacing_reescaled)

        section("waves in code units")
        add("lambda_control", self.lambda_control_m / self.ref_length)
        add("lambda_signal", safe(lambda: self.atom.lambda_signal))
        add("lambda_sw", safe(lambda: self.atom.lambda_sw))
        add("k_control", safe(lambda: self.atom.k_control))
        add("k_signal", safe(lambda: self.atom.k_signal))
        add("k_sw", safe(lambda: self.atom.k_sw))
        add_vec(
            "k_sw vector",
            safe(lambda: self.atom.k_sw_SI_vector * self.ref_length),
            "code_length^-1",
        )

        section("transport and magnetic field in code units")
        add("diffusion coefficient", self.diffusion_coeff_code)
        add("B gradient", safe(lambda: self.B_gradient_z_T_per_code), "T/code_length")

        lines.append("")
        lines.append("=" * 78)

        return "\n".join(lines)



