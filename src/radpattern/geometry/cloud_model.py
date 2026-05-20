#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Creates and stores the geometry distribution """


from dataclasses import dataclass, field, asdict
from radpattern.helpers import io
from .sampling import make_positions

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

@dataclass
class CloudModel:
    geometry: str                  # "box", "sphere", ...
    distribution: str              # "lattice", "random", "gaussian"
    atoms : AtomSpeciment          # Type of atoms and its wavelength, frequencies and such. 

    # distribution parameters
    # Sim density.
    sim_density: float   #field(init=False) 

    # geometry parameters
    Lx:float   = None
    Ly:float   = None
    Lz:float   = None
    R :float   = None

    # anisotropic Gaussian widths
    sigma_x:float  = None
    sigma_y:float  = None
    sigma_z:float  = None

    @property
    def volumen(self): 
        if self.geometry == "box": 
            log.debug("Calculating volume for geom. = box") 
            return  self.Lx * self.Ly * self.Lz
        elif self.geometry ==  "sphere":
            log.debug("Calculating volume for geom. = sphere") 
            return  (4/3) * np.pi * self.R**3 
        elif self.geometry == "cylinder": 
            log.debug("Calculating volume for geom. = cylinder") 
            return  np.pi * self.R**2 * self.Lz
        else: 
            raise ValueError(f"No volumen formula define yet for geometry= {self.geometry}\n Valid current geometries: box, sphere, cylinder")

    @property 
    def n_atoms(self):
        """ sim density. Only for diagnostic"""
        return int(self.sim_density *  self.volumen )

    @property
    def has_any_sigma(self) -> bool:
        return any(s is not None for s in (self.sigma_x, self.sigma_y, self.sigma_z))
    
    @property
    def aspect_ratio(self):
        return self.Lz/ self.Lx
    
    @property 
    def box_size(self): 
        if self.geometry =="box":
            return np.asarray([self.Lx, self.Ly, self.Lz])
        elif self.geometry == "sphere":
            #take the box to which a sphere is inside. 
            D = 2* self.R
            return np.asarray([D,D,D])
        elif self.geometry =="cylinder":
            return np.asarray([2*self.R, 2*self.R, self.Lz])

    @property
    def mean_spacing(self): 
        return 1 / (self.sim_density ** (1/3))

    def mc_density_weight(self, physical_density_code: float) -> float:
        """
        How many real atoms are represented by one simulated atom.
        physical_density_code: atoms / code_length^3
        sim_density: simulated atoms / code_length^3
        """
        if self.sim_density <= 0:
            raise ValueError("sim_density must be > 0")
        return physical_density_code / self.sim_density

    def mc_amplitude_weight(self, physical_density_code: float) -> float:
        """
        Amplitude correction for normalized single-excitation spin waves.
        """
        return np.sqrt(self.mc_density_weight(physical_density_code))

    def generate_cloud(self, rng=None) -> np.ndarray:
        log.info("Constructing atom positions...  rng = %s", rng) 
        self.r_xyz =  make_positions(self, rng=rng)
        return self.r_xyz

    def update_position(self, dt ): 
        """ Balistic motion position update, Updates r(t0 + dt) = r(t0) + vdt""" 
        self.r_xyz = self.r_xyz + self.v_xyz * dt 

    def update_position_diffusive(self, dt_code, D_code, rng=None):
        """ Difussive update position. Updates r(t0 + dt) = r(t0) + sqrt(2 D dt)* randVector """
        if rng is None:
            rng = np.random.default_rng()

        step_std = np.sqrt(2.0 * D_code * dt_code)

        dr = rng.normal(
            loc=0.0,
            scale=step_std,
            size=self.r_xyz.shape,
        )

        self.r_xyz = self.r_xyz + dr

    def cylinder_mask(self): 
        rho2 = self.r_xyz[:, 0]**2 + self.r_xyz[:, 1]**2
        return (rho2 <= 1.3* self.R**2) & (np.abs(self.r_xyz[:, 2]) <=1.5* 0.5*self.Lz)

    def update_motion_phase(
        self,
        dt_s: float,
        B0_T: float = 0.0,
        B_gradient_z_T_per_code: float = 0.0,
        ):
        """
        Update accumulated spin-wave phase from magnetic field.

        Magnetic field model:

            B_j = B0 + Gz * z_j

        where z_j is in code units.

        Phase update:

            phase_j <- phase_j * exp[-i omega_j dt]

        with

            omega_j = (mu_B / hbar) (g_s m_s - g_g m_g) B_j
        """

        if not hasattr(self, "r_xyz"):
            raise ValueError("Call generate_cloud() before update_motion_phase().")

        if not hasattr(self, "motion_phase"):
            self.motion_phase = np.ones(self.r_xyz.shape[0], dtype=np.complex128)

        z_code = self.r_xyz[:, 2]

        B_j = B0_T + B_gradient_z_T_per_code * z_code

        omega_j = self.atoms.magnetic_sensitivity_rad_s_T * B_j

        self.motion_phase *= np.exp(-1j * omega_j * dt_s)

        return self.motion_phase


    def generate_velocity_distribution(self, rng= None ):
        """ generates Velocity distibution according to Boltzman law, normalize to ref velocity == most prob speed"""
        if rng is None:
            rng = np.random.default_rng()
        self.v_xyz = rng.normal(loc = 0.0, scale = 1 / np.sqrt(2), size = (self.n_atoms, 3)) 
        return self.v_xyz


#    def generate_S_profile(self, w0_signal): 
#        """ Generates Spin_wave profile from paper. asymetric distribution skweed to the end of the cloud"""
#
#        z = self.r_xyz[:, 2]
#
#        if self.Lz <= 0:
#            raise ValueError("cloud.Lz must be > 0")
#
#        z = self.r_xyz[:, 2]
#        z_norm = z / (self.Lz/2)          # now in [-1, 1]
#        z_norm = np.clip(z_norm, -1, 1)
#
#        amp = np.sqrt(1 - z_norm**2)
#        amp /= np.linalg.norm(amp)
#
#        k_sw = self.atoms.k_sw * np.array([0,0,1])
#        phase = np.exp(-1j * (self.r_xyz @ k_sw))
#
#        x = self.r_xyz[:, 0]
#        y = self.r_xyz[:, 1]
#        r2_perp = x*x + y*y
#        
#        signal_mode = np.exp(-r2_perp / (w0_signal**2))
#
#        S = amp.astype(np.complex128) * signal_mode * phase
#        S /= np.sqrt( np.sum(np.abs(S)**2))
#        self.S = S
#
#        return self.S 
#
    def gaussian_transverse_mode(self, w0_signal, center=(0, 0, 0)):
        """
        Generates the SW amplitude weight given by a gaussian beam. 
        Generates the transverse amplitude weight. Accounts for Signal Beam direction
        
                        E(r) = exp ( - r_perp^2 / w_oSignal^2)
        """

        # Signal Beam direction 
        k_hat = self.atoms.k_sw_vector / self.atoms.k_sw

        center = np.asarray(center, dtype=float)
        dr = self.r_xyz - center[None, :]

        u_par = dr @ k_hat
        u_perp2 = np.sum(dr * dr, axis=1) - u_par**2

        return np.exp(-u_perp2 / w0_signal**2)

    def generate_S_profile(
        self,
        w0_signal,
        z_span_mode="percentile",
        z_percentiles=(0.5, 99.5),
        profile="sqrt_1_minus_z2",
        retrieval_direction="+z",
        ):
        """
        Generate spin-wave profile using the actual atom cloud extent,
        not the full cell length.

        Parameters
        ----------
        w0_signal : float
            Signal beam waist in code units.

        z_span_mode : str
            "minmax"      -> use z.min(), z.max()
            "percentile"  -> use robust atom-position percentiles

        z_percentiles : tuple
            Percentiles used when z_span_mode="percentile".
            Example: (0.5, 99.5) ignores extreme outlier atoms.

        profile : str
            "sqrt_1_minus_z2"  -> symmetric old profile, but over atom cloud
            "gaussian"         -> Gaussian along actual atom positions
            "forward_sqrt"     -> approximate forward retrieval profile sqrt(z_tilde)

        retrieval_direction : str
            "+z" or "-z" for the forward_sqrt profile.
        """

        if not hasattr(self, "r_xyz"):
            raise ValueError("Call generate_cloud() before generate_S_profile().")

        z = self.r_xyz[:, 2]
        x = self.r_xyz[:, 0]
        y = self.r_xyz[:, 1]

        if z_span_mode == "minmax":
            z_min = float(np.min(z))
            z_max = float(np.max(z))
        elif z_span_mode == "percentile":
            z_min, z_max = np.percentile(z, z_percentiles)
            z_min = float(z_min)
            z_max = float(z_max)
        else:
            raise ValueError("z_span_mode must be 'minmax' or 'percentile'")

        z_center = 0.5 * (z_min + z_max)
        z_half_span = 0.5 * (z_max - z_min)

        if z_half_span <= 0:
            raise ValueError("Atom z span is zero; cannot build longitudinal profile.")

        # coordinate spanning only the atom cloud:
        # z_norm = -1 at lower atom edge, +1 at upper atom edge
        z_norm = (z - z_center) / z_half_span
        z_norm = np.clip(z_norm, -1.0, 1.0)

        if profile == "sqrt_1_minus_z2":
            amp_z = np.sqrt(np.maximum(0.0, 1.0 - z_norm**2))

        elif profile == "gaussian":
            # Choose sigma so that +/- z_half_span corresponds roughly to +/-3 sigma.
            sigma_eff = z_half_span / 3.0
            amp_z = np.exp(-0.5 * ((z - z_center) / sigma_eff)**2)

        elif profile == "forward_sqrt":
            # z_tilde in [0,1] across the actual atom cloud
            z_tilde = 0.5 * (z_norm + 1.0)

            if retrieval_direction == "+z":
                amp_z = np.sqrt(np.maximum(0.0, z_tilde))
            elif retrieval_direction == "-z":
                amp_z = np.sqrt(np.maximum(0.0, 1.0 - z_tilde))
            else:
                raise ValueError("retrieval_direction must be '+z' or '-z'")

        else:
            raise ValueError(
                "profile must be 'sqrt_1_minus_z2', 'gaussian', or 'forward_sqrt'"
            )

        # Spin-wave optical phase
        phase = np.exp(-1j * (self.r_xyz @ self.atoms.k_sw_vector))

        # Transverse signal mode
        signal_mode = self.gaussian_transverse_mode(w0_signal)

        self.S = amp_z.astype(np.complex128) * signal_mode * phase

        # Normalize after applying longitudinal profile, transverse mode, and phase
        norm = np.linalg.norm(self.S)
        if norm <= 0:
            raise ValueError("Spin wave norm is zero.")
        self.S /= norm

        return self.S

    def update_magnetic_phase(self, dt_s, B_gradient_z_T_per_code):
        """
        Accumulate magnetic spin-wave phase for one time step.

        Linear gradient only:
            B_j = Gz * z_j

        phase_B_j <- phase_B_j * exp[-i chi_B B_j dt]

        Parameters
        ----------
        dt_s : float
            Time step in seconds.

        B_gradient_z_T_per_code : float
            Magnetic-field gradient in Tesla / code_length.
        """

        if not hasattr(self, "phase_B"):
            self.phase_B = np.ones(self.n_atoms, dtype=np.complex128)

        z_code = self.r_xyz[:, 2]

        B_j = B_gradient_z_T_per_code * z_code

        omega_B_j = self.atoms.magnetic_sensitivity_rad_s_T * B_j

        self.phase_B *= np.exp(-1j * omega_B_j * dt_s)

        return self.phase_B

    def reflect_cylinder_boundaries(self):
        r = self.r_xyz
        v = self.v_xyz

        zmin = -0.5 * self.Lz
        zmax = +0.5 * self.Lz

        # --- z end caps ---
        mask_hi = r[:, 2] > zmax
        r[mask_hi, 2] = 2*zmax - r[mask_hi, 2]
        v[mask_hi, 2] *= -1

        mask_lo = r[:, 2] < zmin
        r[mask_lo, 2] = 2*zmin - r[mask_lo, 2]
        v[mask_lo, 2] *= -1

        # --- radial cylinder wall ---
        x = r[:, 0]
        y = r[:, 1]
        rho = np.sqrt(x*x + y*y)

        mask_r = rho > self.R

        if np.any(mask_r):
            # outward normal at wall
            nx = x[mask_r] / rho[mask_r]
            ny = y[mask_r] / rho[mask_r]

            # reflect position back inside
            rho_ref = 2*self.R - rho[mask_r]
            r[mask_r, 0] = rho_ref * nx
            r[mask_r, 1] = rho_ref * ny

            # reflect velocity: v' = v - 2(v·n)n
            vx = v[mask_r, 0]
            vy = v[mask_r, 1]
            v_dot_n = vx*nx + vy*ny

            v[mask_r, 0] = vx - 2*v_dot_n*nx
            v[mask_r, 1] = vy - 2*v_dot_n*ny

        self.r_xyz = r
        self.v_xyz = v

    def log_info(self):
        """ summary of the cloud being simulated. All lengths in units of wavelength."""
        log.info("====================================================")
        log.info("Cloud model summary")
        log.info("All length units below are relative to wavelength lambda")
        log.info("geometry         = %s", self.geometry)
        log.info("distribution     = %s", self.distribution)

        if self.geometry == "box":
            log.info("Lx               = %.6g lambda", self.Lx)
            log.info("Ly               = %.6g lambda", self.Ly)
            log.info("Lz               = %.6g lambda", self.Lz)
            if self.aspect_ratio is not None:
                log.info("aspect ratio Lz/Lx = %.6g", self.aspect_ratio)

        elif self.geometry == "sphere":
            log.info("R                = %.6g lambda", self.R)
            j.info("diameter         = %.6g lambda", 2 * self.R)

        elif self.geometry == "cylinder":
            log.info("R                = %.6g lambda", self.R)
            log.info("diameter         = %.6g lambda", 2 * self.R)
            log.info("Lz               = %.6g lambda", self.Lz)

        log.info("volume           = %.6g lambda^3", self.volumen)
        log.info("density          = %.6g lambda^-3", self.sim_density)
        log.info("mean spacing      = %.6g lambda", self.mean_spacing)
        log.info("n_atoms          = %.3e", self.n_atoms)

        if self.distribution == "gaussian":
            log.info("sigma_x          = %s", f"{self.sigma_x:.6g} lambda" if self.sigma_x is not None else "None")
            log.info("sigma_y          = %s", f"{self.sigma_y:.6g} lambda" if self.sigma_y is not None else "None")
            log.info("sigma_z          = %s", f"{self.sigma_z:.6g} lambda" if self.sigma_z is not None else "None")

        log.info("====================================================")



    def report_density_profile(
        self,
        r_xyz,
        cloud_fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
        probe_radius= 1 ,
    ):
        """
        Written diagnostic of the sampled atom density profile.
    
        Reports local density at probe points along:
          - x axis (transverse)
          - z axis (longitudinal)
    
        Parameters
        ----------
        r_xyz : np.ndarray, shape (N, 3)
            Sampled atom positions.
        cloud_fracs : tuple
            Fractions of the cloud extent where density is probed.
        probe_radius : float or None
            Radius of spherical probe volume. If None, use 0.5 * spacing.
        """
        r_xyz = np.asarray(r_xyz, dtype=float)
        if r_xyz.ndim != 2 or r_xyz.shape[1] != 3:
            raise ValueError(f"r_xyz must have shape (N, 3), got {r_xyz.shape}")
    
        if probe_radius is None:
            probe_radius = 0.5 * self.mean_spacing
    
        def local_density(point, radius):
            d2 = np.sum((r_xyz - point[None, :]) ** 2, axis=1)
            n_local = int(np.count_nonzero(d2 <= radius**2))
            vol = (4.0 / 3.0) * np.pi * radius**3
            rho_local = n_local / vol
            return n_local, rho_local
    
        # characteristic cloud extents
        if self.geometry in {"cylinder", "sphere"} and self.R is not None:
            R_cloud = float(self.R)
        else:
            R_cloud = 0.5 * float(self.Lx)
    
        Lz = float(self.Lz)
    
        print("\n=== Cloud density profile report ===")
        print(f"geometry = {self.geometry}")
        print(f"n_atoms = {len(r_xyz)}")
        print(f"target density = {self.sim_density:.6g} atoms / lambda^3")
        print(f"probe_radius = {probe_radius:.6g} lambda")
    
        print("\nTransverse density profile (along +x, y=z=0):")
        for f in cloud_fracs:
            x = f * R_cloud
            point = np.array([x, 0.0, 0.0], dtype=float)
            n_local, rho_local = local_density(point, probe_radius)
            print(
                f"  at {100*f:>5.1f}% of cloud radius:"
                f" x={x:>8.4f}, n_local={n_local:>4d},"
                f" rho_local={rho_local:.6g}"
            )
    
        print("\nLongitudinal density profile (along z, x=y=0):")
        for f in cloud_fracs:
            z = -0.5 * Lz + f * Lz
            point = np.array([0.0, 0.0, z], dtype=float)
            n_local, rho_local = local_density(point, probe_radius)
            print(
                f"  at {100*f:>5.1f}% of cloud length:"
                f" z={z:>8.4f}, n_local={n_local:>4d},"
                f" rho_local={rho_local:.6g}"
            )

    def __str__(self):
        lines = [f"{self.__class__.__name__}("]

        # basic config
        lines.append(f"  geometry      = {self.geometry}")
        lines.append(f"  distribution  = {self.distribution}")

        # atom info
        if self.atoms is not None:
            lines.append(f"  atoms         = {self.atoms.name}")
            lines.append(f"  lambda_ctrl   = {self.atoms.lambda_control_m:.6e} m")
            lines.append(f"  k_signal      = {self.atoms.k_signal:.6g} code units")
            lines.append(f"  k_control     = {self.atoms.k_control:.6g} code units")
            lines.append(f"  k_sw          = {self.atoms.k_sw:.6g} code units")

        # geometry
        lines.append(f"  Lx            = {self.Lx}")
        lines.append(f"  Ly            = {self.Ly}")
        lines.append(f"  Lz            = {self.Lz}")
        lines.append(f"  R             = {self.R}")

        # gaussian widths
        lines.append(f"  sigma_x       = {self.sigma_x}")
        lines.append(f"  sigma_y       = {self.sigma_y}")
        lines.append(f"  sigma_z       = {self.sigma_z}")

        # simulation density
        lines.append(f"  sim_density   = {self.sim_density}")

        try:
            lines.append(f"  volume        = {self.volumen:.6g}")
            lines.append(f"  n_atoms       = {self.n_atoms}")
            lines.append(f"  mean_spacing  = {self.mean_spacing:.6g}")
        except Exception:
            lines.append("  volume        = unavailable")
            lines.append("  n_atoms       = unavailable")

        try:
            lines.append(f"  box_size      = {self.box_size}")
        except Exception:
            lines.append("  box_size      = unavailable")

        # generated arrays, only if they exist
        if hasattr(self, "r_xyz"):
            lines.append(f"  r_xyz.shape   = {self.r_xyz.shape}")

        if hasattr(self, "v_xyz"):
            lines.append(f"  v_xyz.shape   = {self.v_xyz.shape}")

        if hasattr(self, "S"):
            lines.append(f"  S.shape       = {self.S.shape}")

        lines.append(")")
        return "\n".join(lines)
