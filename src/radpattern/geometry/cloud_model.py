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

@dataclass 
class AtomSpeciment: 
    name : str
    lambda_control_m : float
    delta_f_hz : float
    ref_length: float 

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
        return self.k_signal_SI - self.k_control_SI

    @property
    def lambda_sw_SI(self) -> float:
        return 2.0 * math.pi / abs(self.k_sw_SI)

    # wavevectors in chosen units
    @property
    def k_control(self) -> float:
        return self.k_control_SI * self.ref_length 

    @property
    def k_signal(self) -> float:
        return self.k_signal_SI * self.ref_length 

    @property
    def k_sw(self) -> float:
        return self.k_sw_SI * self.ref_length

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

@dataclass
class CloudModel:
    geometry: str                  # "box", "sphere", ...
    distribution: str              # "lattice", "random", "gaussian"
    atoms : AtomSpeciment          # Type of atoms and its wavelength, frequencies and such. 

    # geometry parameters
    Lx:float   = None
    Ly:float   = None
    Lz:float   = None
    R :float   = None

    # distribution parameters
    sim_density: float = 1e11  #field(init=False) 

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

    def generate_cloud(self, rng=None) -> np.ndarray:
        log.info("Constructing atom positions...  rng = %s", rng) 
        self.r_xyz =  make_positions(self, rng=rng)
        return self.r_xyz

    def update_position(self, dt ): 
        self.r_xyz = self.r_xyz + self.v_xyz * dt 

        # now check whether an atom is outside box to make reflection. 
#        if self.geometry == "cylinder":
#            self.reflect_cylinder_boundaries()
#        else:  
#            raise NotImplementedError(f"Reflection not implemented for {self.geometry}")

    def update_position_diffusive(self, dt_code, D_code, rng=None):
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

    def update_motion_phase(self):
        k_sw = self.atoms.k_sw * np.array([0,0,1])
        return  np.exp(-1j * ((self.r_xyz - self.r0_xyz) @ k_sw))

    def generate_velocity_distribution(self ):
        """ generates Velocity distibution according to Boltzman law, normalize to ref velocity == most prob speed"""
        self.v_xyz = np.random.normal(loc = 0.0, scale = 1 / np.sqrt(2), size = (self.n_atoms, 3)) 
        return self.v_xyz


    def generate_S_profile(self, w0_signal): 
        """ Generates Spin_wave profile from paper. asymetric distribution skweed to the end of the cloud"""

        z = self.r_xyz[:, 2]

        if self.Lz <= 0:
            raise ValueError("cloud.Lz must be > 0")

        z = self.r_xyz[:, 2]
        z_norm = z / (self.Lz/2)          # now in [-1, 1]
        z_norm = np.clip(z_norm, -1, 1)

        amp = np.sqrt(1 - z_norm**2)
        amp /= np.linalg.norm(amp)

        k_sw = self.atoms.k_sw * np.array([0,0,1])
        phase = np.exp(-1j * (self.r_xyz @ k_sw))

        x = self.r_xyz[:, 0]
        y = self.r_xyz[:, 1]
        r2_perp = x*x + y*y
        
        signal_mode = np.exp(-r2_perp / (w0_signal**2))

        S = amp.astype(np.complex128) * signal_mode * phase
        S /= np.sqrt( np.sum(np.abs(S)**2))
        self.S = S

        return self.S 


    ## For later. 
    #def make_velocity_distribution
        # return velocty array 
    
    # def update_position( time): 
# Ballistic motion update

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
