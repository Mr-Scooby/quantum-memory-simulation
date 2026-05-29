#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Base cloud model, Parent object """

from dataclasses import dataclass
from .AtomModel import AtomSpeciment
import numpy as np
import logging

log = logging.getLogger(__name__)


@dataclass
class BaseCloud:
    #atoms: AtomSpeciment #Stores type of atoms, wavelength, K-vectors. 

    sim_density: int # Number of atoms being simulated, Processed by the sim_density. 

    @property
    def n_atoms(self):
        return int(self.sim_density )

    @property
    def box_size(self):
        raise NotImplementedError

    def generate_cloud(self, rng=None):
        raise NotImplementedError

    def generate_velocity_distribution(self, rng= None ):
        """ generates Velocity distibution according to Boltzman law, normalize to ref velocity == most prob speed"""
        if rng is None:
            rng = np.random.default_rng()
        self.v_xyz = rng.normal(loc = 0.0, scale = 1 / np.sqrt(2), size = (self.n_atoms, 3)) 
        return self.v_xyz


    def update_position(self, dt):
        self.r_xyz = self.r_xyz + self.v_xyz * dt

    def update_position_diffusive(self, dt_code, D_code, rng=None):
        """ Difussive update position. Updates r(t0 + dt) = r(t0) + sqrt(2 D dt)* randVector """
        if rng is None:
            rng = np.random.default_rng()

        step_std = np.sqrt(2.0 * D_code * dt_code)
        # generates random vecotos displacement. 
        dr = rng.normal(0.0, step_std, size=self.r_xyz.shape)

        self.r_xyz = self.r_xyz + dr
        return self.r_xyz

    def generate_S_profile(self):
        raise NotImplementedError

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

