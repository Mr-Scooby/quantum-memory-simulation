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


    def normalize_coordinates(self, coordinate, span_mode="percentile",
                              percentiles=(0.5, 99.5)):

        u = coordinate
        if span_mode == "minmax":
            u_min = float(np.min(u))
            u_max = float(np.max(u))

        elif span_mode == "percentile":
            u_min, u_max = np.percentile(u, percentiles)
            u_min = float(u_min)
            u_max = float(u_max)

        else:
            raise ValueError("span_mode must be 'minmax' or 'percentile'")

        u_center = 0.5 * (u_min + u_max)
        u_half = 0.5 * (u_max - u_min)

        if u_half <= 0:
            raise ValueError("Cloud span along signal beam is zero.")

        u_norm = (u - u_center) / u_half
        u_norm = np.clip(u_norm, -1.0, 1.0)

        u_tilde = 0.5 * (u_norm + 1.0)

        return u_norm, u_tilde


    def spinwave_protocol_profile(
        self,
        signal_beam,
        profile="sqrt_1_minus_u2",
        span_mode="percentile",
        percentiles=(0.5, 99.5),
        retrieval_direction="+signal",
    ):
        """
        Protocol-dependent spin-wave amplitude profile.

        This is not the optical beam envelope.
        It is the target memory profile along the signal direction.
        """
        # obtaiain the paralel coordinates in the direction of teh beam
        dr = self.r_xyz - signal_beam.center[None, :]
        par_coordinate = dr @ signal_beam.k_in_hat

        u_norm, u_tilde = self.normalize_coordinates(
            coordinate = par_coordinate, 
            span_mode=span_mode,
            percentiles=percentiles,
        )

        if profile in {"flat", "uniform", None}:
            return np.ones_like(u_norm)

        if profile == "sqrt_1_minus_u2":
            return np.sqrt(np.maximum(0.0, 1.0 - u_norm**2))

        if profile in {"forward_sqrt", "gorshkov_forward_high_d"}:
            if retrieval_direction in {"+signal", "+", "forward"}:
                return np.sqrt(np.maximum(0.0, u_tilde))

            if retrieval_direction in {"-signal", "-", "backward"}:
                return np.sqrt(np.maximum(0.0, 1.0 - u_tilde))

            raise ValueError("retrieval_direction must be '+signal' or '-signal'")

        if profile == "gaussian_longitudinal":
            return np.exp(-0.5 * u_norm**2)

        raise ValueError(f"Unknown spin-wave profile: {profile!r}")

    def generate_S_profile(
            self,
            signal_beam,
            control_beam,
            profile="sqrt_1_minus_u2",
            span_mode="percentile",
            percentiles=(0.5, 99.5),
            retrieval_direction="+signal",
        ):
        """
        Generate normalized spin-wave profile from signal/control beams.

        Uses:

            S_j ∝ A_protocol(j)
                  E_signal(r_j)
                  conj(E_control(r_j))

        Since BeamModel uses:

            E(r) ∝ exp(-i k · r)

        the product gives:

            E_signal * conj(E_control)
            ∝ exp[-i (k_signal - k_control) · r]

        which is the spin-wave phase.
        """
        if not hasattr(self, "r_xyz"):
            raise ValueError("Call generate_cloud() before generate_S_profile().")

        # Beam amplitudes and phase
        optical_factor = signal_beam.w * np.conj(control_beam.w)

        # Protocol profile along signal direction
        amp_protocol = self.spinwave_protocol_profile(
            signal_beam=signal_beam,
            profile=profile,
            span_mode=span_mode,
            percentiles=percentiles,
            retrieval_direction=retrieval_direction,
        )

        # Build and normalize spin wave
        S_raw = amp_protocol.astype(np.complex128) * optical_factor

        norm = np.linalg.norm(S_raw)

        if norm <= 0:
            raise ValueError("Spin wave norm is zero.")

        self.S = S_raw / norm

        return self.S


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

