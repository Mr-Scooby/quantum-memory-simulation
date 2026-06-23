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
    atoms: AtomSpeciment #Stores type of atoms, wavelength, K-vectors. 

    def __post_init__(self): 
        if self.n_atoms > 300_000:
            log.warning(
                "Very large warm-vapor cloud: n_atoms=%d. GPU/CPU memory may be high.",
                self.n_atoms,
            )
        self._warned_diffusion_large_step = False
        self.stats = {
                "max_diffusion_step_std_code": 0.0,
                }

    @property
    def n_atoms(self):
       raise NotImplementedError

    @property
    def box_size(self):
        raise NotImplementedError

    @property 
    def char_size(self):
            """Returns charatectiristical size of the cell"""
        return np.min(self.box_size)


    def generate_cloud(self, rng):

        # Wrapper so it can log the memory ussage of the cloud.
        self._generate_cloud_impl(rng)
        log.info(
            "r_xyz size %s,  memory usage : %.2f MB / %.3f GB",
            self.r_xyz.shape,
            self.r_xyz.nbytes / 1024**2,
            self.r_xyz.nbytes / 1024**3,
             )
        return self.r_xyz
    
    def _generate_cloud_impl(self, rng = None):
        raise NotImplementedError

    def generate_velocity_distribution(self, rng= None ):
        """ generates Velocity distibution according to Boltzman law, normalize to ref velocity == most prob speed"""
        log.debug("Generating velocity distribution")
        if rng is None:
            rng = np.random.default_rng()
        self.v_xyz = rng.normal(loc = 0.0, scale = 1 / np.sqrt(2), size = (self.n_atoms, 3)) 
        return self.v_xyz


    def update_position(self, dt):
        self.r_xyz = self.r_xyz + self.v_xyz * dt

    def update_position_diffusive(self, dt_code, D_code, rng):
        """ Difussive update position. Updates r(t0 + dt) = r(t0) + sqrt(2 D dt)* randVector """
        log.debug("Updating atom position. diffusive motion.")
        if rng is None:
            log.debug("Update atom pos() generating rng object") 
            rng = np.random.default_rng()

        # Difusionn step size.
        step_std = np.sqrt(2.0 * D_code * dt_code)
        # Stats update. 
        self.stats["max_diffusion_step_std_code"] = max(
            self.stats["max_diffusion_step_std_code"],
            float(step_std),
        )

        # Checking size not too big 
        if step_std > 0.1 * min(self.box_size) and not self._warned_diffusion_large_step:
            log.warning(
                "Large diffusive step: step_std=%.3e code units, char. size =%.3e. "
                "Boundary reflection may be inaccurate; reduce dt.",
                step_std,
                min(self.box_size),
            )
            self._warned_diffusion_large_step = True

        # generates random vecotos displacement. 
        dr = rng.normal(0.0, step_std, size=self.r_xyz.shape)

        # Updates vector. 
        self.r_xyz = self.r_xyz + dr
        return self.r_xyz


    def normalize_coordinates(self, coordinate, span_mode="percentile",
                              percentiles=(0.5, 99.5)):

        log.debug("Normalizing coordinates...") 
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
        log.debug("SpinWave profile protocol") 
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

        # Check that the numbers of atoms with SW is significant.
        abs_S = np.abs(S_raw)
        active_frac = np.count_nonzero(abs_S > 1e-12 * abs_S.max()) / abs_S.size

        if active_frac < 0.01:
            log.warning(
                "Very small active spin-wave fraction: %.3g. "
                "Beam may miss cloud or waist/units may be wrong.",
                active_frac,
            )

        norm = np.linalg.norm(S_raw)

        if norm <= 0:
            raise ValueError("Spin wave norm is zero.")

        self.S = S_raw / norm
        log.debug("SpinWaveProfile generated")

        log.info(
            "SW  memory usage : %.2f MB / %.3f GB",
            self.S.nbytes / 1024**2,
            self.S.nbytes / 1024**3,
             )

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
        log.debug("Motion magnetic phase update")

        if not hasattr(self, "r_xyz"):
            raise ValueError("Call generate_cloud() before update_motion_phase().")

        if not hasattr(self, "motion_phase"):
            self.motion_phase = np.ones(self.r_xyz.shape[0], dtype=np.complex128)

        z_code = self.r_xyz[:, 2]

        B_j = B0_T + B_gradient_z_T_per_code * z_code

        omega_j = self.atoms.magnetic_sensitivity_rad_s_T * B_j

        # Control that phase step is not too big
        max_phase_step = np.max(np.abs(omega_j * dt_s))
        if max_phase_step > np.pi:
            log.warning(
                "Large magnetic phase step: max Δphi=%.3g rad. "
                "Time step may undersample dephasing.",
                max_phase_step,
            )

        self.motion_phase *= np.exp(-1j * omega_j * dt_s)

        return self.motion_phase

