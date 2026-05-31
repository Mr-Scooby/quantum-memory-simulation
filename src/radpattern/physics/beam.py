#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Calculates the weights for the atoms for a traveling pulse wave incident in the cloud """

from dataclasses import dataclass, field
import numpy as np 
import logging 


log = logging.getLogger(__name__)



@dataclass
class BeamModel: 
    """Generates Beam field amplitude and phase""" 
    # Gaussian-pulse parameters
    w0: float  # gaussian with ( transverse direction ) 
    sigma_long: float # Gaussina width ( longitudinal direction) 
    k_in: float 

    beam_type: str  # "gaussian_pulse" or "plane_wave"

    k_in_hat: np.ndarray #= field(default_factory#=lambda: np.array([0.0, 0.0, 1.0]))
    v_front: float #= 1.0
    box_size: tuple #= (1.0, 1.0, 1.0)
    margin: float #= 0.0
    pulse_center_t0: float #= 0.0
    center: tuple #= (0.0, 0.0, 0.0)
    pcenter_at_origin: bool #= False
    label: str 

    def __post_init__(self):
        #normalize vector
        self.k_in_hat = np.asarray(self.k_in_hat, dtype=float)
        self.k_in_hat /= (np.linalg.norm(self.k_in_hat) + 1e-15)

        self.center = np.asarray(self.center, dtype=float)
        self.box_size = np.asarray(self.box_size, dtype=float)

        if self.beam_type == "gaussian_pulse":
            if self.w0 is None or self.sigma_long is None:
                raise ValueError("gaussian_pulse requires w0 and sigma_long")

            if self.pcenter_at_origin:
                self.r_front0 = self.center.copy()
            else:
                self.r_front0 = self.upstream_front_position(
                    center=self.center,
                    box_size=self.box_size,
                    k_in_hat=self.k_in_hat,
                    margin=self.margin,
                )
                self.r_front0 = self.r_front0 - (
                    1.2 * self.sigma_long + float(self.pulse_center_t0)
                ) * self.k_in_hat

            log.info(
                "BeamModel gaussian_pulse: w0=%.6g, sigma_long=%.6g, v_front=%.6g, k_in=%0.6g",
                self.w0, self.sigma_long, self.v_front, self.k_in
            )

        elif self.beam_type == "plane_wave":
            self.r_front0 = np.zeros(3, dtype=float)
            log.info("BeamModel plane_wave: k_in=%0.6g, k_in_hat=%s", self.k_in, self.k_in_hat)

        else:
            raise ValueError(f"Unsupported beam_type: {self.beam_type!r},\n only gaussian_pulse, plane_wave  supported")

        self.log_info()

    @staticmethod
    def upstream_front_position(center, box_size, k_in_hat, margin=1.0) -> np.array:
        """
        Place the pulse front just outside the upstream edge of a box-shaped cloud.
    
        Parameters
        ----------
        center : array-like, shape (3,)
            Cloud center.
        box_size : array-like, shape (3,)
            Box side lengths (Lx, Ly, Lz).
        k_in_hat : array-like, shape (3,)
            Beam propagation direction.
        margin : float, optional
            Extra distance added upstream.
    
        Returns
        -------
        np.ndarray, shape (3,)
            Initial pulse-front center.
        """
        center = np.asarray(center, dtype=float)
        box_size = np.asarray(box_size, dtype=float)
        k_in_hat = np.asarray(k_in_hat, dtype=float)
        k_in_hat /= (np.linalg.norm(k_in_hat) + 1e-15)
    
        half_extent_along_k = 0.5 * np.sum(np.abs(k_in_hat) * box_size)
        return center - (half_extent_along_k + margin) * k_in_hat

    def pulse_center(self, t: float) -> np.ndarray:
        if self.beam_type == "plane_wave":
            return np.zeros(3, dtype=float)
        return self.r_front0 + self.v_front * float(t) * self.k_in_hat


    def beam_mask(self, r_xyz, radius_factor=2.0):
        """ count how many atoms inside beam"""
        dr = r_xyz - self.pulse_center(0)[None, :]

        u_par = dr @ self.k_in_hat
        dr2 = np.sum(dr * dr, axis=1)
        u_perp2 = dr2 - u_par**2

        return u_perp2 <= (radius_factor * self.w0)**2


    def transverse_gaussian_mode_amplitude(self, r_xyz) -> np.array :
        """
        Generates the Beam amplitude weight given by a gaussian beam. 
        Generates the transverse amplitude weight. Accounts for Beam direction
        
                        E(r) = exp ( - r_perp^2 / w_oSignal^2)

        returns: array
        """

        # Center the beam
        dr = r_xyz - self.center[None, :]

        # getting r_perp.  
        u_par = dr @ self.k_in_hat
        u_perp2 = np.sum(dr * dr, axis=1) - u_par**2

        return np.exp(-u_perp2 / self.w0**2)

    def longitudinal_gaussian_mode_amplitude(self, r_xyz) -> np.array :
        """
        Generates the Beam amplitude weight given by a gaussian beam. 
        Generates the longitudinal amplitude weight. Accounts for Beam direction
        
                        E(r) = exp ( - r_lon^2 / w_oSignal^2)
        returns: array
        """

        # Center the beam
        dr = r_xyz - self.center[None, :]

        # getting r_long.  
        u_par = dr @ self.k_in_hat

        return np.exp(-u_par**2 / self.sigma_long**2)

    def optical_phase(self, r_xyz): 
        """ returns the optical phase inprinted by the beam
        exp ( -1j * k_in @ r_xyz)
        """
        return np.exp(-1j * self.k_in * (r_xyz @ self.k_in_hat))


    def generate_weights(self, r_xyz, t: float = 0.0):
        """
        Compute the complex beam weights on atoms at time t.

        Parameters
        ----------
        r_xyz : np.ndarray, shape (N, 3)
            Atom positions.
        t : float
            Time.

        Returns
        -------
        np.ndarray, shape (N,)
            Complex weights.
        """
        # Dimension check
        if r_xyz.ndim != 2 or r_xyz.shape[1] != 3:
            raise ValueError(f"r_xyz must have shape (N, 3), got {r_xyz.shape}")

        env_perp  = self.transverse_gaussian_mode_amplitude(r_xyz) 
        env_long  = self.longitudinal_gaussian_mode_amplitude(r_xyz) 
        phase     = self.optical_phase(r_xyz) 

        self.w = (env_perp * env_long * phase).astype(np.complex128)
        return self.w

    
    def log_info(self):
        log.info("====================================================")
        log.info("Beam model summary")
        log.info("beam_type         = %s", self.beam_type)
        log.info("k_in              = %.6g", self.k_in)
        log.info("k_in_hat          = %s", np.array2string(self.k_in_hat, precision=6))

        if self.beam_type == "gaussian_pulse":
            log.info("w0                = %.6g", self.w0)
            log.info("sigma_long        = %.6g", self.sigma_long)
            log.info("v_front           = %.6g", self.v_front)
            log.info("box_size          = %s", np.array2string(self.box_size, precision=6))
            log.info("center            = %s", np.array2string(self.center, precision=6))
            log.info("margin            = %.6g", self.margin)
            log.info("pulse_center_t0   = %.6g", self.pulse_center_t0)
            log.info("pcenter_at_origin = %s", self.pcenter_at_origin)
            log.info("r_front0          = %s", np.array2string(self.r_front0, precision=6))

        log.info("====================================================")
    
    def report_weight_profile(
        self,
        cloud,
        t: float = 0.0,
        cloud_fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
    ):
        """
        Short text report of the beam envelope relative to the cloud size.
    
        Reports:
        - transverse |w| at r = fraction * R_cloud
        - longitudinal |w| at u_par = shifted fraction of Lz
        - reference values at 1,2,3 w0 and 1,2,3 sigma_long
    
        Assumes beam axis ~ z and cloud centered at origin.
        Intended as a quick sanity check, not a full general diagnostic.
        """
        import numpy as np
    
        print("\n=== Beam weight profile report ===")
        print(f"t = {t:.6g}")
        print(f"w0 = {self.w0:.6g}")
        print(f"sigma_long = {self.sigma_long:.6g}")
    
        # pulse center in lab frame
        r0 = self.pulse_center(t)
    
        # ---- transverse test ----
        if getattr(cloud, "R", None) is not None:
            R_cloud = float(cloud.R)
        else:
            R_cloud = 0.5 * float(cloud.Lx)
    
        print("\nTransverse profile |w|(u_perp, u_par=0):")
        for f in cloud_fracs:
            u_perp = f * R_cloud
            # point at beam center longitudinally, displaced transversely in x
            r = r0 + np.array([u_perp, 0.0, 0.0], dtype=float)
            w = self.generate_weights(r[None, :], t=t)[0]
            print(f"  at {100*f:>5.1f}% of cloud radius: u_perp={u_perp:>8.4f}, |w|={abs(w):.3e}")
    
        print("\nReference transverse values:")
        for n in (1, 2, 3):
            u_perp = n * self.w0
            r = r0 + np.array([u_perp, 0.0, 0.0], dtype=float)
            w = self.generate_weights(r[None, :], t=t)[0]
            print(f"  at {n} w0: u_perp={u_perp:>8.4f}, |w|={abs(w):.3e}")
    
        # ---- longitudinal test ----
        Lz = float(cloud.Lz)
        print("\nLongitudinal profile |w|(u_perp=0, z through cloud):")
        for f in cloud_fracs:
            z_lab = -0.5 * Lz + f * Lz
            r = np.array([[0.0, 0.0, z_lab]], dtype=float)
            w = self.generate_weights(r, t=t)[0]
    
            # report also beam-frame longitudinal coordinate
            u_par = float((r[0] - r0) @ self.k_in_hat)
            print(
                f"  at {100*f:>5.1f}% of cloud length:"
                f" z={z_lab:>8.4f}, u_par={u_par:>8.4f}, |w|={abs(w):.3e}"
            )
    
        print("\nReference longitudinal values:")
        for n in (1, 2, 3):
            for sgn in (-1, +1):
                u_par = sgn * n * self.sigma_long
                r = r0 + u_par * self.k_in_hat
                w = self.generate_weights(r[None, :], t=t)[0]
                print(f"  at {sgn*n:+d} sigma_long: u_par={u_par:>8.4f}, |w|={abs(w):.3e}")


    def __str__(self):
        lines = [f"{self.__class__.__name__}("]

        lines.append(f"  beam_name_label    = {self.label}")
        lines.append(f"  beam_type          = {self.beam_type}")
        lines.append(f"  k_in               = {self.k_in:.6g}")
        lines.append(f"  k_in_hat           = {self.k_in_hat}")

        lines.append(f"  w0                 = {self.w0}")
        lines.append(f"  sigma_long         = {self.sigma_long}")
        lines.append(f"  v_front            = {self.v_front}")

        lines.append(f"  box_size           = {self.box_size}")
        lines.append(f"  center             = {self.center}")
        lines.append(f"  margin             = {self.margin}")
        lines.append(f"  pulse_center_t0    = {self.pulse_center_t0}")
        lines.append(f"  pcenter_at_origin  = {self.pcenter_at_origin}")

        if hasattr(self, "r_front0"):
            lines.append(f"  r_front0           = {self.r_front0}")

        if hasattr(self, "w"):
            lines.append(f"  w.shape            = {self.w.shape}")
            try:
                lines.append(f"  w min/max          = {self.w.min():.6g} / {self.w.max():.6g}")
                lines.append(f"  w mean             = {self.w.mean():.6g}")
            except Exception:
                pass

        lines.append(")")
        return "\n".join(lines)




