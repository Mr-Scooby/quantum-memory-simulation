#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field, asdict
from radpattern.helpers import io
from radpattern.geometry.cloud_model import CloudModel, AtomSpeciment
from radpattern.geometry.grids import AngleGrid
from radpattern.physics.beam import BeamModel
from radpattern.physics.experimetal_setup import ExperimentalParams
import numpy as np
import logging
import hashlib
import json


log = logging.getLogger(__name__) 

@dataclass
class PhysicalRegime: 
    """ Physical regime of the simualtion.  """ 
    atoms: AtomSpeciment
    geometry: str = "cylinder"
    distribution:str = "gaussian"

    # Box size. how large the cloud is compared to the wavelength, so how much spatial phase can build up across it.
    # L/lambda
    optical_size_z: float = 10.0     # longitudinal size
    aspect_ratio: float  = 100.0     # aspect ratio Lz/ Lxy 

    # Interparticle spacing in units of wavelength. a/ lambda. We want a >sim lambda
    optical_spacing: float = 1.5 

    # illumination ratio: How much of the cloud the beam covers transversely. w0/ L_perp
    illumination_ratio : float = 0.8 

    # Longitudinal filling factor: How much of the cloud is covered along propagation.
    # sigma_long / L_parallel
    filling_factor: float = 0.1

    # Pulse transit. how far the pulse front has traveled through the cloud. for time parameters. 
    # v_front t / L_parrallel. 
#    pulse_transit : float = 1.5 

    # Atom gaussian distributions. 
    transverse_sigma_ratio: float = 0.1  # gaussian distribution sigma on the Lxy plane ratio. Sigma_T / Lxy 
    longitudinal_sigma_ratio: float = 0.3 # Gaussian distribtion along Lz: sigma_l / Lz
    wavelength: float = 1
    # dipole polarization
    p_hat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def __post_init__(self):
        self.log_info()

    @property 
    def L_z (self): 
        return self.optical_size_z * self.wavelength
    @property
    def L_xy (self): 
        return self.L_z / self.aspect_ratio

    @property 
    def spacing (self):
        return self.optical_spacing * self.wavelength
    @property 
    def density (self): 
        return 1 / self.spacing **3 
    @property
    def beam_waist (self): 
        return self.illumination_ratio * self.L_xy
    @property 
    def sigma_long (self): 
        return self.filling_factor * self.L_z

    @property
    def k0 (self): 
        return 2 * np.pi/ self.wavelength

    @property 
    def sigma_T(self): 
        if self.transverse_sigma_ratio == None: 
            return None 
        return self.transverse_sigma_ratio * self.L_xy 
    @property 
    def sigma_z (self): 
        if self.longitudinal_sigma_ratio == None: 
            return None 
        return self.longitudinal_sigma_ratio * self.L_z 

    @property 
    def R_eff (self):
        """efective radius of beam. when r = 3w0 and thus the field is effectivlely zero, 
        even pushing it too much, 2w0 is enough.o"""
        return 3 * self.beam_waist

    @property 
    def L_eff (self):
        """efective radius of beam Longitudinal. when r = 3w0 and thus the field is effectivlely zero, 
        even pushing it too much, 2w0 is enough.o"""
        return 3 * self.sigma_long


    # to implement later
    # Motion / time 
    #v_front: float = 1.0
    #v_thermal: float = 0.001


    def make_cloud(self): 
        log.info("Generating cloud") 
        
        if self.geometry == "cylinder" or self.geometry == "sphere": 
            R = self.L_xy/2
        else:
            R = None

        cloud =  CloudModel(
            geometry=self.geometry,
            distribution=self.distribution,
            Lx=self.L_xy,
            Ly=self.L_xy,
            Lz=self.L_z,
            R = R,
            density=self.density,
            sigma_x= self.sigma_T,
            sigma_y= self.sigma_T, 
            sigma_z= self.sigma_z, 
            atoms = self.atoms
        )
        cloud.log_info()
        return cloud

    def log_info(self):
         log.info("====================================================")
         log.info("Physical regime summary")
         log.info("All length units below are relative to wavelength lambda")

         log.info("geometry                 = %s", self.geometry)
         log.info("distribution             = %s", self.distribution)

         log.info("optical_size_z           = %.6g", self.optical_size_z)
         log.info("aspect_ratio             = %.6g", self.aspect_ratio)
         log.info("optical_spacing          = %.6g", self.optical_spacing)

         log.info("illumination_ratio       = %.6g", self.illumination_ratio)
         log.info("Beam field region ratio R_eff/ Lxy = %.6g", self.R_eff/self.L_xy)
         log.info("filling_factor           = %.6g", self.filling_factor)
         log.info("Beam field region ratio L_eff/ L_z = %.6g", self.L_eff/self.L_z)

         log.info("transverse_sigma_ratio   = %s", self.transverse_sigma_ratio)
         log.info("longitudinal_sigma_ratio = %s", self.longitudinal_sigma_ratio)

         log.info("wavelength               = %.6g", self.wavelength)
         log.info("k0                       = %.6g", self.k0)
         log.info("p_hat                    = %s", np.array2string(self.p_hat, precision=6))

         log.info("L_z                      = %.6g lambda", self.L_z)
         log.info("L_xy                     = %.6g lambda", self.L_xy)
         log.info("spacing                  = %.6g lambda", self.spacing)
         log.info("density                  = %.6g lambda^-3", self.density)
         log.info("beam_waist               = %.6g lambda", self.beam_waist)
         log.info("sigma_long               = %.6g lambda", self.sigma_long)
         log.info("sigma_T                  = %s", "None" if self.sigma_T is None else f"{self.sigma_T:.6g} lambda")
         log.info("sigma_z                  = %s", "None" if self.sigma_z is None else f"{self.sigma_z:.6g} lambda")

         log.info("====================================================")



# 3) Simulation / numerical parameters
# ------------------------------------------------------------------
@dataclass
class SimParams:
    """Numerical controls for the Monte Carlo simulation."""
    n_mc: int 

    # Time sampling
    sim_time_us: float  # microseconds
    char_time: float # Seconds.
    time_divisions: int = 10
    time_spacing: str = "linspace"


    # Angular grid
    n_theta: int = 91
    n_phi: int = 181
    theta_max : float = np.pi
    simulation_window_radius_w0_cutoff: float = 3 #Only simulate atoms within radius = simulation_window_radius_w0_cutoff * w0_control

    # MC atoms sim.
    sim_density :int = 1

    # Performance / implementation
    chunk_atoms: int = 2000
    normalize_each_time: bool = False
    plane_restricted: bool = False
    seed: int = None

    # File naming 
    # Computed run name: human-readable + hash from all params

    @property 
    def grid_shape(self): 
        return (self.n_theta, self.n_phi)
   # @property
   # def times(self) -> np.ndarray:
   #     return np.linspace(0.0, self.t_max, self.n_times)

    def create_grid(self):
       return AngleGrid(self.n_theta, self.n_phi, self.theta_max  )

    def sim_metadataSetUp(self, regime, beam): 
        return SetupParams(regime, self, beam)

    @property 
    def sim_time_s(self):
        return self.sim_time_us * 1e-6
    @property
    def sim_time_code(self): 
        return self.sim_time_s / self.char_time

    def time_array(self): 
        """ returns the time array in code times i.e. time_s / char_time. 
        geomspace array. 
        params: char_time, time_divisions. 
        """
        log.info("Time array creation. Sim_time_window = %f [us], divisions = %i", self.sim_time_us, self.time_divisions )
        if self.time_spacing.upper() == "LINSPACE":
            times_us = np.linspace(0.0, self.sim_time_us, self.time_divisions)
        elif time_spacing.upper() == "GEOMSPACE":
            times_us = np.r_[0.0, np.geomspace(0.05, self.sim_time_us, self.time_divisions - 1 )]
        else:
            raise ValueError("time_spacing must be 'linspace' or 'geomspace'")
        times_code = times_us * 1e-6 / self.char_time
        return times_code




def _k_tag(k_hat) -> str:
    return "k" + "".join(str(round(x)) for x in k_hat)

@dataclass
class SetupParams:
    """ Stores metadata and creates run naming """
    experiment: ExperimentalParams
    sim: SimParams
    beam: BeamModel

    # Computed run name: human-readable + hash from all params
    @property
    def run_name(self) -> str:
        # hash full setup
        d = {
            "experiment": asdict(self.experiment),
            "sim": asdict(self.sim),
            "cloud": asdict(self.sim),
        }
        h = hashlib.sha1(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

        return (
            f"{self.experiment.atoms}_{self.experiment.buffer_pressure_Torr}Torr"
            f"{int(self.experiment.signal_fwhm_diameter_m * 1e6)}SDia_{int(self.experiment.control_fwhm_diameter_m * 1e6) }Cdia"
            f"_simT{self.sim.sim_time_us}us"
            f"_nt{self.sim.time_divisions}"
            f"_{self.sim.n_mc}runs"
            f"_{h}"
        )



