#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field, fields, asdict
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
    def __post_init__(self): 
        if self.seed is None:
            self.seed = int(np.random.default_rng().integers(0, 2**32))
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

    def __str__(self):
        skip_types = (np.ndarray, list, tuple, dict, set)

        lines = [f"{self.__class__.__name__}("]

        for f in fields(self):
            name = f.name
            value = getattr(self, name)

            if isinstance(value, skip_types):
                continue

            lines.append(f"  {name} = {value}")

        lines.append(")")
        return "\n".join(lines)

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



