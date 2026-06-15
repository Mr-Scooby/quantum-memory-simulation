#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from .AtomModel import AtomSpeciment
from .base_cloud import BaseCloud 
import numpy as np
import logging
from radpattern.helpers.timing import debug_timer

log = logging.getLogger(__name__)

@dataclass
class WarmVaporCloud(BaseCloud):
    """ Uniformally distributed atoms in a Cylinder cell. """

    Lz: float
    R: float 
    sim_density: int 
    boundary_condition_apply: bool
    N_bounces: int # Cell_coating_variable:  Number of wall boucnes the atom can take before losing coherence.

    def __post_init__(self):
        super().__post_init__()

        if (not self.boundary_condition_apply) and self.N_bounces is not None:
            log.warning(
                "N_bounces=%s was provided, but boundary reflections are disabled. "
                "Wall decoherence will not be applied.",
                self.N_bounces,
            )


    @property
    def n_atoms(self): 
        return int(self.sim_density * self.volume)

    @property
    def volume(self):
        """ Cylinder cell vloume """
        return np.pi * self.R**2 * self.Lz

    @property
    def box_size(self):
        return np.array([2 * self.R, 2 * self.R, self.Lz])

    @debug_timer()  
    def _generate_cloud_impl(self, rng):
        """ atom cloud distribution generation for warmvapor, where it generates radom uniform distribution
        inside the cylindrical cell dimension  2r x Lz """

        log.debug("generating warmvapor cloud. dimension (r, Lz) = (%.4f, %.4f) [code_units], n_atoms = %d", self.R, self.Lz, self.n_atoms) 
        if rng is None:
            rng = np.random.default_rng()

        # uniform random in cylinder
        rho = self.R * np.sqrt(rng.random(self.n_atoms))
        phi = 2 * np.pi * rng.random(self.n_atoms)
        z = rng.uniform(-self.Lz / 2, self.Lz / 2, self.n_atoms)

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        self.r_xyz = np.column_stack([x, y, z])
        log.debug("cloud points generated. size %s", self.r_xyz.shape) 
        return self.r_xyz

    def _reflect_radial_boundary(self, max_iter=10):
        """ updates de position of the atoms reflecting when outside the cylindrical cell area
        on the radial coordinate. 
        simulating elastic collision with wall.""" 

        log.debug("radial reflection boundary condition") 

        hit_any = np.zeros(self.r_xyz.shape[0], dtype= bool) 
        for it in range(max_iter):
            x = self.r_xyz[:, 0]
            y = self.r_xyz[:, 1]

            rho = np.sqrt(x**2 + y**2)
            outside = rho > self.R
            hit_any |= outside

            n_outside = np.count_nonzero(outside)

            log.debug(
                "radial reflection iter=%d: n_atoms outside : %d / %d (%.3f %%)",
                it,
                n_outside,
                self.n_atoms,
                100.0 * n_outside / self.n_atoms,
            )
            if not np.any(outside):
                log.debug("reflection on radial boundary condition done") 
                return self.r_xyz, hit_any 

            n_hat = self.r_xyz[outside, :2] / rho[outside, None]

            rho_reflected = 2.0 * self.R - rho[outside]

            self.r_xyz[outside, :2] = n_hat * rho_reflected[:, None]

        raise RuntimeError(
              "boundary reflection did not converge. "
              "your diffusive timestep is probably too large."
              )


    def _reflect_z_boundaries(self, max_iter=10):
        """ updates de position of the atoms reflecting when outside the cylindrical cell area
        on the z coordinate. 
        simulating elastic collision with wall.""" 

        log.debug("z coordinate reflection boundary condition") 
        z_min = -0.5 * self.Lz
        z_max =  0.5 * self.Lz
    
        hit_any = np.zeros(self.r_xyz.shape[0], dtype = bool) 
        for it in range(max_iter):
            z = self.r_xyz[:, 2]

            above = z > z_max
            below = z < z_min
            outside = above | below 
            hit_any |= outside
            

            n_outside = np.count_nonzero(outside)
            log.debug(
                "z cap reflection iter=%d: n_atoms outside: %d / %d (%.3f %%)",
                it, 
                n_outside,
                self.n_atoms,
                100.0 * n_outside / self.n_atoms,
            )

            if not (np.any(above) or np.any(below)):
                log.debug("reflection on z boundary condition done") 
                return self.r_xyz, hit_any

            self.r_xyz[above, 2] = 2.0 * z_max - self.r_xyz[above, 2]
            self.r_xyz[below, 2] = 2.0 * z_min - self.r_xyz[below, 2]

        raise RuntimeError(
              "boundary reflection did not converge. "
              "your diffusive timestep is probably too large."
              )

    def wall_decoherence_survival(self, atoms_outside, rng ): 
        """ updates the weights to account for decoherence due to wall collision. 
        takes the probability of survival 1- 1/n_bounces of the coating applied. 
        for decoherence updates S weights to zero ( no longer active in the emissio)
        """
        log.debug("wall decoherence survival calculation") 
        if self.N_bounces is None:
            log.warning("Wall reflection is active but N_bounces=None. "
        "Atoms reflect from the cell wall without wall-induced decoherence.")
            return
        if not np.any(atoms_outside):
            log.debug("No wall hits. Skipping wall decoherence.")
            return

        survive = rng.random(self.n_atoms ) >= ( 1.0 / self.N_bounces) # bool of atoms that survived decoherence 
        n_survive = np.count_nonzero(survive)
        log.debug("survive count = %d / %d ( %.3f %% )  . number of atoms that met the condition >1/n_bounces ( not filered by outside)", n_survive, self.n_atoms, 100* n_survive / self.n_atoms )
        # crossed with atoms that actually hitted the wall. 
        coherence = ( ~atoms_outside ) | survive  # atoms not outside automatically survive 
        n_coh =  np.count_nonzero(coherence)
        log.info("atoms decoherence survival : %d / %d (%3.f %%) ",n_coh , self.n_atoms, 100 * n_coh / self.n_atoms)

        n_depol = self.n_atoms - n_coh
        # Warning if too many atoms depolarize at same time
        if n_depol / self.n_atoms > 0.01:
            log.warning(
                "Large wall decoherence in one timestep: depolarized=%d/%d (%.3f %%). "
                "Check dt, D, cell size, or N_bounces.",
                n_depol,
                self.n_atoms,
                100.0 * n_depol / self.n_atoms,
            )
        # update weights 
        self.S *= coherence 
    
    def update_position_diffusive(self,*args,  **kwargs):
        """ diffusive motion. takes reflection into account if necessary """
        # updates difussion step. parent 
        super().update_position_diffusive(*args,**kwargs)

        # applies boundary conditions. 
        if self.boundary_condition_apply:
            _, cap_out = self._reflect_z_boundaries()
            _, radial_out = self._reflect_radial_boundary()
            log.debug("application of boundary condition met") 
            
            outside = (cap_out | radial_out ) 
            wall_bounce = np.count_nonzero(outside)
            log.info("N_atoms bouncing from wall : %d/ %d (%3.f %%)",wall_bounce, self.n_atoms, 100 * wall_bounce / self.n_atoms )  
            self.wall_decoherence_survival(outside, **kwargs)



        return self.r_xyz



