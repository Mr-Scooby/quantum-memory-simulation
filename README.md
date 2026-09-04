# Quantum Memory Spin-Wave Simulation

3D Monte Carlo simulation framework for studying the survival of the retrievable spin-wave mode in atomic-ensemble quantum memories.

This repository contains the numerical model developed for my Master's thesis:

**Survival of the Retrievable Spin-Wave Mode in Atomic Ensembles: Effects of Motion, Geometry, and Dephasing**

The work was carried out at the **Joint Lab Integrated Quantum Sensors (IQS), Humboldt-Universität zu Berlin**, within the Erasmus Mundus Master in Quantum Technologies and Engineering (QuanTEEM).

---

## Overview

In an atomic-ensemble quantum memory, preserving atomic coherence is not sufficient by itself: the stored collective excitation must also retain the spatial amplitude and phase structure required for retrieval into the desired optical mode.

This project models how that retrievable mode evolves during storage.

The simulation starts after the spin wave has been written and propagates the atomic ensemble during the storage interval. At readout, the optical field is reconstructed from the coherent sum of the atomic contributions and evaluated in the selected collection mode.

The framework compares two experimentally relevant regimes:

- **Warm Cs vapour** — diffusive atomic motion in a buffer-gas cell
- **Cold Rb ensemble** — ballistic expansion after release from a magneto-optical trap

---

## Modelled effects

The framework includes:

- 3D atomic position sampling
- Diffusive motion in warm vapour
- Ballistic thermal motion in cold ensembles
- Gaussian signal and control beam profiles
- Collective phase-matched optical emission
- Magnetic-field-gradient dephasing
- Wall collisions and effective coherence survival
- Finite optical-mode overlap
- Monte Carlo averaging
- Fibre-coupled retrieval calculations
- Parameter sweeps
- Numerical convergence analysis

The model focuses on the **relative survival of the retrievable collective mode** rather than the complete end-to-end memory efficiency.

It does not simulate the full Maxwell-Bloch write process, detector losses, filtering losses, memory noise, or absolute storage-and-retrieval efficiency.

---

## Physical model

Each atom carries a position-dependent spin-wave amplitude and phase.

During storage, its position evolves according to the relevant transport model.

For the warm-vapour system, motion is described diffusively. For the released cold ensemble, atomic velocities are sampled from a thermal distribution and the atoms evolve ballistically.

At readout, the atomic electric-field contributions are summed coherently,

\[
E(\mathbf{k},t)
\propto
\sum_j A_j(t)e^{i\phi_j(t)},
\]

and the intensity is calculated only after the complete collective field has been formed,

\[
I(\mathbf{k},t)
\propto
|E(\mathbf{k},t)|^2.
\]

This allows loss of retrieval to arise naturally from two main mechanisms:

1. **Spatial redistribution** of atoms away from the initially prepared optical mode.
2. **Relative phase evolution** between atomic contributions, which reduces constructive collective interference.

---

## Numerical approach

The physical atomic ensembles contain far more atoms than can be simulated individually.

The code therefore uses Monte Carlo sampling with reduced numerical ensembles.

Each realization samples an independent atomic configuration. Depending on the physical system, this includes:

- initial atomic positions
- thermal velocities
- diffusive trajectories
- wall interactions
- accumulated magnetic phases

Within each realization, all atomic field contributions are summed coherently before the intensity is calculated.

Independent realizations are then averaged to suppress finite-sampling fluctuations.

The numerical sampling density controls simulation resolution only and should not be interpreted as the physical atomic density.

---

## Repository structure

```text
src/radpattern/
├── config/       # simulation parameters and default configurations
├── geometry/     # ensemble and optical geometries
├── helpers/      # sampling and numerical utilities
├── physics/      # physical models and field calculations
├── plotting/     # visualization tools
└── simulation/   # Monte Carlo and time-evolution workflows

scripts/          # simulation and analysis scripts
tests/            # validation and convergence studies
```

The package is organized to keep the physical model, numerical simulation, geometry, configuration, and visualization layers separate.

---

## Installation

Python 3.11 is currently supported.

```bash
git clone https://github.com/Mr-Scooby/quantum-memory-simulation.git
cd quantum-memory-simulation

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

Main dependencies include:

- NumPy
- Matplotlib
- CuPy
- Joblib

GPU acceleration is available through CuPy for compatible CUDA systems.

---

## Research applications

The framework was used to study the sensitivity of the retrievable spin-wave mode to experimentally relevant parameters, including:

- buffer-gas pressure
- cold-atom temperature
- signal beam diameter
- control beam diameter
- vapour-cell geometry
- wall-collision survival
- anti-relaxation coating quality
- magnetic-field gradients

The same framework can also be adapted to related problems involving coherent collective emission from moving or dephasing ensembles.

---

## Thesis

**Radek Vasicek Ruiz**

**Survival of the Retrievable Spin-Wave Mode in Atomic Ensembles: Effects of Motion, Geometry, and Dephasing**

Erasmus Mundus Master in Quantum Technologies and Engineering (QuanTEEM), 2026

Research performed at the **Joint Lab Integrated Quantum Sensors (IQS), Humboldt-Universität zu Berlin**.

---

## Scope and limitations

The simulation begins after the spin wave has already been prepared.

The following effects are therefore outside the present model:

- full Maxwell-Bloch storage dynamics
- absolute write efficiency
- absolute readout efficiency
- optical propagation losses
- spectral-filtering losses
- fibre-transmission losses
- detector efficiency
- memory noise
- recurrent scattering
- near-field dipole-dipole interactions

The calculated observable should therefore be interpreted as the **relative survival of the retrievable collective optical mode**, not as the complete quantum-memory efficiency.

---

## Status

Research code developed during the Master's thesis and subsequently reorganized into a reusable Python package.

The repository is intended primarily for scientific reproducibility, model development, and further research rather than as a production software library.
