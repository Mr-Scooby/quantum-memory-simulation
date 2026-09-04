#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from pathlib import Path
import re
import numpy as np
from scipy.optimize import curve_fit
import logging

#logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
log = logging.getLogger(__name__)



# Symbolic links to the folders containing the .dat files
DATA_FOLDERS = [
    Path("CS_BeamChange/Cs120SbeamVsCbeam.dat"),
    #Path("RB_BeamChange"),
    #Path("path/to/symlink_folder_2"),
]


def decay_model(time_us, amplitude, tau_us, beta):
    """Stretched-exponential decay model."""
    return amplitude * np.exp(-(time_us / tau_us) ** beta)


def read_data(file_path):
    """Read a data file with named columns."""
    return np.genfromtxt(
        file_path,
        names=True,
        dtype=float,
        encoding="utf-8",
    )


def fit_decay(time_us, coupling):
    """Fit the coupling decay and return its 1/e lifetime."""
    time_us = np.asarray(time_us, dtype=float)
    coupling = np.asarray(coupling, dtype=float)

    valid = (
        np.isfinite(time_us)
        & np.isfinite(coupling)
        & (coupling >= 0)
    )

    time_us = time_us[valid]
    coupling = coupling[valid]

    if len(time_us) < 4:
        raise ValueError("Not enough valid points for the fit.")

    amplitude_guess = coupling[0]
    target = amplitude_guess / np.e

    below_target = np.where(coupling <= target)[0]

    if len(below_target) > 0:
        tau_guess = time_us[below_target[0]]
    else:
        tau_guess = 0.7 * time_us[-1]

    initial_guess = [
        amplitude_guess,
        max(tau_guess, 1.0),
        1.5,
    ]

    lower_bounds = [
        0.0,
        1.0e-12,
        0.2,
    ]

    upper_bounds = [
        np.inf,
        np.inf,
        5.0,
    ]

    parameters, covariance = curve_fit(
        decay_model,
        time_us,
        coupling,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50_000,
    )

    amplitude, tau_us, beta = parameters

    parameter_errors = np.sqrt(np.diag(covariance))
    tau_error_us = parameter_errors[1]

    return amplitude, tau_us, beta, parameter_errors

def extract_scanned_beam(column_name):
    """
    Extract the scanned beam size from a column name.

    Example
    -------
    curve_300 -> 300
    """
    match = re.fullmatch(
        r"curve_(\d+)",
        column_name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Cannot determine the beam size from column: "
            f"{column_name}"
        )

    return int(match.group(1))

def build_tau_grid(file_path):
    """
    Fit all curves and construct a two-dimensional lifetime grid.

    Grid convention
    ---------------
    Rows:
        Control-beam size.

    Columns:
        Signal-beam size.

    Values:
        Fitted 1/e lifetime in milliseconds.
    """
    tau_values = defaultdict(list)
    tau_error_values = defaultdict(list)

    try:
        data = read_data(file_path)

    except ValueError as error:
        print(f"Skipping {file_path.name}: {error}")

    column_names = data.dtype.names

    if column_names is None or "time_us" not in column_names:
        print(
            f"Skipping {file_path.name}: "
            f"no time_us column was found."
        )

    time_us = data["time_us"]

    for column_name in column_names:
        if column_name == "time_us":
            continue

        try:

            coupling = data[column_name]

            amplitude, tau_us, beta, parameter_errors = fit_decay(
                time_us,
                coupling,
            )
            
            tau_error_us = parameter_errors[1]

        except (ValueError, RuntimeError) as error:
            print(
                f"Could not fit {file_path.name}, "
                f"column {column_name}: { error}"
            )
            continue

        tau_ms = tau_us #/ 1000.0
        tau_error_ms = tau_error_us #/ 1000.0
        valid = np.isfinite(time_us) & np.isfinite(coupling) & (coupling >= 0)

        t_fit = time_us[valid]
        y_data = coupling[valid]
        y_fit = decay_model(t_fit, amplitude, tau_us, beta)

        residuals = y_data - y_fit

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)

        r_squared = 1.0 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(residuals**2))


        print(
            f"{file_path.name:15s}  "
            f"{column_name:10s}  "
            f"tau = {tau_ms:7.3f} +/- "
            f"{tau_error_ms:7.3f} us  "
            f"beta = {beta:.3f} ({parameter_errors[2]:.3f}) "
            f"amplitude = {amplitude:.3f} ({parameter_errors[0]:.3f}) "
            f"R² = {r_squared:.4f} "
            f" RMSE = {rmse:.4e}"
            )

    if not tau_values:
        raise RuntimeError("No valid lifetime values were obtained.")
    
        return parameters, parameter_errors


if __name__ == "__main__": 

    keepgoing = True
    while keepgoing:
        file = Path(input("File: ").strip()) or False
        if file: 
             build_tau_grid(file)
        else: 
            log.info("Ending program")
            keepgoing = False
        

