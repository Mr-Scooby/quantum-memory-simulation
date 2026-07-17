#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 

Plotting of the 3d pattern 

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 

import logging

log = logging.getLogger(__name__)

# Plotting
def plot_pattern_3d(grid, I, title="", alpha=1.0, stride=2, cmap="viridis", info_text=None, sphere_map = True ,log_plot = True, clip_limit = True, I_ref=None,
    **kwargs ):
    """
    3D surface plot where radius = I^alpha.

    grid object with vector of the directiosn. 
    nx,ny,nz: direction cosines (nt,np)
    I: intensity (nt,np), should be normalized if you want nice scaling

    stride : int, optional
        Downsampling stride used by plot_surface.
    cmap : str, optional
        Matplotlib colormap used to color the surface.
    
    inputs:
    -clip_limit :bool = True, limits the log plot to -60dB as below that is basically noise. set everything below that to 0. 

    """

     # Make sure intensity is a numpy array
    I = np.asarray(I, dtype=float)
    if I.ndim == 3: 
        I = I[0] 

    log.info("Creating pattern 3D plot. Grid_shape = %s, I (max, min) = (%.3e, %.3e) ", grid.shape, I.max(), I.min())
    # Convert intensity into a plotted radius
    # alpha < 1 expands weak regions visually
    # alpha > 1 compresses weak regions
    if sphere_map: 
        R = 1 
    else:
        R = I**alpha
    # Convert spherical direction field into Cartesian surface coordinates
    Xs = R * grid.nx
    Ys = R * grid.ny
    Zs = R * grid.nz


    if log_plot: 
        
        if I_ref is None: 
            I_ref = np.max(I)
        else:
            I_ref= np.max(I_ref)

    
        C = 10 * np.log10(I / I_ref + 1e-12)
        norm = colors.Normalize(vmin=np.min(C), vmax=np.max(C))
        title = title + " Log_plot"
        clabel = "Intensity [dB]"
        if clip_limit: 
            C = np.clip(C, -60, 0)
            norm = colors.Normalize(vmin=-60, vmax=0)
        else:
            norm = colors.Normalize(vmin=np.min(C), vmax=np.max(C))
    else: 
    # Normalize intensity values for the colormap
        norm = colors.Normalize(vmin=np.min(I), vmax=np.max(I))
        C = I 
        clabel = "Intensity"

    #norm = colors.LogNorm(vmin=np.max(I)*1e-6, vmax=np.max(I))

    # Create figure and 3D axes
    fig = plt.figure(figsize = kwargs.get("figsize", (9, 7)),
                     dpi = kwargs.get("dpi", None))

    ax = fig.add_subplot(111, projection="3d")

    # Draw the 3D surface
    # Geometry comes from Xs,Ys,Zs
    # Color comes from the intensity I
    ax.plot_surface(
        Xs, Ys, Zs,
        rstride=stride, cstride=stride,
        facecolors=plt.get_cmap(cmap)(norm(C)),
        linewidth=kwargs.get("linewidth", 0),
        antialiased=kwargs.get("antialiased", True),
        shade=kwargs.get("shade", False)
    )

    # Create a colorbar showing the intensity scale
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(C)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label(clabel)

    # Set equal aspect ratio so the 3D pattern is not distorted
    ax.set_box_aspect([1, 1, 1])

    # Labels and title
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


    # Optional info text box outside the axes
    if info_text is not None:
        fig.text(
            kwargs.get("text_x", 0.02),
            kwargs.get("text_y", 0.05),
            info_text, 
            ha = kwargs.get("text_ha", "left"),
            va = kwargs.get("text_va", "bottom"),
            family="monospace",
            fontsize = kwargs.get("text_fontsize", 10),
            bbox = kwargs.get("text_box",dict(boxstyle="round", facecolor="white", alpha=0.8)),
        )


    log.info("Plotting pattern 3D. Title = %s", title)

    return fig, ax



