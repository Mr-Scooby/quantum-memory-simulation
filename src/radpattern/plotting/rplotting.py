#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Plotting functions for the radiation pattern. 

- 3D radiation pattern 
- animate_pattern_3d
- 2D angular plot of different plane cuts
- 3D plot of the position, velocity, incident wave intensity and direction, direction of the dipole. 
- animation_atoms_with_pulse
- plot_analytic_pattern_3d

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

#from helpers import (make_angle_grid, build_q_vec) 
import logging
log = logging.getLogger(__name__)

# Plotting
def plot_pattern_3d(nx, ny, nz, I, title="", alpha=1.0, stride=2, cmap="viridis", **kwargs):
    """
    3D surface plot where radius = I^alpha.

    nx,ny,nz: direction cosines (nt,np)
    I: intensity (nt,np), should be normalized if you want nice scaling
    """
     # Make sure intensity is a numpy array
    I = np.asarray(I, dtype=float)

    # Convert intensity into a plotted radius
    # alpha < 1 expands weak regions visually
    # alpha > 1 compresses weak regions
    R = I**alpha

    # Convert spherical direction field into Cartesian surface coordinates
    Xs = R * nx
    Ys = R * ny
    Zs = R * nz

    # Normalize intensity values for the colormap
    norm = colors.Normalize(vmin=np.min(I), vmax=np.max(I))

    # Create figure and 3D axes
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the 3D surface
    # Geometry comes from Xs,Ys,Zs
    # Color comes from the intensity I
    ax.plot_surface(
        Xs, Ys, Zs,
        rstride=stride, cstride=stride,
        facecolors=plt.get_cmap(cmap)(norm(I)),
        linewidth=0,
        antialiased=True,
        shade=False
    )

    # Create a colorbar showing the intensity scale
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(I)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Intensity")

    # Set equal aspect ratio so the 3D pattern is not distorted
    ax.set_box_aspect([1, 1, 1])

    # Labels and title
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    log.info("Plotting pattern 3D. Title = %s", title)
    return fig, ax

def animate_pattern_3d(nx, ny, nz, I_series, title="", alpha=1.0, stride=2, cmap="viridis", interval=50):
    """Animate a 3D pattern surface from I_series with shape (T, nt, np_)."""
    nx = np.asarray(nx, dtype=float)
    ny = np.asarray(ny, dtype=float)
    nz = np.asarray(nz, dtype=float)
    I_series = np.asarray(I_series, dtype=float)

    if I_series.ndim != 3:
        raise ValueError("I_series must have shape (T, nt, np_)")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    rmax = np.nanmax(I_series**alpha)
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_zlim(-rmax, rmax)

    # Normalize intensity values for the colormap
    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=np.nanmin(I_series), vmax=np.nanmax(I_series))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)

    I0 = I_series[0]
    R0 = I0**alpha
    surf = [ax.plot_surface(
        R0 * nx, R0 * ny, R0 * nz,
        rstride=stride, cstride=stride,
        facecolors=cmap_obj(I0),
        linewidth=0, antialiased=True, shade=False
    )]
    ax.set_title(f"{title} | frame 0")

    def update(frame):
        I = I_series[frame]
        R = I**alpha
        surf[0].remove()
        surf[0] = ax.plot_surface(
            R * nx, R * ny, R * nz,
            rstride=stride, cstride=stride,
            facecolors=cmap_obj(I),
            linewidth=0, antialiased=True, shade=False
        )
        ax.set_title(f"{title} | frame {frame}")
        return surf[0],

    ani = FuncAnimation(fig, update, frames=I_series.shape[0], interval=interval, blit=False)
    return fig, ax, ani


def _wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    """Map angles to (-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def _nearest_index_periodic(angle_grid: np.ndarray, target: float) -> int:
    """
    Nearest index on a periodic grid (period 2*pi).
    Works even if the grid is not exactly hitting the target value.
    """
    # Compare using wrapped difference so 0 and 2*pi behave correctly
    diff = _wrap_to_pi(angle_grid - target)
    return int(np.argmin(np.abs(diff)))


def _nearest_index(angle_grid: np.ndarray, target: float) -> int:
    """Nearest index on a non-periodic grid."""
    return int(np.argmin(np.abs(angle_grid - target)))


def plot_planar_cuts(theta, phi, I, title_prefix=""):
    """
    Plot planar cuts of an intensity pattern I(theta, phi).

    Inputs
    - theta: (n_theta,) polar angle grid in radians, 0..pi
    - phi:   (n_phi,) azimuth angle grid in radians, 0..2*pi (or similar)
    - I:     (n_theta, n_phi) intensity samples on the grid
    - title_prefix: optional text prefix for subplot titles
    - rmax: optional common radial limit for all polar plots

    Cuts plotted
    1) XY plane: theta = pi/2, angle in the plane is phi
    2) XZ plane: ny = 0, use two half-planes (phi=0 and phi=pi),
       angle in the plane is atan2(nz, nx)
    3) YZ plane: nx = 0, use two half-planes (phi=pi/2 and phi=3pi/2),
       angle in the plane is atan2(nz, ny)

    Returns
    - fig, axes
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    I = np.asarray(I)

    if I.ndim != 2 or I.shape != (theta.size, phi.size):
        raise ValueError(f"I must have shape (len(theta), len(phi)) = "
                         f"({theta.size}, {phi.size}), got {I.shape}")

    # Make sure we do not plot negative radii if I has tiny numerical negatives
    I = np.maximum(I, 0.0)

    # Precompute direction cosines on the grid for consistent angle definitions
    # nx = sin(theta) * cos(phi), ny = sin(theta) * sin(phi), nz = cos(theta)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Helper: build a full-plane cut by stitching two half-planes
    def build_plane_cut(phi_targets):
        """
        Stitch two phi slices into a single (angle_in_plane, intensity) curve.
        phi_targets: list/tuple of two target azimuth angles (radians).
        """
        betas = []
        vals = []

        for target in phi_targets:
            j = _nearest_index_periodic(phi, target)

            # Use the actual phi grid value at the chosen index
            phi_j = phi[j]

            # Compute in-plane angle from direction cosines
            # For XZ plane: beta = atan2(nz, nx) where nx depends on phi_j
            # For YZ plane: beta = atan2(nz, ny) where ny depends on phi_j
            # We decide which based on the target values passed in
            # (Caller ensures correct targets for each plane)
            if np.isclose(_wrap_to_pi(target - 0.0), 0.0) or np.isclose(_wrap_to_pi(target - np.pi), 0.0):
                # XZ plane: use (nx, nz)
                nx = sin_t * np.cos(phi_j)
                nz = cos_t
                beta = np.arctan2(nz, nx)
            else:
                # YZ plane: use (ny, nz)
                ny = sin_t * np.sin(phi_j)
                nz = cos_t
                beta = np.arctan2(nz, ny)

            betas.append(beta)
            vals.append(I[:, j])

        beta_all = np.concatenate(betas)
        val_all = np.concatenate(vals)

        # Sort by angle so the polar plot draws a clean curve
        order = np.argsort(beta_all)
        return beta_all[order], val_all[order]

    # --- Cut 1: XY plane (theta = pi/2) ---
    i_xy = _nearest_index(theta, np.pi / 2)
    beta_xy = phi.copy()  # In the XY plane, in-plane angle equals phi
    I_xy = I[i_xy, :].copy()

    # To avoid a duplicate point if phi includes both 0 and 2*pi, we can drop the last point
    # when it is effectively the same as the first.
    if phi.size >= 2 and np.isclose(_wrap_to_pi(phi[-1] - phi[0]), 0.0, atol=1e-12):
        beta_xy = beta_xy[:-1]
        I_xy = I_xy[:-1]

    # --- Cut 2: XZ plane (ny = 0) ---
    beta_xz, I_xz = build_plane_cut((0.0, np.pi))

    # --- Cut 3: YZ plane (nx = 0) ---
    beta_yz, I_yz = build_plane_cut((np.pi / 2, 3 * np.pi / 2))

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"}, figsize=(14, 4.5))

    # XY
    axes[0].plot(beta_xy, I_xy, lw=2)
    axes[0].set_title(f"{title_prefix} XY cut (theta ~= pi/2)".strip())
    #axes[0].set_rlim(0, rmax)

    # XZ
    axes[1].plot(beta_xz, I_xz, lw=2)
    axes[1].set_title(f"{title_prefix} XZ cut (ny = 0)".strip())
    #axes[1].set_rlim(0, rmax)

    # YZ
    axes[2].plot(beta_yz, I_yz, lw=2)
    axes[2].set_title(f"{title_prefix} YZ cut (nx = 0)".strip())
    #axes[2].set_rlim(0, rmax)

    fig.tight_layout()
    log.info("Plotting plannar cut.")
    return fig, axes


def _validate_plot_atoms_inputs(r_xyz, w=None, v_xyz=None):
    """Validate shapes and convert inputs to arrays."""
    r_xyz = np.asarray(r_xyz, dtype=float)
    if r_xyz.ndim != 2 or r_xyz.shape[1] != 3:
        raise ValueError(f"r_xyz must have shape (N, 3), got {r_xyz.shape}")

    N = r_xyz.shape[0]

    if w is not None:
        w = np.asarray(w)
        if w.shape != (N,):
            raise ValueError(f"w must have shape (N,), got {w.shape}")

    if v_xyz is not None:
        v_xyz = np.asarray(v_xyz, dtype=float)
        if v_xyz.shape != r_xyz.shape:
            raise ValueError(f"v_xyz must have shape {r_xyz.shape}, got {v_xyz.shape}")

    return r_xyz, w, v_xyz


def _compute_cloud_geometry(r_xyz):
    """Return center, bounds, and characteristic span of the cloud."""
    center = np.mean(r_xyz, axis=0)
    mins = r_xyz.min(axis=0)
    maxs = r_xyz.max(axis=0)

    # Use the largest extent to keep axis and arrow scaling consistent.
    span = np.max(maxs - mins)
    if span <= 0:
        span = 1.0

    return {
        "center": center,
        "mins": mins,
        "maxs": maxs,
        "span": span,
    }


def plot_atoms(
    r_xyz,
    title="Atoms",
    s=3,
    alpha=0.6,
    equal_axes=True,
    w=None,
    show_colorbar=True,
    p_hat=None,
    k_in_hat=None,
    arrow_scale=0.25,
    v_xyz=None,
    v_subsample=20,
    v_arrow_scale=0.08,
    seed=0,
    r_subsample=10000,
):
    """Plot atoms in 3D with optional weights and direction arrows."""
    r_xyz, w, v_xyz = _validate_plot_atoms_inputs(r_xyz, w=w, v_xyz=v_xyz)
    geom = _compute_cloud_geometry(r_xyz)

    N = r_xyz.shape[0]
    rng = np.random.default_rng(seed)

    # Subsample atom positions if requested.
    if r_subsample is None or r_subsample >= N:
        idx_scatter = np.arange(N)
    else:
        idx_scatter = rng.choice(N, size=int(r_subsample), replace=False)

    r_plot = r_xyz[idx_scatter]
    x, y, z = r_plot[:, 0], r_plot[:, 1], r_plot[:, 2]

    center = geom["center"]
    mins = geom["mins"]
    maxs = geom["maxs"]
    span = geom["span"]

    def _norm(v):
        """Return a safely normalized 3-vector."""
        v = np.asarray(v, dtype=float)
        return v / (np.linalg.norm(v) + 1e-15)

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter atoms, optionally colored by normalized weight magnitude.
    if w is None:
        ax.scatter(x, y, z, s=s, alpha=alpha)
    else:
        w_plot = w[idx_scatter]
        amp = np.abs(w_plot).astype(float)
        amp_norm = amp / (amp.max() + 1e-15)
        sc = ax.scatter(x, y, z, s=s, alpha=alpha, c=amp_norm)

        if show_colorbar:
            cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
            cb.set_label("|w| (normalized)")

    # Global direction arrows originate at the cloud center.
    arrow_len = arrow_scale * span

    if p_hat is not None:
        pdir = _norm(p_hat)
        ax.quiver(
            center[0], center[1], center[2],
            pdir[0], pdir[1], pdir[2],
            length=arrow_len,
            normalize=True,
            color="blue",
        )

    if k_in_hat is not None:
        kdir = _norm(k_in_hat)
        ax.quiver(
            center[0], center[1], center[2],
            kdir[0], kdir[1], kdir[2],
            length=arrow_len,
            normalize=True,
            color="purple",
        )

    # Per-atom velocity arrows are drawn on a smaller sampled subset.
    if v_xyz is not None:
        M = idx_scatter.size
        if M > v_subsample:
            idx_local = rng.choice(M, size=int(v_subsample), replace=False)
            idx_vel = idx_scatter[idx_local]
        else:
            idx_vel = idx_scatter

        rr = r_xyz[idx_vel]
        vv = v_xyz[idx_vel]
        vel_len = v_arrow_scale * span

        ax.quiver(
            rr[:, 0], rr[:, 1], rr[:, 2],
            vv[:, 0], vv[:, 1], vv[:, 2],
            length=vel_len,
            normalize=True,
            color="red",
            alpha=0.8,
        )

    # Axes labels and optional equal scaling.
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if equal_axes:
        c = 0.5 * (mins + maxs)
        half = 0.55 * span
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_zlim(c[2] - half, c[2] + half)
        ax.set_box_aspect((1, 1, 1))

    # Proxy artists for a compact legend.
    legend_items = []
    if p_hat is not None:
        legend_items.append(Line2D([0], [0], color="blue", lw=3, label="Dipole direction"))
    if k_in_hat is not None:
        legend_items.append(Line2D([0], [0], color="purple", lw=3, label="Incident k-vector"))
    if v_xyz is not None:
        legend_items.append(Line2D([0], [0], color="red", lw=3, label=f"Velocities (sampled <= {v_subsample})"))
    if r_subsample is not None:
        legend_items.append(
            Line2D(
                [0], [0],
                color="black",
                lw=0,
                marker="o",
                label=f"Atoms plotted: {idx_scatter.size}/{N}",
            )
        )
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right")

    plt.tight_layout()
    log.info("Plotting atom positions. Title = %s", title)
    return fig, ax


#### Animations

# Atoms animation. 

def animation_atoms_with_pulse(r_pos,T, weights: np.array = None, pulse_center:np.array = None ):
    """ Creates animation of the atoms movements and the pulse traveling trhough the cloud"""


    if weights is None: 
        weights = np.ones(r_pos[0].shape[0])
        print("weights shape")
        print(weights.shape)
        fig, ax = plot_atoms(r_pos[0],w=weights )
    # Calls atom plot to create the first instance
    else:   
        fig, ax = plot_atoms(r_pos[0],w=weights[0] )
    # Collects the atoms scatter
    scat = ax.collections[0]

    #Creates pulse center point for tracking. 
    if pulse_center is not None:
        pulse_point = ax.scatter(
            [pulse_center[0, 0]],
            [pulse_center[0, 1]],
            [pulse_center[0, 2]],
            color="red",
            s=80,
            marker="o",
            label="Pulse center",
        )   

    def update(frame):
        """ frame update function for the plot_atom 
            updates atom position and weights and plots the center of the pulse 
        """

        xyz = r_pos[frame]
        # update point positions
        scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])

        # update colors if needed
        if weights is not None: 
            amp = np.abs(weights[frame])**2
            amp_norm = amp / (amp.max() + 1e-15)
            scat.set_array(amp_norm)

        # update pulse center
        if pulse_center is not None: 
            rp = pulse_center[frame]
            pulse_point._offsets3d = ([rp[0]], [rp[1]], [rp[2]])

        ax.set_title(f"Frame {frame}")
        return ax,

    # animation function. 
    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=50,
        blit=False
    )   

    return  ani

def plot_cloud_slices(
    r_xyz,
    w=None,
    center=(0.0, 0.0, 0.0),
    slab_thickness_code=None,
    slab_thickness_um=10.0,
    code_unit_m=None,
    s=5,
    alpha=0.7,
    cmap="viridis",
    figsize=(15, 4.8),
):
    """
    Plot thin-slab XY, XZ, and YZ cuts of a cloud.

    Parameters
    ----------
    r_xyz : ndarray, shape (N, 3)
        Cloud positions in code units.
    w : ndarray, shape (N,), optional
        Optional weights for coloring the points.
    center : tuple of float
        Slice center (x0, y0, z0) in code units.
        The cuts are:
            XY at z = z0
            XZ at y = y0
            YZ at x = x0
    slab_thickness_code : float, optional
        Slab thickness in code units.
        If None, it will be computed from slab_thickness_um and code_unit_m.
    slab_thickness_um : float
        Desired slab thickness in microns, used only if slab_thickness_code is None.
    code_unit_m : float, optional
        Size of 1 code unit in meters.
        Needed if you want to convert slab_thickness_um into code units.
    s : float
        Marker size for scatter.
    alpha : float
        Marker transparency.
    cmap : str
        Matplotlib colormap.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes
    """

    r_xyz = np.asarray(r_xyz)
    if r_xyz.ndim != 2 or r_xyz.shape[1] != 3:
        raise ValueError("r_xyz must have shape (N, 3)")

    if w is not None:
        w = np.asarray(w)
        if w.shape[0] != r_xyz.shape[0]:
            raise ValueError("w must have same length as r_xyz")

    x, y, z = r_xyz[:, 0], r_xyz[:, 1], r_xyz[:, 2]
    x0, y0, z0 = center

    # Decide slab thickness
    if slab_thickness_code is None:
        if code_unit_m is not None:
            slab_thickness_code = (slab_thickness_um * 1e-6) / code_unit_m
        else:
            # fallback: pick something small relative to cloud size
            extent = np.max(np.ptp(r_xyz, axis=0))
            slab_thickness_code = 0.02 * extent
            print(
                f"[plot_cloud_slices] No code_unit_m provided. "
                f"Using slab_thickness_code={slab_thickness_code:.4g}"
            )

    half = slab_thickness_code / 2.0

    # Masks for thin slabs
    mask_xy = np.abs(z - z0) <= half   # slice around z=z0
    mask_xz = np.abs(y - y0) <= half   # slice around y=y0
    mask_yz = np.abs(x - x0) <= half   # slice around x=x0

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    def _scatter(ax, mask, a, b, xlabel, ylabel, title):
        if np.count_nonzero(mask) == 0:
            ax.text(0.5, 0.5, "No points in slice", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            return None

        if w is None:
            sc = ax.scatter(a[mask], b[mask], s=s, alpha=alpha)
        else:
            sc = ax.scatter(a[mask], b[mask], c=w[mask], s=s, alpha=alpha, cmap=cmap)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nN={np.count_nonzero(mask)}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("auto")
        return sc

    sc0 = _scatter(
        axes[0], mask_xy, x, y,
        "x (code units)", "y (code units)",
        f"XY slice at z={z0:.4g}\nΔz={slab_thickness_code:.4g}"
    )
    sc1 = _scatter(
        axes[1], mask_xz, x, z,
        "x (code units)", "z (code units)",
        f"XZ slice at y={y0:.4g}\nΔy={slab_thickness_code:.4g}"
    )
    sc2 = _scatter(
        axes[2], mask_yz, y, z,
        "y (code units)", "z (code units)",
        f"YZ slice at x={x0:.4g}\nΔx={slab_thickness_code:.4g}"
    )

    if w is not None:
        # use one common colorbar
        sc_for_cb = sc0 if sc0 is not None else (sc1 if sc1 is not None else sc2)
        if sc_for_cb is not None:
            cbar = fig.colorbar(sc_for_cb, ax=axes, shrink=0.9)
            cbar.set_label("weight")

    return fig, axes
