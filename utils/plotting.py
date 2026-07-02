"""
Shared plotting helpers for Koopman-eigenfunction Lyapunov notebooks.

All functions return the Figure so callers can call fig.savefig(...) or plt.show().
"""

import numpy as np
import matplotlib.pyplot as plt


def surface3d(X, Y, Z, title="", xlabel="x", ylabel="y", zlabel="",
              cmap="viridis", figsize=(6, 5)):
    """
    Create a 3-D surface plot of Z over the meshgrid (X, Y).

    Parameters
    ----------
    X, Y : 2-D meshgrid arrays
    Z    : 2-D array matching X and Y shape
    title, xlabel, ylabel, zlabel : strings for axis/title labels
    cmap : colormap name ('viridis' for V, 'magma' for Vdot)
    figsize : tuple passed to plt.figure

    Returns
    -------
    fig : matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)
    plt.tight_layout()
    return fig


def lyapunov_pair(X, Y, V, Vdot,
                  title_V=r"Learned $V^\star(x)$",
                  title_Vdot=r"Learned $\dot V^\star(x)$",
                  save_V=None, save_Vdot=None, dpi=150):
    """
    Plot V and Vdot side-by-side in two separate figures and optionally save.

    Parameters
    ----------
    X, Y    : meshgrid arrays (shape n_x x n_y)
    V       : 2-D array, Lyapunov function on the grid
    Vdot    : 2-D array, orbital derivative on the grid
    title_V, title_Vdot : plot titles
    save_V, save_Vdot   : file paths to save (None = don't save)
    dpi     : resolution for saved figures

    Returns
    -------
    fig_V, fig_Vdot : the two Figure objects
    """
    fig_V = surface3d(X, Y, V, title=title_V, cmap="viridis")
    if save_V:
        fig_V.savefig(save_V, dpi=dpi, bbox_inches="tight")

    fig_Vdot = surface3d(X, Y, Vdot, title=title_Vdot, cmap="magma")
    if save_Vdot:
        fig_Vdot.savefig(save_Vdot, dpi=dpi, bbox_inches="tight")

    return fig_V, fig_Vdot


def check_lyapunov(XY, V, Vdot, label="V*"):
    """
    Print a quick sanity check: V(0)~0, min V>0, max Vdot<0.

    Parameters
    ----------
    XY   : (N, d) array of grid points
    V    : (N,) Lyapunov values
    Vdot : (N,) orbital derivative values
    label : string label used in print output
    """
    origin_idx = np.argmin(np.sum(XY ** 2, axis=1))
    mask = np.sum(XY ** 2, axis=1) > 1e-6
    print(f"=== {label} ===")
    print(f"  {label}(0)          = {V[origin_idx]:.3e}   (want ~ 0)")
    print(f"  min {label} off 0   = {V[mask].min():.3e}   (want > 0)")
    print(f"  max d{label}/dt off 0 = {Vdot[mask].max():.3e}   (want < 0)")
