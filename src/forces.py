"""Pinning force from the (filled) binned potential: central differences, nearest-bin lookup."""
import numpy as np


def central_difference_force(U, h):
    """F = -grad U with central differences, periodic wrap (matches periodic dynamics).
    Axis 0 = y (rows), axis 1 = x (cols). Units: kBT/um if U in kBT and h in um.
    Call this on the FILLED potential."""
    Fx = -(np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2.0 * h)
    Fy = -(np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2.0 * h)
    return Fx, Fy


def interface_masks(empty):
    """Bins on the finite/empty interface (4-neighbour, periodic).
    finite_side: finite bins with >=1 empty 4-neighbour; filled_side: empty bins with >=1 finite one."""
    nb_empty = np.zeros_like(empty)
    nb_finite = np.zeros_like(empty)
    for ax in (0, 1):
        for s in (-1, 1):
            r = np.roll(empty, s, axis=ax)
            nb_empty |= r
            nb_finite |= ~r
    finite = ~empty
    return finite & nb_empty, empty & nb_finite


def force_at(Fx, Fy, x_um, y_um, h):
    """Nearest-bin lookup (piecewise-constant force), periodic indexing."""
    n = Fx.shape[0]
    c = np.floor(np.asarray(x_um) / h).astype(int) % n
    r = np.floor(np.asarray(y_um) / h).astype(int) % n
    return Fx[r, c], Fy[r, c]
