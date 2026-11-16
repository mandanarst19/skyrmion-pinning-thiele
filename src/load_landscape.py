"""Load the Fig. 1a pinning landscape and trajectory from Supplementary Data 1."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_energy(xlsx, sheet="fig1_a", cache="data/fig1_a.npy"):
    """Return E[row=y, col=x] in kBT; empty (unsampled) bins are +inf."""
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    E = pd.read_excel(xlsx, sheet_name=sheet, header=None).to_numpy(dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, E)
    return E


def load_trajectory(xlsx, sheet="fig1_b"):
    """Return DataFrame with columns frame, x_um, y_um (header row is text)."""
    d = pd.read_excel(xlsx, sheet_name=sheet, header=0)
    d.columns = ["frame", "x_um", "y_um"]
    return d


def empty_mask(E):
    return ~np.isfinite(E)


def fill_empty(E, fill):
    """Replace unsampled (inf) bins by a flat level `fill` (kBT, same zero as E)."""
    U = E.copy()
    U[~np.isfinite(U)] = fill
    return U


def counts_from_energy(E):
    """Inferred raw counts n = exp(-E) for finite bins (assumes E = -ln n)."""
    n = np.zeros_like(E)
    f = np.isfinite(E)
    n[f] = np.exp(-E[f])
    return n


def positions_to_bins(x_um, y_um, h):
    """Bin index (row, col) of positions; row <- y, col <- x. Bin j covers [j*h,(j+1)*h)."""
    c = np.floor(np.asarray(x_um) / h).astype(int)
    r = np.floor(np.asarray(y_um) / h).astype(int)
    return r, c
