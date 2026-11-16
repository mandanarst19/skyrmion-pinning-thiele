"""Dwell-time extraction from site-label time series and exponential fits.

Frame series for conversion factor n (STU/frame = 0.01 n): labels at samples n-1, 2n-1, ... of each segment (0 = no site, 1 = A, 2 = B).
Dwell time t (frames, integer >= 1) = (frame index where the dwell ends) - (frame index where it starts).
A dwell is recorded only if its START was observed (the first frame of a segment is never a start: the skyrmion may have been there since t = 0).
variant 0 ('ign'):  frames with no site are ignored; a dwell at site X ends at the first frame labelled with the other site (hop).
variant 1 ('end'):  a frame with no site ends the dwell (end type 1); a hop is counted only if the very next frame is the other site.
End types: 0 = hop to the other site, 1 = ended by a no-site frame (variant 1 only), 2 = ended by leaving the cluster (segment end).
Records: [site (1=A, 2=B), end_type, t_frames, walker_id].
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar


@njit(cache=True)
def extract_dwells(lab, seg_n, seg_exit, seg_walker, n, variant):
    nseg = seg_n.shape[0]
    tot = 0
    for i in range(nseg):
        tot += seg_n[i] // n
    rec = np.empty((tot + nseg + 10, 4), dtype=np.int64)
    nr = 0; off = 0
    for sidx in range(nseg):
        K = seg_n[sidx] // n
        w = seg_walker[sidx]
        cur = 0; tstart = 0; obs = False; prev = -1
        for k in range(K):
            l = lab[off + k * n + n - 1]
            if variant == 0:
                if l == 0:
                    continue
                if cur == 0:
                    cur = l; tstart = k; obs = (k > 0)
                elif l != cur:
                    if obs:
                        rec[nr, 0] = cur; rec[nr, 1] = 0; rec[nr, 2] = k - tstart; rec[nr, 3] = w; nr += 1
                    cur = l; tstart = k; obs = True
            else:
                if l != prev:
                    if (prev == 1 or prev == 2) and obs:
                        rec[nr, 0] = prev; rec[nr, 1] = 1 if l == 0 else 0; rec[nr, 2] = k - tstart; rec[nr, 3] = w; nr += 1
                    if l == 1 or l == 2:
                        tstart = k; obs = (k > 0)
                    prev = l
        if seg_exit[sidx] == 1:
            if variant == 0:
                if cur != 0 and obs and K > tstart:
                    rec[nr, 0] = cur; rec[nr, 1] = 2; rec[nr, 2] = K - tstart; rec[nr, 3] = w; nr += 1
            else:
                if (prev == 1 or prev == 2) and obs and K > tstart:
                    rec[nr, 0] = prev; rec[nr, 1] = 2; rec[nr, 2] = K - tstart; rec[nr, 3] = w; nr += 1
        off += seg_n[sidx]
    return rec[:nr]


def tau_mle(t):
    """MLE of tau for P(t) = (e^{1/tau}-1) exp(-t/tau), t = 1, 2, ... (geometric: p = 1 - e^{-1/tau}, p_hat = 1/mean)."""
    m = np.mean(t)
    return np.nan if m <= 1.0 else -1.0 / np.log(1.0 - 1.0 / m)


def tau_ls(hist):
    """Least-squares fit of the model pmf to the normalised per-frame histogram hist[t], t = 1..tmax (hist[0] ignored); unweighted."""
    c = np.asarray(hist, dtype=float)[1:]
    p = c / c.sum(); t = np.arange(1, c.size + 1)
    f = lambda lt: np.sum((p - np.expm1(1.0 / np.exp(lt)) * np.exp(-t / np.exp(lt))) ** 2)
    r = minimize_scalar(f, bounds=(np.log(0.3), np.log(2e4)), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))
