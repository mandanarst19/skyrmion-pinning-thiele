"""Overdamped Thiele (Langevin) dynamics, gyrotropic term neglected (G_rel = 0), stochastic Heun scheme.

Paper (Commun. Phys. 9:243, Results + Methods): gamma_sim = 1, kBT_sim = 1, lengths in um,
time in STU, Heun integrator, dt = 1e-4 STU, periodic boundaries, Gaussian white noise
with strength from the fluctuation-dissipation theorem.

Equation used:  gamma dr/dt = F_pin(r) + F_therm,   <F_therm_i(t) F_therm_j(t')> = 2 gamma kBT delta_ij delta(t-t')
Heun (predictor-corrector, same noise increment in both stages; exact for additive noise):
    r~     = r + F(r)/gamma dt + sqrt(2 kBT dt/gamma) xi
    r(t+dt)= r + 0.5 [F(r) + F(r~)]/gamma dt + sqrt(2 kBT dt/gamma) xi
Force modes: 0 = piecewise-constant grid (nearest bin, periodic), 1 = analytic harmonic well U = k r^2 / 2 (validation only).
Each walker has its own seeded RNG stream, so results do not depend on scheduling.
"""
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _force(mode, Fx, Fy, h, k, x, y):
    if mode == 0:
        n = Fx.shape[0]
        c = int(np.floor(x / h)) % n
        r = int(np.floor(y / h)) % n
        return Fx[r, c], Fy[r, c]
    return -k * x, -k * y


@njit(cache=True)
def run_trajectories(mode, Fx, Fy, h, k, x0, y0, n_steps, dt, save_every, seed, gamma=1.0, kT=1.0):
    """W independent walkers. Returns unwrapped positions at every `save_every` steps, shape (W, n_save+1, 2)."""
    W = x0.shape[0]
    n_save = n_steps // save_every
    out = np.empty((W, n_save + 1, 2))
    s = np.sqrt(2.0 * kT * dt / gamma)
    for w in range(W):
        np.random.seed(seed + w)
        x = x0[w]; y = y0[w]
        out[w, 0, 0] = x; out[w, 0, 1] = y
        for i in range(1, n_steps + 1):
            gx = np.random.normal() * s
            gy = np.random.normal() * s
            fx, fy = _force(mode, Fx, Fy, h, k, x, y)
            xp = x + fx * dt / gamma + gx
            yp = y + fy * dt / gamma + gy
            fxp, fyp = _force(mode, Fx, Fy, h, k, xp, yp)
            x += 0.5 * (fx + fxp) * dt / gamma + gx
            y += 0.5 * (fy + fyp) * dt / gamma + gy
            if i % save_every == 0:
                out[w, i // save_every, 0] = x
                out[w, i // save_every, 1] = y
    return out


@njit(cache=True)
def run_histogram(Fx, Fy, h, inside, r0, c0, nr, nc, x0, y0, n_steps, dt, sample_every, seed, gamma=1.0, kT=1.0):
    """Grid-force walkers; accumulate a bin-occupation histogram on the window rows r0:r0+nr, cols c0:c0+nc
    (only samples whose bin lies in the boolean map `inside` are counted). Also returns, per walker,
    the number of samples taken, samples inside `inside`, and the sample index of the first exit (-1 = never)."""
    W = x0.shape[0]
    n = Fx.shape[0]
    hist = np.zeros((W, nr, nc), dtype=np.int64)
    n_samp = np.zeros(W, dtype=np.int64)
    n_in = np.zeros(W, dtype=np.int64)
    first_exit = -np.ones(W, dtype=np.int64)
    s = np.sqrt(2.0 * kT * dt / gamma)
    for w in range(W):
        np.random.seed(seed + w)
        x = x0[w]; y = y0[w]
        for i in range(1, n_steps + 1):
            gx = np.random.normal() * s
            gy = np.random.normal() * s
            fx, fy = _force(0, Fx, Fy, h, 0.0, x, y)
            xp = x + fx * dt / gamma + gx
            yp = y + fy * dt / gamma + gy
            fxp, fyp = _force(0, Fx, Fy, h, 0.0, xp, yp)
            x += 0.5 * (fx + fxp) * dt / gamma + gx
            y += 0.5 * (fy + fyp) * dt / gamma + gy
            if i % sample_every == 0:
                c = int(np.floor(x / h)) % n
                r = int(np.floor(y / h)) % n
                n_samp[w] += 1
                if inside[r, c]:
                    n_in[w] += 1
                    rr = r - r0; cc = c - c0
                    if rr >= 0 and rr < nr and cc >= 0 and cc < nc:
                        hist[w, rr, cc] += 1
                elif first_exit[w] < 0:
                    first_exit[w] = n_samp[w]
    return hist, n_samp, n_in, first_exit


@njit(cache=True)
def run_histogram_confined(Fx, Fy, h, inside, r0, c0, nr, nc, x0, y0, n_steps, dt, sample_every, seed, gamma=1.0, kT=1.0):
    """Like run_histogram, but with a reflecting wall at the boundary of `inside`: a Heun step whose END point lies
    outside `inside` is rejected (walker stays). Interior stationary distribution stays Boltzmann; only wall-adjacent
    bins (within ~ one thermal step, ~0.3 bin at dt=1e-4) are biased. Returns hist, n_rejected per walker."""
    W = x0.shape[0]
    n = Fx.shape[0]
    hist = np.zeros((W, nr, nc), dtype=np.int64)
    n_rej = np.zeros(W, dtype=np.int64)
    s = np.sqrt(2.0 * kT * dt / gamma)
    for w in range(W):
        np.random.seed(seed + w)
        x = x0[w]; y = y0[w]
        for i in range(1, n_steps + 1):
            gx = np.random.normal() * s
            gy = np.random.normal() * s
            fx, fy = _force(0, Fx, Fy, h, 0.0, x, y)
            xp = x + fx * dt / gamma + gx
            yp = y + fy * dt / gamma + gy
            fxp, fyp = _force(0, Fx, Fy, h, 0.0, xp, yp)
            xn = x + 0.5 * (fx + fxp) * dt / gamma + gx
            yn = y + 0.5 * (fy + fyp) * dt / gamma + gy
            c = int(np.floor(xn / h)) % n
            r = int(np.floor(yn / h)) % n
            if inside[r, c]:
                x = xn; y = yn
            else:
                n_rej[w] += 1
            if i % sample_every == 0:
                c = int(np.floor(x / h)) % n
                r = int(np.floor(y / h)) % n
                hist[w, r - r0, c - c0] += 1
    return hist, n_rej


@njit(cache=True)
def run_labels(Fx, Fy, h, inside, labmap, cdf, rr, cc, budget_steps, dt, every, seed, mode, gamma=1.0, kT=1.0):
    """Step 3 kernel: one walker stream, site labels (labmap: 0 none, 1 site A, 2 site B) recorded every `every` steps.
    mode 0 (primary, as in the paper): unconfined; when a step ends outside `inside` (the cluster mask) the segment is
      terminated (that step is discarded) and a new segment starts from a bin drawn from the cluster's Boltzmann
      distribution (cdf over the mask bins rr, cc; position uniform within the bin).
    mode 1 (sensitivity): reflecting wall, a step ending outside is rejected; a single segment.
    budget_steps counts accepted steps inside the mask. Returns lab (int8, concatenated samples), seg_n (samples per
    segment), seg_exit (1 = segment ended by cluster exit, 0 = truncated by the budget), n_rejected."""
    n = Fx.shape[0]
    lab = np.empty(budget_steps // every + 2, dtype=np.int8)
    max_seg = 400000
    seg_n = np.zeros(max_seg, dtype=np.int64)
    seg_exit = np.zeros(max_seg, dtype=np.int8)
    np.random.seed(seed)
    s = np.sqrt(2.0 * kT * dt / gamma)
    nl = 0; nseg = 0; total = 0; n_rej = 0
    while total < budget_steps and nseg < max_seg:
        k = np.searchsorted(cdf, np.random.random())
        if k >= cdf.shape[0]:
            k = cdf.shape[0] - 1
        x = (cc[k] + np.random.random()) * h
        y = (rr[k] + np.random.random()) * h
        steps_seg = 0; ns = 0; exited = False
        while total < budget_steps:
            gx = np.random.normal() * s
            gy = np.random.normal() * s
            c0 = int(np.floor(x / h)) % n; r0 = int(np.floor(y / h)) % n
            fx = Fx[r0, c0]; fy = Fy[r0, c0]
            xp = x + fx * dt / gamma + gx
            yp = y + fy * dt / gamma + gy
            c1 = int(np.floor(xp / h)) % n; r1 = int(np.floor(yp / h)) % n
            xn = x + 0.5 * (fx + Fx[r1, c1]) * dt / gamma + gx
            yn = y + 0.5 * (fy + Fy[r1, c1]) * dt / gamma + gy
            c2 = int(np.floor(xn / h)) % n; r2 = int(np.floor(yn / h)) % n
            if not inside[r2, c2]:
                if mode == 0:
                    exited = True
                    break
                n_rej += 1
            else:
                x = xn; y = yn
            total += 1; steps_seg += 1
            if steps_seg % every == 0:
                lab[nl] = labmap[int(np.floor(y / h)) % n, int(np.floor(x / h)) % n]
                nl += 1; ns += 1
        seg_n[nseg] = ns
        seg_exit[nseg] = 1 if exited else 0
        nseg += 1
    return lab[:nl], seg_n[:nseg], seg_exit[:nseg], n_rej
