"""Optional Step-2 observation: 1D two-level step (Delta = 4 kBT) with the SAME force scheme as the 2D code
(central difference on the binned potential, periodic, piecewise-constant force, nearest-bin lookup, stochastic Heun).
Question: does the occupation near a sharp step show the under/over-occupation pattern seen at finite/filled interface bins?
Domain: 100 bins of 45.7 nm, bins 0-49 at 0 kBT, bins 50-99 at Delta (two steps, periodic). Walker/duration numbers are OUR choices.
Prediction: with piecewise-constant force the 1D stationary density is exactly exp(-U_eff), U_eff = continuous piecewise-linear
integral of the discrete force; compared against the simulation and against exp(-U_input)."""
import json, sys
from pathlib import Path
import numpy as np
from numba import njit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
h, N, Delta = 0.0457, 100, 4.0
U = np.where(np.arange(N) < N // 2, 0.0, Delta)
F = -(np.roll(U, -1) - np.roll(U, 1)) / (2 * h)

@njit(cache=True)
def run(F, h, W, n_steps, dt, every, seed):
    N = F.shape[0]; L = N * h
    hist = np.zeros(N, dtype=np.int64); s = np.sqrt(2.0 * dt)
    for w in range(W):
        np.random.seed(seed + w)
        x = np.random.random() * L
        for i in range(1, n_steps + 1):
            g = np.random.normal() * s
            f0 = F[int(np.floor(x / h)) % N]
            xp = x + f0 * dt + g
            f1 = F[int(np.floor(xp / h)) % N]
            x += 0.5 * (f0 + f1) * dt + g
            if i % every == 0:
                hist[int(np.floor(x / h)) % N] += 1
    return hist

W, T, dt = 100, 100.0, 1e-4
hist = run(F, h, W, int(T / dt), dt, 100, 5150)
P = hist / hist.sum()
Q = np.exp(-U); Q /= Q.sum()
# analytic effective potential: U_eff(x) continuous, slope -F_i inside bin i; bin-integrated Boltzmann weight
sub = 2000; x_in = (np.arange(N * sub) + 0.5) / sub                  # in bin units
Ueff_edges = np.concatenate([[0.0], np.cumsum(-F * h)])               # at bin edges; cyclic check: Ueff_edges[-1] ~ 0
Pe = np.empty(N)
for i in range(N):
    u = Ueff_edges[i] + (-F[i]) * h * (np.arange(sub) + 0.5) / sub
    Pe[i] = np.exp(-u).mean()
Pe /= Pe.sum()
centre_Ueff = np.array([Ueff_edges[i] + (-F[i]) * h * 0.5 for i in range(N)])
centre_Ueff = centre_Ueff - np.median(centre_Ueff - U)            # U_eff has a free additive constant: align far field with the input
se = np.sqrt(P * (1 - P) / (hist.sum() / 30))                       # crude: ~30 correlated samples per independent one (rough)
step_up = N // 2; step_dn = 0                                       # step between bins 49|50 (up) and 99|0 (down)
rows = []
for d in range(-4, 4):
    iu = (step_up + d) % N; idn = (step_dn + d) % N
    rows.append({"bin_rel_to_step(d<0 low side, d>=0 high side)": d,
                 "lnP_sim_over_Qinput_up_step": float(np.log(P[iu] / Q[iu])), "lnP_sim_over_Qinput_down_step": float(np.log(P[idn] / Q[idn])),
                 "lnP_analytic_over_Qinput_up_step": float(np.log(Pe[iu] / Q[iu])),
                 "Ueff_centre_minus_Uinput_up_step_kBT": float(centre_Ueff[iu] - U[iu])})
out = {"Delta_kBT": Delta, "F_in_two_bins_adjacent_to_step_kBT_per_um": float(F[step_up - 1]), "expected_Delta_over_2h": Delta / (2 * h),
       "force_zero_elsewhere": bool(np.allclose(np.delete(F, [step_up - 1, step_up, N - 1, 0]), 0)),
       "walkers": W, "T_STU": T, "dt": dt, "samples": int(hist.sum()), "rows": rows,
       "max_abs_lnP_sim_minus_analytic": float(np.max(np.abs(np.log(P / Pe)))), "cyclic_check_Ueff_edge_end": float(Ueff_edges[-1])}
json.dump(out, open("results/step2_1d_step_test.json", "w"), indent=1); print(json.dumps(out, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
i = np.arange(N); ax[0].plot(i, U, "k-", label="input U (2 levels)"); ax[0].plot(i, centre_Ueff, "r.-", label="effective U (bin centres), same scheme")
ax[0].set(xlim=(40, 60), xlabel="bin", ylabel="[k_BT]", title="central difference spreads the step over 2 bins"); ax[0].legend()
ax[1].plot(i - step_up, np.log(P / Q), "o", label="simulation"); ax[1].plot(i - step_up, np.log(Pe / Q), "-", label="analytic exp(−U_eff)")
ax[1].set(xlim=(-8, 8), xlabel="bin relative to step", ylabel="ln(P / exp(−U_input))", title="occupation near the step"); ax[1].legend(); ax[1].axhline(0, color="k", lw=0.6)
fig.savefig("figures/step2_1d_step_test.png", dpi=140)
