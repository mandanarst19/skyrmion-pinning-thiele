"""Step 2 (a)-(c): free diffusion, harmonic well (Boltzmann + dt halving), damping unit check.
Test parameters (walker numbers, well stiffness, durations) are OUR choices for validation; nothing here is a paper result."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.thiele_heun import run_trajectories

rng = np.random.default_rng(12345)
res = {}
Fz = np.zeros((1, 1)); h = 0.0457
FRAME_S, STU_PER_FRAME, STU_PER_FRAME_ERR = 1 / 16, 1.34, 0.10   # paper: 16 fps; 1 frame = (1.34 +- 0.10) STU

# ---------------- (a) free diffusion ----------------
N, T, dt, save = 10000, 2.0, 1e-4, 1000          # save every 0.1 STU
x0 = np.zeros(N)
traj = run_trajectories(0, Fz, Fz, h, 0.0, x0, x0, int(T / dt), dt, save, 1000)
t = np.arange(traj.shape[1]) * save * dt
r2 = ((traj - traj[:, :1, :]) ** 2).sum(-1)       # (N, nt)
msd = r2.mean(0)
def slope0(r2m, t, tmin=0.2):
    m = t >= tmin
    return (t[m] * r2m[m]).sum() / (t[m] ** 2).sum()
D_fit = slope0(msd, t) / 4
bs = []
for _ in range(300):
    idx = rng.integers(0, N, N)
    bs.append(slope0(r2[idx].mean(0), t) / 4)
D_se = np.std(bs)
res["a_free_diffusion"] = {"N_walkers": N, "T_STU": T, "dt": dt, "D_fit_um2_per_STU": D_fit, "D_se": D_se,
                           "D_expected_um2_per_STU": 1.0, "rel_dev": D_fit - 1.0, "n_sigma": (D_fit - 1) / D_se,
                           "D_MSD_at_T": msd[-1] / (4 * T)}

# ---------------- (b) harmonic well ----------------
k = 25.0                      # kBT/um^2  -> sigma = 0.2 um (inside the 200-400 nm site widths quoted in Suppl. Note 1)
sig = 1 / np.sqrt(k)
Nw, Tw, save_t, burn = 1000, 10.0, 0.1, 1.0
b = {}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
samples_main = None
for dtt in (1e-3, 1e-4, 5e-5):
    sv = int(round(save_t / dtt))
    tr = run_trajectories(1, Fz, Fz, h, k, np.zeros(Nw), np.zeros(Nw), int(Tw / dtt), dtt, sv, 7)
    nb = int(round(burn / save_t))
    xs = tr[:, nb:, :]                                          # (Nw, ns, 2), discard burn-in
    var_w = (xs ** 2).mean(axis=(1, 2))                         # per-walker <x^2> (x and y pooled)
    var = var_w.mean(); se = var_w.std(ddof=1) / np.sqrt(Nw)
    pooled = (xs / sig).ravel()
    ks = stats.kstest(pooled[::7], "norm")                      # thin by 7 to reduce correlation effect
    b[f"dt={dtt:g}"] = {"var_um2": var, "var_expected": 1 / k, "rel_dev": var * k - 1, "se_rel": se * k,
                        "n_sigma": (var * k - 1) / (se * k), "KS_p_thinned": ks.pvalue, "n_samples": int(pooled.size)}
    if dtt == 1e-4:
        samples_main = pooled
    ax[2].errorbar(dtt, var * k - 1, se * k, fmt="o", capsize=3, label=f"dt={dtt:g}")
res["b_harmonic"] = b
# histogram vs Boltzmann (dt = 1e-4)
bins = np.linspace(-4, 4, 65); cen = 0.5 * (bins[1:] + bins[:-1])
hh, _ = np.histogram(samples_main, bins=bins, density=True)
ax[1].semilogy(cen, hh, "o", ms=3, label="simulation, dt=1e-4")
ax[1].semilogy(cen, stats.norm.pdf(cen), "-", label="Boltzmann exp(-E/kBT)")
ax[1].set(xlabel="x / σ   (σ = 0.2 µm, x and y pooled)", ylabel="probability density", title="(b) harmonic well: position histogram")
ax[1].legend()
ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xscale("log")
ax[2].set(xlabel="dt [STU]", ylabel="⟨x²⟩k_BT⁻¹k − 1", title="(b) dt convergence of the variance"); ax[2].legend()
ax[0].plot(t, msd, "o", label="simulation (N=10⁴)"); ax[0].plot(t, 4 * D_fit * t, "-", label=f"fit: D = {D_fit:.4f} ± {D_se:.4f} µm²/STU")
ax[0].plot(t, 4 * t, "k--", lw=0.8, label="4 k_BT t/γ  (D = 1)")
ax[0].set(xlabel="t [STU]", ylabel="MSD [µm²]", title="(a) free diffusion"); ax[0].legend()
fig.savefig("figures/step2_validation_abc.png", dpi=160, bbox_inches="tight")

# ---------------- (c) unit check ----------------
stu_s = FRAME_S / STU_PER_FRAME
res["c_units"] = {"seconds_per_STU": stu_s, "gamma_kBT_s_per_um2": stu_s,
                  "gamma_range_from_conversion_pm0.10": [FRAME_S / (STU_PER_FRAME + STU_PER_FRAME_ERR), FRAME_S / (STU_PER_FRAME - STU_PER_FRAME_ERR)],
                  "paper_gamma": "0.047 +- 0.004 kBT s um^-2 (Commun. Phys. 9:243, Results)",
                  "D_free_um2_per_s": 1.0 / stu_s}
Path("results").mkdir(exist_ok=True)
json.dump(res, open("results/step2_validation.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
