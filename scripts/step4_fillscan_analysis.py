"""Fill-level scan of our single, non-interacting, unlinked skyrmion: D(fill), d ln D / d fill, and the effective fill at which our D
would equal (i) the paper's simulated D at 2.7827 k_BT (0.583, log-interpolated from Suppl. Data 1 'fig3') and (ii) the experimental 0.6141 uM2/s.
Same MSD estimator as step4_analysis.py (5-15 s linear fit with free intercept, d = 2, 1 frame = 62.5 ms).
Fills 2.28, 2.78, 3.28 were requested; 3.78 was ADDED to avoid extrapolating to the target. Walkers: 8 per fill (16 at 2.78)."""
import json
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cfg = yaml.safe_load(open("config.yaml")); XL = cfg["data"]["xlsx"]
FR, L1, L2 = 0.0625, 80, 240
rng = np.random.default_rng(1618); B = 4000
FILLS = ["2.28", "2.78", "3.28", "3.78"]; f = np.array([float(x) for x in FILLS])
def msd_w(p): return np.array([np.mean(np.sum((p[L:] - p[:-L]) ** 2, 1)) for L in range(1, L2 + 1)])
def Dfit(m):
    L = np.arange(L1, L2 + 1); return np.polyfit(L * FR, m[L - 1], 1)[0] / 4.0
M = {}
for lab in FILLS:
    P = np.concatenate([np.load(x)["pos"] for x in sorted(Path("results").glob(f"step4_fill{lab}_chunk*.npz"))]); M[lab] = np.array([msd_w(P[w]) for w in range(P.shape[0])])
D0 = np.array([Dfit(M[l].mean(0)) for l in FILLS])
def boot_D(): return np.array([Dfit(M[l][rng.integers(0, M[l].shape[0], M[l].shape[0])].mean(0)) for l in FILLS])
BD = np.array([boot_D() for _ in range(B)]); sig = np.log(BD).std(0)
def wfit(x, y, s):
    w = 1 / s ** 2; A = np.vstack([np.ones_like(x), x]).T; cov = np.linalg.inv(A.T @ (A * w[:, None])); c = cov @ (A.T @ (w * y)); return c[1], c[0]
def analyse(idx):
    b, a = wfit(f[idx], np.log(D0[idx]), sig[idx])
    bb = np.array([wfit(f[idx], np.log(BD[k, idx]), sig[idx]) for k in range(B)])
    out = {"fills": [float(x) for x in f[idx]], "slope_dlnD_dfill": float(b), "slope_boot_se": float(bb[:, 0].std()), "intercept": float(a)}
    for nm, tgt in (("paper_sim_at_2.7827", 0.58313), ("experiment_0.6141", 0.6141), ("experiment_0.6141-0.0125", 0.6016), ("experiment_0.6141+0.0125", 0.6266)):
        fs = (np.log(tgt) - a) / b; fb = (np.log(tgt) - bb[:, 1]) / bb[:, 0]
        out[f"effective_fill_for_D={nm}"] = {"target_um2_per_s": tgt, "fill_kBT": float(fs), "boot_se": float(fb.std()), "extrapolated_beyond_scan": bool(fs > f[idx].max() or fs < f[idx].min())}
    res = np.log(D0[idx]) - (a + b * f[idx]); out["chi2_linear_fit"] = float(np.sum((res / sig[idx]) ** 2)); out["dof"] = int(idx.sum() - 2) if idx.dtype == bool else int(len(idx) - 2)
    return out
R = {"fills": FILLS, "D_um2_per_s": [float(x) for x in D0], "D_boot_se": [float(x) for x in BD.std(0)], "walkers": [int(M[l].shape[0]) for l in FILLS],
     "per_walker_D_sd": [float(np.std([Dfit(m) for m in M[l]], ddof=1)) for l in FILLS]}
R["fit_all_four_fills"] = analyse(np.array([0, 1, 2, 3])); R["fit_requested_three_fills_2.28_2.78_3.28"] = analyse(np.array([0, 1, 2]))
f3 = pd.read_excel(XL, sheet_name="fig3", header=0).values; pf, pD = f3[:, 0], f3[:, 1]
pinterp = lambda x: float(np.exp(np.interp(x, pf, np.log(pD))))
R["paper_sim_D_at_scan_fills_(log-interp, linked+interacting)"] = [pinterp(x) for x in f]
R["ratio_ours_over_paper_sim"] = [float(D0[i] / pinterp(f[i])) for i in range(4)]
sel = (pf >= 2.25) & (pf <= 3.75); bp, ap = np.polyfit(pf[sel], np.log(pD[sel]), 1); R["paper_sim_local_slope_dlnD_dfill_(2.25-3.75)"] = float(bp)
json.dump(R, open("results/step4_fillscan.json", "w"), indent=1); print(json.dumps(R, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
a_ = ax[0]; a_.errorbar(f, D0, yerr=BD.std(0), fmt="o", color="b", capsize=3, label="ours: single skyrmion, no linking (bootstrap SE)")
ff = np.linspace(2.0, 4.1, 50); fa = R["fit_all_four_fills"]; a_.plot(ff, np.exp(fa["intercept"] + fa["slope_dlnD_dfill"] * ff), "b-", lw=1, label=f"log-linear fit, slope {fa['slope_dlnD_dfill']:.2f} ± {fa['slope_boot_se']:.2f} per k_BT")
a_.plot(pf[(pf >= 2) & (pf <= 4.1)], pD[(pf >= 2) & (pf <= 4.1)], "k.-", label="paper simulation (linked, interacting)")
a_.axhspan(0.6141 - 0.0125, 0.6141 + 0.0125, color="tab:red", alpha=0.3, label="experiment 0.614 ± 0.0125"); a_.axvline(2.78, color="gray", ls=":")
a_.set(yscale="log", xlabel="fill level [k_BT]", ylabel="D [µm²/s]", title="D vs fill level"); a_.legend(fontsize=7)
ax[1].plot(f, R["ratio_ours_over_paper_sim"], "o-"); ax[1].set(xlabel="fill level [k_BT]", ylabel="D_ours / D_paper-sim", title="ratio of our unlinked to the paper's linked simulation", ylim=(1, 3)); ax[1].axhline(1, color="k", lw=0.5)
fig.savefig("figures/step4_fillscan.png", dpi=150)
