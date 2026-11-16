"""Step 4 analysis: MSD-based diffusion coefficient of a single non-interacting skyrmion (no trajectory linking).
Paper (Methods): <[dx(dt)]^2> = 2 d D dt with d = 2; sliding window; per-trajectory MSD then duration-weighted average
(equal durations here -> plain mean over walkers); linear fit over 5-15 s. Time conversion: 1 frame = 1.34 STU = 62.5 ms.
Fit with free intercept is the primary estimator (the paper says 'linear function'); through-origin shown as a check."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cfg = yaml.safe_load(open("config.yaml")); XL = cfg["data"]["xlsx"]
FRAME_S, FRAME_STU = 0.0625, 1.34
L1, L2 = 80, 240                                     # 5 s and 15 s in frames (16 fps)
rng = np.random.default_rng(2718); B = 4000

def load(lab):
    fs = sorted(Path("results").glob(f"step4_fill{lab}_chunk*.npz")); return np.concatenate([np.load(f)["pos"] for f in fs]), len(fs)
def msd_walker(p, Lmax):
    return np.array([np.mean(np.sum((p[L:] - p[:-L]) ** 2, axis=1)) for L in range(1, Lmax + 1)])
def fit(msd):
    L = np.arange(L1, L2 + 1); t = L * FRAME_S; y = msd[L - 1]
    b, a = np.polyfit(t, y, 1); D0 = np.sum(t * y) / (4 * np.sum(t ** 2))
    al = np.polyfit(np.log(t), np.log(y), 1)[0]
    return b / 4.0, a, D0, al                         # D [um^2/s] (free intercept), intercept [um^2], D through origin, log-log exponent
res = {}; M = {}   # "flat_dt0.01" = primary flat control (zero force, dt=1e-2, exact); "flat" = short dt=1e-4 run (consistency check)
for lab in ("2.78", "flat_dt0.01", "flat"):
    pos, nch = load(lab); W = pos.shape[0]
    mw = np.array([msd_walker(pos[w], L2) for w in range(W)]); M[lab] = mw
    D, a, D0, al = fit(mw.mean(0))
    Db = np.array([fit(mw[rng.integers(0, W, W)].mean(0))[0] for _ in range(B)])
    Dw = np.array([fit(mw[w])[0] for w in range(W)])
    res[lab] = {"walkers": int(W), "T_STU_per_walker": float((pos.shape[1] - 1) * FRAME_STU), "T_total_STU": float(W * (pos.shape[1] - 1) * FRAME_STU),
                "frames_per_walker": int(pos.shape[1] - 1), "fit_window_s": [L1 * FRAME_S, L2 * FRAME_S],
                "D_um2_per_s_free_intercept": float(D), "D_bootstrap_se": float(Db.std()), "D_bootstrap_16_84": [float(np.percentile(Db, 16)), float(np.percentile(Db, 84))],
                "intercept_um2": float(a), "D_through_origin": float(D0), "loglog_exponent_5_15s": float(al),
                "per_walker_D": [float(x) for x in Dw], "per_walker_mean": float(Dw.mean()), "per_walker_sd": float(Dw.std(ddof=1)), "per_walker_sem": float(Dw.std(ddof=1) / np.sqrt(W)),
                "D_um2_per_STU": float(D * 0.0625 / FRAME_STU * FRAME_STU / FRAME_STU) if False else float(D * FRAME_S / FRAME_STU)}
# paper values (Supplementary Data 1, sheet fig3; text)
f3 = pd.read_excel(XL, sheet_name="fig3", header=0).values
fill, Dsim, eDsim, Dfit = f3[:, 0], f3[:, 1], f3[:, 2], f3[:, 3]; fm = 2.7827
i = int(np.searchsorted(fill, fm)) - 1; w_ = (fm - fill[i]) / (fill[i + 1] - fill[i])
interp = lambda y: float(np.exp((1 - w_) * np.log(y[i]) + w_ * np.log(y[i + 1])))
res["paper"] = {"fill_for_interp": fm, "sim_points_bracketing": {"fill": [float(fill[i]), float(fill[i + 1])], "D": [float(Dsim[i]), float(Dsim[i + 1])], "err": [float(eDsim[i]), float(eDsim[i + 1])]},
                "paper_sim_D_log_interp_at_2.7827": interp(Dsim), "paper_exp_fit_curve_at_2.7827": interp(Dfit), "paper_exp_D": [float(f3[0, 4]), float(f3[0, 5])],
                "paper_sim_D_at_fill0": [float(Dsim[0]), float(eDsim[0])], "paper_no_pinning_low_density_D": [11.9, 0.9], "free_kBT_over_gamma_um2_per_s": 1.0 / FRAME_S * FRAME_STU,
                "note": "paper sim D = linked, interacting trajectories (70 particles, Crocker-Grier linking); ours = single non-interacting, no linking"}
r = res["2.78"]; P = res["paper"]
res["comparison"] = {"ours_over_paper_sim_interp": r["D_um2_per_s_free_intercept"] / P["paper_sim_D_log_interp_at_2.7827"],
                     "ours_over_experiment": r["D_um2_per_s_free_intercept"] / P["paper_exp_D"][0],
                     "flat_ours_over_experiment": res["flat_dt0.01"]["D_um2_per_s_free_intercept"] / P["paper_exp_D"][0], "paper_no_pinning_over_experiment": 11.9 / P["paper_exp_D"][0],
                     "ours_flat_over_kBT_gamma": res["flat_dt0.01"]["D_um2_per_s_free_intercept"] / P["free_kBT_over_gamma_um2_per_s"],
                     "pinning_suppression_ours_flat_over_pinned": res["flat_dt0.01"]["D_um2_per_s_free_intercept"] / r["D_um2_per_s_free_intercept"],
                     "pinning_suppression_paper_no_pinning_over_sim": 11.9 / P["paper_sim_D_log_interp_at_2.7827"]}
json.dump(res, open("results/step4_analysis.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "paper"}, indent=1)); print(json.dumps(res["paper"], indent=1))
# figures
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
L = np.arange(1, L2 + 1) * FRAME_S
for w in range(M["2.78"].shape[0]): ax[0].plot(L, M["2.78"][w], color="tab:blue", lw=0.6, alpha=0.35)
ax[0].plot(L, M["2.78"].mean(0), "b-", lw=2, label=f"fill 2.78: D = {r['D_um2_per_s_free_intercept']:.3f} µm²/s")
ax[0].plot(L, M["flat_dt0.01"].mean(0), "k-", lw=2, label=f"flat: D = {res['flat_dt0.01']['D_um2_per_s_free_intercept']:.1f} µm²/s")
ax[0].axvspan(L1 * FRAME_S, L2 * FRAME_S, color="gold", alpha=0.25, label="fit window 5–15 s")
ax[0].set(xscale="log", yscale="log", xlabel="Δt [s]", ylabel="MSD [µm²]", title="MSD, 16 walkers × 50,000 STU (thin: single walkers)"); ax[0].legend(fontsize=8)
names = ["ours\nfill 2.78\n(single, no link)", "paper sim\n@2.78 (interp.,\nlinked+interact.)", "experiment\n(Fig. 3)", "ours flat", "paper\nno pinning", "k_BT/γ"]
vals = [r["D_um2_per_s_free_intercept"], P["paper_sim_D_log_interp_at_2.7827"], P["paper_exp_D"][0], res["flat_dt0.01"]["D_um2_per_s_free_intercept"], 11.9, P["free_kBT_over_gamma_um2_per_s"]]
errs = [r["D_bootstrap_se"], 0.0065, P["paper_exp_D"][1], res["flat_dt0.01"]["D_bootstrap_se"], 0.9, 0]
ax[1].errorbar(range(6), vals, yerr=errs, fmt="o", color="k", capsize=3); ax[1].set_xticks(range(6)); ax[1].set_xticklabels(names, fontsize=7); ax[1].set_yscale("log")
ax[1].set(ylabel="D [µm²/s]", title="Diffusion coefficients (errors: bootstrap over walkers / paper)")
ax[2].plot(np.arange(len(r["per_walker_D"])), r["per_walker_D"], "o"); ax[2].axhline(r["D_um2_per_s_free_intercept"], color="b"); ax[2].axhline(P["paper_exp_D"][0], color="tab:red", ls="--", label="experiment 0.614")
ax[2].axhline(P["paper_sim_D_log_interp_at_2.7827"], color="gray", ls=":", label="paper sim (interp.) 0.583"); ax[2].set(xlabel="walker", ylabel="D [µm²/s]", title="per-walker D (50,000 STU = 38 min each)"); ax[2].legend(fontsize=8)
fig.savefig("figures/step4_msd_diffusion.png", dpi=150); print("done")
