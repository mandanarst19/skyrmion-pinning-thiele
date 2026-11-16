"""Step 3 analysis (uses only saved results/step3_*_chunk*.npz and Supplementary Data 1, sheets fig2_a,b,d,e).
Conversion factor conv [STU/frame] = 0.01 n  (paper, Methods). Dwell definitions: see src/dwell.py.
Primary configuration: restart sampling, NaN frames ignored (variant 0), dwells ending in a hop only, MLE fit."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dwell import extract_dwells, tau_mle, tau_ls

cfg = yaml.safe_load(open("config.yaml")); XL = cfg["data"]["xlsx"]
rng = np.random.default_rng(31415); B = 300
SETS = ["base_restart", "base_reflect", "4disk_restart", "8sq_restart"]
GRID_LOG = np.unique(np.round(np.logspace(1, np.log10(995), 34)).astype(int))
NGRID = np.unique(np.concatenate([GRID_LOG, np.arange(100, 252, 2)]))
CONV = 0.01 * NGRID
VN = {0: "NaN ignored", 1: "NaN ends dwell"}; SITE = {1: "A", 2: "B"}

def load_set(tag):
    fs = sorted(Path("results").glob(f"step3_{tag}_chunk*.npz")); d = [np.load(f) for f in fs]
    return dict(lab=np.concatenate([x["lab"] for x in d]), seg_n=np.concatenate([x["seg_n"] for x in d]),
                seg_exit=np.concatenate([x["seg_exit"] for x in d]), seg_walker=np.concatenate([x["seg_walker"] for x in d]),
                budget=sum(float(x["budget_stu"]) * len(np.unique(x["seg_walker"])) for x in d), nfiles=len(fs))

def tau_from(N, S):
    N = np.asarray(N, float); S = np.asarray(S, float)
    with np.errstate(all="ignore"):
        m = S / N; t = -1.0 / np.log(1.0 - 1.0 / m)
    return np.where((N > 0) & (m > 1.0), t, np.nan)

def cross(conv, tau, target, lo=0.3, hi=4.0):
    ok = (conv >= lo) & (conv <= hi) & np.isfinite(tau); c, t = conv[ok], tau[ok]
    s = np.sign(t - target); k = np.where(s[:-1] * s[1:] < 0)[0]
    if k.size == 0: return np.nan
    k = k[0]; return float(np.exp(np.interp(np.log(target), np.log(t[k:k + 2])[::-1] if t[k] > t[k + 1] else np.log(t[k:k + 2]), np.log(c[k:k + 2])[::-1] if t[k] > t[k + 1] else np.log(c[k:k + 2]))))

# ---------- paper data ----------
paper = {}
for site, sb, sh in ((1, "fig2_b", "fig2_a"), (2, "fig2_e", "fig2_d")):
    d = pd.read_excel(XL, sheet_name=sb, header=0).values; hh = pd.read_excel(XL, sheet_name=sh, header=0).values
    paper[site] = dict(conv=d[:, 0], tau=d[:, 1], lo=d[0, 2], hi=d[0, 3], t=hh[:, 0], simh=hh[:, 1], exph=hh[:, 2])
PC = json.load(open("results/step3_paper_crossings.json"))

# ---------- dwell statistics per set ----------
R = {}
data = {}
for tag in SETS:
    D = load_set(tag); widx = np.unique(D["seg_walker"], return_inverse=True)[1]; W = widx.max() + 1
    segw = widx.astype(np.int64)
    cnt = np.zeros((2, 2, 3, NGRID.size, W)); sm = np.zeros_like(cnt)
    for v in (0, 1):
        for i, n in enumerate(NGRID):
            rec = extract_dwells(D["lab"], D["seg_n"], D["seg_exit"], segw, int(n), v)
            for si, site in enumerate((1, 2)):
                for et in range(3):
                    m = (rec[:, 0] == site) & (rec[:, 1] == et)
                    cnt[v, si, et, i] = np.bincount(rec[m, 3], minlength=W); sm[v, si, et, i] = np.bincount(rec[m, 3], weights=rec[m, 2], minlength=W)
    data[tag] = dict(D=D, cnt=cnt, sm=sm, W=W, segw=segw)
    R[tag] = {"walkers": int(W), "inside_time_STU_total": D["budget"], "n_segments": int(D["seg_n"].size), "n_exits": int(D["seg_exit"].sum()),
              "mean_residence_STU": float(D["budget"] / max(D["seg_n"].size, 1)) if D["seg_exit"].sum() else None}

SEL = {"hop only": [0], "all endings": [0, 1, 2]}
def NS(tag, v, si, sel, idx=None, ni=slice(None)):
    c = data[tag]["cnt"][v, si][SEL[sel]][:, ni].sum(0); s = data[tag]["sm"][v, si][SEL[sel]][:, ni].sum(0)
    if idx is not None: c, s = c[..., idx], s[..., idx]
    return c.sum(-1), s.sum(-1)

def tau_boot(tag, v, si, sel, ni, B=B):
    W = data[tag]["W"]; out = []
    for _ in range(B):
        idx = rng.integers(0, W, W); out.append(tau_from(*NS(tag, v, si, sel, idx, ni)))
    return np.array(out)

# ---------- tau_sim(conv), crossings: primary (restart) and reflecting-wall sensitivity ----------
cross_out = {}; curves = {}
for tag in ("base_restart", "base_reflect"):
    for si, site in enumerate((1, 2)):
        P = paper[site]
        for v in (0, 1):
            for sel in SEL:
                tau = tau_from(*NS(tag, v, si, sel)); tb = tau_boot(tag, v, si, sel, slice(None))
                key = f"{tag}|{SITE[site]}|{VN[v]}|{sel}"; curves[key] = (tau, np.nanpercentile(tb, 16, axis=0), np.nanpercentile(tb, 84, axis=0))
                row = {}
                for nm, target in (("tau_exp_lower", P["lo"]), ("tau_exp_mid", 0.5 * (P["lo"] + P["hi"])), ("tau_exp_upper", P["hi"])):
                    c0 = cross(CONV, tau, target); cb = np.array([cross(CONV, tb[b], target) for b in range(B)])
                    row[nm] = {"target_frames": float(target), "conv_STU_per_frame": c0, "bootstrap_se": float(np.nanstd(cb))}
                cross_out[key] = row
R["crossings"] = cross_out
R["paper_crossings_from_xlsx_curves"] = PC

# ---------- fixed-conversion tables (all sets) ----------
def nidx(c): return int(np.argmin(np.abs(CONV - c)))
tables = {}
for tg in SETS:
    for c in (1.34, 1.40, 1.48):
        ni = nidx(c)
        for si, site in enumerate((1, 2)):
            for v in (0, 1):
                for sel in SEL:
                    N, S = NS(tg, v, si, sel, None, ni); tb = tau_boot(tg, v, si, sel, ni, B=200)
                    ctot = data[tg]["cnt"][v, si][:, ni].sum(0).sum(); ex = data[tg]["cnt"][v, si][2, ni].sum()
                    tables[f"{tg}|conv={CONV[ni]:.2f}|{SITE[site]}|{VN[v]}|{sel}"] = {
                        "N_dwells": int(N), "N_hops": int(data[tg]["cnt"][v, si][0, ni].sum()), "exit_fraction_of_all_dwells": float(ex / ctot) if ctot else None,
                        "tau_mle": float(tau_from(N, S)), "tau_mle_boot_se": float(np.nanstd(tb))}
R["tables_fixed_conversion"] = tables

# ---------- least-squares fits (+bootstrap) at fixed conversions, primary and reflect sets ----------
def per_walker_hists(tag, v, si, sel, n, tmax):
    D = data[tag]["D"]; rec = extract_dwells(D["lab"], D["seg_n"], D["seg_exit"], data[tag]["segw"], int(n), v)
    m = (rec[:, 0] == (si + 1)) & np.isin(rec[:, 1], SEL[sel]); W = data[tag]["W"]
    H = np.zeros((W, tmax + 1))
    for w in range(W):
        t = rec[m & (rec[:, 3] == w), 2]; t = t[t <= tmax]; H[w] = np.bincount(t, minlength=tmax + 1)
    return H, rec[m, 2]
ls = {}
for tg in ("base_restart", "base_reflect"):
    for c in (1.34, 1.40, 1.48):
        n = int(round(c * 100))
        for si, site in enumerate((1, 2)):
            for v in (0, 1):
                for sel in SEL:
                    H0, tall = per_walker_hists(tg, v, si, sel, n, 1000)
                    tmax = int(min(1000, tall.max())); H = H0[:, :tmax + 1]
                    t_ls = tau_ls(H.sum(0)); bs = [tau_ls(H[rng.integers(0, H.shape[0], H.shape[0])].sum(0)) for _ in range(100)]
                    ls[f"{tg}|conv={c:.2f}|{SITE[site]}|{VN[v]}|{sel}"] = {"tau_ls": t_ls, "tau_ls_boot_se": float(np.std(bs)), "tau_mle": float(tau_mle(tall)), "N": int(tall.size), "t_max_used": tmax}
R["least_squares_fits"] = ls

# ---------- paper comparison of curves ----------
cmp_ = {}
for tg in ("base_restart", "base_reflect"):
    for si, site in enumerate((1, 2)):
        P = paper[site]; tau = curves[f"{tg}|{SITE[site]}|NaN ignored|hop only"][0]
        for c in (0.2, 0.5, 1.0, 1.34, 1.5, 3.0, 5.0, 9.9):
            ni = nidx(c); pt = float(np.interp(CONV[ni], P["conv"], P["tau"]))
            cmp_[f"{tg}|{SITE[site]}|conv={CONV[ni]:.2f}"] = {"tau_ours_frames": float(tau[ni]), "tau_paper_frames": pt, "ratio_ours_over_paper": float(tau[ni] / pt)}
R["curve_comparison_NaN_ignored_hop_only_vs_paper"] = cmp_
# histogram-shape comparison at each set's own best-match conversion (tau_exp mid)
hist_out = {}; HIST = {}
for tg in ("base_restart", "base_reflect"):
    for si, site in enumerate((1, 2)):
        P = paper[site]; cm = R["crossings"][f"{tg}|{SITE[site]}|NaN ignored|hop only"]["tau_exp_mid"]["conv_STU_per_frame"]; n = int(round(cm * 100 / 2) * 2)
        tmax_h = int(P["t"].max())                                   # 98 frames (site A sheet), 48 frames (site B sheet)
        H, tall = per_walker_hists(tg, 0, si, "hop only", n, tmax_h); Hs = H.sum(0); p = Hs[1:] / tall.size
        t = np.arange(1, tmax_h + 1)
        pe, ps = np.asarray(P["exph"][1:tmax_h + 1], float), np.asarray(P["simh"][1:tmax_h + 1], float)
        tm = float(tau_mle(tall)); HIST[(tg, site)] = (t, p, pe, ps, n, tm)
        hist_out[f"{tg}|{SITE[site]}"] = {"our_best_match_conv": n / 100, "N_dwells_ours": int(tall.size), "tau_mle_ours_frames": tm,
            "L1_ours_vs_experiment": float(np.abs(p - pe).sum()), "L1_paper_sim_vs_experiment": float(np.abs(ps - pe).sum()), "L1_ours_vs_paper_sim": float(np.abs(p - ps).sum()),
            "exp_hist_mean_t_(t<=tmax)": float((t * pe).sum() / pe.sum()), "ours_mean_t_(t<=tmax)": float((t * p).sum() / p.sum()), "paper_sim_mean_t_(t<=tmax)": float((t * ps).sum() / ps.sum())}
R["histogram_shape"] = hist_out
json.dump(R, open("results/step3_analysis.json", "w"), indent=1, default=float)
np.savez_compressed("results/step3_tau_curves.npz", conv=CONV, n=NGRID, **{k.replace("|", "__").replace(" ", "_"): np.array(v) for k, v in curves.items()})

# ---------- figures ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
sty = [("base_restart", "NaN ignored", "hop only", "tab:blue", "-", "restart, NaN ignored, hop only (PRIMARY)"), ("base_restart", "NaN ignored", "all endings", "tab:orange", "--", "restart, NaN ignored, all endings"),
       ("base_restart", "NaN ends dwell", "hop only", "tab:cyan", "-", "restart, NaN ends dwell, hop only"), ("base_reflect", "NaN ignored", "hop only", "tab:green", "-", "reflecting wall, NaN ignored (sensitivity)"),
       ("base_reflect", "NaN ends dwell", "hop only", "tab:olive", "--", "reflecting wall, NaN ends dwell")]
for si, site in enumerate((1, 2)):
    a = ax[si]; P = paper[site]
    a.plot(P["conv"], P["tau"], "k-", lw=2.2, label="paper τ_sim (Supp. Data 1)")
    a.axhspan(P["lo"], P["hi"], color="tab:red", alpha=0.3, label="τ_exp bounds (paper)")
    for tg, vn, sel, col, ls_, lab in sty:
        t, lo, hi = curves[f"{tg}|{SITE[site]}|{vn}|{sel}"]; a.plot(CONV, t, ls_, color=col, label="ours MLE: " + lab)
        if col in ("tab:blue", "tab:green"): a.fill_between(CONV, lo, hi, color=col, alpha=0.2)
    a.set(xscale="log", yscale="log", xlim=(0.1, 10), xlabel="time conversion [STU/frame]", ylabel="τ_dwell [frames]", title=f"Site {SITE[site]} ({'A→B' if site == 1 else 'B→A'})")
    a.legend(fontsize=7)
fig.savefig("figures/step3_tau_vs_conversion.png", dpi=150)
fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for r_, tg in enumerate(("base_restart", "base_reflect")):
    for si, site in enumerate((1, 2)):
        t, p, pe, ps, n, tm = HIST[(tg, site)]; a = ax[r_, si]
        a.bar(t, pe, width=1, color="tab:red", alpha=0.6, label="experiment (Supp. Data 1)"); a.step(t - 0.5, ps, where="post", color="k", label="paper sim best match")
        a.step(t - 0.5, p, where="post", color="tab:blue" if r_ == 0 else "tab:green", label=f"ours ({'restart' if r_ == 0 else 'reflect'}), conv {n/100:.2f}")
        a.plot(t, np.expm1(1 / tm) * np.exp(-t / tm), "b--" if r_ == 0 else "g--", lw=1, label=f"ours MLE τ = {tm:.1f}")
        a.set(yscale="log", ylim=(1e-4, 0.3), xlim=(0, t.max()), xlabel="dwell time [frames]", ylabel="probability per frame", title=f"Site {SITE[site]} — {tg}"); a.legend(fontsize=7)
fig.savefig("figures/step3_dwell_hist.png", dpi=150)
print("done")
