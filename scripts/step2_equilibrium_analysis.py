"""Step 2 (d)/(e) analysis: confined-walker bin occupation vs exp(-U) of the input landscape (fill 4 kBT),
at dt = 1e-4 (paper's value) and dt/2. Uses only saved results/step2_confined_dt*_chunk*.npz."""
import json, sys
from pathlib import Path
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_landscape import load_energy, fill_empty
from src.analysis import label_clusters
from src.forces import interface_masks

cfg = yaml.safe_load(open("config.yaml")); h = cfg["landscape"]["bin_size_um"]; fill = cfg["fill"]["dwell_time_kBT"]
E = load_energy(cfg["data"]["xlsx"])
CL = json.load(open("results/step2_cluster_AB.json"))["cluster_id"]
_, _, clab, _, _, _ = label_clusters(E, h)
Mfull = clab == CL
files = {lab: sorted(Path("results").glob(f"step2_confined_dt{lab}_chunk*.npz")) for lab in ("0.0001", "5e-05")}
assert all(len(v) == 4 for v in files.values()), "missing chunk files"
r0, r1, c0, c1 = [int(v) for v in np.load(files["0.0001"][0])["window"]]
sl = (slice(r0, r1), slice(c0, c1))
M = Mfull[sl]; Ew = E[sl]; Uw = fill_empty(E, fill)[sl]
fin_side = interface_masks(~np.isfinite(E))[0][sl]
wall = ndi.distance_transform_edt(np.pad(M, 1))[1:-1, 1:-1] <= 2           # within 2 bins of the reflecting wall
fin = np.isfinite(Ew)
cat = np.full(M.shape, "", dtype=object)
cat[M & fin & (Ew < -3.5)] = "core (E<-3.5)"
cat[M & fin & (Ew >= -3.5) & (Ew < -2.5)] = "periphery (-3.5..-2.5)"
cat[M & fin & (Ew >= -2.5)] = "shallow (E>=-2.5)"
cat[M & fin & fin_side] = "finite interface bins"
cat[M & ~fin] = "filled bins (4 kBT)"
cat[M & wall] = "wall-adjacent (<=2 bins)"
cats = ["core (E<-3.5)", "periphery (-3.5..-2.5)", "shallow (E>=-2.5)", "finite interface bins", "filled bins (4 kBT)", "wall-adjacent (<=2 bins)"]

def load(lab):
    d = [np.load(f) for f in files[lab]]
    return np.concatenate([x["hist"] for x in d]), float(d[0]["T"]), float(d[0]["dt_sample"]), float(d[0]["dt"])
rng = np.random.default_rng(99); B = 400
q = np.where(M, np.exp(-Uw), 0.0); q /= q.sum()

def pdist(Hs): H = Hs.sum(0); return H / H[M].sum(), H
def boot(H):
    W = H.shape[0]; out = []
    for _ in range(B):
        P, _ = pdist(H[rng.integers(0, W, W)]); out.append(P)
    return np.array(out)

data = {}
for lab in files:
    H, T, ds, dt = load(lab); P, Hp = pdist(H); Pb = boot(H)
    data[lab] = dict(H=H, P=P, Hp=Hp, Pb=Pb, dt=dt, T=T, W=H.shape[0], samples=int(Hp.sum()))
res = {"cluster_id": CL, "window_rows_cols": [r0, r1, c0, c1], "fill_kBT": fill, "n_mask_bins": int(M.sum()),
       "walkers": 8, "T_STU_per_walker": data["0.0001"]["T"], "sample_interval_STU": ds,
       "min_pooled_count_for_bin_statistics": 30, "bootstrap": f"{B} resamples over the 8 walkers (crude: only 8 walkers)"}

def catmass(P, Pb=None):
    out = {}
    for c in cats:
        m = cat == c
        out[c] = {"n_bins": int(m.sum()), "expected_Q": float(q[m].sum()), "observed_P": float(P[m].sum()),
                  "observed_se": float(Pb[:, m].sum(1).std()) if Pb is not None else None}
    return out

# ---------------- (d) occupation vs exp(-U) ----------------
for lab, dd in data.items():
    P, Hp, Pb = dd["P"], dd["Hp"], dd["Pb"]
    sel = M & ~wall & (Hp >= 30)
    x = np.log(q[sel]); y = np.log(P[sel]); w = Hp[sel]
    b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
    bs = []
    for Pbi in Pb:
        yb = np.log(np.maximum(Pbi[sel], 1e-300)); bs.append(np.polyfit(x, yb, 1, w=np.sqrt(w))[0])
    se_y = np.log(np.maximum(Pb[:, sel], 1e-300)).std(0)
    r = y - x
    z = r / se_y
    rows = {}
    for c in cats[:5]:
        m = (cat[sel] == c)
        if m.sum() == 0: continue
        rows[c] = {"n_bins": int(m.sum()), "mean_resid_lnP_minus_lnQ": float(np.average(r[m], weights=w[m])),
                   "rms_resid": float(np.sqrt(np.mean(r[m] ** 2))), "median_resid": float(np.median(r[m])),
                   "std_z_vs_walker_bootstrap": float(z[m].std()), "median_pooled_count": float(np.median(w[m]))}
    dd["sel"] = sel; dd["resid"] = r
    res[f"d_dt={dd['dt']:g}"] = {"samples_total": dd["samples"], "n_bins_used": int(sel.sum()),
        "slope_weighted": float(b), "slope_se_bootstrap": float(np.std(bs)), "intercept": float(a),
        "weighted_mean_resid": float(np.average(r, weights=w)), "rms_resid": float(np.sqrt(np.mean(r ** 2))),
        "std_z": float(z.std()), "frac_|z|<2": float((np.abs(z) < 2).mean()), "by_category": rows,
        "category_probability_mass": catmass(P, Pb)}

# ---------------- (e) dt vs dt/2 ----------------
d1, d2 = data["0.0001"], data["5e-05"]
sel = M & ~wall & (d1["Hp"] >= 30) & (d2["Hp"] >= 30)
l1, l2 = np.log(d1["P"][sel]), np.log(d2["P"][sel])
se = np.sqrt(np.log(np.maximum(d1["Pb"][:, sel], 1e-300)).std(0) ** 2 + np.log(np.maximum(d2["Pb"][:, sel], 1e-300)).std(0) ** 2)
dlt = l1 - l2; zz = dlt / se
w = np.minimum(d1["Hp"][sel], d2["Hp"][sel])
b12, a12 = np.polyfit(l1, l2, 1, w=np.sqrt(w))
rows = {}
for c in cats[:5]:
    m = cat[sel] == c
    if m.sum(): rows[c] = {"n_bins": int(m.sum()), "mean_lnP_dt_minus_lnP_dt/2": float(np.average(dlt[m], weights=w[m])),
                           "rms": float(np.sqrt(np.mean(dlt[m] ** 2))), "mean_z": float(zz[m].mean()), "std_z": float(zz[m].std())}
res["e_dt_vs_halfdt"] = {"n_bins_used": int(sel.sum()), "slope_lnP(dt/2)_vs_lnP(dt)": float(b12), "intercept": float(a12),
    "mean_diff": float(np.average(dlt, weights=w)), "rms_diff": float(np.sqrt(np.mean(dlt ** 2))),
    "mean_z": float(zz.mean()), "std_z": float(zz.std()), "frac_|z|<2": float((np.abs(zz) < 2).mean()), "by_category": rows,
    "note": "independent noise streams (seeds 424242 vs 737373); z uses walker-bootstrap SEs of both runs"}
json.dump(res, open("results/step2_equilibrium.json", "w"), indent=2)
print(json.dumps(res, indent=1))

# ---------------- plots ----------------
colors = {"core (E<-3.5)": "#08306b", "periphery (-3.5..-2.5)": "#2171b5", "shallow (E>=-2.5)": "#6baed6",
          "finite interface bins": "#e6550d", "filled bins (4 kBT)": "#969696"}
fig, ax = plt.subplots(2, 2, figsize=(13, 10.5), constrained_layout=True)
for lab, a_ in (("0.0001", ax[0, 0]),):
    dd = data[lab]; sel_ = dd["sel"]
    for c, col in colors.items():
        m = sel_ & (cat == c)
        a_.plot(-Uw[m], np.log(dd["P"][m]), ".", ms=4, color=col, label=f"{c} (n={m.sum()})")
    xx = np.linspace(-Uw[sel_].min() * -1 - 0.2, 0, 2); 
    ln0 = np.log(q[sel_]) + Uw[sel_]               # = -ln Z
    xs = np.array([(-Uw[sel_]).min(), (-Uw[sel_]).max()])
    a_.plot(xs, xs + ln0[0], "k-", lw=1, label="slope 1 (exact Boltzmann)")
    r_ = res["d_dt=0.0001"]
    a_.set(xlabel="−U = −E_filled [k_BT]", ylabel="ln(time-averaged occupation fraction)",
           title=f"(d) dt=1e-4: weighted slope {r_['slope_weighted']:.3f} ± {r_['slope_se_bootstrap']:.3f}")
    a_.legend(fontsize=7)
dd = data["0.0001"]; sel_ = dd["sel"]
for c, col in colors.items():
    cm = (cat[sel_] == c)
    ax[0, 1].plot(-Uw[sel_][cm], dd["resid"][cm], ".", ms=4, color=col)
ax[0, 1].axhline(0, color="k", lw=0.8)
ax[0, 1].set(xlabel="−U [k_BT]", ylabel="ln P_sim − ln Q_Boltzmann", title="(d) residuals (dt = 1e-4)")
sel2 = sel & True
for c, col in colors.items():
    m = (cat[sel] == c)
    ax[1, 0].plot(l1[m], l2[m], ".", ms=4, color=col)
lim = [l1.min(), l1.max()]; ax[1, 0].plot(lim, lim, "k-", lw=1)
rr = res["e_dt_vs_halfdt"]
ax[1, 0].set(xlabel="ln P (dt = 1e-4)", ylabel="ln P (dt = 5e-5)", title=f"(e) dt vs dt/2: slope {rr['slope_lnP(dt/2)_vs_lnP(dt)']:.3f}, std z = {rr['std_z']:.2f}")
rmap = np.full(M.shape, np.nan); rmap[dd["sel"]] = dd["resid"]
im = ax[1, 1].imshow(rmap, origin="lower", extent=[c0 * h, c1 * h, r0 * h, r1 * h], cmap="RdBu_r", vmin=-1, vmax=1)
ax[1, 1].contour(np.linspace(c0 * h, c1 * h, M.shape[1]), np.linspace(r0 * h, r1 * h, M.shape[0]), M.astype(float), [0.5], colors="k", linewidths=0.6)
fig.colorbar(im, ax=ax[1, 1], label="ln P_sim − ln Q (bins with ≥30 counts)")
ax[1, 1].set(xlabel="x [µm]", ylabel="y [µm]", title="(d) residual map (cluster mask outlined)")
fig.savefig("figures/step2_equilibrium.png", dpi=150, bbox_inches="tight")
