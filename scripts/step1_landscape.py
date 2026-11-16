"""Step 1: load Fig. 1a landscape, fill empty bins, build forces, plot, run sanity checks."""
import json, sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_landscape import (load_energy, load_trajectory, empty_mask, fill_empty,
                                counts_from_energy, positions_to_bins)
from src.forces import central_difference_force, interface_masks, force_at

cfg = yaml.safe_load(open("config.yaml"))
h = cfg["landscape"]["bin_size_um"]
xlsx = cfg["data"]["xlsx"]
fills = {"final": cfg["fill"]["final_kBT"], "dwell": cfg["fill"]["dwell_time_kBT"]}

E = load_energy(xlsx)
n = E.shape[0]
L = n * h
empty = empty_mask(E)
res = {"n_bins": n, "box_um": L, "empty_fraction": float(empty.mean()),
       "E_min": float(E[~empty].min()), "E_max_finite": float(E[~empty].max())}

# --- encoding check: E = -ln(counts) ---
cnt = counts_from_energy(E)
res["inferred_total_counts"] = float(cnt.sum())
res["expected_counts_70sk_x_115200frames"] = 70 * 2 * 3600 * 16
res["n_bins_below_-2.5"] = int((E < -2.5).sum())
res["n_bins_below_-3.5"] = int((E < -3.5).sum())

# --- trajectory overlay / orientation check ---
tr = load_trajectory(xlsx)
def traj_stats(x, y, name):
    r, c = positions_to_bins(x, y, h)
    ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
    e = E[r[ok], c[ok]]
    return {"orientation": name, "n_points": int(len(x)), "frac_on_finite_bin": float(np.isfinite(e).mean()),
            "mean_E_on_finite": float(e[np.isfinite(e)].mean()) if np.isfinite(e).any() else None}
x, y = tr.x_um.values, tr.y_um.values
res["trajectory_checks"] = [
    traj_stats(x, y, "rows=y, cols=x (adopted)"),
    traj_stats(y, x, "transposed (control)"),
    traj_stats(x, L - y, "y flipped (control)"),
]

# --- forces ---
bnd = {}
F = {}
for k, f in fills.items():
    U = fill_empty(E, f)
    Fx, Fy = central_difference_force(U, h)
    mag = np.hypot(Fx, Fy)
    F[k] = (U, Fx, Fy, mag)
    fin_side, fil_side = interface_masks(empty)
    interior = (~empty) & ~fin_side
    q = lambda a: {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "n": int(a.size)}
    bnd[k] = {"fill_kBT": f,
              "finite_side_of_interface": q(mag[fin_side]),
              "filled_side_of_interface": q(mag[fil_side]),
              "finite_interior": q(mag[interior]),
              "flat_filled_interior_max": float(mag[empty & ~fil_side].max()) if (empty & ~fil_side).any() else 0.0}
res["force_magnitude_kBT_per_um"] = bnd
# periodic-wrap seam: finite bins on the map edge create artificial forces across the seam
edge = np.zeros_like(empty); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
res["finite_bins_on_map_edge"] = int((edge & ~empty).sum())
# step-size context for dt choice
res["noise_step_um_at_dt1e-4"] = float(np.sqrt(2 * 1e-4))
res["bin_um"] = h
res["max_drift_step_um_at_dt1e-4_fill4"] = float(F["dwell"][3].max() * 1e-4)
Path("results").mkdir(exist_ok=True)
json.dump(res, open("results/step1_numbers.json", "w"), indent=2)
print(json.dumps(res, indent=2))

# --- plots ---
cmap = LinearSegmentedColormap.from_list("pin", ["#0a0a14", "#24465a", "#5b8793", "#a6c4c6", "#e8f0ef"])
cmap.set_bad("white")
vmin, vmax = cfg["landscape"]["colour_range_kBT"]
Em = np.ma.masked_invalid(E)
lo, hi = 1.0, 10.0
i0, i1 = int(lo / h), int(hi / h)

fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1]})
im = ax[0].imshow(Em, origin="lower", extent=[0, L, 0, L], cmap=cmap, vmin=vmin, vmax=0, interpolation="nearest")
ax[0].add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec="k", lw=1.2))
ax[0].set(xlabel="x [µm]", ylabel="y [µm]", title=f"Fig. 1a data (empty bins white: {100*empty.mean():.2f}%)")
ax[1].imshow(Em[i0:i1, i0:i1], origin="lower", extent=[lo, hi, lo, hi], cmap=cmap, vmin=vmin, vmax=0, interpolation="nearest")
m = (tr.x_um > lo) & (tr.x_um < hi) & (tr.y_um > lo) & (tr.y_um < hi)
ax[1].plot(tr.x_um, tr.y_um, "-", color="#d9534f", lw=0.8, alpha=0.7, label="fig1_b trajectory")
ax[1].plot(tr.x_um, tr.y_um, ".", color="#a02020", ms=2)
ax[1].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="x [µm]", ylabel="y [µm]", title="1–10 µm box + fig1_b trajectory (orientation check)")
ax[1].legend(loc="upper right", fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("E [k$_B$T]")
fig.savefig("figures/step1_landscape.png", dpi=170, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
k = "final"                                              # README figure: fill = 2.78 kBT only (the fill-4 maps are in results/step1_numbers.json)
mag = F[k][3][i0:i1, i0:i1]; vmaxF = np.percentile(mag, 99.5)
imf = ax[0].imshow(mag, origin="lower", extent=[lo, hi, lo, hi], cmap="magma", vmin=0, vmax=vmaxF, interpolation="nearest")
ax[0].set(xlabel="x [µm]", ylabel="y [µm]", title=f"|F| (central differences), fill = {fills[k]} k$_B$T")
fig.colorbar(imf, ax=ax[0], fraction=0.046, pad=0.02, label="|F| [k$_B$T/µm] (clipped at 99.5th pct)")
fin_side, fil_side = interface_masks(empty); interior = (~empty) & ~fin_side
bins = np.linspace(0, 100, 101); mfull = F[k][3]
ax[1].hist(mfull[fin_side], bins=bins, histtype="step", color="tab:blue", label="finite bins next to an empty bin")
ax[1].hist(mfull[fil_side], bins=bins, histtype="step", color="tab:orange", label="filled bins next to a finite bin")
ax[1].hist(mfull[interior], bins=bins, histtype="step", color="gray", label="other finite bins")
ax[1].axvline(fills[k] / (2 * h), color="k", ls=":", lw=0.8)
ax[1].annotate("single-count bin (E = 0) beside a\nfilled bin: |F| = 2.78/(2h) = 30.4", xy=(fills[k] / (2 * h), 3e4), xytext=(45, 2e4), fontsize=8, arrowprops=dict(arrowstyle="->"))
ax[1].set(yscale="log", xlabel="|F| [k$_B$T/µm]", ylabel="bins", title="|F| histogram (full map); spikes: E = −ln(n), integer n"); ax[1].legend(fontsize=8)
fig.savefig("figures/step1_forces.png", dpi=170, bbox_inches="tight")
print("figures written")
