"""Step 2 (d)/(e) simulation: W independent non-interacting walkers on the real landscape (fill 4 kBT),
started inside the A/B cluster, accumulating bin occupation inside that cluster.
Usage: python scripts/step2_equilibrium_run.py <dt> <chunk> [free|confined]   (default confined)
  free     = unconfined walkers (used only to measure how fast walkers leave the cluster at fill 4 kBT)
  confined = reflecting wall at the cluster-mask boundary (used for the occupation-vs-exp(-E) test)
Usage: python scripts/step2_equilibrium_run.py <dt> <chunk>   (dt e.g. 1e-4 or 5e-5; chunk 0..3 = walkers 2*chunk, 2*chunk+1)
Writes results/step2_equil_dt<dt>_chunk<k>.npz atomically; skips if it already exists (resumable).
Run parameters (walkers, duration, sampling interval) are OUR choices; the paper's only constraints used are
dt = 1e-4 STU, fill = 4 kBT for dwell-time simulations, periodic boundaries, bin = 45.7 nm."""
import json, os, sys, time
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_landscape import load_energy, fill_empty
from src.forces import central_difference_force
from src.analysis import label_clusters, bbox
from src.thiele_heun import run_histogram, run_histogram_confined

dt = float(sys.argv[1]); chunk = int(sys.argv[2]); CH = 2
mode = sys.argv[3] if len(sys.argv) > 3 else "confined"
out = Path(f"results/step2_{mode}_dt{dt:g}_chunk{chunk}.npz")
if out.exists():
    print("exists, skipping", out); sys.exit(0)
cfg = yaml.safe_load(open("config.yaml"))
h = cfg["landscape"]["bin_size_um"]; fill = cfg["fill"]["dwell_time_kBT"]
E = load_energy(cfg["data"]["xlsx"])
U = fill_empty(E, fill)
Fx, Fy = central_difference_force(U, h)
sites, ns, clab, nc, edge, base = label_clusters(E, h)
CL = int(clab[int(43.2322 / h), int(7.3577 / h)])             # cluster containing site A (x=7.36, y=43.23 um)
assert CL > 0 and clab[int(43.2322 / h), int(6.1695 / h)] == CL, "A and B must be in the same cluster"
mask = clab == CL
r0, r1, c0, c1 = bbox(mask)
json.dump({"cluster_id": CL, "n_bins": int(mask.sum()), "n_finite_bins": int(np.isfinite(E[mask]).sum()),
           "n_empty_bins_in_dilation_ring": int((~np.isfinite(E[mask])).sum()),
           "x_range_um": [c0 * h, c1 * h], "y_range_um": [r0 * h, r1 * h], "touches_edge": bool(edge[CL]),
           "site_ids_inside": [int(s) for s in np.unique(sites[mask & (sites > 0)])],
           "n_sites_total": ns, "n_clusters_total": nc, "connectivity": 8, "dilation": "euclidean disk 250 nm"},
          open("results/step2_cluster_AB.json", "w"), indent=2)

W, T, dt_sample = 8, 2.0e4, 0.01                              # walkers, STU per walker, sampling interval [STU]
rng = np.random.default_rng(2026)
rr, cc = np.where(mask)
p = np.exp(-U[rr, cc]); p /= p.sum()
pick = rng.choice(rr.size, size=W, p=p)                        # start bins ~ Boltzmann inside the cluster
x0 = (cc[pick] + rng.random(W)) * h; y0 = (rr[pick] + rng.random(W)) * h
x0 = x0[CH*chunk:CH*(chunk+1)]; y0 = y0[CH*chunk:CH*(chunk+1)]       # this chunk's walkers (global index w = CH*chunk + local)
n_steps = int(round(T / dt)); every = int(round(dt_sample / dt))
SEED = 424242 if dt >= 1e-4 else 737373   # independent noise streams for different dt (avoid common random numbers)
t0 = time.time()
if mode == "free":
    hist, n_samp, n_in, first_exit = run_histogram(Fx, Fy, h, mask, r0, c0, r1 - r0, c1 - c0, x0, y0,
                                                    n_steps, dt, every, SEED + CH * chunk)
    extra = dict(n_samp=n_samp, n_in=n_in, first_exit=first_exit)
    msg = f"inside fraction {n_in/n_samp}; first exit (STU) {np.where(first_exit>=0, first_exit*dt_sample, -1)}"
else:
    hist, n_rej = run_histogram_confined(Fx, Fy, h, mask, r0, c0, r1 - r0, c1 - c0, x0, y0,
                                         n_steps, dt, every, SEED + CH * chunk)
    extra = dict(n_rej=n_rej)
    msg = f"rejected steps per walker {n_rej} ({n_rej/n_steps})"
tmp = out.with_suffix(".tmp.npz")
np.savez_compressed(tmp, hist=hist, window=np.array([r0, r1, c0, c1]), dt=dt, T=T, dt_sample=dt_sample, W=W, chunk=chunk,
                    fill=fill, cluster=CL, runtime_s=time.time() - t0, mode=mode, **extra)
os.replace(tmp, out)
print(f"{mode} dt={dt:g} chunk {chunk}: runtime {time.time()-t0:.0f} s; {msg}")
