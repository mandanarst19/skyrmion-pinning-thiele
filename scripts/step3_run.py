"""Step 3 simulation: site-label time series for sites A and B (cluster containing both), fill 4 kBT, central-difference forces.
Usage: python scripts/step3_run.py <maskvariant: base|4disk|8sq> <mode: restart|reflect> <chunk> <walkers_per_chunk> <budget_STU_per_walker> [tag]
 restart (primary, paper-like): unconfined; on leaving the cluster mask the walker is terminated and a new one is started inside the cluster,
          drawn from the cluster's Boltzmann distribution (weights exp(-U_filled) over mask bins).
 reflect (sensitivity): reflecting wall at the mask boundary.
 base = 8-neighbour connectivity + Euclidean 250 nm disk; 4disk = 4-neighbour + disk; 8sq = 8-neighbour + square dilation (sensitivity, our choice).
Writes results/step3_<maskvariant>_<mode>_chunk<k>.npz atomically (skips if present). Walker w = chunk*walkers_per_chunk + j uses seed 910000 + 1000*w.
Walker count/duration are OUR choices; dt = 1e-4 STU, fill = 4 kBT, bin = 45.7 nm and periodic boundaries are the paper's values (Commun. Phys. 9:243, Methods)."""
import os, sys, time
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_landscape import load_energy, fill_empty
from src.forces import central_difference_force
from src.analysis import label_clusters
from src.thiele_heun import run_labels

variant, mode, chunk, CHW, budget_stu = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
tag = sys.argv[6] if len(sys.argv) > 6 else f"{variant}_{mode}"
out = Path(f"results/step3_{tag}_chunk{chunk}.npz")
if out.exists():
    print("exists, skipping", out); sys.exit(0)
cfg = yaml.safe_load(open("config.yaml")); h = cfg["landscape"]["bin_size_um"]; fill = cfg["fill"]["dwell_time_kBT"]
E = load_energy(cfg["data"]["xlsx"]); U = fill_empty(E, fill)
Fx, Fy = central_difference_force(U, h)
kw = {"base": {}, "4disk": dict(connectivity=4), "8sq": dict(dilation="square")}[variant]
sites, ns, clab, nc, edge, base = label_clusters(E, h, **kw)
rA, cA = int(43.2322 / h), int(7.3577 / h); rB, cB = int(43.2322 / h), int(6.1695 / h)      # minima of sites A and B in the landscape (see README, Assumptions)
idA, idB, CL = int(sites[rA, cA]), int(sites[rB, cB]), int(clab[rA, cA])
assert idA > 0 and idB > 0 and idA != idB and clab[rB, cB] == CL
mask = clab == CL
labmap = np.zeros(E.shape, dtype=np.int8); labmap[sites == idA] = 1; labmap[sites == idB] = 2
rr, cc = np.where(mask); p = np.exp(-U[rr, cc]); cdf = np.cumsum(p / p.sum()); cdf[-1] = 1.0
dt, every = 1e-4, 100                                       # 0.01 STU per write-out (paper, Methods)
budget = int(round(budget_stu / dt)); t0 = time.time()
labs, segn, segx, segw, nrej = [], [], [], [], []
for j in range(CHW):
    w = chunk * CHW + j
    lab, sn, sx, nr = run_labels(Fx, Fy, h, mask, labmap, cdf, rr, cc, budget, dt, every, 910000 + 1000 * w, 0 if mode == "restart" else 1)
    labs.append(lab); segn.append(sn); segx.append(sx); segw.append(np.full(sn.size, w)); nrej.append(nr)
tmp = out.with_suffix(".tmp.npz")
np.savez_compressed(tmp, lab=np.concatenate(labs), seg_n=np.concatenate(segn), seg_exit=np.concatenate(segx), seg_walker=np.concatenate(segw),
                    n_rejected=np.array(nrej), dt=dt, every=every, budget_stu=budget_stu, fill=fill, variant=variant, mode=mode,
                    cluster_id=CL, siteA_id=idA, siteB_id=idB, mask_bins=int(mask.sum()), runtime_s=time.time() - t0)
os.replace(tmp, out)
nseg = sum(s.size for s in segn); nex = sum(int(s.sum()) for s in segx)
print(f"{tag} chunk {chunk}: {time.time()-t0:.0f} s | segments {nseg}, exits {nex}, mean inside-time per segment {CHW*budget_stu/max(nseg,1):.1f} STU | mask bins {int(mask.sum())}, A bins {(labmap==1).sum()}, B bins {(labmap==2).sum()}")
