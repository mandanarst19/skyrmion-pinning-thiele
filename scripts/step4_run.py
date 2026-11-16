"""Step 4 simulation: single non-interacting skyrmion(s) on the full periodic landscape (no linking, no interactions).
Usage: python scripts/step4_run.py <fill_kBT | flat | flat_dt0.01> <chunk> <walkers_per_chunk> <T_STU_per_walker> [dt]
(label flat*: zero force; Heun with F = 0 is exact for ANY dt, so the flat control may use dt = 1e-2 for statistics; never used with forces)
Positions are recorded (unwrapped) once per camera frame = 1.34 STU = 13400 steps at dt = 1e-4.
Paper values used: dt = 1e-4 STU, 1 frame = 1.34 STU, frame time 62.5 ms, fill 2.78 kBT, bin 45.7 nm, periodic boundaries
(Commun. Phys. 9:243, Results/Methods). Walker count/duration are OUR choices. Start positions ~ Boltzmann over the whole map (no burn-in).
Writes results/step4_fill<label>_chunk<k>.npz atomically; skips if present."""
import os, sys, time
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_landscape import load_energy, fill_empty
from src.forces import central_difference_force
from src.thiele_heun import run_trajectories

lab, chunk, CHW, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
out = Path(f"results/step4_fill{lab}_chunk{chunk}.npz")
if out.exists():
    print("exists, skipping", out); sys.exit(0)
cfg = yaml.safe_load(open("config.yaml")); h = cfg["landscape"]["bin_size_um"]
dt = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
frame_steps = int(round(1.34 / dt))                              # 1 frame = 1.34 STU (13400 steps at dt = 1e-4)
n_steps = int(round(T / dt)); n_steps -= n_steps % frame_steps
rng = np.random.default_rng(777 + chunk)
if lab.startswith("flat"):
    Fx = Fy = np.zeros((1, 1)); x0 = rng.random(CHW) * 90.0; y0 = rng.random(CHW) * 90.0
else:
    E = load_energy(cfg["data"]["xlsx"]); U = fill_empty(E, float(lab)); Fx, Fy = central_difference_force(U, h)
    p = np.exp(-(U - U.min())).ravel(); p /= p.sum()
    k = rng.choice(p.size, size=CHW, p=p); n = U.shape[0]
    x0 = ((k % n) + rng.random(CHW)) * h; y0 = ((k // n) + rng.random(CHW)) * h
seed = 5500000 + 1000 * chunk * CHW + (0 if lab.startswith("flat") else int(round(float(lab) * 100)) * 7)
t0 = time.time()
pos = run_trajectories(0, Fx, Fy, h, 0.0, x0, y0, n_steps, dt, frame_steps, seed)
tmp = out.with_suffix(".tmp.npz")
np.savez_compressed(tmp, pos=pos, dt=dt, frame_STU=1.34, T_STU=n_steps * dt, fill=lab, chunk=chunk, runtime_s=time.time() - t0)
os.replace(tmp, out)
print(f"fill {lab} chunk {chunk}: {time.time()-t0:.0f} s, {CHW} walkers x {n_steps*dt:.0f} STU, frames/walker {pos.shape[1]-1}")
