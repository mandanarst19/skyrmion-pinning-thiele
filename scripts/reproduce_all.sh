#!/bin/sh
# Command sequence that regenerates every result and figure (single CPU core; times are approximate).
# Prerequisite: pip install -r requirements.txt ; put Supplementary Data 1 at data/42005_2026_2766_MOESM4_ESM.xlsx
# Runs save one file per chunk and skip chunks that already exist, so an interrupted run can simply be restarted.
set -e
python3 scripts/step1_landscape.py                       # ~1-2 min (first call reads the 15 MB xlsx and caches data/fig1_a.npy)
python3 scripts/step2_validation.py                      # ~2 min  (free diffusion, harmonic well, dt halving, unit check)
python3 scripts/step2_1d_step_test.py                    # ~10 s   (1D step observation)
for dt in 1e-4 5e-5; do for k in 0 1 2 3; do python3 scripts/step2_equilibrium_run.py $dt $k confined; done; done   # ~15 min
python3 scripts/step2_equilibrium_analysis.py            # seconds
for k in 0 1 2 3 4 5; do python3 scripts/step3_run.py base restart $k 4 25000; done     # ~14 min (primary dwell-time run)
for k in 0 1; do python3 scripts/step3_run.py base reflect $k 4 20000; done             # ~4 min  (sensitivity)
for v in 4disk 8sq; do for k in 0 1; do python3 scripts/step3_run.py $v restart $k 4 10000; done; done   # ~4 min (sensitivity)
python3 scripts/step3_paper_curves.py                    # seconds (reference numbers from the paper's Supplementary Data 1)
python3 scripts/step3_analysis.py                        # ~5 min
for k in 0 1 2 3; do python3 scripts/step4_run.py 2.78 $k 4 50000; done                 # ~13 min
python3 scripts/step4_run.py flat_dt0.01 0 200 10000 1e-2                                # ~20 s (flat control, exact for F = 0)
python3 scripts/step4_run.py flat 0 8 2000                                              # ~16 s (short dt = 1e-4 consistency check)
python3 scripts/step4_analysis.py
for f in 2.28 3.28 3.78; do for k in 0 1; do python3 scripts/step4_run.py $f $k 4 50000; done; done   # ~16 min
python3 scripts/step4_fillscan_analysis.py
