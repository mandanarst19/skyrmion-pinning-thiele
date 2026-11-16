"""Reference numbers read from the paper's Supplementary Data 1 (sheets fig2_b and fig2_e): the conversion factors at which the paper's
tau_sim(conversion) curve crosses the tau_exp bounds, and tau_sim at 1.34 STU/frame. Usage: python scripts/step3_paper_curves.py [output.json]"""
import json, sys
import numpy as np, pandas as pd, yaml
cfg = yaml.safe_load(open("config.yaml")); XL = cfg["data"]["xlsx"]
out = sys.argv[1] if len(sys.argv) > 1 else "results/step3_paper_crossings.json"

def cross(conv, tau, v):
    """conversion at which tau(conv) = v; tau decreases with conv, interpolation in log-log."""
    k = np.where(np.diff(np.sign(tau - v)) != 0)[0][0]
    return float(np.exp(np.interp(np.log(v), np.log(tau[k:k + 2])[::-1], np.log(conv[k:k + 2])[::-1])))

res = {}
for site, sheet in (("A", "fig2_b"), ("B", "fig2_e")):
    d = pd.read_excel(XL, sheet_name=sheet, header=0).values
    conv, tau, lo, hi = d[:, 0], d[:, 1], d[0, 2], d[0, 3]
    res[site] = {"tau_exp_bounds": [float(lo), float(hi)], "conv_at_lower": cross(conv, tau, lo), "conv_at_upper": cross(conv, tau, hi),
                 "conv_mid": cross(conv, tau, 0.5 * (lo + hi)), "tau_sim_at_conv_1.34": float(np.interp(1.34, conv, tau))}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
