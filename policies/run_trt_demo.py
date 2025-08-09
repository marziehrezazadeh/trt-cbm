# -*- coding: utf-8 -*-
# Run a small grid over CoF and epsilon to compute TRT on Wiener demo outputs.

import numpy as np, os, csv
from trt_policy import compute_risk_curve, rcrit_from_rule, infer_trt

def main():
    os.makedirs("tables", exist_ok=True)
    data = np.load("tables/wiener_demo.npz")
    times, pof = data["times"], data["pof"]

    cof_grid = [1.0, 5.0, 20.0]
    eps_grid = [0.01, 0.03, 0.05]

    rows = [("cof","epsilon","rcrit","trt")]
    for cof in cof_grid:
        for eps in eps_grid:
            R = compute_risk_curve(times, pof, cof)
            rcrit = rcrit_from_rule(cof, eps)
            trt = infer_trt(times, R, rcrit)
            rows.append((cof, eps, rcrit, trt))

    out_csv = "tables/trt_demo_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[TRT] Wrote results to {out_csv}")

if __name__ == "__main__":
    main()
