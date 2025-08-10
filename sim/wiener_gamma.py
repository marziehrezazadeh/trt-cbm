# Parametric Wiener simulation + first-passage + PoF(t) export.
import argparse
import numpy as np
import os

def simulate_wiener(mu, sigma, L, n_paths, H, dt, seed=123):
    """Simulate n_paths Wiener-with-drift trajectories on [0, H] with step dt."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(H / dt)) + 1
    times = np.arange(n_steps, dtype=float) * dt
    paths = np.zeros((n_paths, n_steps), dtype=float)
    for i in range(n_paths):
        x = 0.0
        for k in range(1, n_steps):
            # Euler step: x += mu*dt + sigma*sqrt(dt)*N(0,1)
            x += mu * dt + sigma * np.sqrt(dt) * rng.standard_normal()
            paths[i, k] = x
    return times, paths

def first_passage_times(paths, times, L=1.0, refine="linear"):
    """
    Return first-passage time for each path crossing level L.
    refine: "none" | "linear"  (linear reduces late-count bias)
    """
    n = paths.shape[0]
    fpt = np.full(n, np.inf, dtype=float)

    for i in range(n):
        xi = paths[i]
        crossed = np.where(xi >= L)[0]
        if crossed.size == 0:
            continue
        k = crossed[0]
        if refine == "none" or k == 0:
            fpt[i] = times[k]
            continue

        # linear interpolation inside [times[k-1], times[k]]
        t0, t1 = times[k-1], times[k]
        x0, x1 = xi[k-1], xi[k]
        if x1 == x0:
            fpt[i] = t1
        else:
            frac = (L - x0) / (x1 - x0)
            if frac < 0.0: frac = 0.0
            if frac > 1.0: frac = 1.0
            fpt[i] = t0 + frac * (t1 - t0)
    return fpt

def pof_from_fpt(fpt, t_grid):
    """Empirical PoF(t): fraction of paths with FPT <= t."""
    finite = np.isfinite(fpt)
    return np.array([(fpt[finite] <= tt).mean() if finite.any() else 0.0 for tt in t_grid], dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.20)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--n_paths", type=int, default=10000)
    ap.add_argument("--H", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    times, paths = simulate_wiener(args.mu, args.sigma, args.L, args.n_paths, args.H, args.dt, args.seed)
    fpt = first_passage_times(paths, times, L=args.L, refine="linear")
    pof = pof_from_fpt(fpt, times)
    # ensure monotonic (helpful for plotting/reporting)
    pof = np.maximum.accumulate(pof)

    os.makedirs("tables", exist_ok=True)
    np.savez("tables/wiener_demo.npz", times=times, pof=pof, fpt=fpt)
    print("[SIM] Saved PoF/FPT to tables/wiener_demo.npz")

if __name__ == "__main__":
    main()
