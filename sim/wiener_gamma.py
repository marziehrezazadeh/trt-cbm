# Parametric Wiener simulation + first-passage + PoF(t) export.
import argparse, numpy as np, os

def simulate_wiener(mu, sigma, L, n_paths, H, dt, seed=123):
    rng = np.random.RandomState(seed)
    n_steps = int(H/dt) + 1
    t = np.linspace(0.0, H, n_steps)
    X = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        x = 0.0
        for k in range(1, n_steps):
            x += mu*dt + sigma*np.sqrt(dt)*rng.randn()
            X[i, k] = x
    return t, X

def first_passage_times(paths, times, L=1.0):
    fpt = np.full(paths.shape[0], np.inf)
    hit = (paths >= L)
    for i in range(paths.shape[0]):
        idx = np.argmax(hit[i])
        if hit[i, idx]:
            fpt[i] = times[idx]
    return fpt

def pof_from_fpt(fpt, t_grid):
    finite = np.isfinite(fpt)
    return np.array([(fpt[finite] <= tt).mean() if finite.any() else 0.0 for tt in t_grid])

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

    t, X = simulate_wiener(args.mu, args.sigma, args.L, args.n_paths, args.H, args.dt, args.seed)
    fpt = first_passage_times(X, t, args.L)
    pof = pof_from_fpt(fpt, t)

    os.makedirs("tables", exist_ok=True)
    np.savez("tables/wiener_demo.npz", times=t, pof=pof, fpt=fpt)
    print("[SIM] Saved PoF/FPT to tables/wiener_demo.npz")

if __name__ == "__main__":
    main()
