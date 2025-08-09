# -*- coding: utf-8 -*-
# Minimal Wiener/Gamma simulation + first-passage to produce PoF(t)
# All comments in English per your convention.

import numpy as np, os

def simulate_wiener(mu=0.05, sigma=0.1, L=1.0, n_paths=100, H=10.0, dt=0.1, seed=123):
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
    for i in range(paths.shape[0]):
        idx = np.argmax(paths[i] >= L)
        if paths[i, idx] >= L:
            fpt[i] = times[idx]
    return fpt

def pof_from_fpt(fpt, t_grid):
    finite = np.isfinite(fpt)
    return np.array([(fpt[finite] <= tt).mean() if finite.any() else 0.0 for tt in t_grid])

if __name__ == "__main__":
    t, X = simulate_wiener()
    fpt = first_passage_times(X, t, L=1.0)
    pof = pof_from_fpt(fpt, t)

    os.makedirs("tables", exist_ok=True)
    np.savez("tables/wiener_demo.npz", times=t, pof=pof, fpt=fpt)
    print("[SIM] Saved PoF/FPT to tables/wiener_demo.npz")
