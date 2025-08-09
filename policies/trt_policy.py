# Compute R(t) = PoF(t) * CoF, get Rcrit, and infer TRT (linear interpolation).
import numpy as np

def compute_risk_curve(times, pof, cof):
    """Return R(t) = PoF(t) * CoF."""
    return pof * float(cof)

def rcrit_from_rule(cof, epsilon):
    """Primary rule: Rcrit = CoF * epsilon."""
    return float(cof) * float(epsilon)

def infer_trt(times, risk, rcrit):
    """
    Earliest time t where risk(t) >= rcrit, using linear interpolation
    between the two grid points that straddle rcrit.
    Assumes risk is non-decreasing (true when risk = CoF * PoF).
    """
    if risk[-1] < rcrit:
        return float("inf")
    idx = int(np.searchsorted(risk, rcrit, side="left"))
    if idx == 0:
        return float(times[0])
    x0, x1 = times[idx-1], times[idx]
    y0, y1 = risk[idx-1], risk[idx]
    if y1 == y0:  # flat segment
        return float(x1)
    return float(x0 + (rcrit - y0) * (x1 - y0) / (y1 - y0))

    if risk[idx] >= rcrit:
        return float(times[idx])
    return float("inf")
