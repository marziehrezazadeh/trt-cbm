# -*- coding: utf-8 -*-
# Compute R(t) = PoF(t) * CoF, get Rcrit, and infer TRT.

import numpy as np
from typing import Tuple

def compute_risk_curve(times: np.ndarray, pof: np.ndarray, cof: float) -> np.ndarray:
    """Return R(t) = PoF(t) * CoF."""
    return pof * float(cof)

def rcrit_from_rule(cof: float, epsilon: float) -> float:
    """Primary rule: Rcrit = CoF * epsilon."""
    return float(cof) * float(epsilon)

def infer_trt(times: np.ndarray, risk: np.ndarray, rcrit: float) -> float:
    """Earliest time when risk >= rcrit; return +inf if never occurs."""
    idx = np.argmax(risk >= rcrit)
    if risk[idx] >= rcrit:
        return float(times[idx])
    return float("inf")
