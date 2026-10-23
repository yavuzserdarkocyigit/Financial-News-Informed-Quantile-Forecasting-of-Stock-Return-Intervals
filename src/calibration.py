"""Past-only, stock-specific conformal calibration and interval scores.

For one stock and one nominal level, the lower and upper errors of past forecasts are normalized by the raw
interval width: (L - y) / w below and (y - U) / w above (positive = outside the bound). Their empirical
(1 - alpha/2) quantiles give the correction factors, and the calibrated interval is
[L - q_lo * w, U + q_hi * w]. The history starts with the validation period and grows with every test
session, after that session's forecast has been made.
"""
import math

import numpy as np


def conf_q(scores, level):
    """Finite-sample conformal quantile of the finite scores."""
    s = np.sort(scores[np.isfinite(scores)])
    if not len(s):
        return 0.0
    lvl = min(1.0, math.ceil((len(s) + 1) * level) / len(s))
    return float(np.quantile(s, lvl, method="higher"))


def calibrate(val, test, lo_col, hi_col, cov):
    """Calibrated (lower, upper) bounds for every test row. `val` and `test` are date-sorted
    prediction frames of one stock with columns lo_col, hi_col and y."""
    a2 = (1 - cov) / 2.0
    vl, vh, vy = (val[c].to_numpy(float) for c in (lo_col, hi_col, "y"))
    tl, th, ty = (test[c].to_numpy(float) for c in (lo_col, hi_col, "y"))
    out_lo, out_hi = np.empty(len(test)), np.empty(len(test))
    for i in range(len(test)):
        lo, hi, y = np.r_[vl, tl[:i]], np.r_[vh, th[:i]], np.r_[vy, ty[:i]]   # outcomes known before i
        w = np.maximum(hi - lo, 1e-9)
        q_lo, q_hi = conf_q((lo - y) / w, 1 - a2), conf_q((y - hi) / w, 1 - a2)
        wc = max(th[i] - tl[i], 1e-9)
        out_lo[i], out_hi[i] = tl[i] - q_lo * wc, th[i] + q_hi * wc
    return out_lo, out_hi


def winkler(y, lo, hi, alpha):
    """Winkler interval score of every observation (log-return units)."""
    return (hi - lo) + np.where(y < lo, (2 / alpha) * (lo - y), 0.0) + np.where(y > hi, (2 / alpha) * (y - hi), 0.0)
