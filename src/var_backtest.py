"""Value-at-Risk backtest of the lower interval bounds (Fig. 4).

The lower bound of the p95 interval is a one-step 2.5% VaR and that of the p98 interval a 1% VaR.
Per stock, the breach rate (return below the VaR) is computed over all test sessions and over sessions
preceded by negative news (signed sentiment < -0.1; stocks with at least ten such sessions), then averaged
over stocks. Methods: the news-aware and no-news models, calibrated (CAL) and raw (RAW), and three
uncalibrated benchmarks estimated from the preceding 252 returns: historical simulation, a Gaussian model
and RiskMetrics EWMA (decay 0.94).

Usage:
    python src/var_backtest.py
"""
import argparse
import os
from statistics import NormalDist

import numpy as np
import pandas as pd

import config as C
from calibration import calibrate
from evaluate import load_preds
from paired_stats import tickers_with

SENT_THR = 0.1
VAR_LEVELS = [("2.5%", "q025", "q975", 0.95, 0.025), ("1%", "q01", "q99", 0.98, 0.01)]
METHODS = ["news (CAL)", "no-news (CAL)", "news (RAW)", "no-news (RAW)", "Historical", "Normal", "EWMA"]


def benchmarks(y, is_test, alpha):
    """Rolling-252 benchmark VaRs, each known at the previous close."""
    s = pd.Series(y)
    z = NormalDist().inv_cdf(alpha)
    hist = s.rolling(252, min_periods=100).quantile(alpha).shift(1)
    mu = s.rolling(252, min_periods=100).mean().shift(1)
    sd = s.rolling(252, min_periods=100).std().shift(1)
    ew = s.pow(2).ewm(alpha=1 - 0.94).mean().pow(0.5).shift(1)
    return {"Historical": hist.to_numpy()[is_test], "Normal": (mu + z * sd).to_numpy()[is_test],
            "EWMA": (z * ew).to_numpy()[is_test]}


def stock_var(tk, seed, preds_dir, level):
    _, lc, hc, cov, alpha = level
    news = load_preds(C.pred_file(preds_dir, tk, C.run_name("news", seed)))
    base = load_preds(C.pred_file(preds_dir, tk, C.run_name("nonews", seed)))
    if news is None or base is None or len(news[1]) < 150 or len(news[0]) < 60:
        return []
    v, t = news
    vn, tn = base
    y = t["y"].to_numpy(float)
    neg = t["sent"].to_numpy(float) < -SENT_THR
    var = {"news (CAL)": calibrate(v, t, lc, hc, cov)[0], "news (RAW)": t[lc].to_numpy(float),
           "no-news (CAL)": calibrate(vn, tn, lc, hc, cov)[0], "no-news (RAW)": tn[lc].to_numpy(float)}
    vt = pd.concat([v, t])                                 # benchmark history starts with validation
    var.update(benchmarks(vt["y"].to_numpy(float), (vt["split"] == "test").to_numpy(), alpha))
    rows = []
    for m, q in var.items():
        ok = np.isfinite(q)
        if ok.sum() < 200:
            continue
        br = y[ok] < q[ok]
        nm = neg[ok]
        rows.append({"ticker": tk, "seed": seed, "level": level[0], "method": m, "breach": br.mean() * 100,
                     "neg_breach": br[nm].mean() * 100 if nm.sum() >= 10 else np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--preds-dir", default=C.PREDS_DIR)
    ap.add_argument("--out-dir", default=C.RESULTS_DIR)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    tks = tickers_with(args.preds_dir, C.run_name("news", seeds[0]))
    d = pd.DataFrame([r for lvl in VAR_LEVELS for s in seeds for tk in tks
                      for r in stock_var(tk, s, args.preds_dir, lvl)])
    os.makedirs(args.out_dir, exist_ok=True)
    d.to_csv(os.path.join(args.out_dir, "var_per_stock.csv"), index=False)
    per_seed = d.groupby(["level", "method", "seed"])[["breach", "neg_breach"]].mean().reset_index()
    per_seed.to_csv(os.path.join(args.out_dir, "var_backtest.csv"), index=False)
    s = per_seed.groupby(["level", "method"])[["breach", "neg_breach"]].mean()

    for lvl, *_ , alpha in VAR_LEVELS:
        print(f"\n[var] {lvl} VaR (target {100 * alpha:.1f}%), means over stocks then seeds")
        print(f"{'method':<15} {'all sessions':>13} {'negative news':>14}")
        for m in METHODS:
            if (lvl, m) in s.index:
                print(f"{m:<15} {s.loc[(lvl, m), 'breach']:>12.2f}% {s.loc[(lvl, m), 'neg_breach']:>13.2f}%")
    print(f"\n-> {os.path.join(args.out_dir, 'var_backtest.csv')}")


if __name__ == "__main__":
    main()
