"""Per-stock interval evaluation of raw (RAW) and calibrated (CAL) forecasts at 90/95/98% (Table V).

For every run and stock: coverage, normalized mean width (NMW), mean log-return width and Winkler score,
raw and calibrated, plus calibrated p95 Winkler on event sessions and p95 breach rates of each side on
negative- and positive-sentiment sessions. The summary averages stocks within a seed, then seeds.

Usage:
    python src/evaluate.py                          # news and no-news arms, seeds 7, 42, 123
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as C
from calibration import calibrate, winkler

SENT_THR = 0.1                          # sentiment regimes: NEG < -0.1 < neutral < 0.1 < POS


def load_preds(path):
    """(validation, test) frames of one prediction file, or None."""
    if not os.path.exists(path):
        return None
    p = pd.read_parquet(path)
    p["Date"] = pd.to_datetime(p["Date"])
    p = p.sort_values("Date")
    return p[p["split"] == "val"].reset_index(drop=True), p[p["split"] == "test"].reset_index(drop=True)


def event_flags(news_dir, tk):
    nw = pd.read_parquet(C.news_file(news_dir, tk), columns=["Date", "ev_hi"])
    return nw.set_index(pd.to_datetime(nw["Date"]))["ev_hi"]


def stock_metrics(v, t, ev_map):
    y, price, sent = t["y"].to_numpy(float), t["price"].to_numpy(float), t["sent"].to_numpy(float)
    rng = max(price.max() - price.min(), 1e-9)             # NMW denominator: test price range
    ev = ev_map.reindex(pd.DatetimeIndex(t["Date"].values)).fillna(0.0).to_numpy() > 0
    out = {"n": len(t), "n_ev": int(ev.sum())}
    for name, lc, hc, cov in C.LEVELS:
        a = 1 - cov
        rlo, rhi = t[lc].to_numpy(float), t[hc].to_numpy(float)
        clo, chi = calibrate(v, t, lc, hc, cov)
        for pre, lo, hi in (("r", rlo, rhi), ("", clo, chi)):          # r = RAW, '' = CAL
            out[f"{name}_{pre}cov"] = float(((y >= lo) & (y <= hi)).mean())
            out[f"{name}_{pre}width"] = float(np.mean(hi - lo))
            out[f"{name}_{pre}nmw"] = float(np.mean(price * (np.exp(hi) - np.exp(lo))) / rng)
            out[f"{name}_{pre}wink"] = float(np.mean(winkler(y, lo, hi, a)))
        out[f"{name}_wink_ev"] = float(np.mean(winkler(y[ev], clo[ev], chi[ev], a))) if ev.sum() >= 5 else np.nan
        if name == "p95":                                  # side-specific breaches by sentiment regime
            for g, gm in (("NEG", sent < -SENT_THR), ("POS", sent > SENT_THR)):
                if gm.sum() >= 10:
                    for pre, lo, hi in (("r", rlo, rhi), ("", clo, chi)):
                        out[f"{pre}loBr_{g}"] = float((y[gm] < lo[gm]).mean()) * 100
                        out[f"{pre}upBr_{g}"] = float((y[gm] > hi[gm]).mean()) * 100
    return out


def evaluate_run(run, tickers, preds_dir, news_dir):
    rows = []
    for tk in tickers:
        vt = load_preds(C.pred_file(preds_dir, tk, run))
        if vt is None or len(vt[1]) < 150 or len(vt[0]) < 60:
            continue
        rows.append({"ticker": tk, **stock_metrics(*vt, event_flags(news_dir, tk))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="news,nonews")
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--data", default=C.DATASET, help="dataset (for the ticker list)")
    ap.add_argument("--preds-dir", default=C.PREDS_DIR)
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--out-dir", default=C.RESULTS_DIR)
    args = ap.parse_args()
    import torch
    tickers = torch.load(args.data, weights_only=False)["tickers"]
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for arm in args.arms.split(","):
        per_seed = []
        for s in seeds:
            run = C.run_name(arm, s)
            d = evaluate_run(run, tickers, args.preds_dir, args.news_dir)
            if d.empty:
                print(f"[eval] no predictions for {run}"); continue
            d.round(5).to_csv(os.path.join(args.out_dir, f"eval_{run}.csv"), index=False)
            per_seed.append(d.drop(columns="ticker").mean())
        if per_seed:
            summary[arm] = pd.concat(per_seed, axis=1).mean(axis=1)

    for arm, m in summary.items():
        print(f"\n[eval] {arm}: means over stocks, then seeds   (RAW / CAL)")
        for name, _, _, cov in C.LEVELS:
            print(f"  {name}  coverage {100 * m[f'{name}_rcov']:.2f} / {100 * m[f'{name}_cov']:.2f} %"
                  f"   NMW {m[f'{name}_rnmw']:.4f} / {m[f'{name}_nmw']:.4f}"
                  f"   Winkler {m[f'{name}_rwink']:.4f} / {m[f'{name}_wink']:.4f}   (target {cov:.2f})")
    if {"news", "nonews"} <= set(summary):
        print(f"\n[eval] calibrated p95 Winkler: news {summary['news']['p95_wink']:.4f}  "
              f"no-news {summary['nonews']['p95_wink']:.4f}")


if __name__ == "__main__":
    main()
