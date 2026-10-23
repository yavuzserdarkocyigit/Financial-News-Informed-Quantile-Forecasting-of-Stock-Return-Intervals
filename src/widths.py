"""Event-selective width reallocation (Table VII).

For every stock, the mean p95 interval width of each arm is computed on non-event and on event sessions,
averaged over seeds, and the news / no-news ratio is formed. Reported: the mean ratio over stocks and the
number of stocks whose ratio is below / above one, for raw (RAW) and calibrated (CAL) intervals.

Usage:
    python src/widths.py
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as C
from calibration import calibrate
from evaluate import load_preds
from paired_stats import tickers_with

LC, HC, COV = "q025", "q975", 0.95


def stock_widths(tk, seeds, preds_dir, news_dir):
    """Seed-averaged mean widths per (kind, scope, arm) of one stock, or None."""
    nw = pd.read_parquet(C.news_file(news_dir, tk), columns=["Date", "ev_hi"])
    ev_map = nw.groupby(pd.to_datetime(nw["Date"]), sort=False)["ev_hi"].max()
    rows = []
    for s in seeds:
        news = load_preds(C.pred_file(preds_dir, tk, C.run_name("news", s)))
        if news is None or len(news[1]) < 150 or len(news[0]) < 60:
            continue
        ev = news[1]["Date"].map(ev_map).fillna(0.0).to_numpy(float) > 0
        arms = [("news", news)]
        base = load_preds(C.pred_file(preds_dir, tk, C.run_name("nonews", s)))
        if base is not None and len(base[1]) == len(news[1]):
            arms.append(("nonews", base))
        for arm, (v, t) in arms:
            rlo, rhi = t[LC].to_numpy(float), t[HC].to_numpy(float)
            clo, chi = calibrate(v, t, LC, HC, COV)
            for kind, lo, hi in (("RAW", rlo, rhi), ("CAL", clo, chi)):
                w = hi - lo
                for scope, sel in (("non-event", ~ev), ("event", ev)):
                    rows.append({"seed": s, "arm": arm, "kind": kind, "scope": scope,
                                 "width": w[sel].mean() if sel.any() else np.nan})
    return pd.DataFrame(rows).assign(ticker=tk) if rows else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--preds-dir", default=C.PREDS_DIR)
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--out-dir", default=C.RESULTS_DIR)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    parts = [stock_widths(tk, seeds, args.preds_dir, args.news_dir)
             for tk in tickers_with(args.preds_dir, C.run_name("news", seeds[0]))]
    d = pd.concat([p for p in parts if p is not None], ignore_index=True)
    w = (d.groupby(["arm", "ticker", "kind", "scope"])["width"].mean().reset_index()        # seed mean
          .pivot_table(index=["ticker", "kind", "scope"], columns="arm", values="width").reset_index().dropna())
    w["ratio"] = w["news"] / w["nonews"]
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "width_ratios.csv")
    w.to_csv(out, index=False)

    print(f"\n[widths] news / no-news p95 width ratio per stock, seed-averaged (seeds {seeds})")
    print(f"{'kind':<5} {'scope':<6} {'mean':>7} {'narrower':>10} {'wider':>10}")
    for kind in ("CAL", "RAW"):
        for scope in ("non-event", "event"):
            r = w[(w["kind"] == kind) & (w["scope"] == scope)]["ratio"]
            print(f"{kind:<5} {scope:<6} {r.mean():>7.3f} {int((r < 1).sum()):>6}/{len(r):<3} {int((r > 1).sum()):>6}/{len(r):<3}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
