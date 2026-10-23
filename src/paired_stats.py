"""Paired comparison of two model arms (Table VI, the annual effects and the pinball comparison).

For every stock and test session the loss of each arm is averaged over seeds. Within a scope
(all / event / non-event sessions) the losses are averaged across the stocks of each date, giving a daily
series of both arms. Reported per scope:
    loss        mean over dates of the daily mean loss of each arm
    change %    100 * mean(daily difference) / mean(daily reference loss)
    95% CI      moving-block bootstrap over dates (21-session blocks, 2,000 resamples), recomputing the
                percentage change in every resample
    p           two-sided Newey-West HAC test (12 lags) of a zero mean daily difference
Losses: CAL and RAW p95 Winkler, and the mean pinball loss of the seven raw quantiles (PINBALL).
Event sessions have at least one article in the event groups (ev_hi > 0); non-event = all other sessions.

Usage:
    python src/paired_stats.py                                  # news vs no-news, seeds 7, 42, 123
"""
import argparse
import glob
import os
from math import erfc, sqrt
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config as C
from calibration import calibrate, winkler
from evaluate import load_preds

LC, HC, COV = "q025", "q975", 0.95
STAGES = ("cal", "raw", "pinball")
SCOPES = [("All", lambda d: np.ones(len(d), bool)), ("Event", lambda d: d["event"].to_numpy()),
          ("Non-event", lambda d: ~d["event"].to_numpy())]


def run_losses(path, stage):
    """Date-indexed loss of one prediction file: calibrated or raw p95 Winkler, or raw pinball."""
    vt = load_preds(path)
    if vt is None:
        return None
    v, t = vt
    y = t["y"].to_numpy(float)
    if stage == "pinball":
        loss = sum(np.maximum(q * (y - t[c].to_numpy(float)), (q - 1) * (y - t[c].to_numpy(float)))
                   for c, q in zip(C.QCOLS, C.QUANTILES)) / len(C.QUANTILES)
    else:
        lo, hi = calibrate(v, t, LC, HC, COV) if stage == "cal" else (t[LC].to_numpy(float), t[HC].to_numpy(float))
        loss = winkler(y, lo, hi, 1 - COV)
    return pd.Series(loss, index=t["Date"])


def stock_job(args):
    """Seed-averaged losses of both arms for one stock, with its event flags."""
    tk, arms, seeds, stage, preds_dir, news_dir = args
    loss = {}
    for key, arm in zip(("a", "b"), arms):
        parts = [run_losses(C.pred_file(preds_dir, tk, C.run_name(arm, s)), stage) for s in seeds]
        if any(p is None for p in parts):
            return None
        loss[key] = sum(parts) / len(parts)
    nw = pd.read_parquet(C.news_file(news_dir, tk), columns=["Date", "ev_hi"])
    ev = (nw.set_index(pd.to_datetime(nw["Date"]))["ev_hi"] > 0)
    f = pd.DataFrame(loss).dropna()
    f["event"] = ev.reindex(f.index).fillna(False).to_numpy().astype(bool)
    return f.rename_axis("Date").reset_index()


def tickers_with(preds_dir, run):
    return sorted(os.path.basename(p)[:-len(f"_{run}.parquet")].upper()
                  for p in glob.glob(os.path.join(preds_dir, f"*_{run}.parquet")))


def collect(arms, seeds, stage, preds_dir, news_dir, workers):
    """Stock-date losses of arm a (compared) and arm b (reference) for all stocks with both arms."""
    tks = tickers_with(preds_dir, C.run_name(arms[0], seeds[0]))
    with Pool(workers) as pool:
        parts = pool.map(stock_job, [(tk, arms, seeds, stage, preds_dir, news_dir) for tk in tks])
    return pd.concat([p for p in parts if p is not None], ignore_index=True)


def boot_idx(n, block, reps, rng):
    starts = np.arange(max(n - block + 1, 1)); k = int(np.ceil(n / block))
    for _ in range(reps):
        yield (rng.choice(starts, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]


def change_ci(ma, mb, block=21, reps=2000, seed=0):
    """Percentage change of the daily means and its moving-block bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    boot = [100 * (ma[ix] - mb[ix]).mean() / mb[ix].mean() for ix in boot_idx(len(ma), block, reps, rng)]
    return 100 * (ma - mb).mean() / mb.mean(), np.percentile(boot, [2.5, 97.5])


def newey_west_p(x, lags=12):
    """Two-sided p-value of mean(x) = 0 with a Bartlett-kernel HAC variance."""
    x = np.asarray(x, float); n = len(x); u = x - x.mean()
    s = u @ u / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return erfc(abs(x.mean() / sqrt(s / n)) / sqrt(2))


def summarize(d):
    g = d.groupby("Date")[["a", "b"]].mean()
    ma, mb = g["a"].to_numpy(), g["b"].to_numpy()
    chg, ci = change_ci(ma, mb)
    return {"dates": len(g), "stock_days": len(d), "loss_b": mb.mean(), "loss_a": ma.mean(),
            "diff": (ma - mb).mean(), "change%": chg, "ci_lo%": ci[0], "ci_hi%": ci[1],
            "p": newey_west_p(ma - mb)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="news,nonews", help="compared arm, reference arm")
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--preds-dir", default=C.PREDS_DIR)
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--out-dir", default=C.RESULTS_DIR)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for stage in args.stages.split(","):
        df = collect(arms, seeds, stage, args.preds_dir, args.news_dir, args.workers)
        df["year"] = df["Date"].dt.year
        for scope, sel in SCOPES:
            d = df[sel(df)]
            rows.append({"stage": stage.upper(), "scope": scope, "year": "all", **summarize(d)})
            if scope != "Non-event" and stage != "pinball":
                for yr, dy in d.groupby("year"):
                    rows.append({"stage": stage.upper(), "scope": scope, "year": yr, **summarize(dy)})
    T = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"paired_{arms[0]}_vs_{arms[1]}.csv")
    T.to_csv(out, index=False)

    print(f"\n[paired] {arms[0]} vs {arms[1]}, seeds {seeds}")
    print(T.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nTable VI rows:")
    for _, r in T[(T["year"] == "all") & T["stage"].isin(["CAL", "RAW"])].iterrows():
        p = "$<0.001$" if r["p"] < 0.001 else f"${r['p']:.2f}$"
        print(f"{r['stage']} & {r['scope']} & {r['loss_b']:.4f} & {r['loss_a']:.4f} & ${r['change%']:.1f}$ "
              f"& $[{r['ci_lo%']:.1f},\\,{r['ci_hi%']:+.1f}]$ & {p}\\\\")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
