"""Paper figures at IEEE print size (Fig. 1, the architecture diagram, is drawn by hand).

    cases       Fig. 2  event sessions covered by the news-aware p95 interval and missed by the no-news
                        one under every seed; four pinned stocks, bands of the first seed
    asymmetry   Fig. 3  raw p95 distances below and above the median by sentiment regime (and Table S8,
                        including the content placebo when its predictions exist)
    var         Fig. 4  VaR breach rates on negative-news sessions (reads var_backtest.csv)

Usage:
    python src/figures.py --figs cases,asymmetry,var
"""
import argparse
import glob
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from calibration import calibrate
from evaluate import load_preds
from news_features import CHAN_CATS, MAX_TICKERS, _as_list, _templated, assign_sessions
from paired_stats import tickers_with

LC, HC, COV, THR = "q025", "q975", 0.95, 0.1
COL_W, FULL_W = 3.5, 7.16                                  # IEEE column and text widths (inches)
C_NEWS, C_NONEWS, C_MISS, C_EVENT, C_BENCH = "#2e7d32", "#767676", "#c62828", "#e08214", "#b0524d"
plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "savefig.bbox": "tight"})


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}.pdf")


def bands(tk, run, preds_dir):
    """(test frame, calibrated lower, calibrated upper) of one stock and run, or None."""
    vt = load_preds(C.pred_file(preds_dir, tk, run))
    if vt is None or len(vt[1]) < 150 or len(vt[0]) < 60:
        return None
    v, t = vt
    return (t, *calibrate(v, t, LC, HC, COV))


def event_mask(tk, dates, news_dir):
    nw = pd.read_parquet(C.news_file(news_dir, tk), columns=["Date", "ev_hi"])
    nw["Date"] = pd.to_datetime(nw["Date"])
    return nw.set_index("Date")["ev_hi"].reindex(pd.DatetimeIndex(dates)).fillna(0).to_numpy() > 0


# --------------------------------------------------------------------------- Fig. 2: case studies
def scan_catches(tickers, seed, preds_dir, news_dir, cache):
    """Sessions covered by exactly one of the two calibrated p95 intervals (cached per seed)."""
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=["Date"])
    rows = []
    for tk in tickers:
        F, N = bands(tk, C.run_name("news", seed), preds_dir), bands(tk, C.run_name("nonews", seed), preds_dir)
        if F is None or N is None or len(F[0]) != len(N[0]):
            continue
        (tf, lof, hif), (_, lon, hin) = F, N
        y = tf["y"].to_numpy(float)
        cov_f, cov_n = (y >= lof) & (y <= hif), (y >= lon) & (y <= hin)
        ev = event_mask(tk, tf["Date"].values, news_dir)
        sev_n = np.maximum(lon - y, y - hin) / np.maximum(hin - lon, 1e-9)   # depth of the miss, in widths
        sev_f = np.maximum(lof - y, y - hif) / np.maximum(hif - lof, 1e-9)
        for i in np.where(cov_f & ~cov_n)[0]:
            rows.append({"ticker": tk, "Date": tf["Date"].values[i], "kind": "catch", "y": y[i],
                         "sev": sev_n[i], "event": bool(ev[i])})
        for i in np.where(cov_n & ~cov_f)[0]:
            rows.append({"ticker": tk, "Date": tf["Date"].values[i], "kind": "reverse", "y": y[i],
                         "sev": sev_f[i], "event": bool(ev[i])})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def session_headlines(benzinga_dir, tk, td):
    """date -> pre-open headlines, ordered by event group (earnings first), then freshest first."""
    recs = []
    for fp in sorted(glob.glob(os.path.join(benzinga_dir, tk.upper(), f"{tk.upper()}_*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                try:
                    a = json.loads(ln)
                except ValueError:
                    continue
                if _templated(a) or len(_as_list(a.get("tickers"))) > MAX_TICKERS:
                    continue
                ttl = (a.get("title") or "").strip()
                if ttl:
                    ch = {str(c).lower() for c in _as_list(a.get("channels"))}
                    rank = (0 if ch & CHAN_CATS["earn"] else 1 if ch & CHAN_CATS["corp"] else
                            2 if ch & CHAN_CATS["analyst"] else 3 if ch & CHAN_CATS["macro"] else 4)
                    recs.append({"published": a.get("published"), "title": ttl, "rank": rank})
    if not recs:
        return {}
    df = assign_sessions(pd.DataFrame(recs), td).sort_values("et")
    return {pd.Timestamp(s).normalize(): list(g.iloc[::-1].sort_values("rank", kind="stable")["title"])
            for s, g in df.groupby("sess")}


def fig_cases(tickers, seeds, pin, args):
    scans = {s: scan_catches(tickers, s, args.preds_dir, args.news_dir,
                             os.path.join(args.out_dir, f"case_scan_s{s}.csv")) for s in seeds}
    keys = None                                       # event catches present under every seed
    for d in scans.values():
        k = set(map(tuple, d.loc[(d["kind"] == "catch") & d["event"], ["ticker", "Date"]]
                    .astype({"Date": "datetime64[ns]"}).itertuples(index=False, name=None)))
        keys = k if keys is None else keys & k
    ref = scans[seeds[0]]
    cand = ref[(ref["kind"] == "catch") & ref["event"]]
    cand = cand[[(t, pd.Timestamp(d)) in keys for t, d in zip(cand["ticker"], cand["Date"])]]
    cand = cand.sort_values("sev", ascending=False)
    print(f"  [cases] {len(cand):,} event sessions caught under all {len(seeds)} seeds")
    picks = [cand[cand["ticker"] == tk].iloc[0] for tk in pin if (cand["ticker"] == tk).any()]

    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.4))
    run = C.run_name("news", seeds[0])
    for ax, r in zip(axes.ravel(), picks):
        tk, day = r["ticker"], pd.Timestamp(r["Date"])
        tf, lof, hif = bands(tk, run, args.preds_dir)
        _, lon, hin = bands(tk, C.run_name("nonews", seeds[0]), args.preds_dir)
        dts = pd.DatetimeIndex(tf["Date"])
        j = int(np.argmin(np.abs(dts - day)))
        a, b = max(0, j - 10), min(len(dts), j + 8)
        px = tf["price"].to_numpy(float)[a:b]
        real = px * np.exp(tf["y"].to_numpy(float)[a:b])
        d = dts[a:b]
        ax.fill_between(d, px * np.exp(lof[a:b]), px * np.exp(hif[a:b]), color=C_NEWS, alpha=0.22,
                        label="news band (95%)")
        ax.plot(d, px * np.exp(lon[a:b]), ls="--", lw=1.1, color=C_NONEWS, label="no-news band")
        ax.plot(d, px * np.exp(hin[a:b]), ls="--", lw=1.1, color=C_NONEWS)
        ax.plot(d, real, "-o", color="black", ms=2.6, lw=1.0, label="realized close")
        ax.axvline(day, color=C_EVENT, lw=1.4, alpha=0.9)
        ax.plot([day], [real[j - a]], "o", color=C_MISS, ms=7, zorder=5)
        hl = session_headlines(args.benzinga_dir, tk,
                               np.sort(tf["Date"].values.astype("datetime64[ns]"))).get(day.normalize(), [])
        ttl = textwrap.shorten(hl[0] if hl else "(event session)", width=76, placeholder=" …")
        pct = 100.0 * (np.exp(float(r["y"])) - 1.0)        # simple return, as in public reporting
        ax.set_title(f"{tk} · {day.date()} · return {pct:+.1f}% · {len(hl)} pre-open "
                     f"article{'s' if len(hl) != 1 else ''}\n“{ttl}”",
                     fontsize=6.6, loc="left", linespacing=1.35)
        ax.set_ylabel("price ($)")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    axes.ravel()[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, args.out_dir, "fig2_case_studies")


# --------------------------------------------------------------------------- Fig. 3: band asymmetry
def band_shape(tickers, seeds, preds_dir, arms=("news", "nonews")):
    """Mean (downside, upside) raw p95 distance from the median per arm and sentiment regime:
    averaged over stocks within a seed, then over seeds. Regimes always use the original sentiment
    (that of the news-aware model's inputs), also for the content placebo, whose own sentiment is permuted."""
    per_seed = []
    for s in seeds:
        acc = {a: {r: [[], []] for r in ("NEG", "NEU", "POS")} for a in arms}
        for tk in tickers:
            ref = pd.read_parquet(C.pred_file(preds_dir, tk, C.run_name("news", s)), columns=["Date", "split", "sent"])
            ref = ref[ref["split"] == "test"].set_index("Date")["sent"]
            for arm in arms:
                p = C.pred_file(preds_dir, tk, C.run_name(arm, s))
                if not os.path.exists(p):
                    continue
                t = pd.read_parquet(p)
                t = t[t["split"] == "test"]
                if len(t) < 150:
                    continue
                med, sent = t["q50"].to_numpy(float), ref.reindex(t["Date"]).to_numpy(float)
                dn, up = med - t[LC].to_numpy(float), t[HC].to_numpy(float) - med
                for reg, m in (("NEG", sent < -THR), ("NEU", np.abs(sent) <= THR), ("POS", sent > THR)):
                    if m.sum():
                        acc[arm][reg][0].append(dn[m].mean()); acc[arm][reg][1].append(up[m].mean())
        per_seed.append({a: {r: (np.mean(acc[a][r][0]), np.mean(acc[a][r][1])) for r in acc[a]} for a in acc})
    return {a: {r: (np.mean([p[a][r][0] for p in per_seed]), np.mean([p[a][r][1] for p in per_seed]))
                for r in ("NEG", "NEU", "POS")} for a in arms}


def regime_counts(tickers, run, preds_dir):
    """Test stock-sessions per sentiment regime."""
    n = dict.fromkeys(("NEG", "NEU", "POS"), 0)
    for tk in tickers:
        t = pd.read_parquet(C.pred_file(preds_dir, tk, run), columns=["split", "sent"])
        sent = t.loc[t["split"] == "test", "sent"].to_numpy(float)
        n["NEG"] += int((sent < -THR).sum()); n["NEU"] += int((np.abs(sent) <= THR).sum()); n["POS"] += int((sent > THR).sum())
    return n


def fig_asymmetry(tickers, seeds, args):
    has_placebo = all(os.path.exists(C.pred_file(args.preds_dir, tickers[0], C.run_name("placebo", s))) for s in seeds)
    acc = band_shape(tickers, seeds, args.preds_dir, ("news", "nonews", "placebo") if has_placebo else ("news", "nonews"))
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    ypos, labels, y = [], [], 0
    for reg in ("NEG", "NEU", "POS"):
        for arm, col in (("news", C_NEWS), ("nonews", C_NONEWS)):
            dn, up = acc[arm][reg]
            ax.barh(y, dn + up, left=-dn, height=0.62, color=col, alpha=0.85, edgecolor="black", linewidth=0.35)
            ypos.append(y); labels.append(f"{reg} · {'news-aware' if arm == 'news' else 'no-news'}")
            y -= 1
        y -= 0.5
    ax.axvline(0, color="black", lw=1.1)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xlabel("distance from median (log-return)\n" r"$\leftarrow$ downside | upside $\rightarrow$")
    fig.tight_layout()
    save(fig, args.out_dir, "fig3_band_asymmetry")

    skew = {a: {r: (acc[a][r][1] - acc[a][r][0]) / (acc[a][r][1] + acc[a][r][0]) for r in acc[a]} for a in acc}
    n = regime_counts(tickers, C.run_name("news", seeds[0]), args.preds_dir)
    print("\n  [asymmetry] Table S8: raw p95 distances below / above the median and band skew")
    print(f"  {'regime':<7} {'sessions':>9} {'news d-':>9} {'news d+':>9} " + " ".join(f"{a + ' skew':>13}" for a in acc))
    for r in ("NEG", "NEU", "POS"):
        print(f"  {r:<7} {n[r]:>9,} {acc['news'][r][0]:>9.5f} {acc['news'][r][1]:>9.5f} "
              + " ".join(f"{skew[a][r]:>+13.4f}" for a in acc))


# --------------------------------------------------------------------------- Fig. 4: VaR
def fig_var(args):
    d = pd.read_csv(os.path.join(args.results_dir, "var_backtest.csv"))
    d = d.groupby(["level", "method"])["neg_breach"].mean()                # mean over seeds
    order = ["news (CAL)", "no-news (CAL)", "news (RAW)", "no-news (RAW)", "Historical", "Normal", "EWMA"]
    label = {"news (CAL)": "News-aware (CAL)", "no-news (CAL)": "No-news (CAL)",
             "news (RAW)": "News-aware (RAW)", "no-news (RAW)": "No-news (RAW)"}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
    for ax, (lvl, tgt) in zip(axes, (("2.5%", 2.5), ("1%", 1.0))):
        ms = [m for m in order if (lvl, m) in d.index]
        vals = np.array([d[(lvl, m)] for m in ms])
        for i, (m, v) in enumerate(zip(ms, vals)):
            ax.bar(i, v, 0.65, color=C_NEWS if m.startswith("news") else C_NONEWS if "news" in m else C_BENCH,
                   alpha=0.45 if "RAW" in m else 0.90, edgecolor="black", linewidth=0.4)
            ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=7.5)
        ax.axhline(tgt, ls="--", color="black", lw=1.2)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([label.get(m, m) for m in ms], rotation=28, ha="right", fontsize=6.8)
        ax.set_title(f"{tgt}% VaR", fontsize=8.5)
        ax.set_ylim(0, max(vals.max() * 1.08, tgt * 1.12))
    axes[0].set_ylabel("breach rate on negative-news sessions (%)")
    fig.tight_layout()
    save(fig, args.out_dir, "fig4_var_negative_news")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figs", default="cases,asymmetry,var")
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--cases", default="NKE,CRM,ANET,RMD", help="stocks pinned in the case-study figure")
    ap.add_argument("--preds-dir", default=C.PREDS_DIR)
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--benzinga-dir", default=C.BENZINGA_DIR)
    ap.add_argument("--results-dir", default=C.RESULTS_DIR)
    ap.add_argument("--out-dir", default=os.path.join(C.ARTIFACTS, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    want = set(args.figs.split(","))
    tickers = tickers_with(args.preds_dir, C.run_name("news", seeds[0])) if want & {"cases", "asymmetry"} else []
    if "cases" in want:
        fig_cases(tickers, seeds, [t.strip().upper() for t in args.cases.split(",")], args)
    if "asymmetry" in want:
        fig_asymmetry(tickers, seeds, args)
    if "var" in want:
        fig_var(args)


if __name__ == "__main__":
    main()
