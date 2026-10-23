"""Session-level news representation: 24 features per (stock, trading session).

Articles are assigned to the session they precede: an article published before 09:30 ET on a trading day
belongs to that day's session, every other article to the next trading session. Rolling statistics use
preceding sessions only, and novelty compares each headline only with earlier articles of the same stock.

    FLOW (4)       news_z, novel_flow_z, decay_flow_z, n_log
    SENT (8)       title_neg, title_pos, body_neg, body_pos, sent_signed, neg_share, sent_dispersion,
                   tb_disagreement
    NOVELTY (3)    novelty, surprise, novel_negative
    EVENT (5)      ev_earn, ev_analyst, ev_corp, ev_macro, ev_hi
    META (4)       has_news, premarket_share, relevance_mean, finbert_cov

Inputs: raw Benzinga articles, the FinBERT score database and the trading calendar of each stock.
Output: one parquet per stock (Date + the 24 features); sessions without news are zero.

Usage:
    python src/news_features.py --tickers-file data/tickers.txt
"""
import argparse
import glob
import json
import os
import sqlite3
from collections import deque

import numpy as np
import pandas as pd

import config as C

OPEN_SECONDS = int(9.5 * 3600)          # 09:30 ET
MAX_TICKERS = 3                         # articles tagged with more tickers are not firm-specific
NOV_DAYS, NOV_CAP = 30, 100             # novelty: compare with the last 30 days, at most 100 articles
ZWIN, ZMIN = 63, 21                     # window of the abnormal-flow z-scores

# Benzinga channels of the four event groups
CHAN_CATS = {
    "earn":    {"earnings", "guidance", "reg fd"},
    "analyst": {"analyst ratings", "price target", "upgrades", "downgrades"},
    "corp":    {"m&a", "contracts", "asset sales", "offerings", "buybacks", "dividends",
                "ipos", "financing", "partnership", "products"},
    "macro":   {"fda", "clinical trials", "legal", "regulations", "government", "federal reserve"},
}
HIGH_IMPACT = set().union(*CHAN_CATS.values())          # any event group -> event session

FEATURES = ["news_z", "novel_flow_z", "decay_flow_z", "n_log",
            "title_neg", "title_pos", "body_neg", "body_pos", "sent_signed", "neg_share",
            "sent_dispersion", "tb_disagreement",
            "novelty", "surprise", "novel_negative",
            "ev_earn", "ev_analyst", "ev_corp", "ev_macro", "ev_hi",
            "has_news", "premarket_share", "relevance_mean", "finbert_cov"]

# feature groups of the ablation study (EMBED = everything computed from the title embeddings)
GROUPS = {
    "FLOW":    ["news_z", "novel_flow_z", "decay_flow_z", "n_log"],
    "SENT":    ["title_neg", "title_pos", "body_neg", "body_pos", "sent_signed", "neg_share",
                "sent_dispersion", "tb_disagreement"],
    "NOVELTY": ["novelty", "surprise", "novel_negative"],
    "EVENT":   ["ev_earn", "ev_analyst", "ev_corp", "ev_macro", "ev_hi"],
    "META":    ["has_news", "premarket_share", "relevance_mean", "finbert_cov"],
    "EMBED":   ["novel_flow_z", "novelty", "surprise", "novel_negative"],
}


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return list(x)
    return [x]


def _templated(a):
    """Automatically generated Benzinga articles carry no firm-specific information."""
    author = (a.get("author") or "").strip().lower()
    return author == "benzinga insights" or any(
        str(t).lower().startswith("bzi-") for t in _as_list(a.get("tags")))


def load_finbert(db_path):
    """benzinga_id -> six sentiment scores, and benzinga_id -> unit-norm title embedding."""
    scores, embs = {}, {}
    con = sqlite3.connect(db_path)
    sc = pd.read_sql("SELECT benzinga_id, title_negative, title_positive, title_sentiment, "
                     "body_negative, body_positive, body_sentiment FROM scores", con)
    sc["benzinga_id"] = pd.to_numeric(sc["benzinga_id"], errors="coerce")
    for r in sc.itertuples():
        if np.isfinite(r.benzinga_id):
            scores[int(r.benzinga_id)] = (r.title_negative, r.title_positive, r.title_sentiment,
                                          r.body_negative, r.body_positive, r.body_sentiment)
    for bid, blob in con.execute("SELECT benzinga_id, semantic_embedding FROM scores"):
        try:
            embs[int(bid)] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        except (TypeError, ValueError):
            pass
    con.close()
    print(f"[news] {len(scores):,} scored articles from {os.path.basename(db_path)}", flush=True)
    return scores, embs


def collect_articles(benzinga_dir, tk):
    recs = []
    for fp in sorted(glob.glob(os.path.join(benzinga_dir, tk.upper(), f"{tk.upper()}_*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                a = json.loads(ln)
                if _templated(a):
                    continue
                tks = _as_list(a.get("tickers"))
                if len(tks) > MAX_TICKERS:
                    continue
                chans = {str(c).lower() for c in _as_list(a.get("channels"))}
                try:
                    bid = int(a.get("benzinga_id"))
                except (TypeError, ValueError):
                    bid = -1
                recs.append({"bid": bid, "published": a.get("published"),
                             "chans": chans, "n_tk": max(len(tks), 1)})
    return pd.DataFrame(recs)


def assign_sessions(df, td):
    """Map every article to the trading session it precedes (td = sorted trading days)."""
    et = pd.to_datetime(df["published"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    et_date = et.dt.tz_localize(None).dt.normalize().values.astype("datetime64[ns]")
    sec = (et.dt.hour * 3600 + et.dt.minute * 60 + et.dt.second).values
    premkt = sec < OPEN_SECONDS
    li = np.searchsorted(td, et_date, "left"); lic = np.clip(li, 0, len(td) - 1)
    is_td = (li < len(td)) & (td[lic] == et_date)
    ri = np.searchsorted(td, et_date, "right")
    nxt = np.where(ri < len(td), td[np.clip(ri, 0, len(td) - 1)],
                   np.datetime64("NaT")).astype("datetime64[ns]")
    df = df.copy()
    df["sess"] = np.where(premkt & is_td, et_date, nxt).astype("datetime64[ns]")
    df["et"] = et.values
    # published on the session's own calendar day before the open (not carried over from earlier)
    df["premkt_same_day"] = (df["sess"].values == et_date) & premkt
    return df.dropna(subset=["sess"])


def novelty_pass(df, embs):
    """Chronological and past-only: novelty = 1 - max cosine to the preceding 30 days (at most 100
    articles), surprise = 1 - cosine to the centroid of those articles."""
    nov = np.ones(len(df)); sur = np.ones(len(df))
    order = np.argsort(df["et"].values, kind="stable")
    win = pd.Timedelta(days=NOV_DAYS)
    hist = deque()                                   # (timestamp, unit vector)
    dim = len(next(iter(embs.values()))) if embs else 96
    csum = np.zeros(dim, dtype=np.float64)
    for idx in order:
        bid = df["bid"].iloc[idx]
        ts = df["et"].iloc[idx]
        v = embs.get(bid)
        while hist and (ts - hist[0][0]) > win:
            csum -= hist.popleft()[1]
        if v is not None and len(hist):
            m = 0.0
            for k, (_, hv) in enumerate(reversed(hist)):
                if k >= NOV_CAP:
                    break
                s = float(np.dot(v, hv))
                if s > m:
                    m = s
            nov[idx] = 1.0 - max(min(m, 1.0), 0.0)
            c = csum / max(len(hist), 1)
            cn = np.linalg.norm(c)
            sur[idx] = 1.0 - max(min(float(np.dot(v, c / cn)) if cn > 1e-9 else 0.0, 1.0), 0.0)
        if v is not None:
            hist.append((ts, v)); csum += v
    return nov, sur


def build_ticker(tk, prices_dir, benzinga_dir, scores, embs):
    """Return (features DataFrame, None) or (None, reason)."""
    cs = os.path.join(prices_dir, f"{tk.lower()}.csv")
    if not os.path.exists(cs):
        return None, "no price csv"
    px = pd.read_csv(cs, usecols=["Date"])
    td = np.array(sorted(pd.to_datetime(px["Date"]).dt.normalize().unique())).astype("datetime64[ns]")
    art = collect_articles(benzinga_dir, tk)
    if art.empty:
        return None, "no articles"
    art = assign_sessions(art, td)
    if art.empty:
        return None, "no session-assignable articles"

    art["novelty"], art["surprise"] = novelty_pass(art, embs)
    S = np.array([scores.get(b, (np.nan,) * 6) for b in art["bid"]], dtype=float)
    art["t_neg"], art["t_pos"], art["t_sent"] = S[:, 0], S[:, 1], S[:, 2]
    art["b_neg"], art["b_pos"], art["b_sent"] = S[:, 3], S[:, 4], S[:, 5]
    art["c_sent"] = (art["t_sent"] + art["b_sent"]) / 2.0
    art["tb_dis"] = np.abs(art["t_sent"] - art["b_sent"])
    art["nov_neg"] = art["novelty"] * art["b_neg"]
    for cat, cset in CHAN_CATS.items():
        art[f"c_{cat}"] = art["chans"].apply(lambda s: len(s & cset) > 0)
    art["c_hi"] = art["chans"].apply(lambda s: len(s & HIGH_IMPACT) > 0)
    art["rel"] = 1.0 / art["n_tk"]
    art["scored"] = np.isfinite(art["t_sent"])

    g = art.groupby("sess")
    d = pd.DataFrame({
        "n": g.size().astype(float),
        "nov_flow": g["novelty"].sum(),
        "title_neg": g["t_neg"].mean(), "title_pos": g["t_pos"].mean(),
        "body_neg": g["b_neg"].mean(), "body_pos": g["b_pos"].mean(),
        "sent_signed": g["c_sent"].mean(),
        "neg_share": g["c_sent"].apply(lambda s: float((s < -0.1).mean())),
        "sent_dispersion": g["c_sent"].std(),
        "tb_disagreement": g["tb_dis"].mean(),
        "novelty": g["novelty"].mean(), "surprise": g["surprise"].mean(),
        "novel_negative": g["nov_neg"].mean(),
        "ev_earn": g["c_earn"].sum(), "ev_analyst": g["c_analyst"].sum(),
        "ev_corp": g["c_corp"].sum(), "ev_macro": g["c_macro"].sum(), "ev_hi": g["c_hi"].sum(),
        "premarket_share": g["premkt_same_day"].mean(),
        "relevance_mean": g["rel"].mean(),
        "finbert_cov": g["scored"].mean(),
    })
    d.index = pd.DatetimeIndex(d.index); d.index.name = "Date"
    d = d.reindex(pd.DatetimeIndex(td)).rename_axis("Date")

    def abnormal(x):                                  # z-score against the preceding 63 sessions
        m = x.rolling(ZWIN, min_periods=ZMIN).mean().shift(1)
        s = x.rolling(ZWIN, min_periods=ZMIN).std().shift(1).replace(0, np.nan)
        return ((x - m) / s).clip(-3, 6).fillna(0.0)

    n = d["n"].fillna(0.0)
    d["news_z"] = abnormal(n)
    d["novel_flow_z"] = abnormal(d["nov_flow"].fillna(0.0))
    d["decay_flow_z"] = abnormal(n.ewm(halflife=5).mean())     # persistent attention
    d["n_log"] = np.log1p(n)
    d["has_news"] = (n > 0).astype(float)
    for c in ["ev_earn", "ev_analyst", "ev_corp", "ev_macro", "ev_hi"]:
        d[c] = 1.0 - np.exp(-d[c].fillna(0.0))                 # saturating event intensity
    d["sent_dispersion"] = d["sent_dispersion"].fillna(0.0)     # single-article session
    return d[FEATURES].fillna(0.0).reset_index(), None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers-file", default=C.TICKERS_FILE)
    ap.add_argument("--ticker", default="", help="build a single ticker instead of the list")
    ap.add_argument("--prices-dir", default=C.PRICES_DIR)
    ap.add_argument("--benzinga-dir", default=C.BENZINGA_DIR)
    ap.add_argument("--finbert-db", default=C.FINBERT_DB)
    ap.add_argument("--out-dir", default=C.NEWS_DIR)
    args = ap.parse_args()
    tks = [args.ticker.upper()] if args.ticker else C.read_tickers(args.tickers_file)
    os.makedirs(args.out_dir, exist_ok=True)
    scores, embs = load_finbert(args.finbert_db)

    done, skipped = 0, []
    for tk in tks:
        d, err = build_ticker(tk, args.prices_dir, args.benzinga_dir, scores, embs)
        if d is None:
            skipped.append((tk, err)); continue
        d.to_parquet(C.news_file(args.out_dir, tk), index=False)
        print(f"  [news] {tk:<6} sessions={len(d):>5} news sessions={int((d['has_news'] > 0).sum()):>5}",
              flush=True)
        done += 1
    print(f"\n[news] {done} tickers -> {args.out_dir}")
    if skipped:
        print(f"  skipped {len(skipped)}: " + ", ".join(f"{t} ({e})" for t, e in skipped))


if __name__ == "__main__":
    main()
