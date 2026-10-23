"""Pooled dataset: 20-session windows of 21 market features and 24 news features for every stock.

Market features are shifted by one session (known at the previous close); news features are the current
session's pre-open values. The target is the session's close-to-close log return. Every feature is
z-scored per stock with that stock's training-period mean and standard deviation.
Splits: train <= 2017, validation 2018-2019, test 2020-2025. Windows are stored as float16.

Usage:
    python src/dataset.py --tickers-file data/tickers.txt
    python src/dataset.py --drop-group SENT --out artifacts/dataset_no_sent.pt     # ablation dataset
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import config as C
from news_features import FEATURES as NEWS_FEATURES, GROUPS

EXCLUDE = {"RTX"}                       # price history merged across a ticker change

PRICE_FEATURES = ["ret1", "ret5", "rv5", "rv20", "rsi14", "sma13_ratio", "sma21_ratio",
                  "sma50_ratio", "ema5_ratio", "bb_width", "bb_pos", "vol_z", "gap", "range",
                  "vix_norm", "vix_ret",
                  "f_mkt_ret1", "f_mkt_ret5", "f_mkt_vol10", "f_mkt_vol20", "f_mkt_drawdown20"]


def load_market(prices_dir, vix_file):
    """VIX features and S&P 500 context features, shared by all stocks."""
    vx = pd.read_parquet(vix_file)
    vx["Date"] = pd.to_datetime(vx["Date"]).dt.normalize()
    vc = pd.to_numeric(vx["VIX_Close"], errors="coerce")
    vix = pd.DataFrame({"Date": vx["Date"],
                        "vix_norm": pd.to_numeric(vx.get("VIX_Norm"), errors="coerce"),
                        "vix_ret": np.log(vc / vc.shift(1))})
    mk = pd.read_csv(os.path.join(prices_dir, "gspc.csv"))
    mk["Date"] = pd.to_datetime(mk["Date"]).dt.normalize()
    mc = pd.to_numeric(mk["Adj Close"], errors="coerce")
    mr = np.log(mc / mc.shift(1))
    mkt = pd.DataFrame({"Date": mk["Date"], "f_mkt_ret1": mr,
                        "f_mkt_ret5": mr.rolling(5, min_periods=3).sum(),
                        "f_mkt_vol10": mr.rolling(10, min_periods=5).std(),
                        "f_mkt_vol20": mr.rolling(20, min_periods=10).std(),
                        "f_mkt_drawdown20": mc / mc.rolling(20, min_periods=10).max() - 1.0})
    return vix, mkt


def price_frame(tk, prices_dir, vix, mkt):
    """The 21 market features of one stock (not yet shifted)."""
    px = pd.read_csv(os.path.join(prices_dir, f"{tk.lower()}.csv"))
    px["Date"] = pd.to_datetime(px["Date"]).dt.normalize()
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in px:
            px[c] = pd.to_numeric(px[c], errors="coerce")
    px = px.sort_values("Date").reset_index(drop=True)
    ratio = (px["Adj Close"] / px["Close"]).replace([np.inf, -np.inf], np.nan)
    o, h, l, c = px["Open"] * ratio, px["High"] * ratio, px["Low"] * ratio, px["Adj Close"]
    d = pd.DataFrame({"Date": px["Date"], "close": c})
    ret = np.log(c / c.shift(1))
    d["ret1"] = ret
    d["ret5"] = ret.rolling(5, min_periods=3).sum()
    d["rv5"] = ret.rolling(5, min_periods=3).std()
    d["rv20"] = ret.rolling(20, min_periods=10).std()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    d["rsi14"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50.0) / 100.0
    for w, nm in [(13, "sma13_ratio"), (21, "sma21_ratio"), (50, "sma50_ratio")]:
        d[nm] = c / c.rolling(w, min_periods=w // 2).mean() - 1.0
    d["ema5_ratio"] = c / c.ewm(span=5).mean() - 1.0
    s20 = c.rolling(20, min_periods=10).mean(); sd20 = c.rolling(20, min_periods=10).std()
    d["bb_width"] = (4 * sd20 / s20).replace([np.inf, -np.inf], np.nan)
    d["bb_pos"] = ((c - (s20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)).clip(-0.5, 1.5)
    if "Volume" in px and px["Volume"].notna().any():
        lv = np.log1p(px["Volume"])
        d["vol_z"] = ((lv - lv.rolling(63, min_periods=21).mean())
                      / lv.rolling(63, min_periods=21).std().replace(0, np.nan)).clip(-3, 6)
    else:
        d["vol_z"] = 0.0
    d["gap"] = np.log(o / c.shift(1))
    d["range"] = np.log(h / l)
    return d.merge(vix, on="Date", how="left").merge(mkt, on="Date", how="left")


def build_ticker(tk, news, prices_dir, vix, mkt, news_features=NEWS_FEATURES):
    """Windows of one stock. `news` is that stock's session-level news frame (news_features.py).
    Returns (dict of arrays, None) or (None, reason)."""
    d = price_frame(tk, prices_dir, vix, mkt)
    nw = news.copy()
    nw["Date"] = pd.to_datetime(nw["Date"]).dt.normalize()
    d = d.merge(nw, on="Date", how="left")
    d[news_features] = d[news_features].fillna(0.0)
    d["sent_signed"] = d["sent_signed"].fillna(0.0)       # kept for the sentiment-regime analyses

    d = d[d["Date"] >= C.START].reset_index(drop=True)
    pre_p = d[PRICE_FEATURES].shift(1)                    # market: through the previous close
    cur_n = d[news_features]                              # news: this session's pre-open features
    d["y"] = np.log(d["close"] / d["close"].shift(1))
    d["price_base"] = d["close"].shift(1)
    ok = pre_p.notna().all(1) & cur_n.notna().all(1) & d["y"].notna() & d["price_base"].notna()
    d, pre_p, cur_n = d[ok].reset_index(drop=True), pre_p[ok].reset_index(drop=True), cur_n[ok].reset_index(drop=True)
    if len(d) < 800:
        return None, f"too short ({len(d)})"

    dates = d["Date"].values.astype("datetime64[ns]")
    split = np.where(dates <= np.datetime64(C.TRAIN_END), "train",
                     np.where(dates <= np.datetime64(C.VAL_END), "val",
                              np.where(dates <= np.datetime64(C.TEST_END), "test", "drop")))
    tr = split == "train"
    if tr.sum() < 400:
        return None, "short train"

    def z(block):                                         # training-period standardization
        X = block.to_numpy(np.float64)
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-8] = 1.0
        return ((X - mu) / sd).astype(np.float32)

    Xp, Xn = z(pre_p), z(cur_n)
    idx = [i for i in range(C.WINDOW - 1, len(d)) if split[i] in ("train", "val", "test")]
    Wp = np.stack([Xp[i - C.WINDOW + 1: i + 1] for i in idx]).astype(np.float16)
    Wn = np.stack([Xn[i - C.WINDOW + 1: i + 1] for i in idx]).astype(np.float16)
    return {"Xp": Wp, "Xn": Wn, "y": d["y"].to_numpy(np.float32)[idx],
            "price": d["price_base"].to_numpy(np.float32)[idx],
            "dates": dates[idx], "split": split[idx],
            "sent_signed": d["sent_signed"].to_numpy(np.float32)[idx]}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers-file", default=C.TICKERS_FILE)
    ap.add_argument("--prices-dir", default=C.PRICES_DIR)
    ap.add_argument("--vix-file", default=C.VIX_FILE)
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--drop-group", default="", choices=[""] + list(GROUPS),
                    help="leave one news feature group out (ablation)")
    ap.add_argument("--out", default=C.DATASET)
    args = ap.parse_args()
    features = [f for f in NEWS_FEATURES if f not in GROUPS.get(args.drop_group, [])]
    if args.drop_group:
        print(f"[data] dropping {args.drop_group}: {len(features)} news features remain")
    tks = [t for t in C.read_tickers(args.tickers_file) if t not in EXCLUDE]
    vix, mkt = load_market(args.prices_dir, args.vix_file)

    parts, kept, skipped = [], [], []
    for tk in tks:
        nf = C.news_file(args.news_dir, tk)
        if not os.path.exists(nf):
            skipped.append((tk, "no news file")); continue
        b, err = build_ticker(tk, pd.read_parquet(nf), args.prices_dir, vix, mkt, features)
        if b is None:
            skipped.append((tk, err)); continue
        b["stock_id"] = np.full(len(b["y"]), len(kept), dtype=np.int64)
        parts.append(b); kept.append(tk)
        print(f"  [data] {tk:<6} n={len(b['y'])}", flush=True)
    if not parts:
        raise SystemExit("[data] nothing built")
    blob = {k: (torch.from_numpy(np.concatenate([p[k] for p in parts]))
                if k not in ("dates", "split") else np.concatenate([p[k] for p in parts]))
            for k in ["Xp", "Xn", "y", "price", "dates", "split", "stock_id", "sent_signed"]}
    blob.update({"tickers": kept, "price_features": PRICE_FEATURES,
                 "news_features": features, "window": C.WINDOW})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(blob, args.out)
    n = {s: int((blob["split"] == s).sum()) for s in ("train", "val", "test")}
    print(f"\n[data] {len(kept)} stocks -> {args.out}  train={n['train']:,} val={n['val']:,} test={n['test']:,}")
    if skipped:
        print("  skipped: " + ", ".join(f"{t} ({e})" for t, e in skipped))


if __name__ == "__main__":
    main()
