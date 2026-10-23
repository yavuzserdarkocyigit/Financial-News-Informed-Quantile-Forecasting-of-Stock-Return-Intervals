"""Daily prices from Yahoo Finance: one CSV per ticker, plus the S&P 500 index saved as gspc.csv.

Columns: Date, Adj Close, Close, Dividends, High, Low, Open, Stock Splits, Volume. The full history is
downloaded in one call so the split/dividend adjustment is consistent across the whole series.

Usage:
    python src/data_collection/prices.py --tickers-file data/tickers.txt
"""
import argparse
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

MARKET = "^GSPC"
COLUMNS = ["Date", "Adj Close", "Close", "Dividends", "High", "Low", "Open", "Stock Splits", "Volume"]


def _flat(d):
    if d is None or d.empty:
        raise RuntimeError("empty download")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index()
    d["Date"] = pd.to_datetime(d["Date"], utc=True).dt.tz_convert(None).dt.normalize()
    return d


def fetch(ticker, start, end_excl, retries=3):
    for attempt in range(1, retries + 1):
        try:
            df = _flat(yf.download(ticker, start=start, end=end_excl, progress=False, auto_adjust=False,
                                   actions=True, timeout=30))
            if "Adj Close" not in df or df["Adj Close"].isna().all():     # fall back to adjusted Close
                adj = _flat(yf.download(ticker, start=start, end=end_excl, progress=False, auto_adjust=True,
                                        timeout=30))[["Date", "Close"]].rename(columns={"Close": "Adj Close"})
                df = df.drop(columns=[c for c in ["Adj Close"] if c in df]).merge(adj, on="Date", how="left")
            for c in COLUMNS:
                if c not in df:
                    df[c] = pd.NA
            if df["Adj Close"].isna().all():
                raise RuntimeError("no adjusted close")
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            return df[COLUMNS]
        except Exception as e:                                           # network hiccups
            print(f"  [{ticker}] attempt {attempt}/{retries} failed: {e}", flush=True)
            time.sleep(2.0)
    raise RuntimeError(f"{ticker}: download failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers-file", default=C.TICKERS_FILE)
    ap.add_argument("--start", default="1970-01-01")
    ap.add_argument("--end", default="2025-12-31", help="last date kept (inclusive)")
    ap.add_argument("--out-dir", default=C.PRICES_DIR)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    end_excl = (pd.Timestamp(args.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for tk in C.read_tickers(args.tickers_file) + [MARKET]:
        out = os.path.join(args.out_dir, "gspc.csv" if tk == MARKET else f"{tk.lower()}.csv")
        if os.path.exists(out) and not args.force:
            continue
        try:
            df = fetch(tk, args.start, end_excl)
        except RuntimeError as e:
            print(f"  [{tk}] FAILED: {e}", flush=True)
            continue
        df[pd.to_datetime(df["Date"]) <= pd.Timestamp(args.end)].to_csv(out, index=False)
        print(f"  [{tk}] {len(df):>6} rows -> {os.path.basename(out)}", flush=True)


if __name__ == "__main__":
    main()
