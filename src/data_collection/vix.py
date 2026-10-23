"""Daily VIX close from FRED (series VIXCLS) with its return and a min-max normalized level.

VIX_Norm uses the minimum and maximum of the whole downloaded series. This does not leak information
into the model: dataset.py z-scores every feature with training-period statistics, and a z-score is
unchanged by an affine rescaling, so the model sees the same values as for the raw VIX level.

Usage:
    python src/data_collection/vix.py
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--end", default="2025-12-31", help="last date kept (inclusive)")
    ap.add_argument("--out", default=C.VIX_FILE)
    args = ap.parse_args()
    df = pd.read_csv(FRED_URL)
    df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "VIX_Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["VIX_Close"] = pd.to_numeric(df["VIX_Close"], errors="coerce")
    df = df.dropna().sort_values("Date")
    df = df[df["Date"] <= pd.Timestamp(args.end)].reset_index(drop=True)
    df["VIX_Return"] = df["VIX_Close"].pct_change()
    lo, hi = df["VIX_Close"].min(), df["VIX_Close"].max()
    df["VIX_Norm"] = (df["VIX_Close"] - lo) / (hi - lo)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[vix] {len(df):,} days {df['Date'].min().date()} -> {df['Date'].max().date()} -> {args.out}")


if __name__ == "__main__":
    main()
