"""Content-placebo news features (Tables S8-S9).

The 16 content features (sentiment 8, semantic change 3, event content 5) are permuted as one block among
each stock's own news sessions within the same split, with seed 2026 plus the stock's position in the
ticker list. Features describing news arrival stay in place (news flow, news presence, premarket share,
firm specificity, FinBERT completeness), so a model trained on these features knows when and how much news
arrived but not what it said.

Usage:
    python src/placebo.py
    python src/dataset.py --news-dir artifacts/news_placebo --out artifacts/dataset_placebo.pt
    python src/train.py --data artifacts/dataset_placebo.pt --arm placebo --seed 7      # and 42, 123
    python src/paired_stats.py --arms placebo,nonews --stages cal                         # Table S9
    python src/paired_stats.py --arms news,placebo --stages cal                           # Table S9
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as C
from news_features import GROUPS

SEED = 2026
CONTENT = GROUPS["SENT"] + GROUPS["NOVELTY"] + GROUPS["EVENT"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tickers-file", default=C.TICKERS_FILE, help="dataset ticker order (fixes the seeds)")
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--out-dir", default=os.path.join(C.ARTIFACTS, "news_placebo"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    moved = 0
    for j, tk in enumerate(C.read_tickers(args.tickers_file)):
        d = pd.read_parquet(C.news_file(args.news_dir, tk))
        dt = pd.to_datetime(d["Date"])
        split = np.where(dt <= C.TRAIN_END, "train", np.where(dt <= C.VAL_END, "val", "test"))
        rng = np.random.default_rng(SEED + j)
        for s in ("train", "val", "test"):
            idx = np.where((split == s) & (d["has_news"].to_numpy() > 0))[0]
            if len(idx) > 1:
                d.loc[idx, CONTENT] = d.loc[rng.permutation(idx), CONTENT].to_numpy()
                moved += len(idx)
        d.to_parquet(C.news_file(args.out_dir, tk), index=False)
    print(f"[placebo] content of {moved:,} news sessions permuted within stock and split -> {args.out_dir}")


if __name__ == "__main__":
    main()
