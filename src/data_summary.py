"""Sample composition (Table IV) and the size of the filtered news corpus.

Table IV: per split, the number of dates and stock-session observations of the dataset and the share of
observations with eligible pre-open news (has_news) and with event news (ev_hi > 0). Corpus: unique
articles of the panel stocks that are not templated, carry at most three tickers and were published in
2010-2025, and the share of them without a usable body.

Usage:
    python src/data_summary.py
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

import config as C
from news_features import MAX_TICKERS, _as_list, _templated

TAG_RE = re.compile(r"<[^>]+>")


def table_ii(blob, news_dir):
    tk = np.asarray(blob["tickers"])[blob["stock_id"].numpy()]
    d = pd.DataFrame({"ticker": tk, "Date": pd.to_datetime(blob["dates"]), "split": blob["split"]})
    flags = []
    for t in blob["tickers"]:
        nw = pd.read_parquet(C.news_file(news_dir, t), columns=["Date", "has_news", "ev_hi"])
        flags.append(nw.assign(ticker=t, Date=pd.to_datetime(nw["Date"])))
    d = d.merge(pd.concat(flags), on=["ticker", "Date"], how="left").fillna({"has_news": 0, "ev_hi": 0})
    return pd.DataFrame([{"split": s, "dates": g["Date"].nunique(), "observations": len(g),
                          "with_news%": 100 * (g["has_news"] > 0).mean(), "with_event%": 100 * (g["ev_hi"] > 0).mean()}
                         for s, g in d.groupby("split", sort=False)])


def corpus(tickers, benzinga_dir):
    ids, no_body = set(), set()
    for tk in tickers:
        for fp in sorted(glob.glob(os.path.join(benzinga_dir, tk, f"{tk}_*.jsonl"))):
            with open(fp, encoding="utf-8") as f:
                for ln in f:
                    try:
                        a = json.loads(ln)
                        bid = int(a.get("benzinga_id"))
                    except (ValueError, TypeError):
                        continue
                    pub = str(a.get("published") or "")
                    if (not str(a.get("title") or "").strip() or not pub or _templated(a)
                            or len(_as_list(a.get("tickers"))) > MAX_TICKERS or not "2010" <= pub[:4] <= "2025"):
                        continue
                    ids.add(bid)
                    if not TAG_RE.sub(" ", str(a.get("body") or "")).strip():
                        no_body.add(bid)
    return len(ids), 100 * len(no_body & ids) / max(len(ids), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=C.DATASET)
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--benzinga-dir", default=C.BENZINGA_DIR)
    args = ap.parse_args()
    blob = torch.load(args.data, weights_only=False)
    t = table_ii(blob, args.news_dir)
    print(f"\n[summary] {len(blob['tickers'])} stocks")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    n, nb = corpus(blob["tickers"], args.benzinga_dir)
    print(f"\n[summary] filtered corpus: {n:,} unique articles, {nb:.1f}% without a usable body")


if __name__ == "__main__":
    main()
