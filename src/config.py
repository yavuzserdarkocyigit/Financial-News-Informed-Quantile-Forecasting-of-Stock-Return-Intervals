"""Default paths and experiment constants shared by every pipeline step.

All paths can be overridden on the command line of each script. By default the raw inputs live under
`data/` and everything the pipeline produces goes to `artifacts/`, both at the repository root.

    data/prices/{ticker}.csv          daily OHLCV from Yahoo Finance (gspc.csv = S&P 500 index)
    data/vix.parquet                  daily VIX from FRED (columns Date, VIX_Close, VIX_Norm)
    data/benzinga/{TICKER}/*.jsonl    raw Benzinga articles (licensed; not redistributed)
    data/finbert.sqlite               article-level FinBERT scores (see src/data/finbert_panel.py)
    data/tickers.txt                  one ticker per line
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
ARTIFACTS = os.path.join(REPO, "artifacts")

PRICES_DIR = os.path.join(DATA, "prices")
VIX_FILE = os.path.join(DATA, "vix.parquet")
BENZINGA_DIR = os.path.join(DATA, "benzinga")
FINBERT_DB = os.path.join(DATA, "finbert.sqlite")
TICKERS_FILE = os.path.join(DATA, "tickers.txt")

NEWS_DIR = os.path.join(ARTIFACTS, "news")          # {ticker}_news.parquet, one row per session
DATASET = os.path.join(ARTIFACTS, "dataset.pt")     # pooled windows of all stocks
MODELS_DIR = os.path.join(ARTIFACTS, "models")      # {run}.pt
PREDS_DIR = os.path.join(ARTIFACTS, "preds")        # {ticker}_{run}.parquet, 7 quantiles per session
RESULTS_DIR = os.path.join(ARTIFACTS, "results")

# chronological splits
START, TRAIN_END, VAL_END, TEST_END = "2010-01-01", "2017-12-31", "2019-12-31", "2025-12-31"
WINDOW = 20                                         # input sessions per forecast
SEEDS = [7, 42, 123]

QUANTILES = [0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99]
QCOLS = ["q01", "q025", "q05", "q50", "q95", "q975", "q99"]
LEVELS = [("p90", "q05", "q95", 0.90), ("p95", "q025", "q975", 0.95), ("p98", "q01", "q99", 0.98)]


def run_name(arm, seed):
    """Model / prediction name, e.g. run_name('news', 7) -> 'news_s7'."""
    return f"{arm}_s{seed}"


def pred_file(preds_dir, ticker, run):
    return os.path.join(preds_dir, f"{ticker.lower()}_{run}.parquet")


def news_file(news_dir, ticker):
    return os.path.join(news_dir, f"{ticker.lower()}_news.parquet")


def read_tickers(path):
    with open(path, encoding="utf-8") as f:
        tks = [ln.split("#")[0].strip().upper() for ln in f if ln.split("#")[0].strip()]
    return list(dict.fromkeys(tks))
