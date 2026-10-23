# Data

The raw inputs are not part of this repository. Price and VIX data are public and are downloaded by the
scripts in `src/data_collection/`. The Benzinga articles are licensed and cannot be redistributed; they
require your own access to the Benzinga news API.

| Path | Content | Created by |
|---|---|---|
| `tickers.txt` | the 457 panel members, in the order that fixes the stock ids | included |
| `llm_tickers.txt` | the 60 stocks of the LLM comparison | included |
| `prices/{ticker}.csv`, `prices/gspc.csv` | daily OHLCV from Yahoo Finance (S&P 500 index as `gspc.csv`) | `data_collection/prices.py` |
| `vix.parquet` | daily VIX from FRED (`Date`, `VIX_Close`, `VIX_Return`, `VIX_Norm`) | `data_collection/vix.py` |
| `benzinga/{TICKER}/{TICKER}_{year}.jsonl` | raw articles, one JSON object per line | `data_collection/benzinga.py` |
| `finbert.sqlite` | article-level FinBERT scores and title embeddings | `data_collection/finbert_scores.py` |

Fields used from each Benzinga article: `benzinga_id`, `published` (UTC timestamp), `title`, `body` (HTML),
`tickers`, `channels`, `author`, `tags`.
