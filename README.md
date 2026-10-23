# Financial News-Informed Quantile Forecasting of Stock Return Intervals

Code for the paper *Financial News-Informed Quantile Forecasting of Stock Return Intervals*
(Y. S. Koçyiğit, İ. Karadeniz, O. T. Yıldız).

A two-stem quantile Transformer combines 20 sessions of market features with a 24-feature session-level
representation of firm-specific Benzinga news and predicts seven ordered quantiles of the next session's
close-to-close log return. Intervals are corrected with a stock-specific, past-only conformal step. The
news-aware model is compared with an otherwise identical model trained without news on 456 U.S. stocks
and SPY, with out-of-sample evaluation over 2020-2025.

## Repository layout

```
src/
  config.py              paths, splits, quantile levels
  data_collection/       prices (Yahoo Finance), VIX (FRED), Benzinga articles, FinBERT scores
  news_features.py       24 session-level news features per stock
  dataset.py             pooled 20-session windows, per-stock standardization, splits
  model.py               two-stem quantile Transformer and pinball loss
  train.py               training and per-stock quantile predictions
  calibration.py         past-only conformal calibration, Winkler score
  evaluate.py            coverage, width and Winkler per stock               (Table V)
  paired_stats.py        paired news vs no-news comparison                   (Table VI, Tables S5-S6)
  widths.py              width reallocation                                  (Table VII)
  ablation.py            feature-group ablations                             (Table VIII, Table S7)
  placebo.py             content-placebo news features                       (Tables S8-S9)
  var_backtest.py        Value-at-Risk breach rates                          (Fig. 4)
  llm_baseline.py        zero-shot LLM interval forecasts                    (Table IX)
  figures.py             paper figures and band asymmetry table              (Figs. 2-4, Table S8)
  data_summary.py        sample composition and corpus size                  (Table IV)
data/                    inputs (not distributed, see data/README.md)
artifacts/               everything the pipeline writes
```

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended for training, FinBERT scoring and the LLM baseline. Training on a GPU is not
bit-reproducible (cuDNN and scatter kernels are non-deterministic); results agree to the reported precision.

## Data

Prices and VIX are public. The Benzinga articles are licensed, so neither the articles nor anything derived
from their text (FinBERT scores, news features, datasets, predictions) is included; reproducing the news
features requires your own Benzinga API access. See [data/README.md](data/README.md).

```bash
python src/data_collection/prices.py
python src/data_collection/vix.py
export BENZINGA_API_URL=... BENZINGA_API_KEY=... # your Benzinga news API endpoint and key
python src/data_collection/benzinga.py
python src/data_collection/finbert_scores.py
```

## Reproducing the paper

```bash
# features, dataset, models (news-aware and no-news, three seeds)
python src/news_features.py
python src/dataset.py
for s in 7 42 123; do
  python src/train.py --seed $s
  python src/train.py --seed $s --no-news
done

# results
python src/data_summary.py                       # Table IV
python src/evaluate.py                           # Table V
python src/paired_stats.py                       # Table VI, annual effects, pinball comparison
python src/widths.py                             # Table VII
python src/var_backtest.py                       # data of Fig. 4

# feature-group ablations (Table VIII, Table S7): print and run the dataset/training commands, then summarize
python src/ablation.py --commands
python src/ablation.py

# content placebo (Tables S8-S9)
python src/placebo.py
python src/dataset.py --news-dir artifacts/news_placebo --out artifacts/dataset_placebo.pt
for s in 7 42 123; do
  python src/train.py --data artifacts/dataset_placebo.pt --arm placebo --seed $s
done
python src/paired_stats.py --arms placebo,nonews --stages cal
python src/paired_stats.py --arms news,placebo --stages cal

python src/figures.py                            # Figs. 2-4 and Table S8

# zero-shot LLM baselines (Table IX)
python src/llm_baseline.py --model Qwen/Qwen3-1.7B
python src/llm_baseline.py --model google/gemma-3-1b-it
python src/llm_baseline.py --model meta-llama/Llama-3.2-1B-Instruct --accept-sorted
```

Final model: d = 96, 3 encoder layers, 4 heads, dropout 0.10, 363,527 parameters (the defaults of
`train.py`). Splits: training 2010-2017, validation 2018-2019, test 2020-2025.

## License

The code is released under the [MIT License](LICENSE). The license does not cover the Benzinga articles
or anything derived from them, which remain subject to Benzinga's terms.
