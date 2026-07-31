# Nifty50 Trade Decision Support: News-Sentiment to TFT Forecasting Pipeline

## What This Project Does

This repository builds an end-to-end forecasting workflow for Indian equities:

1. Scrape market news + stock data.
2. Clean and normalize articles.
3. Split article impact into `direct`, `sectoral`, and `global`.
4. Score sentiment with FinBERT.
5. Merge sentiment with OHLCV and engineer features.
6. Train/evaluate TFT across multiple windows.
7. Generate evaluation visualizations.


---

## Public Dataset and Model Links

- Hugging Face Dataset: `https://huggingface.co/datasets/mveen3/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker`
- Hugging Face Model: `https://huggingface.co/mveen3/TFT_Model_For_Stock_Price_Prediction`

---

## Repository Structure (Current View)

```text
root/
|-- src/
|   |-- 0_preprocess_news.py
|   |-- 1_news_segregation.py
|   |-- 2_finbert_sentiment.py
|   |-- 3_feature_engineering.py
|   |-- 4_tft_hpt_train_test.py
|   |-- 5_tft_visualize.py
|   `-- scrapers/
|       |-- businessstandard_scraper.py
|       |-- economictimes_scraper.py
|       |-- financialexpress_scraper.py
|       |-- moneycontrol_scraper.py
|       `-- nifty_yfinance_scraper.py
|-- dataset/
|   |-- raw_dataset/
|   |-- stock_dataset/
|   |-- news_segregation_checkpoints/
|   |-- processed_news_dataset.csv
|   |-- tier_segregated_news.csv
|   |-- news_sentiment.csv
|   `-- tft_ready.csv
|-- artifacts/
|   |-- tft/
|   |-- tft_tune/
|   `-- visualizations/
`-- environment.yml
```

---

## Pipeline Flow (Stage by Stage)

### Stage 0: Data Collection

- News scraping from Moneycontrol, Financial Express, Economic Times, Business Standard.
- Ticker OHLCV scraping from Yahoo Finance for Nifty constituents.

### Stage 1: Preprocessing

- `src/0_preprocess_news.py`
- Merges `*_raw.csv`, cleans text, validates dates, deduplicates by URL, applies length filtering.
- Output: `dataset/processed_news_dataset.csv`.

### Stage 2: News Segregation

Pipeline: `src/1_news_segregation.py`. Rule-based, no model and no network.

Each article is assigned to three channels against the 50 constituents:

- **direct** — a company named in the headline. Headline anchoring is deliberate:
  a company named only in the body is usually a passing mention in a market
  round-up, and accepting those drops ticker precision from 0.89 to 0.68.
- **sectoral** — sector keywords, plus the sector of any company named anywhere,
  so round-ups still contribute sector signal.
- **global** — macro and geopolitical keywords, anchored on the headline when the
  article is about a company.

Two mechanisms do the heavy lifting for reliability. Company names are matched
longest-first against the aliases *and* against a list of decoy names, so
`Reliance Power` (Anil Ambani) never resolves to Reliance Industries, and
`Tata Power`, `SBI Cards`, `ICICI Lombard`, `Adani Green`, `JSW Energy`,
`L&T Finance` and `Apollo Tyres` all resolve to nothing. Aliases that are also
ordinary English words (`reliance`, `titan`, `eternal`, `trent`, `indigo`) must
additionally be capitalised in the source text and sit beside a corporate cue.

Stored text is a keyword-centred window: the headline, the sentence carrying the
matched term, and two sentences either side.

`python src/1_news_segregation.py` then option `4` runs a self-test over the
constituent table and the ambiguous-name traps.

Output: `dataset/tier_segregated_news.csv`.

### Stage 3: Sentiment Scoring

- `src/2_finbert_sentiment.py`
- FinBERT sentiment over all three news channels, with unique-text dedup and sliding windows for long texts.
- Output: `dataset/news_sentiment.csv`.

### Stage 4: Feature Engineering

- `src/3_feature_engineering.py`
- Merges sentiment + ticker data, adds technical features (RSI, MACD, volatility, MA ratios), builds target and `time_idx`.
- Output: `dataset/tft_ready.csv`.

Two properties matter downstream:

- **Leakage-safe scaling.** `StandardScaler` is fit per symbol on rows before
  `2025-01-01` only, then applied to the whole series. Validation and test rows
  never influence the fitted mean/variance.
- **`adj_close_inr` is left unscaled.** Every other price column is standardized,
  so reconstructing a predicted price from them would produce z-score units
  rather than rupees. This column is the base price the evaluation stage uses to
  express predictions in real money.

### Stage 5: Modeling (TFT)

- `src/4_tft_hpt_train_test.py`
- Unified menu for tune/train/test/inference.
- Artifacts written under `artifacts/tft_tune` and `artifacts/tft`.

### Stage 6: Visualization

- `src/5_tft_visualize.py`
- Recomputes all metrics from `test_predictions.csv` instead of reading
  `metrics.csv`, so the plots independently verify the training stage's numbers.
- Picks the best window by recomputed MAPE (currently W7) and generates:
  - `best_window_actual_vs_predicted.png` — all 49 symbols, actual vs predicted
  - `per_window_metrics.png` — the six metrics across W7/W10/W15/W30
  - `best_window_stock_mape.png` — per-stock MAPE, ranked ascending
  - `best_window_lowest_4_mape_actual_vs_predicted.png`
  - `best_window_middle_4_mape_actual_vs_predicted.png`
  - `best_window_highest_4_mape_actual_vs_predicted.png`
- Prices come from the raw ticker file when present, otherwise from the rupee
  prices recorded in the prediction audit.

---

## Dataset Snapshot (Current Files)

Observed local row counts (excluding header):

- `dataset/raw_dataset/businessstandard_raw.csv`: 49,046
- `dataset/raw_dataset/economictimes_raw.csv`: 90,417
- `dataset/raw_dataset/financialexpress_raw.csv`: 156,214
- `dataset/raw_dataset/moneycontrol_raw.csv`: 15,855
- `dataset/processed_news_dataset.csv`: 309,041
- `dataset/tier_segregated_news.csv`: 114,100
- `dataset/news_sentiment.csv`: 114,100
- `dataset/stock_dataset/nifty50_ticker.csv`: 74,438
- `dataset/tft_ready.csv`: 71,938

The segregated table is exactly 2,282 trading dates x 50 constituents. Every
date/ticker pair carries a global cell, so the row count is fixed by the
calendar rather than by how many articles happened to mention a company.

Schema progression:

- Raw: `date,title,news,url`
- Segregated: `date,symbol,direct_news,sectoral_news,global_news`
- Sentiment: 9 sentiment probabilities + 3 count columns
- TFT-ready: market features + sentiment features + `target_pct_change` + `time_idx` + `adj_close_inr`

### Splits

| Split | Date range | Purpose |
| --- | --- | --- |
| Train | ... to 2024-12-31 | model fitting, scaler fitting |
| Validation | 2025-01-01 to 2025-06-30 | early stopping, Optuna objective |
| Test | 2025-07-01 onward | reported metrics only |

The split is strictly chronological, so no future bar ever informs a past
prediction.

---

## Environment and Dependency Installation (Conflict-Safe)

All commands must run in conda env `MarketFusion`.

```bash
conda activate MarketFusion
```

### Why these pinned versions

TFT reliability here depends on version alignment among:

- `torch`
- `lightning` / `pytorch-lightning`
- `pytorch-forecasting`
- `optuna` + `optuna-integration`
- `numpy` / `pandas` / `scikit-learn`

`pytorch-forecasting==1.1.1` is sensitive to major-version drift, so this README keeps a locked compatibility set.

### Installation

The full pinned stack lives in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate MarketFusion

# Dependency conflict check
python -m pip check

# Quick compatibility check
python -c "import torch, lightning, pytorch_lightning, pytorch_forecasting, optuna, optuna_integration, numpy, pandas, sklearn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available()); print('lightning', lightning.__version__, 'pl', pytorch_lightning.__version__, 'pf', pytorch_forecasting.__version__)"
```

Without an NVIDIA GPU, swap `pytorch-cuda=11.8` for `cpuonly` in
`environment.yml` — every stage detects the device and falls back to CPU.

`accelerate` is a hard requirement of stage 3, not an optional extra: FinBERT is
loaded with `from_pretrained(low_cpu_mem_usage=True)`, which raises `ImportError`
without it.

---

## How to Run (End-to-End Commands)

```bash
conda activate MarketFusion
```

### 0) Optional: refresh source datasets

```bash
python src/scrapers/moneycontrol_scraper.py
python src/scrapers/financialexpress_scraper.py
python src/scrapers/economictimes_scraper.py
python src/scrapers/businessstandard_scraper.py
python src/scrapers/nifty_yfinance_scraper.py --index nifty50 --start 2020-01-01 --end 2026-03-31
```

### 1) Preprocess merged news

```bash
python src/0_preprocess_news.py
```

### 2) Segregate news into channels

No server and no model to start; this stage is pure CPU and runs in roughly
eight minutes over the full 309k-article corpus.

```bash
python src/1_news_segregation.py
```

Menu:
- `1` Segregate only
- `2` CSV only
- `3` Full pipeline

### 3) Run FinBERT sentiment

```bash
python src/2_finbert_sentiment.py
```

### 4) Build TFT-ready features

```bash
python src/3_feature_engineering.py
```

### 5) TFT run (menu-based)

```bash
python src/4_tft_hpt_train_test.py --data-path dataset/tft_ready.csv --windows 7,10,15,30
```

Menu:
- `0` tune + train + test
- `1` train + test
- `2` inference only
- `3` exit

### 6) Generate plots

```bash
python src/5_tft_visualize.py \
  --artifact-root artifacts/tft \
  --ticker-path dataset/stock_dataset/nifty50_ticker.csv \
  --windows 7,10,15,30 \
  --output-dir artifacts/visualizations \
  --dpi 300
```

---

## What Each Main File Does

### Core pipeline

- `src/0_preprocess_news.py`: merges and cleans raw article files.
- `src/1_news_segregation.py`: rule-based direct/sectoral/global segregation against the 50 constituents, with checkpointed aggregation.
- `src/2_finbert_sentiment.py`: FinBERT sentiment feature generation.
- `src/3_feature_engineering.py`: feature engineering and final training table creation.
- `src/4_tft_hpt_train_test.py`: unified TFT menu runner for tune/train/test/inference.
- `src/5_tft_visualize.py`: final metric and chart generation.

### Scrapers

- `src/scrapers/moneycontrol_scraper.py`: async scraper with TLS impersonation.
- `src/scrapers/financialexpress_scraper.py`: async + process-pool scraper with checkpointing.
- `src/scrapers/economictimes_scraper.py`: sitemap-based historical scraper.
- `src/scrapers/businessstandard_scraper.py`: hybrid `curl_cffi` + Selenium scraper.
- `src/scrapers/nifty_yfinance_scraper.py`: index constituent + price downloader.

---

## Results

From `artifacts/tft/metrics_summary.csv` (test period 2025-07-01 onward, 9,061
predictions across 49 eligible symbols):

| Window | MAPE (price, %) | MAE (return) | Directional Acc. | Precision (up) | Recall (up) | F1 (up) |
| --- | --- | --- | --- | --- | --- | --- |
| W7  | **1.0494** | 0.01048 | 0.5146 | 0.4939 | 0.1752 | 0.2586 |
| W10 | 1.0548 | 0.01053 | **0.5178** | 0.5010 | 0.6409 | 0.5624 |
| W15 | 1.0590 | 0.01057 | 0.4815 | 0.4697 | 0.5672 | 0.5139 |
| W30 | 1.0501 | 0.01047 | 0.4914 | 0.4864 | 0.9292 | 0.6385 |

`src/5_tft_visualize.py` recomputes every one of these from
`test_predictions.csv` rather than reading `metrics.csv`, and reproduces them
exactly — so the table is self-verifying.

Hyperparameters come from `artifacts/tft_tune/`, 30 Optuna trials per window.
Those studies were run against an earlier build of the feature table and were
reused unchanged here rather than repeated, which keeps this a controlled
comparison: the price columns are byte-identical to the previous run, so any
movement in the table above is attributable to the news features alone.

That reuse is also defensible on its own terms, and the evidence is in
[What tuning was worth](#what-tuning-was-worth) below.

### How to read these numbers

Read against a naive baseline, the headline metrics do **not** show forecasting
skill. Stating that plainly is the point of this section — the numbers above look
respectable in isolation and are easy to misread.

Comparison against predicting zero return on every bar (n = 9,061; 48.3% of
actual next-day returns are positive):

| Window | Model MAE | Zero-prediction MAE | Ratio | Pred. std | True std | % predicted "up" |
| --- | --- | --- | --- | --- | --- | --- |
| W7  | 0.01048 | 0.01044 | 1.004 | 0.00097 | 0.01480 | 17.1% |
| W10 | 0.01053 | 0.01044 | 1.008 | 0.00175 | 0.01481 | 61.8% |
| W15 | 0.01057 | 0.01044 | 1.012 | 0.00150 | 0.01481 | 58.3% |
| W30 | 0.01047 | 0.01044 | 1.003 | 0.00043 | 0.01481 | 92.4% |

Four things follow.

**The model has collapsed toward a constant.** Predicted returns have a standard
deviation of 0.00043–0.00175 against a true standard deviation of 0.0148, which
is 8x to 35x too flat. Under quantile loss on a near-unpredictable target the
optimum is to sit near the conditional median, and that is what the network
found. It is not learning to time the market; it is learning that the safest
guess is "almost no move".

**Every window is worse than predicting zero.** The MAE ratios are 1.003–1.012,
all above 1. MAPE near 1.05% therefore measures the volatility of the target, not
the quality of the forecast: prices are reconstructed as
`base_price x (1 + return)` for both series, so the base cancels in the ratio and
MAPE reduces to approximately the mean absolute return error.

**The four windows are not meaningfully distinguishable.** MAPE spans
1.0494–1.0590, a spread of 0.01 percentage points, and the ranking is unstable
both across metrics and across runs: W7 wins on MAPE, W10 on directional
accuracy, and the previous run of this same pipeline had W30 best on MAPE and W7
best on directional accuracy. A genuine effect would not reshuffle like this.
Encoder length does not matter here, because there is little signal for any
length to capture.

**Validation loss carries no information about test performance.** The four
windows finish within 0.000010 of each other on validation loss — a 0.14% spread
— while their test MAPE ordering is unrelated to it. W15 has both the worst
validation loss and the worst test MAPE, but W30 has the second-worst validation
loss and the second-best test MAPE. Selecting on this quantity is selecting on
noise.

The directional metrics deserve one caution: F1 is easy to misread when a model
is degenerate, and this run illustrates it starkly. **W30 posts the best F1 of
the four (0.6385) while having the second-worst directional accuracy (0.4914,
below chance).** It gets there by predicting "up" on 92.4% of bars, which forces
recall to 0.9292 while precision (0.4864) simply reproduces the base rate. At the
other extreme W7 predicts "up" on 17.1% of bars, dragging recall to 0.1752 and F1
to 0.2586 despite the best MAPE. Neither is skill — both are artefacts of where a
collapsed predictor's near-constant output happens to sit relative to zero. F1
alone is never used to pick a window here.

Per-stock MAPE on the best window (W7) spans 0.761%–1.570% with a median of
1.021%, and all 49 symbols land under 2% (`best_window_stock_mape.png`). This
looks like uniform accuracy but is not: since MAPE here reduces to mean absolute
return error, the chart is essentially a ranking of each stock's realised
volatility over the test period. Low-volatility names score "well" for reasons
that have nothing to do with the model.

### What tuning was worth

Two independent 30-trial Optuna studies were run over this pipeline. They
selected substantially different architectures — hidden size 16 against 48 at
W7, learning rates 11x apart at W30 — and landed within 0.5% of each other on
validation loss. A search that returns unrelated answers at equal objective
values is sampling a flat surface, not finding structure.

An accidental control settles it. One run trained W7 and W10 on this exact
dataset with the CLI defaults rather than the tuned values:

| Window | Default HPs | Tuned HPs | Better |
| --- | --- | --- | --- |
| W7  | 0.007143 | 0.007199 | default |
| W10 | 0.007176 | 0.007204 | default |

Untuned defaults beat the tuned hyperparameters on both windows. Taken with the
0.14% spread across all four tuned windows, roughly eight hours of Bayesian
search bought nothing measurable on this problem. That is a property of the
target, not of the search: there is no hyperparameter setting that extracts
signal that is not there.

### Does better news segregation help?

An earlier build of stage 1 resolved company names by raw substring match against
the alias table. That scheme had real defects: every company ending in "Ltd"
resolved to LT and "Biocon Ltd" to IOC, seven index constituents had no entry at
all and could never be credited with news, and twelve entries were not in the
index. The current engine matches longest-first against aliases and decoys,
requires capitalisation and a corporate cue for ordinary-word aliases, and
anchors direct attribution on the headline. Direct-attribution precision went
from 0.68 to 0.89, and all 50 constituents became reachable.

It made no difference to forecasting:

| Window | MAPE before | MAPE after | Change |
| --- | --- | --- | --- |
| W7  | 1.0505 | 1.0494 | -0.0011 |
| W10 | 1.0475 | 1.0548 | +0.0074 |
| W15 | 1.0570 | 1.0590 | +0.0020 |
| W30 | 1.0470 | 1.0501 | +0.0031 |

The price columns are byte-identical between the two runs, so this isolates the
news features. Mean change is +0.003 percentage points, third-decimal movement
well inside run-to-run variation, and every window remains worse than predicting
zero.

This is the most useful negative result in the project. The earlier version could
be dismissed on the grounds that its features were corrupted; this one cannot.
The news channel does not carry next-day cross-sectional signal that this
architecture can use, and cleaning it up does not change that.

### What this project does demonstrate

The engineering result stands on its own terms, independent of the modelling
outcome: an end-to-end, reproducible, leakage-free pipeline that takes 309k raw
scraped articles to a tuned multi-series forecaster — deterministic entity and
sector segregation, transformer sentiment scoring, chronologically-safe feature
construction, per-window Bayesian hyperparameter search with resumable studies,
and evaluation independently recomputed from the prediction audit.

Specific properties worth pointing at:

- **No leakage by construction.** Scalers are fit only on pre-2025 rows; splits
  are strictly chronological; the price used to reconstruct a prediction is the
  last observation *before* the decoder starts.
- **Every reported number is reproducible.** Each window writes a
  `run_manifest.json` recording git revision, library versions, dataset
  fingerprint, precision, batch size, seed and the full hyperparameter set —
  including the optimizer, which pytorch-forecasting otherwise resolves
  differently depending on what happens to be installed.
- **The evaluation is independently recomputed.** Stage 6 rebuilds prices and
  metrics from the raw ticker file and reproduces `metrics_summary.csv` exactly.
- **Predictions are auditable in rupees.** `test_predictions.csv` carries the
  base price, predicted and actual price per symbol per day.

- **Segregation is deterministic and self-testing.** Stage 1 uses no model and no
  network, so the same corpus always yields the same channels. `option 4` runs 26
  assertions over the constituent table and the ambiguous-name traps.

A negative result on next-day equity direction from public news is the expected
outcome and is reported as such. The interesting content here is the diagnosis —
predictions 8–35x too flat, no window beating a zero-return baseline, validation
loss carrying no information about test performance, and a large improvement in
feature quality moving nothing — not the headline MAPE.

### Limitations

- Sentiment is scored per article and averaged per day; article importance,
  publication time within the trading day, and source reliability are ignored.
- Sectoral news is fanned out to every constituent of a sector uniformly, so a
  sector-level signal is diluted across symbols with very different exposures.
- Global news is broadcast identically to every symbol on a given date, which
  gives the model no cross-sectional information from that channel. It is also
  capped at `MAX_GLOBAL_SNIPPETS_PER_DATE` (30) snippets per date, because that
  cell is duplicated across all 50 tickers and otherwise dominates output size.
- Direct attribution is anchored on the headline, which trades recall for
  precision: a company named only in the body contributes to its sector rather
  than to itself. This roughly halves how often the direct channel is populated
  compared with accepting any mention, and the direct channel is the only one
  carrying genuine cross-sectional information.
- Sector membership is a fixed present-day mapping applied to the whole history,
  so index changes over the 2020–2026 window are not tracked.
- 49 of the 50 constituents are evaluated. `TMPV.NS` is dropped by the
  eligibility filter because it has no pre-2025 history (post-demerger listing,
  62 rows total), so it cannot supply a training window.
- Transaction costs, slippage and liquidity are not modelled, so directional
  accuracy above 0.5 would not by itself imply a tradeable strategy.
- A single chronological test split is used; results are not averaged over
  multiple folds or market regimes.
- Quantile loss on a near-unpredictable target drives the model toward the
  conditional median. A loss that penalises under-dispersion, or a reformulation
  as direction classification rather than return regression, would be the natural
  next experiment.


