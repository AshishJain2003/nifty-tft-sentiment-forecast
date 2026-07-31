#!/usr/bin/env python3
"""
Build the flat master training table for the TFT stage.

- Engineers ticker features and the next-day return target.
- Rolls non-trading-day news back to the previous market trading day.
- Merges news + ticker, creates time_idx, drops NaNs from warm-up/cool-down.
- Applies per-symbol scaling fit only on rows before TRAIN_CUTOFF_EXCLUSIVE.
- Preserves one unscaled price column (adj_close_inr) so the evaluation stage
  can reconstruct predictions in rupees.

Output: dataset/tft_ready.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "Missing dependency `scikit-learn`. Install inside conda env `MarketFusion` "
        "(priority: conda install scikit-learn; fallback: conda-forge; last: pip)."
    ) from exc

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
except ImportError as exc:
    raise ImportError(
        "Missing dependency `ta`. Install inside conda env `MarketFusion` "
        "(priority: conda install -c conda-forge ta; last resort: pip install ta)."
    ) from exc


NEWS_PATH = Path("dataset/news_sentiment.csv")
TICKER_PRIMARY_PATH = Path("dataset/stock_dataset/nifty50_ticker.csv")
OUTPUT_PATH = Path("dataset/tft_ready.csv")

TRAIN_CUTOFF_EXCLUSIVE = pd.Timestamp("2025-01-01")  # fit scalers on rows before this date
MA_WINDOWS = (5, 10, 20, 50)

# Unscaled copy of `Adj Close`, kept out of the standardization step so the
# evaluation stage has a genuine INR base price to reconstruct from.
PRICE_INR_COL = "adj_close_inr"

USELESS_TICKER_COLS = ["Company", "Index", "Dividends", "Stock Splits"]
OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

COUNT_COLS = ["direct_news_count", "sectoral_news_count", "global_news_count"]
SENTIMENT_COLS = [
    "direct_news_pos",
    "direct_news_neu",
    "direct_news_neg",
    "sectoral_news_pos",
    "sectoral_news_neu",
    "sectoral_news_neg",
    "global_news_pos",
    "global_news_neu",
    "global_news_neg",
]
SENTIMENT_TO_COUNT = {
    "direct_news_pos": "direct_news_count",
    "direct_news_neu": "direct_news_count",
    "direct_news_neg": "direct_news_count",
    "sectoral_news_pos": "sectoral_news_count",
    "sectoral_news_neu": "sectoral_news_count",
    "sectoral_news_neg": "sectoral_news_count",
    "global_news_pos": "global_news_count",
    "global_news_neu": "global_news_count",
    "global_news_neg": "global_news_count",
}

REQUIRED_NEWS_COLS = ["Date", "Symbol"] + SENTIMENT_COLS + COUNT_COLS
REQUIRED_TICKER_COLS = [
    "Date",
    "Symbol",
    "Company",
    "Index",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]


def resolve_ticker_path() -> Path:
    if TICKER_PRIMARY_PATH.exists():
        return TICKER_PRIMARY_PATH
    raise FileNotFoundError(
        f"Ticker file not found. Checked: {TICKER_PRIMARY_PATH}"
    )


def parse_dates(news_df: pd.DataFrame, ticker_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    news_df = news_df.copy()
    ticker_df = ticker_df.copy()

    news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce").dt.normalize()
    # Parse timezone-aware ticker timestamps and normalize to local calendar date.
    ticker_df["Date"] = (
        pd.to_datetime(ticker_df["Date"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if news_df["Date"].isna().any():
        raise ValueError("Found invalid Date values in news_sentiment.csv.")
    if ticker_df["Date"].isna().any():
        raise ValueError("Found invalid Date values in nifty50_ticker.csv.")

    return news_df, ticker_df


def validate_required_columns(df: pd.DataFrame, required_cols: Sequence[str], dataset_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def add_ticker_features(ticker_df: pd.DataFrame, ma_windows: Sequence[int]) -> pd.DataFrame:
    df = ticker_df.copy()
    df = df.sort_values(["Symbol", "Date"], kind="mergesort").reset_index(drop=True)

    # Safety fill for occasional missing OHLCV values (per symbol, chronological).
    df[OHLCV_COLS] = df.groupby("Symbol", sort=False)[OHLCV_COLS].ffill()

    cols_to_drop = [c for c in USELESS_TICKER_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    g = df.groupby("Symbol", sort=False)

    prev_close = g["Close"].shift(1)
    df["target_pct_change"] = g["Adj Close"].shift(-1).div(df["Adj Close"]).sub(1.0)
    df["adj_ret_1d"] = g["Adj Close"].pct_change()

    df["price_range_pct"] = df["High"].sub(df["Low"]).div(df["Close"])
    df["gap_pct"] = df["Open"].sub(prev_close).div(prev_close)
    df["rolling_volatility_5d"] = g["adj_ret_1d"].transform(
        lambda s: s.rolling(5, min_periods=5).std(ddof=0)
    )

    for window in ma_windows:
        price_sma = g["Adj Close"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        vol_sma = g["Volume"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        df[f"adjclose_sma_ratio_{window}"] = df["Adj Close"].div(price_sma).sub(1.0)
        df[f"volume_momentum_{window}"] = df["Volume"].div(vol_sma)

    # Compute TA indicators per symbol without groupby.apply (avoids pandas deprecation warning).
    rsi_14 = np.full(df.shape[0], np.nan, dtype=np.float64)
    macd_line = np.full(df.shape[0], np.nan, dtype=np.float64)
    macd_signal = np.full(df.shape[0], np.nan, dtype=np.float64)
    macd_diff = np.full(df.shape[0], np.nan, dtype=np.float64)
    close_series = df["Adj Close"]
    for idx in df.groupby("Symbol", sort=False).groups.values():
        idx_arr = np.asarray(idx, dtype=np.int64)
        close = close_series.iloc[idx_arr]
        macd_obj = MACD(
            close=close, window_fast=12, window_slow=26, window_sign=9, fillna=False
        )
        rsi_14[idx_arr] = RSIIndicator(close=close, window=14, fillna=False).rsi().to_numpy(
            dtype=np.float64
        )
        macd_line[idx_arr] = macd_obj.macd().to_numpy(dtype=np.float64)
        macd_signal[idx_arr] = macd_obj.macd_signal().to_numpy(dtype=np.float64)
        macd_diff[idx_arr] = macd_obj.macd_diff().to_numpy(dtype=np.float64)

    df["rsi_14"] = rsi_14
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_diff"] = macd_diff

    df["symbol_base"] = df["Symbol"].str.replace(r"\.NS$", "", regex=True)
    return df


def aggregate_news_to_prev_trading_day(
    news_df: pd.DataFrame, trading_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    if trading_dates.empty:
        raise ValueError("Trading calendar is empty; cannot roll news dates.")

    df = news_df.copy()
    trading_values = trading_dates.values.astype("datetime64[ns]")
    news_values = df["Date"].values.astype("datetime64[ns]")

    # Map each news date -> previous available market trading date.
    map_idx = np.searchsorted(trading_values, news_values, side="right") - 1
    valid = map_idx >= 0
    if not np.all(valid):
        df = df.loc[valid].copy()
        map_idx = map_idx[valid]

    df["Date"] = pd.to_datetime(trading_values[map_idx])

    weighted_cols: Dict[str, str] = {}
    for sent_col in SENTIMENT_COLS:
        count_col = SENTIMENT_TO_COUNT[sent_col]
        weighted_col = f"__weighted__{sent_col}"
        df[weighted_col] = df[sent_col].astype(np.float64) * df[count_col].astype(np.float64)
        weighted_cols[sent_col] = weighted_col

    agg_sum_cols = COUNT_COLS + list(weighted_cols.values())
    agg = df.groupby(["Date", "Symbol"], as_index=False, sort=False)[agg_sum_cols].sum()

    # Weighted average for ALL 9 sentiment columns with their respective count columns.
    for sent_col in SENTIMENT_COLS:
        count_col = SENTIMENT_TO_COUNT[sent_col]
        numerator = agg[weighted_cols[sent_col]].to_numpy(dtype=np.float64)
        denominator = agg[count_col].to_numpy(dtype=np.float64)
        result = np.zeros_like(numerator, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator > 0.0)
        agg[sent_col] = result

    agg = agg.drop(columns=list(weighted_cols.values()))
    ordered_cols = ["Date", "Symbol"] + SENTIMENT_COLS + COUNT_COLS
    return agg[ordered_cols]


def add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    unique_dates = pd.Index(np.sort(out["Date"].unique()))
    date_to_idx = pd.Series(np.arange(unique_dates.size, dtype=np.int32), index=unique_dates)
    out["time_idx"] = out["Date"].map(date_to_idx).astype(np.int32)
    return out


def scale_ticker_numeric_features(
    df: pd.DataFrame, train_cutoff_exclusive: pd.Timestamp
) -> pd.DataFrame:
    out = df.copy()

    exclude = set(
        SENTIMENT_COLS + COUNT_COLS + ["target_pct_change", "time_idx", PRICE_INR_COL]
    )
    numeric_cols = [
        col
        for col in out.columns
        if pd.api.types.is_numeric_dtype(out[col]) and col not in exclude
    ]
    # Ensure scaled columns are float before assignment to avoid dtype incompatibility warnings.
    out[numeric_cols] = out[numeric_cols].astype(np.float64)

    grouped_indices = out.groupby("Symbol", sort=False).groups
    for symbol, idx in grouped_indices.items():
        sym_idx = pd.Index(idx)
        sym_dates = out.loc[sym_idx, "Date"]
        fit_idx = sym_idx[sym_dates < train_cutoff_exclusive]
        if fit_idx.empty:
            fit_idx = sym_idx
            print(
                f"[WARN] No pre-2025 rows for {symbol}; fitting scaler on available rows "
                f"({len(sym_idx)} rows)."
            )

        scaler = StandardScaler()
        x_fit = out.loc[fit_idx, numeric_cols].to_numpy(dtype=np.float64)
        x_all = out.loc[sym_idx, numeric_cols].to_numpy(dtype=np.float64)

        scaler.fit(x_fit)
        out.loc[sym_idx, numeric_cols] = scaler.transform(x_all)

    return out


def downcast_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64", "float32", "float16"]).columns
    int_cols = out.select_dtypes(
        include=["int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"]
    ).columns

    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].astype(np.float32)
    if len(int_cols) > 0:
        out[int_cols] = out[int_cols].astype(np.int32)
    return out


def run_pipeline() -> pd.DataFrame:
    ticker_path = resolve_ticker_path()
    print(f"[INFO] Loading news: {NEWS_PATH}")
    news_df = pd.read_csv(NEWS_PATH)
    print(f"[INFO] Loading ticker: {ticker_path}")
    ticker_df = pd.read_csv(ticker_path)

    validate_required_columns(news_df, REQUIRED_NEWS_COLS, "news_sentiment.csv")
    validate_required_columns(ticker_df, REQUIRED_TICKER_COLS, "nifty50_ticker.csv")

    news_df, ticker_df = parse_dates(news_df, ticker_df)
    ticker_df = add_ticker_features(ticker_df, MA_WINDOWS)

    trading_dates = pd.DatetimeIndex(np.sort(ticker_df["Date"].unique()))
    news_agg = aggregate_news_to_prev_trading_day(news_df, trading_dates)
    news_agg = news_agg.rename(columns={"Symbol": "symbol_base"})

    merged = ticker_df.merge(
        news_agg,
        on=["Date", "symbol_base"],
        how="left",
        validate="m:1",
    )

    merged[SENTIMENT_COLS + COUNT_COLS] = merged[SENTIMENT_COLS + COUNT_COLS].fillna(0.0)

    before_drop = len(merged)
    # Training dataset rule: drop warm-up rows from rolling indicators and final unlabeled T+1 rows.
    merged = merged.dropna(axis=0, how="any").reset_index(drop=True)
    print(f"[INFO] Dropped {before_drop - len(merged)} rows with NaN (warm-up/cool-down and invalids).")
    merged = add_time_idx(merged)

    # Snapshot the real rupee price before standardization consumes `Adj Close`.
    merged[PRICE_INR_COL] = merged["Adj Close"].astype(np.float64)

    merged = scale_ticker_numeric_features(merged, TRAIN_CUTOFF_EXCLUSIVE)
    merged = downcast_numeric_dtypes(merged)

    # Contract checks.
    if merged["time_idx"].min() != 0:
        raise ValueError("time_idx must start at 0.")
    if merged["time_idx"].max() != merged["time_idx"].nunique() - 1:
        raise ValueError("time_idx must be continuous with no gaps.")
    if merged.isna().any().any():
        raise ValueError("Final dataset still contains NaN values.")
    if (merged[PRICE_INR_COL] <= 0).any():
        raise ValueError(
            f"{PRICE_INR_COL} must stay positive — it is the unscaled base price "
            "used for price-space evaluation."
        )

    return merged


def main() -> None:
    final_df = run_pipeline()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Wrote master dataset to: {OUTPUT_PATH}")
    print(f"[INFO] Shape: {final_df.shape}")


if __name__ == "__main__":
    main()
