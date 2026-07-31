#!/usr/bin/env python3
"""
Build TFT evaluation visualizations from the per-window test predictions.

Metrics are recomputed here from `test_predictions.csv` rather than read from
`metrics.csv`, so the plots independently verify the numbers the training stage
reported.

Prices are rebuilt from the raw ticker file when it is available; otherwise the
rupee prices already recorded in the prediction audit are used.

    conda activate MarketFusion
    python src/5_tft_visualize.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_ARTIFACT_ROOT = Path("artifacts/tft")
DEFAULT_TICKER_PATH = Path("dataset/stock_dataset/nifty50_ticker.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/visualizations")
DEFAULT_WINDOWS = (7, 10, 15, 30)
PREDICTION_REQUIRED_COLS = {"Symbol", "decoder_start_date", "pred_return", "true_return"}
TICKER_REQUIRED_COLS = {"Date", "Symbol", "Adj Close"}
ACTUAL_PRICE_COLOR = "#0D47A1"
PREDICTED_PRICE_COLOR = "#81C784"


def parse_windows(text: str) -> List[int]:
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("No windows provided. Use --windows like '7,10,15,30'.")
    return out


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def compute_mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No valid samples for MAPE computation.")
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    denom = np.where(np.abs(y_true) < eps, np.nan, y_true)
    ape = np.abs((y_pred - y_true) / denom)
    ape = ape[np.isfinite(ape)]
    if ape.size == 0:
        raise ValueError("MAPE undefined: all denominators are near zero or invalid.")
    return float(np.mean(ape) * 100.0)


def compute_return_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No valid return samples for metric computation.")

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    err = y_pred - y_true

    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))

    true_up = y_true > 0.0
    pred_up = y_pred > 0.0
    directional_accuracy = float(np.mean(true_up == pred_up))

    tp = int(np.sum(true_up & pred_up))
    fp = int(np.sum(~true_up & pred_up))
    fn = int(np.sum(true_up & ~pred_up))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_up = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "directional_accuracy": directional_accuracy,
        "f1_up": float(f1_up),
    }


def select_middle_ranks(total: int, count: int = 4) -> Tuple[int, ...]:
    """Centre `count` ranks of a `total`-length ranking, 1-indexed."""
    if total < count:
        raise ValueError(
            f"Need at least {count} ranked stocks for a grouped plot, found {total}."
        )
    start = (total - count) // 2 + 1
    return tuple(range(start, start + count))


def load_ticker_prices(ticker_path: Path) -> pd.DataFrame:
    if not ticker_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_path}")

    ticker = pd.read_csv(ticker_path)
    ensure_required_columns(ticker, TICKER_REQUIRED_COLS, ticker_path)

    ticker = ticker[["Date", "Symbol", "Adj Close"]].copy()
    ticker["Date"] = (
        pd.to_datetime(ticker["Date"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    ticker["Adj Close"] = pd.to_numeric(ticker["Adj Close"], errors="coerce")
    ticker = ticker.dropna(subset=["Date", "Symbol", "Adj Close"]).copy()
    ticker = ticker.sort_values(["Symbol", "Date"], kind="mergesort").reset_index(drop=True)
    return ticker


def load_window_predictions(window: int, artifact_root: Path) -> pd.DataFrame:
    pred_path = artifact_root / f"window_{window}" / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    pred = pd.read_csv(pred_path)
    ensure_required_columns(pred, PREDICTION_REQUIRED_COLS, pred_path)

    pred["Symbol"] = pred["Symbol"].astype(str)
    pred["decoder_start_date"] = pd.to_datetime(pred["decoder_start_date"], errors="coerce").dt.normalize()
    pred["pred_return"] = pd.to_numeric(pred["pred_return"], errors="coerce")
    pred["true_return"] = pd.to_numeric(pred["true_return"], errors="coerce")

    pred = pred.dropna(subset=["Symbol", "decoder_start_date", "pred_return", "true_return"]).copy()
    pred = pred.sort_values(["Symbol", "decoder_start_date"], kind="mergesort").reset_index(drop=True)
    pred["window"] = int(window)
    return pred


def prices_from_prediction_audit(pred_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Use the price columns the evaluation stage already wrote.

    ``src/4_tft_hpt_train_test.py`` reconstructs both prices from the unscaled
    ``adj_close_inr`` base, so this is the same arithmetic as
    ``reconstruct_prices_inr`` without needing the raw ticker file, which is
    gitignored and therefore absent in a fresh clone.
    """
    missing = sorted({"pred_price", "true_price"} - set(pred_df.columns))
    if missing:
        raise ValueError(
            f"window={window}: prediction audit lacks {missing}. Pass --ticker-path "
            "pointing at the raw ticker CSV to rebuild prices instead."
        )

    out = pred_df.copy()
    out["actual_price_inr"] = pd.to_numeric(out["true_price"], errors="coerce")
    out["pred_price_inr"] = pd.to_numeric(out["pred_price"], errors="coerce")

    invalid = out["actual_price_inr"].isna() | out["pred_price_inr"].isna()
    if bool(invalid.any()):
        raise ValueError(
            f"window={window}: {int(invalid.sum())} prediction rows have unusable "
            "pred_price/true_price values."
        )
    # A z-scored base price (pre-fix datasets) produces near-zero, sometimes
    # negative "prices". Refuse to plot those silently as rupees.
    if (out["actual_price_inr"] <= 0).any():
        raise ValueError(
            f"window={window}: test_predictions.csv holds non-positive prices, which "
            "means it was produced from a standardized price column. Re-run "
            "src/3_feature_engineering.py and src/4_tft_hpt_train_test.py, or pass "
            "--ticker-path to rebuild prices from the raw ticker file."
        )
    return out.sort_values(["Symbol", "decoder_start_date"], kind="mergesort").reset_index(drop=True)


def reconstruct_prices_inr(pred_df: pd.DataFrame, ticker_df: pd.DataFrame, window: int) -> pd.DataFrame:
    merged_parts: List[pd.DataFrame] = []

    for symbol, sym_pred in pred_df.groupby("Symbol", sort=True):
        sym_price = ticker_df.loc[ticker_df["Symbol"] == symbol, ["Date", "Adj Close"]].copy()
        if sym_price.empty:
            missing = sym_pred[["Symbol", "decoder_start_date"]].copy()
            missing["base_price_inr"] = np.nan
            missing["base_date"] = pd.NaT
            merged_parts.append(missing)
            continue

        sym_pred = sym_pred.sort_values("decoder_start_date", kind="mergesort").copy()
        sym_price = sym_price.sort_values("Date", kind="mergesort").copy()
        sym_price = sym_price.rename(columns={"Date": "base_date", "Adj Close": "base_price_inr"})

        merged = pd.merge_asof(
            sym_pred,
            sym_price,
            left_on="decoder_start_date",
            right_on="base_date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_parts.append(merged)

    out = pd.concat(merged_parts, axis=0, ignore_index=True)
    missing_base = out["base_price_inr"].isna()
    if bool(missing_base.any()):
        sample = (
            out.loc[missing_base, ["Symbol", "decoder_start_date"]]
            .drop_duplicates()
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"window={window}: missing previous-trading-day base Adj Close for "
            f"{int(missing_base.sum())} prediction rows. Sample: {sample}"
        )

    out["actual_price_inr"] = out["base_price_inr"] * (1.0 + out["true_return"])
    out["pred_price_inr"] = out["base_price_inr"] * (1.0 + out["pred_return"])
    out = out.sort_values(["Symbol", "decoder_start_date"], kind="mergesort").reset_index(drop=True)
    return out


def compute_window_metrics(window_df: pd.DataFrame, window: int) -> Dict[str, float]:
    y_true_return = window_df["true_return"].to_numpy(dtype=np.float64)
    y_pred_return = window_df["pred_return"].to_numpy(dtype=np.float64)
    y_true_price = window_df["actual_price_inr"].to_numpy(dtype=np.float64)
    y_pred_price = window_df["pred_price_inr"].to_numpy(dtype=np.float64)

    ret_metrics = compute_return_metrics(y_true_return, y_pred_return)
    mape = compute_mape_percent(y_true_price, y_pred_price)
    return {
        "window": int(window),
        "mape": mape,
        "mae": ret_metrics["mae"],
        "rmse": ret_metrics["rmse"],
        "mse": ret_metrics["mse"],
        "directional_accuracy": ret_metrics["directional_accuracy"],
        "f1_up": ret_metrics["f1_up"],
    }


def compute_stock_mape(window_df: pd.DataFrame, window: int) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for symbol, sym_df in window_df.groupby("Symbol", sort=True):
        y_true = sym_df["actual_price_inr"].to_numpy(dtype=np.float64)
        y_pred = sym_df["pred_price_inr"].to_numpy(dtype=np.float64)
        mape = compute_mape_percent(y_true, y_pred)
        rows.append(
            {
                "window": int(window),
                "Symbol": str(symbol),
                "mape": mape,
                "n_samples": int(len(sym_df)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["window", "mape", "Symbol"], kind="mergesort").reset_index(drop=True)


def validate_metric_ranges(metrics_df: pd.DataFrame) -> None:
    if (metrics_df["mape"] < 0).any():
        raise ValueError("Invalid metrics: MAPE must be >= 0.")
    if (metrics_df["mse"] < 0).any():
        raise ValueError("Invalid metrics: MSE must be >= 0.")
    if ((metrics_df["directional_accuracy"] < 0) | (metrics_df["directional_accuracy"] > 1)).any():
        raise ValueError("Invalid metrics: directional_accuracy must be in [0, 1].")
    if ((metrics_df["f1_up"] < 0) | (metrics_df["f1_up"] > 1)).any():
        raise ValueError("Invalid metrics: f1_up must be in [0, 1].")


def plot_symbol_actual_vs_predicted(
    ax,
    sym_df: pd.DataFrame,
    symbol: str,
    linewidth: float,
    title_fontsize: int,
    tick_fontsize: int,
    title_pad: float,
) -> None:
    ax.plot(
        sym_df["decoder_start_date"],
        sym_df["actual_price_inr"],
        color=ACTUAL_PRICE_COLOR,
        linewidth=linewidth,
        linestyle="-",
        label="Actual Price",
    )
    ax.plot(
        sym_df["decoder_start_date"],
        sym_df["pred_price_inr"],
        color=PREDICTED_PRICE_COLOR,
        linewidth=linewidth,
        linestyle="-",
        label="Predicted Price",
    )
    ax.set_title(symbol, fontsize=title_fontsize, pad=title_pad)
    ax.grid(alpha=0.22, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=tick_fontsize, length=2)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")


def build_price_legend_handles(linewidth: float) -> List[Line2D]:
    return [
        Line2D([0], [0], color=ACTUAL_PRICE_COLOR, lw=linewidth, linestyle="-", label="Actual Price"),
        Line2D([0], [0], color=PREDICTED_PRICE_COLOR, lw=linewidth, linestyle="-", label="Predicted Price"),
    ]


def select_ranked_stock_group(best_stock_mape_df: pd.DataFrame, ranks: Iterable[int]) -> pd.DataFrame:
    part = best_stock_mape_df.sort_values(["mape", "Symbol"], kind="mergesort").reset_index(drop=True).copy()
    if part.empty:
        raise ValueError("No stock-MAPE data found for grouped price plotting.")

    ordered_ranks = [int(rank) for rank in ranks]
    part["rank"] = np.arange(1, len(part) + 1, dtype=int)
    selected = part.loc[part["rank"].isin(ordered_ranks)].copy()

    if len(selected) != len(ordered_ranks):
        found_ranks = set(selected["rank"].tolist())
        missing = [rank for rank in ordered_ranks if rank not in found_ranks]
        raise ValueError(f"Missing ranked stocks for requested MAPE positions: {missing}")

    rank_order = {rank: idx for idx, rank in enumerate(ordered_ranks)}
    selected["_rank_order"] = selected["rank"].map(rank_order)
    selected = (
        selected.sort_values("_rank_order", kind="mergesort")
        .drop(columns="_rank_order")
        .reset_index(drop=True)
    )
    return selected


def plot_best_window_actual_vs_predicted(best_df: pd.DataFrame, output_path: Path, dpi: int, best_window: int) -> None:
    symbols = sorted(best_df["Symbol"].unique().tolist())
    if not symbols:
        raise ValueError("No symbols found for best-window plotting.")

    ncols = 5
    nrows = math.ceil(len(symbols) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 2.3 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()

    for i, symbol in enumerate(symbols):
        ax = axes_arr[i]
        sym = (
            best_df.loc[best_df["Symbol"] == symbol]
            .sort_values("decoder_start_date", kind="mergesort")
            .copy()
        )
        plot_symbol_actual_vs_predicted(
            ax=ax,
            sym_df=sym,
            symbol=symbol,
            linewidth=0.95,
            title_fontsize=8,
            tick_fontsize=6,
            title_pad=2.5,
        )

    for j in range(len(symbols), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(
        f"Best Window (W{best_window}) - Actual vs Predicted Price (INR)",
        fontsize=14,
        weight="bold",
        y=0.996,
    )
    fig.text(0.004, 0.5, "Price (INR)", va="center", rotation=90, fontsize=10)

    legend_handles = build_price_legend_handles(linewidth=1.0)
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.982))

    fig.tight_layout(rect=[0.015, 0.02, 1.0, 0.972])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_ranked_stock_group_actual_vs_predicted(
    best_df: pd.DataFrame,
    ranked_group_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    best_window: int,
    group_title: str,
) -> None:
    if ranked_group_df.empty:
        raise ValueError(f"No stock data found for grouped plot: {group_title}")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    axes_arr = np.atleast_1d(axes).ravel()

    for i, row in enumerate(ranked_group_df.itertuples(index=False)):
        ax = axes_arr[i]
        sym = (
            best_df.loc[best_df["Symbol"] == row.Symbol]
            .sort_values("decoder_start_date", kind="mergesort")
            .copy()
        )
        if sym.empty:
            raise ValueError(f"Missing best-window price series for symbol: {row.Symbol}")

        plot_symbol_actual_vs_predicted(
            ax=ax,
            sym_df=sym,
            symbol=str(row.Symbol),
            linewidth=1.75,
            title_fontsize=18,
            tick_fontsize=11,
            title_pad=5.5,
        )

    for j in range(len(ranked_group_df), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(
        f"{group_title} - Best Window (W{best_window})",
        fontsize=20,
        weight="bold",
        y=0.992,
    )
    fig.text(0.015, 0.5, "Price (INR)", va="center", rotation=90, fontsize=12)
    fig.legend(
        handles=build_price_legend_handles(linewidth=1.9),
        loc="upper center",
        ncol=2,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.962),
    )

    fig.tight_layout(rect=[0.02, 0.04, 1.0, 0.93])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_per_window_metrics(metrics_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    metrics_df = metrics_df.sort_values("window", kind="mergesort").reset_index(drop=True)
    x_labels = [f"W{int(w)}" for w in metrics_df["window"].tolist()]
    x = np.arange(len(x_labels))

    panels = [
        ("mape", "MAPE on Price (%)"),
        ("mae", "MAE (Return)"),
        ("rmse", "RMSE (Return)"),
        ("mse", "MSE (Return)"),
        ("directional_accuracy", "Directional Accuracy"),
        ("f1_up", "F1_up"),
    ]
    colors = ["#1565C0", "#1E88E5", "#42A5F5", "#64B5F6"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes_arr = axes.ravel()

    for ax, (col, title) in zip(axes_arr, panels):
        vals = metrics_df[col].to_numpy(dtype=np.float64)
        bars = ax.bar(x, vals, color=[colors[i % len(colors)] for i in range(len(vals))], width=0.64)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        y_max = float(np.nanmax(vals)) if len(vals) else 0.0
        if col in {"directional_accuracy", "f1_up"}:
            upper = max(1.0, y_max * 1.12 + 0.02)
            ax.set_ylim(0.0, upper)
        elif y_max > 0:
            ax.set_ylim(0.0, y_max * 1.16)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (ax.get_ylim()[1] * 0.012),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )

    fig.suptitle("Per-Window TFT Model - Recomputed Test Metrics", fontsize=16, weight="bold", y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_best_window_stock_mape(
    best_stock_mape_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    best_window: int,
) -> None:
    part = best_stock_mape_df.sort_values(["mape", "Symbol"], kind="mergesort").reset_index(drop=True)
    if part.empty:
        raise ValueError(f"No stock-MAPE data found for best window W{best_window}.")

    colors = np.where(
        part["mape"].to_numpy(dtype=np.float64) < 2.0,
        "#2CA02C",
        np.where(part["mape"].to_numpy(dtype=np.float64) <= 5.0, "#F1C232", "#D62728"),
    )

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(part["Symbol"], part["mape"], color=colors, width=0.82, edgecolor="none")
    ax.set_title(
        f"Per-Stock Real MAPE Ranking (Ascending) - Best Window (W{best_window})",
        fontsize=24,
        weight="bold",
    )
    ax.set_xlabel("Symbol", fontsize=26)
    ax.set_ylabel("MAPE (%)", fontsize=26)
    ax.tick_params(axis="x", labelsize=11, rotation=75)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)

    ax.axhline(2.0, color="#66BB6A", linestyle="--", linewidth=1.4, alpha=0.9)
    ax.axhline(5.0, color="#E57373", linestyle="--", linewidth=1.4, alpha=0.9)
    y_max = float(np.nanmax(part["mape"].to_numpy(dtype=np.float64)))
    ax.set_ylim(0.0, max(5.25, y_max * 1.15))

    legend_handles = [
        Patch(facecolor="#2CA02C", label="MAPE < 2%"),
        Patch(facecolor="#F1C232", label="2% <= MAPE <= 5%"),
        Patch(facecolor="#D62728", label="MAPE > 5%"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=16, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TFT evaluation visualizations (non-notebook).")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--ticker-path", type=Path, default=DEFAULT_TICKER_PATH)
    parser.add_argument("--windows", type=str, default="7,10,15,30")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    args = build_arg_parser().parse_args()

    artifact_root = Path(args.artifact_root)
    ticker_path = Path(args.ticker_path)
    output_dir = Path(args.output_dir)
    windows = parse_windows(args.windows)
    dpi = int(args.dpi)

    output_dir.mkdir(parents=True, exist_ok=True)

    # The raw ticker file is the preferred source of truth: rebuilding prices
    # from it independently cross-checks the evaluation stage. It is gitignored,
    # so fall back to the prices the evaluation stage already wrote.
    ticker_df = load_ticker_prices(ticker_path) if ticker_path.exists() else None
    if ticker_df is None:
        print(
            f"[INFO] Ticker file not found ({ticker_path}); reconstructing prices "
            "from the per-window prediction audit instead."
        )

    all_windows_data: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, float]] = []
    stock_rows: List[pd.DataFrame] = []

    for window in windows:
        pred_df = load_window_predictions(window=window, artifact_root=artifact_root)
        if ticker_df is not None:
            enriched_df = reconstruct_prices_inr(pred_df=pred_df, ticker_df=ticker_df, window=window)
        else:
            enriched_df = prices_from_prediction_audit(pred_df=pred_df, window=window)
        all_windows_data.append(enriched_df)
        metric_rows.append(compute_window_metrics(enriched_df, window=window))
        stock_rows.append(compute_stock_mape(enriched_df, window=window))

    metrics_df = pd.DataFrame(metric_rows).sort_values("window", kind="mergesort").reset_index(drop=True)
    stock_mape_df = pd.concat(stock_rows, axis=0, ignore_index=True)
    stock_mape_df = stock_mape_df.sort_values(["window", "mape", "Symbol"], kind="mergesort").reset_index(drop=True)
    all_data_df = pd.concat(all_windows_data, axis=0, ignore_index=True)

    validate_metric_ranges(metrics_df)
    best_window = int(metrics_df.loc[metrics_df["mape"].idxmin(), "window"])
    best_df = (
        all_data_df.loc[all_data_df["window"] == best_window]
        .sort_values(["Symbol", "decoder_start_date"], kind="mergesort")
        .reset_index(drop=True)
    )

    best_plot = output_dir / "best_window_actual_vs_predicted.png"
    metrics_plot = output_dir / "per_window_metrics.png"
    stock_plot = output_dir / "best_window_stock_mape.png"
    lowest_mape_plot = output_dir / "best_window_lowest_4_mape_actual_vs_predicted.png"
    middle_mape_plot = output_dir / "best_window_middle_4_mape_actual_vs_predicted.png"
    highest_mape_plot = output_dir / "best_window_highest_4_mape_actual_vs_predicted.png"
    best_stock_mape_df = stock_mape_df.loc[stock_mape_df["window"] == best_window].copy()
    total_ranked_stocks = int(len(best_stock_mape_df))
    lowest_mape_ranks = (1, 2, 3, 4)
    middle_mape_ranks = select_middle_ranks(total_ranked_stocks, count=4)
    highest_mape_ranks = tuple(range(max(1, total_ranked_stocks - 3), total_ranked_stocks + 1))
    ranked_group_specs = [
        ("4 Lowest-MAPE Stocks", lowest_mape_ranks, lowest_mape_plot),
        ("4 Middle-MAPE Stocks", middle_mape_ranks, middle_mape_plot),
        ("4 Highest-MAPE Stocks", highest_mape_ranks, highest_mape_plot),
    ]

    plot_best_window_actual_vs_predicted(best_df=best_df, output_path=best_plot, dpi=dpi, best_window=best_window)
    plot_per_window_metrics(metrics_df=metrics_df, output_path=metrics_plot, dpi=dpi)
    plot_best_window_stock_mape(
        best_stock_mape_df=best_stock_mape_df,
        output_path=stock_plot,
        dpi=dpi,
        best_window=best_window,
    )
    for group_title, ranks, group_output_path in ranked_group_specs:
        ranked_group_df = select_ranked_stock_group(best_stock_mape_df=best_stock_mape_df, ranks=ranks)
        plot_ranked_stock_group_actual_vs_predicted(
            best_df=best_df,
            ranked_group_df=ranked_group_df,
            output_path=group_output_path,
            dpi=dpi,
            best_window=best_window,
            group_title=group_title,
        )

    legacy_csvs = [
        output_dir / "recomputed_window_metrics.csv",
        output_dir / "stock_mape_by_window.csv",
        output_dir / "best_window_price_series.csv",
    ]
    for csv_path in legacy_csvs:
        if csv_path.exists():
            csv_path.unlink()

    print(f"[INFO] Best window by recomputed MAPE: W{best_window}")
    print(f"[INFO] Saved: {best_plot}")
    print(f"[INFO] Saved: {metrics_plot}")
    print(f"[INFO] Saved: {stock_plot}")
    print(f"[INFO] Saved: {lowest_mape_plot}")
    print(f"[INFO] Saved: {middle_mape_plot}")
    print(f"[INFO] Saved: {highest_mape_plot}")


if __name__ == "__main__":
    main()
