#!/usr/bin/env python3
"""
Download historical OHLCV for Nifty 50/100/200 constituents from Yahoo Finance.

Constituent lists come from the NSE index archives (CSV), cached locally so a
stale-list fallback is always available. Ticker renames are resolved lazily
during download, symbols are fetched in parallel with rate limiting, and
progress is checkpointed so a run can resume without losing earlier rows.

    conda activate MarketFusion
    python src/scrapers/nifty_yfinance_scraper.py --index nifty50
    python src/scrapers/nifty_yfinance_scraper.py --index all
"""

import argparse
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


# Date range for historical data
START_DATE = "2020-01-01"
END_DATE = "2026-03-31"

# Parallel processing (utilize all 12 threads)
MAX_WORKERS = 12

# Rate limiting (requests per second to avoid Yahoo Finance blocks)
RATE_LIMIT_DELAY = 0.5  # seconds between requests

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Output directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "stock_dataset"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoint"
CONSTITUENTS_DIR = OUTPUT_DIR / "constituents"

# NSE publishes index constituents as CSVs on its archive host. The older
# www.nseindia.com/api/equity-stockIndices endpoint now returns 404.
NSE_INDEX_FILES = {
    "nifty50": "ind_nifty50list.csv",
    "nifty100": "ind_nifty100list.csv",
    "nifty200": "ind_nifty200list.csv",
}
NSE_ARCHIVE_BASE = "https://nsearchives.nseindia.com/content/indices"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nifty_scraper")


# NSE symbols that Yahoo Finance lists under a different ticker.
# Keys are bare NSE symbols, values are Yahoo symbols (suffix included).
TICKER_CHANGES = {
    # Tata Motors demerged: the original entity (and its full price history)
    # is now Tata Motors Passenger Vehicles; the CV business trades as TMCV.
    "TATAMOTORS": "TMPV.NS",
}


@dataclass
class Stock:
    symbol: str
    company: str
    index: str  # nifty50, nifty100, or nifty200

@dataclass
class DownloadStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.time)

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        return (
            f"[{m:02d}:{s:02d}] total={self.total} "
            f"success={self.success} failed={self.failed} skipped={self.skipped}"
        )


def _parse_constituents_csv(text: str, index_name: str) -> list[Stock]:
    """Turn an NSE index CSV into Stock records with Yahoo Finance symbols."""
    df = pd.read_csv(io.StringIO(text))

    if "Symbol" not in df.columns:
        raise ValueError(f"unexpected columns in NSE CSV: {list(df.columns)}")

    stocks = []
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        if not symbol or symbol.lower() == "nan":
            continue

        company = str(row.get("Company Name", symbol)).strip() or symbol
        # Yahoo Finance lists NSE equities with a .NS suffix.
        yf_symbol = TICKER_CHANGES.get(symbol, f"{symbol}.NS")
        stocks.append(Stock(yf_symbol, company, index_name))

    return stocks


def fetch_constituents(index_name: str) -> list[Stock]:
    """Fetch an index's constituents from NSE, falling back to the local cache."""
    csv_name = NSE_INDEX_FILES[index_name]
    cache_file = CONSTITUENTS_DIR / csv_name

    log.info(f"Fetching {index_name} constituents from NSE archives...")
    try:
        response = requests.get(
            f"{NSE_ARCHIVE_BASE}/{csv_name}", headers=NSE_HEADERS, timeout=20
        )
        response.raise_for_status()
        stocks = _parse_constituents_csv(response.text, index_name)

        if not stocks:
            raise ValueError("NSE returned an empty constituent list")

        # Cache the raw CSV so a later network failure still has a usable list.
        CONSTITUENTS_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(response.text)

        log.info(f"  ✓ Fetched {len(stocks)} {index_name} stocks")
        return stocks

    except Exception as e:
        log.error(f"Failed to fetch {index_name} from NSE: {e}")

        if cache_file.exists():
            log.warning(f"  ↻ Falling back to cached list: {cache_file}")
            try:
                stocks = _parse_constituents_csv(cache_file.read_text(), index_name)
                log.info(f"  ✓ Loaded {len(stocks)} {index_name} stocks from cache")
                return stocks
            except Exception as cache_error:
                log.error(f"Cached list is unusable: {cache_error}")

        log.error(f"No constituent list available for {index_name} - skipping")
        return []


def _fetch_history(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch one symbol's OHLCV, retrying transient failures. None if no data."""
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
            time.sleep(RATE_LIMIT_DELAY)
            return None if df.empty else df

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                log.warning(f"  ⚠ Attempt {attempt + 1} failed for {symbol}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"  ✗ Failed to download {symbol} after {MAX_RETRIES} attempts: {e}")

    return None


def download_stock_data(stock: Stock, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download historical data for a stock, trying the BSE listing as a fallback.

    Resolution happens here rather than in a separate validation pass: probing
    with Ticker.info costs an extra request per symbol and gets rate-limited,
    and history() already tells us whether the symbol resolves.
    """
    base_symbol = stock.symbol.rsplit(".", 1)[0]
    candidates = [stock.symbol, f"{base_symbol}.BO"]

    for candidate in dict.fromkeys(candidates):
        df = _fetch_history(candidate, start, end)
        if df is None:
            continue

        if candidate != stock.symbol:
            log.info(f"  ↻ Using alternative listing: {stock.symbol} → {candidate}")

        df = df.reset_index()
        df["Symbol"] = candidate
        df["Company"] = stock.company
        df["Index"] = stock.index

        cols = [
            "Date", "Symbol", "Company", "Index", "Open", "High", "Low",
            "Close", "Adj Close", "Volume", "Dividends", "Stock Splits",
        ]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            log.warning(f"  ⚠ {candidate} missing columns {missing}")
            return None

        return df[cols]

    log.warning(f"  ⚠ No data for {stock.symbol} ({stock.company})")
    return None


def load_checkpoint(index_name: str) -> set[str]:
    """Load list of already-downloaded symbols from checkpoint."""
    checkpoint_file = CHECKPOINT_DIR / f"{index_name}_checkpoint.json"
    
    if not checkpoint_file.exists():
        return set()
    
    try:
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            return set(data.get("completed", []))
    except Exception as e:
        log.warning(f"Could not load checkpoint: {e}")
        return set()

def save_checkpoint(index_name: str, completed_symbols: set[str]):
    """Save checkpoint of completed downloads."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{index_name}_checkpoint.json"
    
    try:
        with open(checkpoint_file, "w") as f:
            json.dump({"completed": list(completed_symbols)}, f, indent=2)
    except Exception as e:
        log.error(f"Could not save checkpoint: {e}")


def write_output(output_file: Path, new_data: list[pd.DataFrame]) -> int:
    """Merge freshly downloaded frames into the index CSV, returning total rows.

    A resumed run only downloads the symbols missing from the checkpoint, so the
    existing rows must be preserved instead of overwritten.
    """
    combined = pd.concat(new_data, ignore_index=True)

    if output_file.exists():
        try:
            existing = pd.read_csv(output_file)
            # Re-downloaded symbols win; everything else is carried forward.
            kept = existing[~existing["Symbol"].isin(combined["Symbol"].unique())]
            combined = pd.concat([kept, combined], ignore_index=True)
        except Exception as e:
            log.warning(f"Could not merge with existing {output_file.name}: {e}")

    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    combined.to_csv(output_file, index=False)
    return len(combined)


def scrape_index(index_name: str, stocks: list[Stock], start_date: str, end_date: str) -> bool:
    """Download historical data for all stocks in an index. True if data landed."""

    if not stocks:
        log.error(f"No stocks found for {index_name}")
        return False

    log.info(f"═══ Starting {index_name.upper()} Download ═══")
    log.info(f"  Stocks: {len(stocks)}")
    log.info(f"  Date Range: {start_date} to {end_date}")
    log.info(f"  Workers: {MAX_WORKERS}")

    # Load checkpoint
    completed = load_checkpoint(index_name)
    if completed:
        log.info(f"  ↻ Resuming: {len(completed)} stocks already downloaded")

    pending = [s for s in stocks if s.symbol not in completed]
    log.info(f"  ✓ {len(pending)} stocks to download")

    if not pending:
        log.info("  All stocks already downloaded!")
        return True

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{index_name}_ticker.csv"

    # Download in parallel
    stats = DownloadStats(total=len(pending))
    all_data = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {
            executor.submit(download_stock_data, stock, start_date, end_date): stock
            for stock in pending
        }

        for future in as_completed(future_to_stock):
            stock = future_to_stock[future]

            try:
                df = future.result()

                if df is not None and not df.empty:
                    all_data.append(df)
                    completed.add(stock.symbol)
                    stats.success += 1
                    log.info(f"  ✓ [{stats.success}/{stats.total}] {stock.symbol} ({len(df)} rows)")

                    # Checkpoint periodically so a crash keeps most progress.
                    if stats.success % 10 == 0:
                        save_checkpoint(index_name, completed)
                else:
                    stats.failed += 1
                    log.warning(f"  ✗ [{stats.success + stats.failed}/{stats.total}] {stock.symbol} - No data")

            except Exception as e:
                stats.failed += 1
                log.error(f"  ✗ [{stats.success + stats.failed}/{stats.total}] {stock.symbol} - {e}")

    if all_data:
        log.info(f"Saving data to {output_file}...")
        total_rows = write_output(output_file, all_data)
        log.info(f"  ✓ Saved {total_rows} rows to {output_file}")

    save_checkpoint(index_name, completed)

    log.info(f"═══ DONE ═══  {stats.summary()}")
    return bool(all_data)


def run_indices(choice: str, start_date: str, end_date: str) -> tuple[list[str], list[str]]:
    """Fetch and scrape the requested indices. Returns (written, skipped) names."""
    targets = list(NSE_INDEX_FILES) if choice == "all" else [choice]

    written, skipped = [], []
    for index_name in targets:
        stocks = fetch_constituents(index_name)
        if stocks and scrape_index(index_name, stocks, start_date, end_date):
            written.append(index_name)
        else:
            skipped.append(index_name)
        print()  # blank line between indices

    return written, skipped


def print_menu():
    """Display the interactive menu."""
    print("\n" + "═" * 70)
    print("  NIFTY STOCK PRICE SCRAPER - Yahoo Finance")
    print("═" * 70)
    print(f"\n📅 Date Range: {START_DATE} to {END_DATE}")
    print(f"💾 Output Directory: {OUTPUT_DIR}")
    print(f"⚡ Parallel Workers: {MAX_WORKERS}")
    print("\n" + "─" * 70)
    print("  SELECT INDEX TO DOWNLOAD:")
    print("─" * 70)
    print("  [0] Nifty 50   (50 stocks)")
    print("  [1] Nifty 100  (100 stocks)")
    print("  [2] Nifty 200  (200 stocks)")
    print("  [3] All Indices (all three)")
    print("  [4] Exit")
    print("─" * 70)

def get_user_choice() -> str:
    """Get and validate user input."""
    while True:
        try:
            choice = input("\n👉 Enter your choice (0-4): ").strip()
            
            if choice in ["0", "1", "2", "3", "4"]:
                return choice
            else:
                print("❌ Invalid choice! Please enter 0, 1, 2, 3, or 4.")
        except (EOFError, KeyboardInterrupt):
            print("\n\n⚠️  Interrupted by user. Exiting...")
            exit(0)

def confirm_download(index_name: str) -> bool:
    """Ask user to confirm the download."""
    index_map = {
        "nifty50": "Nifty 50",
        "nifty100": "Nifty 100",
        "nifty200": "Nifty 200",
        "all": "All Indices (Nifty 50, 100, 200)"
    }
    
    print(f"\n{'─' * 70}")
    print(f"  You selected: {index_map[index_name]}")
    print(f"  Date Range: {START_DATE} to {END_DATE}")
    print(f"{'─' * 70}")
    
    while True:
        confirm = input("\n✓ Start download? (y/n): ").strip().lower()
        if confirm in ["y", "yes"]:
            return True
        elif confirm in ["n", "no"]:
            return False
        else:
            print("❌ Please enter 'y' or 'n'")

def interactive_mode():
    """Run the scraper in interactive menu mode."""
    while True:
        print_menu()
        choice = get_user_choice()
        
        # Map choice to index
        choice_map = {
            "0": "nifty50",
            "1": "nifty100",
            "2": "nifty200",
            "3": "all",
            "4": "exit"
        }
        
        index_name = choice_map[choice]
        
        if index_name == "exit":
            print("\n👋 Thank you for using Nifty Stock Scraper. Goodbye!\n")
            break
        
        # Confirm before starting download
        if not confirm_download(index_name):
            print("\n❌ Download cancelled. Returning to menu...\n")
            continue
        
        # Start download
        print("\n" + "═" * 70)
        print("  STARTING DOWNLOAD...")
        print("═" * 70 + "\n")
        
        done, skipped = run_indices(index_name, START_DATE, END_DATE)

        print("\n" + "═" * 70)
        if done:
            print("  ✅ DOWNLOAD COMPLETED!")
        else:
            print("  ⚠️  DOWNLOAD FAILED - no data saved")
        print("═" * 70)
        if done:
            print(f"  📂 Files saved in: {OUTPUT_DIR}")
            print(f"  ✓ Indices written: {', '.join(done)}")
        if skipped:
            print(f"  ✗ Indices skipped: {', '.join(skipped)}")
        print("═" * 70 + "\n")

        # Ask if user wants to continue
        while True:
            again = input("📥 Download another index? (y/n): ").strip().lower()
            if again in ["y", "yes"]:
                break
            elif again in ["n", "no"]:
                print("\n👋 Thank you for using Nifty Stock Scraper. Goodbye!\n")
                return
            else:
                print("❌ Please enter 'y' or 'n'")


def cli_mode():
    """Run the scraper in command-line mode (with arguments)."""
    parser = argparse.ArgumentParser(
        description="Download Nifty stock historical prices from Yahoo Finance"
    )
    parser.add_argument(
        "--index",
        choices=["nifty50", "nifty100", "nifty200", "all"],
        required=True,
        help="Which index to download (nifty50, nifty100, nifty200, or all)"
    )
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"Start date (YYYY-MM-DD, default: {START_DATE})"
    )
    parser.add_argument(
        "--end",
        default=END_DATE,
        help=f"End date (YYYY-MM-DD, default: {END_DATE})"
    )
    
    args = parser.parse_args()
    
    log.info("Nifty Stock Price Scraper")
    log.info(f"  Date Range: {args.start} to {args.end}")
    log.info(f"  Output Dir: {OUTPUT_DIR}")
    
    done, skipped = run_indices(args.index, args.start, args.end)

    if skipped:
        log.warning(f"Indices skipped: {', '.join(skipped)}")
    if not done:
        log.error("No data was saved")
        raise SystemExit(1)

    log.info(f"Indices written: {', '.join(done)}")

def main():
    """Main entry point - choose between interactive or CLI mode."""
    import sys
    
    # If arguments are provided, use CLI mode
    if len(sys.argv) > 1:
        cli_mode()
    else:
        # No arguments - use interactive menu mode
        try:
            interactive_mode()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            exit(0)

if __name__ == "__main__":
    main()
