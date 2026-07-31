#!/usr/bin/env python3
"""
Merge the raw scraper CSVs into one dataset and clean it in chunks.

Phase 1 concatenates dataset/raw_dataset/*_raw.csv onto a common column layout.
Phase 2 drops nulls and duplicate URLs, strips HTML entities, boilerplate, URLs
and email addresses out of the text, filters by date and length, and sorts the
result chronologically.
"""

import pandas as pd
from pathlib import Path
import html
import re
import shutil
from typing import Set
import time


CHUNK_SIZE = 100_000
START_DATE = "2020-01-01"
END_DATE = "2026-03-31"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw_dataset"
OUTPUT_FILE = PROJECT_ROOT / "dataset" / "processed_news_dataset.csv"

STANDARD_COLUMNS = ["date", "title", "news", "url"]

# HTML entities and special characters to normalize
HTML_ENTITIES_EXTRA = {
    "&#8211;": "–",
    "&#8212;": "—",
    "&#8216;": "'",
    "&#8217;": "'",
    "&#8220;": """,
    "&#8221;": """,
    "&#8230;": "...",
    "&#039;": "'",
    "&nbsp;": " ",
    "&ndash;": "–",
    "&mdash;": "—",
    "&lsquo;": "'",
    "&rsquo;": "'",
    "&ldquo;": """,
    "&rdquo;": """,
    "&hellip;": "...",
}

# Smart quotes and unicode to ASCII
TYPOGRAPHY_MAP = {
    """: '"',
    """: '"',
    "'": "'",
    "'": "'",
    "—": "-",
    "–": "-",
    "…": "...",
    "−": "-",
}

# Boilerplate patterns to remove (case-insensitive)
BOILERPLATE_PATTERNS = [
    r"download\s+(?:the\s+)?app.*?(?:\.|$)",
    r"subscribe\s+to\s+(?:our\s+)?newsletter.*?(?:\.|$)",
    r"disclaimer\s*:.*?(?:\.|$)",
    r"follow\s+us\s+on.*?(?:\.|$)",
    r"join\s+us\s+on.*?(?:\.|$)",
    r"get\s+(?:the\s+)?latest\s+news.*?(?:\.|$)",
    r"read\s+more\s+at.*?(?:\.|$)",
    r"click\s+here\s+to.*?(?:\.|$)",
    r"for\s+more\s+information.*?(?:\.|$)",
    r"visit\s+our\s+website.*?(?:\.|$)",
]

# Promotional artifact pattern (ALSO READ...)
PROMOTIONAL_PATTERN = r"ALSO\s+READ\s*:?.*?(?:\.|\n|$)"

# URL and email patterns
URL_PATTERN = r"https?://[^\s<>\"]+|www\.[^\s<>\"]+"
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"


def decode_html_entities(text: pd.Series) -> pd.Series:
    """Decode HTML entities in text using vectorized operations."""
    text = text.apply(lambda x: html.unescape(x) if pd.notna(x) else x)
    
    for entity, replacement in HTML_ENTITIES_EXTRA.items():
        text = text.str.replace(entity, replacement, regex=False)
    
    return text


def normalize_typography(text: pd.Series) -> pd.Series:
    """Convert smart quotes and unicode dashes to ASCII."""
    for unicode_char, ascii_char in TYPOGRAPHY_MAP.items():
        text = text.str.replace(unicode_char, ascii_char, regex=False)
    return text


def remove_promotional_artifacts(text: pd.Series) -> pd.Series:
    """Remove 'ALSO READ...' promotional text."""
    return text.str.replace(PROMOTIONAL_PATTERN, " ", flags=re.IGNORECASE, regex=True)


def strip_urls_and_emails(text: pd.Series) -> pd.Series:
    """Remove URLs and email addresses from text."""
    text = text.str.replace(URL_PATTERN, " ", regex=True)
    text = text.str.replace(EMAIL_PATTERN, " ", regex=True)
    return text


def remove_boilerplate(text: pd.Series) -> pd.Series:
    """Remove common scraper boilerplate text."""
    for pattern in BOILERPLATE_PATTERNS:
        text = text.str.replace(pattern, " ", flags=re.IGNORECASE, regex=True)
    return text


def normalize_whitespace(text: pd.Series) -> pd.Series:
    """Normalize newlines and collapse multiple spaces."""
    text = text.str.replace(r"[\n\r]+", ". ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    text = text.str.strip()
    text = text.str.replace(r"\.{2,}", ".", regex=True)
    text = text.str.replace(r"\.\s+\.", ".", regex=True)
    return text


def preprocess_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all text preprocessing transformations to title and news columns."""
    for col in ["title", "news"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = decode_html_entities(df[col])
            df[col] = normalize_typography(df[col])
            df[col] = remove_promotional_artifacts(df[col])
            df[col] = strip_urls_and_emails(df[col])
            df[col] = remove_boilerplate(df[col])
            df[col] = normalize_whitespace(df[col])
    
    return df


def parse_and_filter_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Parse dates to YYYY-MM-DD format and filter by date range."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    before_count = len(df)
    df = df.dropna(subset=["date"])
    invalid_count = before_count - len(df)
    
    if invalid_count > 0:
        print(f"    ⚠ Dropped {invalid_count:,} rows with invalid dates")
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    before_filter = len(df)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    filtered_count = before_filter - len(df)
    
    if filtered_count > 0:
        print(f"    ⚠ Filtered {filtered_count:,} rows outside date range [{start_date}, {end_date}]")
    
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase and keep only required columns."""
    df.columns = df.columns.str.lower().str.strip()
    
    existing_cols = [col for col in STANDARD_COLUMNS if col in df.columns]
    df = df[existing_cols]
    
    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null values in critical columns."""
    before_count = len(df)
    df = df.dropna(subset=["date", "title", "news"])
    dropped_count = before_count - len(df)
    
    if dropped_count > 0:
        print(f"    ⚠ Dropped {dropped_count:,} rows with null values")
    
    return df


def filter_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """Drop articles with title < 3 chars or news < 100 chars."""
    before_count = len(df)
    
    df = df[
        (df["title"].str.len() >= 3) & 
        (df["news"].str.len() >= 100)
    ]
    
    dropped_count = before_count - len(df)
    if dropped_count > 0:
        print(f"    ⚠ Filtered {dropped_count:,} rows by length (title<3 or news<100)")
    
    return df


def deduplicate_chunk(df: pd.DataFrame, seen_urls: Set[str]) -> tuple[pd.DataFrame, Set[str]]:
    """Remove duplicate URLs using global hash set."""
    before_count = len(df)
    
    df = df[~df["url"].isin(seen_urls)]
    
    new_urls = set(df["url"].values)
    seen_urls.update(new_urls)
    
    duplicates_removed = before_count - len(df)
    if duplicates_removed > 0:
        print(f"    ⚠ Removed {duplicates_removed:,} duplicate URLs")
    
    return df, seen_urls


def merge_raw_files():
    """Merge all *_raw.csv files from raw_dataset directory into single CSV."""
    print("=" * 80)
    print("PHASE 1: CHUNKED MERGING")
    print("=" * 80)
    
    raw_files = sorted(RAW_DATA_DIR.glob("*_raw.csv"))
    
    if not raw_files:
        print("⚠ No *_raw.csv files found in", RAW_DATA_DIR)
        return
    
    print(f"Found {len(raw_files)} CSV file(s) to merge:")
    for f in raw_files:
        file_size = f.stat().st_size / (1024 * 1024)  # MB
        print(f"  • {f.name} ({file_size:.1f} MB)")
    
    print(f"\nMerging into: {OUTPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows\n")
    
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print("  ✓ Removed existing output file\n")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    total_rows = 0
    header_written = False
    
    for file_idx, raw_file in enumerate(raw_files, 1):
        print(f"[{file_idx}/{len(raw_files)}] Processing {raw_file.name}...")
        file_rows = 0
        chunk_num = 0
        
        try:
            for chunk in pd.read_csv(raw_file, chunksize=CHUNK_SIZE, low_memory=False):
                chunk_num += 1
                chunk_rows = len(chunk)
                file_rows += chunk_rows

                # Force every chunk onto the standard layout before appending.
                chunk.columns = chunk.columns.str.lower().str.strip()
                chunk = chunk.reindex(columns=STANDARD_COLUMNS)

                mode = "w" if not header_written else "a"
                chunk.to_csv(OUTPUT_FILE, mode=mode, index=False, header=not header_written)
                
                if not header_written:
                    header_written = True
                
                print(f"  Chunk {chunk_num}: {chunk_rows:,} rows → Total: {file_rows:,}")
        
        except Exception as e:
            print(f"  ✗ Error reading {raw_file.name}: {e}")
            continue
        
        total_rows += file_rows
        print(f"  ✓ {raw_file.name}: {file_rows:,} rows\n")
    
    print("=" * 80)
    print(f"✓ PHASE 1 COMPLETE: {total_rows:,} total rows merged")
    print("=" * 80)
    print()


def sort_csv_by_date(csv_path: Path) -> None:
    """Sort a CSV chronologically without holding it all in memory.

    Two-pass bucket sort on the `YYYY-MM` prefix: pass one scatters rows into
    one temp file per month, pass two sorts each bucket in memory and appends it
    in calendar order. Peak memory is one bucket rather than one dataset, which
    matters because the cleaned news file is ~800 MB on disk.
    """
    bucket_dir = csv_path.parent / f".{csv_path.stem}_sort_buckets"
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    bucket_dir.mkdir(parents=True)

    sorted_path = csv_path.parent / f"sorted_{csv_path.name}"
    total_rows = 0

    try:
        print("  → Pass 1/2: partitioning rows by month...")
        bucket_has_header: Set[str] = set()

        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
            # Dates were already normalized to YYYY-MM-DD in parse_and_filter_dates,
            # so the first 7 characters are a sortable month key.
            keys = chunk["date"].astype(str).str.slice(0, 7)
            for month, part in chunk.groupby(keys, sort=False):
                bucket_path = bucket_dir / f"{month}.csv"
                write_header = month not in bucket_has_header
                part.to_csv(
                    bucket_path,
                    mode="w" if write_header else "a",
                    index=False,
                    header=write_header,
                )
                bucket_has_header.add(month)
            total_rows += len(chunk)

        print(f"  ✓ Partitioned {total_rows:,} rows into {len(bucket_has_header)} monthly buckets")

        print("  → Pass 2/2: sorting buckets and concatenating...")
        header_written = False
        for bucket_path in sorted(bucket_dir.glob("*.csv")):
            bucket = pd.read_csv(bucket_path, low_memory=False)
            bucket = bucket.sort_values("date", ascending=True, kind="mergesort")
            bucket.to_csv(
                sorted_path,
                mode="w" if not header_written else "a",
                index=False,
                header=not header_written,
            )
            header_written = True

        if not header_written:
            raise RuntimeError("External sort produced no output rows.")

        shutil.move(str(sorted_path), str(csv_path))
        print("  ✓ Sorting complete\n")

    finally:
        shutil.rmtree(bucket_dir, ignore_errors=True)
        if sorted_path.exists():
            sorted_path.unlink()


def preprocess_merged_file():
    """Apply all preprocessing transformations to merged file using chunks."""
    print("=" * 80)
    print("PHASE 2: IN-PLACE PREPROCESSING")
    print("=" * 80)
    
    if not OUTPUT_FILE.exists():
        print(f"⚠ {OUTPUT_FILE} not found. Run Phase 1 first.")
        return
    
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Input file: {OUTPUT_FILE.name} ({file_size:.1f} MB)")
    print(f"Chunk size: {CHUNK_SIZE:,} rows\n")
    
    temp_file = OUTPUT_FILE.parent / f"temp_{OUTPUT_FILE.name}"
    
    seen_urls: Set[str] = set()
    total_input_rows = 0
    total_output_rows = 0
    chunk_num = 0
    header_written = False
    
    start_time = time.time()
    
    print("Processing chunks...\n")
    
    try:
        for chunk in pd.read_csv(OUTPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
            chunk_num += 1
            chunk_start = time.time()
            input_rows = len(chunk)
            total_input_rows += input_rows
            
            print(f"Chunk {chunk_num}: {input_rows:,} rows")
            
            chunk = standardize_columns(chunk)
            
            chunk = drop_nulls(chunk)
            
            chunk, seen_urls = deduplicate_chunk(chunk, seen_urls)
            
            if len(chunk) == 0:
                print(f"  ⚠ Chunk empty after deduplication, skipping\n")
                continue
            
            chunk = parse_and_filter_dates(chunk, START_DATE, END_DATE)
            
            if len(chunk) == 0:
                print(f"  ⚠ Chunk empty after date filtering, skipping\n")
                continue
            
            print(f"    → Applying text preprocessing...")
            chunk = preprocess_text_columns(chunk)
            
            chunk = filter_by_length(chunk)
            
            if len(chunk) == 0:
                print(f"  ⚠ Chunk empty after length filtering, skipping\n")
                continue
            
            mode = "w" if not header_written else "a"
            chunk.to_csv(temp_file, mode=mode, index=False, header=not header_written)
            
            if not header_written:
                header_written = True
            
            output_rows = len(chunk)
            total_output_rows += output_rows
            chunk_time = time.time() - chunk_start
            
            print(f"    ✓ Output: {output_rows:,} rows ({chunk_time:.2f}s)")
            print(f"    → Running total: {total_output_rows:,} / {total_input_rows:,} rows ({100 * total_output_rows / total_input_rows:.1f}% retained)\n")
    
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return
    
    print("=" * 80)
    print("SORTING BY DATE (Chronological Order)...")
    print("=" * 80)

    try:
        sort_csv_by_date(temp_file)
    except Exception as e:
        print(f"  ⚠ Could not sort file: {e}\n")

    print("Finalizing...")
    shutil.move(str(temp_file), str(OUTPUT_FILE))
    
    elapsed_time = time.time() - start_time
    
    print("=" * 80)
    print("✓ PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"Input rows:    {total_input_rows:,}")
    print(f"Output rows:   {total_output_rows:,}")
    print(f"Removed:       {total_input_rows - total_output_rows:,} ({100 * (total_input_rows - total_output_rows) / total_input_rows:.1f}%)")
    print(f"Unique URLs:   {len(seen_urls):,}")
    print(f"Processing time: {elapsed_time / 60:.1f} minutes")
    print(f"\nFinal output: {OUTPUT_FILE}")
    print("=" * 80)


def main():
    """Execute full preprocessing pipeline."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 16 + "NEWS DATASET PREPROCESSING PIPELINE" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    overall_start = time.time()
    
    merge_raw_files()
    
    preprocess_merged_file()
    
    overall_time = time.time() - overall_start
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 27 + "PIPELINE COMPLETE" + " " * 34 + "║")
    print("║" + f" Total execution time: {overall_time / 60:.1f} minutes" + " " * (78 - 29 - len(f"{overall_time / 60:.1f}")) + "║")
    print("╚" + "=" * 78 + "╝")
    print()


if __name__ == "__main__":
    main()
