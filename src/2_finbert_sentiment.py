"""
Sentiment scoring for the segregated news table using ProsusAI/finbert.

Every unique article string is scored once and cached; cell values are then
assembled from that cache, which matters because `global_news` repeats the same
cell for all 50 tickers on a date. Texts above the token limit are split into
overlapping windows and averaged, and batches are packed to a token budget
rather than a fixed row count so padding waste stays low.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock between HuggingFace Rust tokenizer and PyTorch DataLoader workers
import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


INPUT_CSV   = Path("dataset/tier_segregated_news.csv")
OUTPUT_CSV  = Path("dataset/news_sentiment.csv")
MODEL_NAME  = "ProsusAI/finbert"

# Tokenizer / sliding-window knobs
MAX_TOKENS      = 500   # treat texts above this length as "long"
WINDOW_TOKENS   = 400   # chunk size for the sliding window
OVERLAP_TOKENS  = 100   # overlap between consecutive chunks
STRIDE_TOKENS   = WINDOW_TOKENS - OVERLAP_TOKENS  # effective stride = 300

# Inference knobs, sized for a 4 GB RTX 3050 / 16 GB RAM host. Batches are built
# to a token budget rather than a fixed row count, which keeps peak activation
# memory (batch_rows x seq_len) roughly constant across very uneven article
# lengths.
MAX_BATCH_TOKENS = int(os.environ.get("FINBERT_MAX_BATCH_TOKENS", "12288"))
MAX_BATCH_ROWS   = int(os.environ.get("FINBERT_MAX_BATCH_ROWS", "192"))
# Collate is a cheap pad-and-stack; extra workers cost more in IPC than they save.
NUM_WORKERS = int(os.environ.get("FINBERT_NUM_WORKERS", "2"))
# Slice size for batched (Rust) tokenization of the unique article strings.
TOKENIZE_BATCH = 512

# Separator used between articles inside a cell
ARTICLE_SEP = " || "

# Column names in the input file (lower-cased for robustness)
COL_DATE   = "date"
COL_SYMBOL = "symbol"
NEWS_COLS  = ["direct_news", "sectoral_news", "global_news"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def count_articles(cell: str) -> int:
    """Return the number of articles in a cell (number of `||` separators + 1)."""
    if not cell or not cell.strip():
        return 0
    return cell.count("||") + 1


def chunks_from_token_ids(
    ids: List[int],
    cls_id: int,
    sep_id: int,
    max_tokens: int = MAX_TOKENS,
    window: int = WINDOW_TOKENS,
    stride: int = STRIDE_TOKENS,
) -> List[Dict]:
    """
    Turn one article's raw token ids into model-ready chunks.

    Texts within `max_tokens` become a single chunk. Longer texts become
    overlapping chunks of `window` tokens stepped by `stride`. Each chunk is a
    dict with 'input_ids' and 'attention_mask' as plain Python lists (no batch
    dimension); special tokens are added here, not by the tokenizer.
    """
    if len(ids) <= max_tokens:
        chunk_ids  = [cls_id] + ids + [sep_id]
        return [{"input_ids": chunk_ids, "attention_mask": [1] * len(chunk_ids)}]

    chunks = []
    start = 0
    while start < len(ids):
        end        = min(start + window, len(ids))
        chunk_ids  = [cls_id] + ids[start:end] + [sep_id]
        chunks.append({"input_ids": chunk_ids, "attention_mask": [1] * len(chunk_ids)})
        if end == len(ids):
            break
        start += stride

    return chunks


def build_token_budget_batches(
    lengths: np.ndarray,
    max_batch_tokens: int = MAX_BATCH_TOKENS,
    max_batch_rows: int = MAX_BATCH_ROWS,
) -> List[List[int]]:
    """
    Group chunk indices into batches of similar length under a token budget.

    Chunks are visited shortest-first, so each batch is padded to a length close
    to its own members rather than to the longest chunk in the whole corpus. The
    padded token count of a batch is `rows * max_len_in_batch`, which is what
    actually drives GPU memory and time.

    Returns a list of index lists suitable for `DataLoader(batch_sampler=...)`.
    """
    order = np.argsort(lengths, kind="stable")

    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0

    for idx in order:
        length = int(lengths[idx])
        candidate_max = max(current_max, length)
        # Close the batch when adding this chunk would breach either limit.
        if current and (
            len(current) + 1 > max_batch_rows
            or (len(current) + 1) * candidate_max > max_batch_tokens
        ):
            batches.append(current)
            current, current_max = [], 0
            candidate_max = length

        current.append(int(idx))
        current_max = candidate_max

    if current:
        batches.append(current)
    return batches


class ChunkDataset(Dataset):
    """
    Holds a flat list of tokenised chunks (each a dict with 'input_ids' and
    'attention_mask'). The collate function pads them to the batch maximum.
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict:
        return self.chunks[idx]


def collate_pad(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length chunks to the same length."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_batch  = []
    attn_mask_batch  = []

    for item in batch:
        ids  = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - len(ids)
        input_ids_batch.append(ids  + [pad_id] * pad_len)
        attn_mask_batch.append(mask + [0]      * pad_len)

    return {
        "input_ids":      torch.tensor(input_ids_batch,  dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask_batch,  dtype=torch.long),
    }


def run_inference(
    model,
    chunks: List[Dict],
    device: torch.device,
    num_workers: int = NUM_WORKERS,
    desc: str = "FinBERT inference",
) -> np.ndarray:
    """
    Run FinBERT on `chunks` (flat list of tokenised dicts).

    Returns an (N, 3) float32 array of softmax probabilities in the model's own
    label order (0=positive, 1=negative, 2=neutral), aligned row-for-row with
    `chunks`. Batches are length-sorted internally; results are scattered back to
    the caller's ordering, so callers never see the permutation.
    """
    lengths = np.fromiter(
        (len(c["input_ids"]) for c in chunks), dtype=np.int64, count=len(chunks)
    )
    batches = build_token_budget_batches(lengths)
    padded = sum(len(b) * int(lengths[b].max()) for b in batches)
    log.info(
        "  %s batches | %.1f%% padding waste | max seq %d",
        f"{len(batches):,}",
        100.0 * (1.0 - lengths.sum() / max(padded, 1)),
        int(lengths.max()) if len(lengths) else 0,
    )

    loader = DataLoader(
        ChunkDataset(chunks),
        batch_sampler=batches,
        num_workers=num_workers,
        collate_fn=lambda b: collate_pad(b, pad_id=0),
        pin_memory=(device.type == "cuda"),
    )

    out = np.empty((len(chunks), 3), dtype=np.float32)
    model.eval()

    with torch.no_grad():
        # DataLoader yields batches in `batches` order, so zipping restores the
        # mapping from each result row back to its original chunk index.
        for indices, batch in zip(
            batches, tqdm(loader, desc=desc, leave=False, unit="batch")
        ):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Weights stay fp32; autocast runs the matmuls in fp16. A blanket
            # .half() on the weights is what risks NaN in LayerNorm.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(**batch).logits             # (B, 3)

            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            out[indices] = probs

    return out  # (N, 3)


def build_article_cache(
    unique_articles: List[str],
    tokenizer,
    model,
    device: torch.device,
) -> Dict[str, Tuple[float, float, float]]:
    """
    For every unique article string:
      1. Tokenise with sliding window -> N_chunks chunks
      2. Run inference -> N_chunks x 3 probability vectors
      3. Average across chunks (equal weight) -> (pos, neu, neg)

    Returns a dict article_text -> (pos, neu, neg). FinBERT's own label order is
    0=positive, 1=negative, 2=neutral; the output convention here is pos/neu/neg.
    """
    log.info(f"Building article-level cache for {len(unique_articles):,} unique articles …")

    # tokenise all articles, recording the slice boundaries
    all_chunks: List[Dict] = []
    article_slices: List[Tuple[int, int]] = []  # (start_idx, end_idx) into all_chunks

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    log.info("Tokenising unique articles (sliding window) …")
    # Encode in slices: the Rust tokenizer amortises far better over a list.
    for start in tqdm(
        range(0, len(unique_articles), TOKENIZE_BATCH), desc="Tokenising", unit="batch"
    ):
        slice_texts = unique_articles[start:start + TOKENIZE_BATCH]
        encoded = tokenizer(
            slice_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,   # plain Python lists
        )
        for ids in encoded["input_ids"]:
            chunk_start = len(all_chunks)
            all_chunks.extend(chunks_from_token_ids(ids, cls_id, sep_id))
            article_slices.append((chunk_start, len(all_chunks)))

    log.info(f"Total chunks to infer: {len(all_chunks):,}")

    # run inference in one pass
    probs_all = run_inference(model, all_chunks, device, desc="FinBERT → articles")
    # probs_all shape: (total_chunks, 3)   columns: pos=0, neg=1, neu=2

    # aggregate & cache
    article_cache: Dict[str, Tuple[float, float, float]] = {}
    for article, (s, e) in zip(unique_articles, article_slices):
        chunk_probs = probs_all[s:e]           # (n_chunks, 3)
        mean_probs  = chunk_probs.mean(axis=0) # (3,)
        # FinBERT label order: 0=positive, 1=negative, 2=neutral
        pos = float(mean_probs[0])
        neg = float(mean_probs[1])
        neu = float(mean_probs[2])
        article_cache[article] = (pos, neu, neg)  # store as (pos, neu, neg)

    log.info("Article cache built ✓")
    return article_cache


def score_cell(
    cell_value,
    article_cache: Dict[str, Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """
    Split `cell_value` by ARTICLE_SEP.
    Retrieve each article's (pos, neu, neg) from `article_cache`.
    Return the mean across all articles in the cell.

    Returns (0.0, 0.0, 0.0) for empty/NaN cells.
    """
    if cell_value is None or (isinstance(cell_value, float) and math.isnan(cell_value)):
        return (0.0, 0.0, 0.0)
    cell_str = str(cell_value).strip()
    if not cell_str:
        return (0.0, 0.0, 0.0)

    articles = [a.strip() for a in cell_str.split(ARTICLE_SEP) if a.strip()]
    if not articles:
        return (0.0, 0.0, 0.0)

    pos_vals, neu_vals, neg_vals = [], [], []
    for art in articles:
        scored = article_cache.get(art)
        if scored is None:
            # Would mean the cell splits differently than the collection pass did.
            raise KeyError(
                f"Article missing from FinBERT cache (len={len(art)}): {art[:120]!r}"
            )
        p, n, g = scored
        pos_vals.append(p)
        neu_vals.append(n)
        neg_vals.append(g)

    return (
        float(np.mean(pos_vals)),
        float(np.mean(neu_vals)),
        float(np.mean(neg_vals)),
    )


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading dataset: {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.lower().str.strip()

    required = [COL_DATE, COL_SYMBOL] + NEWS_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}. Found: {list(df.columns)}")

    log.info(f"Dataset loaded: {len(df):,} rows × {len(df.columns)} cols")

    log.info("Collecting unique article strings …")
    unique_articles_set: set = set()

    for col in NEWS_COLS:
        for cell in df[col].dropna():
            cell_str = str(cell).strip()
            if not cell_str:
                continue
            for art in cell_str.split(ARTICLE_SEP):
                art = art.strip()
                if art:
                    unique_articles_set.add(art)

    unique_articles = sorted(unique_articles_set)  # deterministic order
    log.info(f"Unique article strings: {len(unique_articles):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Loading tokenizer & model: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Weights load in fp32 and stay there; mixed precision is applied per-op by
    # torch.autocast inside run_inference. Loading with torch_dtype=float16 is
    # equivalent to model.half(), and fp16 LayerNorm can produce NaN.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()

    article_cache = build_article_cache(unique_articles, tokenizer, model, device)

    # Free GPU memory - no more inference needed after this point
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    log.info("Assembling output DataFrame …")

    rows = []
    append_row = rows.append
    date_idx = df.columns.get_loc(COL_DATE)
    symbol_idx = df.columns.get_loc(COL_SYMBOL)
    news_indices = {col: df.columns.get_loc(col) for col in NEWS_COLS}

    # Cell-level cache. `global_news` carries the same joined string for every
    # ticker on a date, and empty cells are the most common value overall.
    cell_cache: Dict[str, Tuple[float, float, float, int]] = {}

    def score_and_count(cell) -> Tuple[float, float, float, int]:
        cell_str = (
            ""
            if cell is None or (isinstance(cell, float) and math.isnan(cell))
            else str(cell)
        )
        cached = cell_cache.get(cell_str)
        if cached is None:
            pos, neu, neg = score_cell(cell_str, article_cache)
            cached = (pos, neu, neg, count_articles(cell_str))
            cell_cache[cell_str] = cached
        return cached

    for row in tqdm(df.itertuples(index=False, name=None), total=len(df), desc="Building output", unit="row"):
        record = {
            "Date":   row[date_idx],
            "Symbol": row[symbol_idx],
        }

        for col in NEWS_COLS:
            pos, neu, neg, count = score_and_count(row[news_indices[col]])
            record[f"{col}_pos"] = pos
            record[f"{col}_neu"] = neu
            record[f"{col}_neg"] = neg
            record[f"{col}_count"] = count

        append_row(record)

    log.info(f"Cell cache: {len(cell_cache):,} distinct cells across {len(df) * len(NEWS_COLS):,} lookups")
    out_df = pd.DataFrame(rows)

    output_columns = [
        "Date", "Symbol",
        "direct_news_pos",    "direct_news_neu",    "direct_news_neg",
        "sectoral_news_pos",  "sectoral_news_neu",  "sectoral_news_neg",
        "global_news_pos",    "global_news_neu",    "global_news_neg",
        "direct_news_count",  "sectoral_news_count", "global_news_count",
    ]

    out_df = out_df[output_columns]

    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Output saved → {OUTPUT_CSV.resolve()}")
    log.info(f"Rows: {len(out_df):,}  |  Columns: {list(out_df.columns)}")

    log.info("\n--- Sample output (first 3 rows) ---")
    log.info("\n" + out_df.head(3).to_string())


if __name__ == "__main__":
    main()
