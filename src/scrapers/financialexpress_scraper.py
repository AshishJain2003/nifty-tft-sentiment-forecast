#!/usr/bin/env python3
"""
Financial Express async scraper.

Collects date, title, news and url from paginated listing pages on
financialexpress.com and streams the rows into a CSV. Network I/O runs on
asyncio + aiohttp under a semaphore with exponential backoff; HTML parsing runs
in a ProcessPoolExecutor.

A run resumes after a crash: a JSON checkpoint tracks the section index and last
completed page, and URLs already present in the CSV are loaded at startup so
nothing is scraped twice. Delete the checkpoint to force a fresh start.

    conda activate MarketFusion
    python src/scrapers/financialexpress_scraper.py
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import random
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup


# Date threshold: stop paginating a section once articles fall before this date.
# Change to e.g. "2020-01-01" for deep historical scraping.
STOP_DATE: str = "2020-01-01"

# Source identifier (used in output filename)
WEBSITE_NAME: str = "financialexpress"

# Base URLs to scrape (each section is paginated independently)
SECTION_URLS: list[str] = [
    "https://www.financialexpress.com/business/page/{page}/",
    "https://www.financialexpress.com/market/page/{page}/",
]

# Concurrency knobs
MAX_CONCURRENT_REQUESTS: int = 60      # Semaphore limit for aiohttp calls
TCP_CONNECTOR_LIMIT: int = 100         # aiohttp TCPConnector pool size
TCP_CONNECTOR_PER_HOST: int = 30       # per-host connection ceiling
ARTICLES_PER_BATCH: int = 20           # articles processed concurrently per page

# Retry / backoff
MAX_RETRIES: int = 5
INITIAL_BACKOFF_S: float = 1.0
BACKOFF_MULTIPLIER: float = 2.0
BACKOFF_JITTER: float = 0.5           # random +/-jitter added to backoff

# CPU offloading
# None -> defaults to os.cpu_count()
PARSER_WORKERS: int | None = None

# Politeness delay (seconds) between successive pagination requests
PAGE_DELAY_S: float = 0.3

# Timeout per request (seconds)
REQUEST_TIMEOUT_S: int = 30

# User-Agent rotation pool
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.86 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fe_scraper")


# The script lives at  <root>/src/scrapers/financialexpress_scraper.py
# Output goes to       <root>/dataset/raw_dataset/<website>_raw.csv
SCRIPT_DIR = Path(__file__).resolve().parent             # src/scrapers/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                  # root/
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "raw_dataset"
OUTPUT_FILE = OUTPUT_DIR / f"{WEBSITE_NAME}_raw.csv"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoint"
CHECKPOINT_FILE = CHECKPOINT_DIR / f"{WEBSITE_NAME}_checkpoint.json"


@dataclass
class Checkpoint:
    """Tracks progress so the scraper can resume after crash/shutdown.

    State is saved as JSON after every fully-processed page:
        {
            "section_pages": {
                "0": 6190,    # section 0 (business) -> last completed page 6190
                "1": 1367     # section 1 (market) -> last completed page 1367
            },
            "completed_sections": [0],   # section indices that are fully done
            "version": 2
        }
    """
    section_pages: dict[int, int] = field(default_factory=dict)  # section_idx -> last_completed_page
    completed_sections: list[int] = field(default_factory=list)
    version: int = 2

    # Persistence
    def save(self, path: Path) -> None:
        """Atomically write checkpoint to disk (write-then-rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "section_pages": {str(k): v for k, v in self.section_pages.items()},
                "completed_sections": self.completed_sections,
                "version": self.version,
            }, f, indent=2)
        tmp.replace(path)   # atomic on POSIX

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        """Load checkpoint from disk, or return a fresh one if missing/corrupt."""
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            version = int(data.get("version", 1))
            
            # Handle old format (version 1) - convert to new format
            if version == 1:
                section_idx = int(data.get("section_index", 0))
                last_page = int(data.get("last_completed_page", 0))
                section_pages = {section_idx: last_page} if last_page > 0 else {}
                return cls(
                    section_pages=section_pages,
                    completed_sections=list(data.get("completed_sections", [])),
                    version=2,
                )
            
            # Handle new format (version 2)
            section_pages_raw = data.get("section_pages", {})
            section_pages = {int(k): int(v) for k, v in section_pages_raw.items()}
            
            return cls(
                section_pages=section_pages,
                completed_sections=list(data.get("completed_sections", [])),
                version=2,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            log.warning("Corrupt checkpoint file (%s) — starting fresh: %s", path, e)
            return cls()

    @staticmethod
    def delete(path: Path) -> None:
        """Remove the checkpoint file (called on successful completion)."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    
    def get_last_page(self, section_idx: int) -> int:
        """Get the last completed page for a section (0 if none)."""
        return self.section_pages.get(section_idx, 0)
    
    def set_last_page(self, section_idx: int, page_num: int) -> None:
        """Set the last completed page for a section."""
        self.section_pages[section_idx] = page_num


def load_scraped_urls(csv_path: Path) -> set[str]:
    """Read the 'url' column from an existing CSV to build a dedup set.

    Called once at startup so an article already in the output file is never
    re-scraped. The set holds URL strings only, so even 500K articles cost
    roughly 100 MB.
    """
    urls: set[str] = set()
    if not csv_path.exists():
        return urls
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("url", "").strip()
                if url:
                    urls.add(url)
    except Exception as e:
        log.warning("Could not read existing CSV for dedup: %s", e)
    if urls:
        log.info("  Loaded %d already-scraped URLs from CSV for dedup", len(urls))
    return urls


@dataclass
class Article:
    date: str
    title: str
    news: str
    url: str


@dataclass
class ScraperStats:
    """Mutable live statistics for console feedback."""
    pages_crawled: int = 0
    articles_scraped: int = 0
    articles_skipped: int = 0          # already in CSV (dedup)
    articles_failed: int = 0
    retries_total: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> str:
        secs = int(time.monotonic() - self.start_time)
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def summary(self) -> str:
        return (
            f"[{self.elapsed}] pages={self.pages_crawled}  "
            f"articles={self.articles_scraped}  "
            f"skipped={self.articles_skipped}  "
            f"failed={self.articles_failed}  "
            f"retries={self.retries_total}"
        )


def _parse_listing_page(html: str, base_url: str) -> list[str]:
    """Extract article URLs from a section listing page.

    Financial Express uses <h2><a href="..."> for article cards on listing
    pages.  We filter to keep only links that look like article permalinks
    (they contain a trailing numeric ID pattern like ``-1234567/``).
    """
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    urls: list[str] = []

    # Primary: h2 > a (article headlines on listing page)
    for a_tag in soup.select("h2 a[href]"):
        href = a_tag["href"]
        if isinstance(href, list):
            href = href[0]
        href = href.strip()
        # Must be an absolute financialexpress.com article link
        if not href.startswith("https://www.financialexpress.com/"):
            continue
        # Skip pagination / category links - real articles end with digit + /
        if not _is_article_url(href):
            continue
        if href not in seen:
            seen.add(href)
            urls.append(href)

    return urls


def _is_article_url(url: str) -> bool:
    """Heuristic: article URLs end with ``/<slug>-<numeric_id>/``."""
    # e.g. .../industry-draft-it-rule-changes-...-4190943/
    parts = url.rstrip("/").split("/")
    if not parts:
        return False
    last = parts[-1]
    # Must contain a hyphen and end with digits (article id)
    segments = last.rsplit("-", 1)
    if len(segments) == 2 and segments[1].isdigit():
        return True
    # Fallback: last segment is purely numeric (some older articles)
    if last.isdigit():
        return True
    return False


def _parse_article_page(html: str, url: str) -> Optional[dict]:
    """Extract date, title, and body text from an individual article page.

    Returns a plain dict (not Article) because this runs in a subprocess and
    must be pickle-friendly.
    """
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    else:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()

    if not title:
        return None  # unusable article

    # Date
    pub_date = ""
    meta_time = soup.find("meta", property="article:published_time")
    if meta_time and meta_time.get("content"):
        raw = meta_time["content"].strip()
        try:
            dt = datetime.fromisoformat(raw)
            pub_date = dt.strftime("%Y-%m-%d")
        except ValueError:
            pub_date = raw[:10]   # best-effort ISO prefix
    else:
        # Fallback: look for a <time> element
        time_el = soup.find("time", attrs={"datetime": True})
        if time_el:
            raw = time_el["datetime"].strip()
            try:
                dt = datetime.fromisoformat(raw)
                pub_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                pub_date = raw[:10]

    # Body content
    paragraphs: list[str] = []

    # Primary container: .wp-block-post-content or .entry-content
    content_div = (
        soup.select_one("div.wp-block-post-content")
        or soup.select_one("div.entry-content")
        or soup.select_one("div.pcl-full-content")
        or soup.select_one("article")
    )

    if content_div:
        for p in content_div.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if not text:
                continue
            # Skip "ALSO READ" / promotional lines
            lower = text.lower()
            if lower.startswith("also read") or lower.startswith("recommended"):
                continue
            paragraphs.append(text)
    else:
        # Absolute fallback: grab all <p> tags in the page
        for p in soup.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text and len(text) > 40:
                paragraphs.append(text)

    news_body = " ".join(paragraphs).strip()
    if not news_body:
        return None

    return {"date": pub_date, "title": title, "news": news_body, "url": url}


class AsyncFetcher:
    """Thin wrapper around aiohttp with semaphore + exponential backoff."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        stats: ScraperStats,
    ):
        self._session = session
        self._sem = semaphore
        self._stats = stats

    async def fetch(self, url: str) -> Optional[str]:
        """GET *url* and return response text, or None on permanent failure."""
        backoff = INITIAL_BACKOFF_S
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self._sem:
                    headers = {"User-Agent": random.choice(USER_AGENTS)}
                    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
                    async with self._session.get(
                        url, headers=headers, timeout=timeout,
                        allow_redirects=True,
                    ) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        if resp.status == 404:
                            log.debug("404 → %s", url)
                            return None
                        if resp.status in (429, 503):
                            # Rate-limited or overloaded - back off
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=f"HTTP {resp.status}",
                            )
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=f"HTTP {resp.status}",
                            )
                        # Other 4xx - don't retry
                        log.warning("HTTP %d → %s (not retrying)", resp.status, url)
                        return None

            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                self._stats.retries_total += 1
                jitter = random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
                sleep_time = max(0.1, backoff + jitter)
                if attempt < MAX_RETRIES:
                    log.debug(
                        "Retry %d/%d (%.1fs) %s ← %s",
                        attempt, MAX_RETRIES, sleep_time, url, exc,
                    )
                    await asyncio.sleep(sleep_time)
                    backoff *= BACKOFF_MULTIPLIER
                else:
                    log.warning("Gave up after %d retries → %s", MAX_RETRIES, url)
                    return None
        return None


class CSVWriter:
    """Appends Article rows to a CSV file, flushing after every write."""

    def __init__(self, filepath: Path):
        self._path = filepath
        self._lock = asyncio.Lock()
        self._file: Optional[io.TextIOWrapper] = None
        self._writer: Optional[csv.writer] = None

    async def open(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._path.exists() and self._path.stat().st_size > 0
        # Open in append mode; if brand-new file, write header
        self._file = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, quoting=csv.QUOTE_ALL)
        if not file_exists:
            self._writer.writerow(["date", "title", "news", "url"])
            self._file.flush()
        log.info("CSV output → %s", self._path)

    async def write_many(self, articles: list[Article]):
        async with self._lock:
            if self._writer and self._file:
                for a in articles:
                    self._writer.writerow([a.date, a.title, a.news, a.url])
                self._file.flush()

    async def close(self):
        if self._file:
            self._file.close()


class FinancialExpressScraper:
    """Coordinates pagination, link extraction, article parsing and CSV I/O."""

    def __init__(self):
        self.stop_date: date = datetime.strptime(STOP_DATE, "%Y-%m-%d").date()
        self.stats = ScraperStats()
        self.csv_writer = CSVWriter(OUTPUT_FILE)
        self._shutdown = False

        # Resume state
        self.checkpoint = Checkpoint.load(CHECKPOINT_FILE)
        self.scraped_urls: set[str] = load_scraped_urls(OUTPUT_FILE)

    # Graceful shutdown on Ctrl-C
    def _install_signal_handlers(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum, frame):
        log.warning("Received signal %s — finishing current batch then saving checkpoint…", signum)
        self._shutdown = True

    # Entry point
    async def run(self):
        self._install_signal_handlers()
        await self.csv_writer.open()

        # Log resume info
        if self.checkpoint.section_pages or self.checkpoint.completed_sections:
            log.info("  ↻ RESUMING from checkpoint:")
            for idx, last_page in self.checkpoint.section_pages.items():
                log.info("    Section %d → resuming from page %d", idx, last_page + 1)
            if self.checkpoint.completed_sections:
                log.info("    Completed sections: %s", self.checkpoint.completed_sections)
            log.info("    Dedup set: %d URLs", len(self.scraped_urls))
        else:
            log.info("  ☐ Fresh run (no checkpoint found)")

        connector = aiohttp.TCPConnector(
            limit=TCP_CONNECTOR_LIMIT,
            limit_per_host=TCP_CONNECTOR_PER_HOST,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async with aiohttp.ClientSession(connector=connector) as session:
            fetcher = AsyncFetcher(session, semaphore, self.stats)

            with ProcessPoolExecutor(max_workers=PARSER_WORKERS) as pool:
                loop = asyncio.get_running_loop()

                for section_idx, section_template in enumerate(SECTION_URLS):
                    if self._shutdown:
                        break

                    # Skip sections that are fully completed
                    if section_idx in self.checkpoint.completed_sections:
                        log.info(
                            "  ⏭ Skipping section %d (already completed): %s",
                            section_idx, section_template,
                        )
                        continue

                    # Get the starting page for this section from checkpoint
                    last_completed = self.checkpoint.get_last_page(section_idx)
                    start_page = last_completed + 1

                    await self._scrape_section(
                        section_idx, section_template, fetcher, pool, loop,
                        start_page=start_page,
                    )

        await self.csv_writer.close()

        # Preserve checkpoint even on successful completion
        if not self._shutdown:
            log.info("✓ Checkpoint preserved — full run complete")
        else:
            # Save final checkpoint on graceful shutdown
            self.checkpoint.save(CHECKPOINT_FILE)
            log.info("✓ Checkpoint saved — re-run to resume from where you left off")

        log.info("═══ DONE ═══  %s", self.stats.summary())

    # Section-level pagination loop
    async def _scrape_section(
        self,
        section_idx: int,
        url_template: str,
        fetcher: AsyncFetcher,
        pool: ProcessPoolExecutor,
        loop: asyncio.AbstractEventLoop,
        start_page: int = 1,
    ):
        section_name = url_template.split("financialexpress.com/")[1].split("/page/")[0]
        if start_page > 1:
            log.info("▶ Resuming section: %s  (from page %d)", section_name, start_page)
        else:
            log.info("▶ Starting section: %s", section_name)

        page_num = start_page

        while not self._shutdown:
            page_url = url_template.format(page=page_num)
            log.info("  Page %d → %s", page_num, page_url)

            html = await fetcher.fetch(page_url)
            if html is None:
                log.warning("  Could not fetch page %d — stopping section", page_num)
                break

            # Parse the listing page in a subprocess
            article_urls = await loop.run_in_executor(
                pool, _parse_listing_page, html, page_url,
            )
            self.stats.pages_crawled += 1

            if not article_urls:
                log.info("  No articles found on page %d — end of section", page_num)
                break

            # Filter out already-scraped URLs (dedup from previous run)
            new_urls = [u for u in article_urls if u not in self.scraped_urls]
            skipped = len(article_urls) - len(new_urls)
            if skipped:
                self.stats.articles_skipped += skipped
                log.info(
                    "  Found %d article URLs on page %d (%d new, %d already scraped)",
                    len(article_urls), page_num, len(new_urls), skipped,
                )
            else:
                log.info("  Found %d article URLs on page %d", len(article_urls), page_num)

            if new_urls:
                # Fetch + parse all NEW articles on this page concurrently
                should_stop = await self._process_articles(
                    new_urls, fetcher, pool, loop,
                )
            else:
                should_stop = False

            # Save checkpoint after each page
            self.checkpoint.set_last_page(section_idx, page_num)
            self.checkpoint.save(CHECKPOINT_FILE)

            if should_stop:
                log.info(
                    "  Reached STOP_DATE (%s) in section '%s' at page %d",
                    STOP_DATE, section_name, page_num,
                )
                break

            # Politeness delay before the next page
            await asyncio.sleep(PAGE_DELAY_S)
            page_num += 1

        # Mark this section as completed (unless we were interrupted)
        if not self._shutdown:
            if section_idx not in self.checkpoint.completed_sections:
                self.checkpoint.completed_sections.append(section_idx)
            self.checkpoint.save(CHECKPOINT_FILE)

        log.info("◀ Finished section: %s  %s", section_name, self.stats.summary())

    # Batch article processing
    async def _process_articles(
        self,
        urls: list[str],
        fetcher: AsyncFetcher,
        pool: ProcessPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """Fetch and parse articles concurrently.

        Returns True if the scraper should stop paginating this section
        (because a MAJORITY of articles are older than STOP_DATE).
        """
        # Track dated articles on this page
        dated_articles_count = 0
        old_articles_count = 0

        # Process in batches to avoid blasting thousands of requests at once
        for i in range(0, len(urls), ARTICLES_PER_BATCH):
            if self._shutdown:
                return True

            batch = urls[i : i + ARTICLES_PER_BATCH]
            tasks = [
                self._fetch_and_parse_one(url, fetcher, pool, loop)
                for url in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            articles_to_write: list[Article] = []
            for result in results:
                if isinstance(result, Exception):
                    log.debug("Article task raised: %s", result)
                    self.stats.articles_failed += 1
                    continue
                if result is None:
                    self.stats.articles_failed += 1
                    continue

                article: Article = result

                # Final dedup guard (race-condition safe)
                if article.url in self.scraped_urls:
                    self.stats.articles_skipped += 1
                    continue

                articles_to_write.append(article)
                self.scraped_urls.add(article.url)
                self.stats.articles_scraped += 1

                # Track articles by date for stop condition
                if article.date:
                    try:
                        art_date = datetime.strptime(article.date, "%Y-%m-%d").date()
                        dated_articles_count += 1
                        if art_date < self.stop_date:
                            old_articles_count += 1
                    except ValueError:
                        pass

            if articles_to_write:
                await self.csv_writer.write_many(articles_to_write)

            # Live progress feedback
            if self.stats.articles_scraped % 50 == 0:
                log.info("  ├─ %s", self.stats.summary())

        # Only stop if MAJORITY (>60%) of dated articles are before STOP_DATE
        if dated_articles_count > 0:
            old_ratio = old_articles_count / dated_articles_count
            if old_ratio > 0.60:
                log.info(
                    "  ├─ Stop condition met: %d/%d (%.1f%%) articles before %s",
                    old_articles_count, dated_articles_count, old_ratio * 100, STOP_DATE
                )
                return True

        return False

    async def _fetch_and_parse_one(
        self,
        url: str,
        fetcher: AsyncFetcher,
        pool: ProcessPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> Optional[Article]:
        """Fetch a single article page and parse it in a subprocess."""
        html = await fetcher.fetch(url)
        if html is None:
            return None

        parsed = await loop.run_in_executor(pool, _parse_article_page, html, url)
        if parsed is None:
            return None

        return Article(**parsed)


def main():
    # Pre-flight checks
    log.info("Financial Express Scraper starting…")
    log.info("  STOP_DATE  = %s", STOP_DATE)
    log.info("  OUTPUT     = %s", OUTPUT_FILE)
    log.info("  CHECKPOINT = %s", CHECKPOINT_FILE)
    log.info("  SECTIONS   = %d", len(SECTION_URLS))
    log.info("  WORKERS    = %s", PARSER_WORKERS or os.cpu_count())
    log.info("  SEMAPHORE  = %d", MAX_CONCURRENT_REQUESTS)
    log.info("  TCP_POOL   = %d (per-host %d)", TCP_CONNECTOR_LIMIT, TCP_CONNECTOR_PER_HOST)

    scraper = FinancialExpressScraper()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(scraper.run())


if __name__ == "__main__":
    main()
