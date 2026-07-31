#!/usr/bin/env python3
"""
Economic Times historical scraper, 2020 to present.

The lazy-load API stops at roughly 140 pages, so this crawls the monthly XML
sitemaps instead: sitemap index -> monthly XML -> article URLs -> article pages.
Only URLs under TARGET_PATH_PREFIX are kept, and article bodies come from
JSON-LD with an HTML fallback. Rows are appended to the same
economictimes_raw.csv the other scrapers write.

A run resumes after a crash: a JSON checkpoint records the last completed
sitemap file, and URLs already present in the CSV are skipped at startup.

    conda activate MarketFusion
    python src/scrapers/economictimes_scraper.py
"""

from __future__ import annotations

import asyncio
import csv
import html as html_mod
import io
import json
import logging
import os
import random
import re
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
from curl_cffi.requests import AsyncSession

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ET-sitemap")

# Paths
SCRIPT_DIR   = Path(__file__).resolve().parent
RAW_DIR      = (SCRIPT_DIR / ".." / ".." / "dataset" / "raw_dataset").resolve()
CHECKPOINT_DIR = RAW_DIR / "checkpoint"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Append to the SAME CSV that the lazy-load scraper uses
OUTPUT_CSV      = RAW_DIR / "economictimes_raw.csv"
CHECKPOINT_FILE = CHECKPOINT_DIR / "economictimes_checkpoint.json"

# Config
STOP_DATE   = date(2020, 1, 1)       # do not scrape articles older than this
BASE_URL    = "https://economictimes.indiatimes.com"
SITEMAP_INDEX = (
    "https://economictimes.indiatimes.com"
    "/etstatic/sitemaps/et/news/sitemap-index.xml"
)

# Only keep articles from this section
TARGET_PATH_PREFIX = "/markets/stocks/news/"

# Concurrency
MAX_CONCURRENT_REQUESTS: int = 20
ARTICLES_PER_BATCH:       int = 25
PARSER_WORKERS:     int | None = None   # None -> os.cpu_count()

# Retry / rate
PAGE_DELAY_S: tuple[float, float] = (0.15, 0.4)
MAX_RETRIES:   int   = 5
BACKOFF_BASE:  float = 2.0
JITTER_MAX:    float = 1.0

# TLS impersonation
IMPERSONATE_BROWSER = "chrome124"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

ARTICLE_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

XML_HEADERS: dict[str, str] = {
    "Accept": "application/xml,text/xml,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
}

# Namespaces in ET sitemap XML
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


@dataclass
class Article:
    date: str
    title: str
    news: str
    url: str


@dataclass
class SitemapCheckpoint:
    """Progress tracking across sitemap files."""
    completed_sitemaps: list[str] = field(default_factory=list)
    articles_done:      int = 0


@dataclass
class Stats:
    sitemaps_done:    int = 0
    articles_scraped: int = 0
    articles_skipped: int = 0
    articles_failed:  int = 0
    full_body_count:  int = 0
    retries_total:    int = 0

    def summary(self) -> str:
        return (
            f"sitemaps={self.sitemaps_done} "
            f"scraped={self.articles_scraped} "
            f"skipped={self.articles_skipped} "
            f"failed={self.articles_failed} "
            f"full_body={self.full_body_count} "
            f"retries={self.retries_total}"
        )


def _parse_article(html: str, url: str) -> tuple[Optional[str], Optional[str]]:
    """Parse title and body from an ET article page HTML.

    Returns (title, body) or (None, None) if extraction fails.
    """
    soup = BeautifulSoup(html, "lxml")
    title: Optional[str] = None
    body:  Optional[str] = None

    # JSON-LD (highest quality)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "", strict=False)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") in ("NewsArticle", "Article"):
                    if not title:
                        title = item.get("headline") or item.get("name", "")
                    art_body = item.get("articleBody", "")
                    if art_body and len(art_body) > 100:
                        body = re.sub(r"[\n\r\t]+", " ", art_body).strip()
                        break
        except Exception:
            pass
        if body:
            break

    # Title fallback
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True).split("|")[0].strip()

    # Body fallback: ET article selectors
    if not body:
        selectors = [
            ".artText p",
            ".article-txt p",
            ".Normal p",
            "article p",
            "[data-commentid] p",
            ".story-content p",
        ]
        skip = {
            "also read", "subscribe", "click here", "download app",
            "disclaimer", "follow us", "share this", "prime",
            "unlock this", "et prime",
        }
        for sel in selectors:
            paras = soup.select(sel)
            if paras:
                texts = []
                for p in paras:
                    t = p.get_text(separator=" ", strip=True)
                    if t and len(t) > 30:
                        lower = t.lower()
                        if not any(s in lower for s in skip):
                            texts.append(t)
                if texts:
                    body = re.sub(r"[\n\r\t]+", " ", " ".join(texts)).strip()
                    break

    return (title or None, body or None)


def _load_checkpoint() -> SitemapCheckpoint:
    if CHECKPOINT_FILE.exists():
        try:
            raw = json.loads(CHECKPOINT_FILE.read_text())
            return SitemapCheckpoint(
                completed_sitemaps=raw.get("completed_sitemaps", []),
                articles_done=raw.get("articles_done", 0),
            )
        except Exception:
            pass
    return SitemapCheckpoint()


def _save_checkpoint(cp: SitemapCheckpoint) -> None:
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "completed_sitemaps": cp.completed_sitemaps,
        "articles_done":      cp.articles_done,
    }, indent=2))
    tmp.replace(CHECKPOINT_FILE)


class CsvWriter:
    FIELDS = ["date", "title", "news", "url"]

    def __init__(self, path: Path, existing_urls: set[str]) -> None:
        self._path = path
        self._existing = existing_urls
        is_new = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS,
                                      quoting=csv.QUOTE_ALL)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    async def write_many(self, articles: list[Article]) -> None:
        for art in articles:
            if art.url in self._existing:
                continue
            self._writer.writerow({
                "date":  art.date,
                "title": art.title,
                "news":  art.news,
                "url":   art.url,
            })
            self._existing.add(art.url)
        self._fh.flush()

    async def close(self) -> None:
        self._fh.close()


class AsyncFetcher:
    def __init__(self, session: AsyncSession, sem: asyncio.Semaphore,
                 stats: Stats) -> None:
        self._session = session
        self._sem     = sem
        self._stats   = stats

    def _random_ua(self) -> str:
        return random.choice(USER_AGENTS)

    async def fetch_xml(self, url: str) -> Optional[str]:
        """Fetch a sitemap XML file."""
        headers = {**XML_HEADERS, "User-Agent": self._random_ua()}
        backoff = BACKOFF_BASE
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self._sem:
                    resp = await self._session.get(
                        url, headers=headers, timeout=30,
                    )
                    if resp.status_code == 200:
                        return resp.text
                    if resp.status_code == 404:
                        return None
                    raise Exception(f"HTTP {resp.status_code}")
            except Exception as exc:
                self._stats.retries_total += 1
                jitter = random.uniform(0, JITTER_MAX)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(backoff + jitter)
                    backoff *= 2
                else:
                    log.warning("fetch_xml failed %s: %s", url, exc)
                    return None
        return None

    async def fetch_article_html(self, url: str) -> Optional[str]:
        """Fetch a full article HTML page."""
        headers = {
            **ARTICLE_HEADERS,
            "User-Agent": self._random_ua(),
            "Referer": "https://economictimes.indiatimes.com/markets/stocks/news",
        }
        backoff = BACKOFF_BASE
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self._sem:
                    delay = random.uniform(*PAGE_DELAY_S)
                    await asyncio.sleep(delay)
                    resp = await self._session.get(
                        url, headers=headers, timeout=30,
                    )
                    if resp.status_code == 200:
                        return resp.text
                    if resp.status_code in (404, 410, 451):
                        return None
                    raise Exception(f"HTTP {resp.status_code}")
            except Exception as exc:
                self._stats.retries_total += 1
                jitter = random.uniform(0, JITTER_MAX)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(backoff + jitter)
                    backoff *= 2
                else:
                    log.debug("article fetch failed %s: %s", url, exc)
                    return None
        return None


def _parse_sitemap_index(xml_text: str) -> list[str]:
    """Extract all monthly sitemap URLs from the sitemap index XML."""
    root = ET.fromstring(xml_text.encode())
    urls = []
    for sitemap in root.findall("sm:sitemap", NS):
        loc = sitemap.findtext("sm:loc", namespaces=NS)
        if loc:
            urls.append(loc.strip())
    return urls


def _parse_monthly_sitemap(xml_text: str) -> list[tuple[str, str]]:
    """Extract (url, lastmod) pairs from a monthly sitemap XML.

    Returns only articles matching TARGET_PATH_PREFIX.
    """
    root = ET.fromstring(xml_text.encode())
    results = []
    for url_el in root.findall("sm:url", NS):
        loc     = url_el.findtext("sm:loc",     namespaces=NS) or ""
        lastmod = url_el.findtext("sm:lastmod", namespaces=NS) or ""
        loc = loc.strip()
        # filter to only the stocks/news section
        if TARGET_PATH_PREFIX in loc:
            results.append((loc, lastmod.strip()))
    return results


def _sitemap_date(url: str) -> Optional[date]:
    """Extract the date a sitemap file covers from its URL.

    e.g. .../2020-January-1.xml -> date(2020, 1, 1)
         .../2025-March-2.xml   -> date(2025, 3, 1)
    """
    m = re.search(r"/(\d{4})-([A-Za-z]+)-\d+\.xml", url)
    if not m:
        return None
    year = int(m.group(1))
    month_str = m.group(2)
    try:
        month = datetime.strptime(month_str, "%B").month
        return date(year, month, 1)
    except ValueError:
        return None


def _article_date_from_lastmod(lastmod: str) -> str:
    """Convert ISO8601 lastmod to YYYY-MM-DD."""
    try:
        dt = datetime.fromisoformat(lastmod[:19])
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return lastmod[:10] if len(lastmod) >= 10 else ""


class EtSitemapScraper:
    def __init__(self) -> None:
        self._shutdown = False
        self.stats = Stats()
        self.checkpoint = _load_checkpoint()

        # Load existing URLs for dedup
        self.scraped_urls: set[str] = set()
        if OUTPUT_CSV.exists():
            try:
                with OUTPUT_CSV.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("url"):
                            self.scraped_urls.add(row["url"])
                log.info("  ✓ Loaded %d existing URLs from CSV", len(self.scraped_urls))
            except Exception as e:
                log.warning("Could not read existing CSV: %s", e)

        self.csv_writer = CsvWriter(OUTPUT_CSV, self.scraped_urls)
        self._setup_signals()

    def _setup_signals(self) -> None:
        def handler(sig, frame):
            log.info("⚠ Signal received — finishing current batch then stopping…")
            self._shutdown = True
        signal.signal(signal.SIGINT,  handler)
        signal.signal(signal.SIGTERM, handler)

    async def run(self) -> None:
        log.info("ET Sitemap-Based Historical Scraper")
        log.info("  STOP_DATE  = %s", STOP_DATE.isoformat())
        log.info("  OUTPUT     = %s", OUTPUT_CSV)
        log.info("  CHECKPOINT = %s", CHECKPOINT_FILE)
        log.info("  FILTER     = %s", TARGET_PATH_PREFIX)
        log.info("  SEMAPHORE  = %d", MAX_CONCURRENT_REQUESTS)

        if self.checkpoint.completed_sitemaps:
            log.info("  ↩ Resuming — %d sitemaps already done, %d articles so far",
                     len(self.checkpoint.completed_sitemaps),
                     self.checkpoint.articles_done)
        else:
            log.info("  ☐ Fresh run")

        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async with AsyncSession(impersonate=IMPERSONATE_BROWSER) as session:
            fetcher = AsyncFetcher(session, sem, self.stats)

            # Warm up
            try:
                await session.get(
                    "https://economictimes.indiatimes.com/",
                    timeout=20,
                )
                log.info("  ✓ Warmup OK")
            except Exception as e:
                log.warning("  Warmup failed: %s", e)

            # Step 1: Fetch sitemap index
            log.info("Fetching sitemap index…")
            index_xml = await fetcher.fetch_xml(SITEMAP_INDEX)
            if not index_xml:
                log.error("Could not fetch sitemap index. Exiting.")
                return

            all_sitemaps = _parse_sitemap_index(index_xml)
            log.info("  Found %d monthly sitemap files", len(all_sitemaps))

            # Filter to >= STOP_DATE and not already done, most-recent-first order
            pending = []
            for sm_url in all_sitemaps:
                if sm_url in self.checkpoint.completed_sitemaps:
                    continue
                sm_date = _sitemap_date(sm_url)
                if sm_date is None or sm_date < STOP_DATE:
                    continue
                pending.append((sm_date, sm_url))

            # Sort newest first (so if scraper is stopped early, newest data is captured)
            pending.sort(key=lambda x: x[0], reverse=True)
            log.info("  Pending sitemaps: %d (after dedup + date filter)", len(pending))

            # Step 2: Process each monthly sitemap
            with ProcessPoolExecutor(max_workers=PARSER_WORKERS) as pool:
                loop = asyncio.get_running_loop()

                for sm_date, sm_url in pending:
                    if self._shutdown:
                        break

                    sm_name = sm_url.split("/")[-1]
                    log.info("▶ Sitemap: %s  [%s]", sm_name, sm_date.strftime("%B %Y"))

                    # Fetch the monthly sitemap XML
                    sm_xml = await fetcher.fetch_xml(sm_url)
                    if not sm_xml:
                        log.warning("  Could not fetch %s — skipping", sm_name)
                        continue

                    # Parse article entries
                    entries = _parse_monthly_sitemap(sm_xml)
                    log.info("  Found %d target articles in sitemap", len(entries))

                    # Skip already-seen URLs
                    new_entries = [
                        (url, lm) for url, lm in entries
                        if url not in self.scraped_urls
                    ]
                    log.info("  New (not in CSV): %d", len(new_entries))

                    if new_entries:
                        await self._process_articles(new_entries, fetcher, pool, loop)

                    # Mark sitemap as done
                    self.checkpoint.completed_sitemaps.append(sm_url)
                    self.checkpoint.articles_done = self.stats.articles_scraped
                    _save_checkpoint(self.checkpoint)
                    self.stats.sitemaps_done += 1

                    log.info("  ✓ Done  │ %s", self.stats.summary())
                    await asyncio.sleep(random.uniform(0.5, 1.5))

        await self.csv_writer.close()
        log.info("═" * 60)
        log.info("FINISHED │ %s", self.stats.summary())

    async def _process_articles(
        self,
        entries: list[tuple[str, str]],
        fetcher: AsyncFetcher,
        pool: ProcessPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Fetch and parse articles in concurrent batches."""

        for i in range(0, len(entries), ARTICLES_PER_BATCH):
            if self._shutdown:
                return

            batch = entries[i: i + ARTICLES_PER_BATCH]

            # Concurrent fetch
            fetch_tasks = [fetcher.fetch_article_html(url) for url, _ in batch]
            html_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            articles_to_write: list[Article] = []

            for (url, lastmod), html_result in zip(batch, html_results):
                if url in self.scraped_urls:
                    self.stats.articles_skipped += 1
                    continue

                art_date = _article_date_from_lastmod(lastmod)

                if isinstance(html_result, str) and html_result:
                    try:
                        title, body = await loop.run_in_executor(
                            pool, _parse_article, html_result, url
                        )
                    except Exception:
                        title, body = None, None
                else:
                    title, body = None, None

                if not title:
                    # Extract from URL slug as last resort
                    slug = url.rstrip("/").rsplit("/", 1)[-1]
                    slug = slug.replace("-", " ").replace(".cms", "").strip()
                    slug = re.sub(r"articleshow\s*\d+", "", slug).strip()
                    title = slug.title() if slug else "Unknown"

                if not body:
                    self.stats.articles_failed += 1
                    continue

                self.stats.full_body_count += 1
                article = Article(
                    date=art_date,
                    title=title,
                    news=body,
                    url=url,
                )
                articles_to_write.append(article)
                # NOTE: do NOT add to scraped_urls here -
                # write_many adds to self._existing (same set) after the disk write.
                self.stats.articles_scraped += 1

            if articles_to_write:
                await self.csv_writer.write_many(articles_to_write)

            if self.stats.articles_scraped % 100 == 0 and self.stats.articles_scraped > 0:
                log.info("  ├─ %s", self.stats.summary())


if __name__ == "__main__":
    scraper = EtSitemapScraper()
    asyncio.run(scraper.run())
