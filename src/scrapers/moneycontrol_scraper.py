#!/usr/bin/env python3
"""
Moneycontrol async scraper.

Collects date, title, news and url from paginated video news listing pages on
moneycontrol.com and streams the rows into a CSV. Requests go through
`curl_cffi` for TLS/JA3 impersonation, which is what gets past the WAF; HTML
parsing runs in a ProcessPoolExecutor.

A run resumes after a crash: a JSON checkpoint is written after every page
(write-then-rename), and URLs already present in the CSV are loaded at startup
so nothing is scraped twice.

    conda activate MarketFusion
    python src/scrapers/moneycontrol_scraper.py
"""

from __future__ import annotations

import asyncio
import csv
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

from bs4 import BeautifulSoup
from curl_cffi import requests


STOP_DATE: str = "2020-01-01"
WEBSITE_NAME: str = "moneycontrol"

SECTION_URLS: list[str] = [
    "https://www.moneycontrol.com/news/videos",
]

# Concurrency knobs
MAX_CONCURRENT_REQUESTS: int = 15      # Semaphore - conservative for MC
ARTICLES_PER_BATCH: int = 8

# Retry / backoff
MAX_RETRIES: int = 5
INITIAL_BACKOFF_S: float = 2.0
BACKOFF_MULTIPLIER: float = 2.0
BACKOFF_JITTER: float = 1.0

# CPU offloading
PARSER_WORKERS: int | None = None      # None -> os.cpu_count() (12 threads)

# Delays
PAGE_DELAY_S: float = 1.0
ARTICLE_FETCH_DELAY_S: tuple[float, float] = (0.2, 0.8)

# Timeout per request (seconds)
REQUEST_TIMEOUT_S: int = 30

# curl_cffi impersonation
IMPERSONATE_BROWSER = "chrome120"

BASE_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mc_scraper")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "raw_dataset"
OUTPUT_FILE = OUTPUT_DIR / f"{WEBSITE_NAME}_raw.csv"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoint"
CHECKPOINT_FILE = CHECKPOINT_DIR / f"{WEBSITE_NAME}_checkpoint.json"


@dataclass
class Checkpoint:
    section_index: int = 0
    last_completed_page: int = 0
    completed_sections: list[int] = field(default_factory=list)
    version: int = 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "section_index": self.section_index,
                "last_completed_page": self.last_completed_page,
                "completed_sections": self.completed_sections,
                "version": self.version,
            }, f, indent=2)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                section_index=int(data.get("section_index", 0)),
                last_completed_page=int(data.get("last_completed_page", 0)),
                completed_sections=list(data.get("completed_sections", [])),
                version=int(data.get("version", 1)),
            )
        except Exception as e:
            log.warning("Corrupt checkpoint — starting fresh: %s", e)
            return cls()


def load_scraped_urls(csv_path: Path) -> set[str]:
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
    pages_crawled: int = 0
    articles_scraped: int = 0
    articles_skipped: int = 0
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


# Since we only scrape from /news/videos/page-N/ listing pages,
# every article link on these pages IS a video - regardless of its URL path.
# Older video articles (pre-2022) use category paths like /news/business/stocks/
# instead of /news/videos/, so we match /news/ and /world/ paths broadly.
_MC_ARTICLE_RE = re.compile(
    r"^https?://www\.moneycontrol\.com/(news|world)/.+?-\d+\.html$"
)

# Exclude non-article patterns (liveblogs, tags, promos, etc.)
_MC_EXCLUDE_RE = re.compile(
    r"(liveblog|/topic/|/tag/|/author/|/photos/|/microsite/|/podcast/)"
)


def _parse_listing_page(html: str, base_url: str) -> list[str]:
    """Extract article URLs from a Moneycontrol video listing page.

    Since every link on the /news/videos/ listing IS a video,
    we match broadly by domain + numeric-ID.html pattern.
    """
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    urls: list[str] = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if isinstance(href, list):
            href = href[0]
        href = href.strip()
        if href.startswith("/"):
            href = "https://www.moneycontrol.com" + href
        if (
            _MC_ARTICLE_RE.match(href)
            and not _MC_EXCLUDE_RE.search(href)
            and href not in seen
        ):
            seen.add(href)
            urls.append(href)

    return urls


def _parse_date_str(date_str: str) -> str:
    """Parse various MC date formats to YYYY-MM-DD."""
    if not date_str:
        return ""
    try:
        clean = re.sub(r"[+-]\d{2}:\d{2}$", "", date_str)
        clean = re.sub(r"\.\d{3}Z$", "", clean)
        clean = clean.replace("Z", "")
        return datetime.fromisoformat(clean).strftime("%Y-%m-%d")
    except ValueError:
        pass
    for fmt in [
        "%B %d, %Y %I:%M %p IST", "%B %d, %Y %I:%M %p",
        "%b %d, %Y %I:%M %p IST", "%b %d, %Y %I:%M %p",
        "%b %d, %Y", "%B %d, %Y",
    ]:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def _parse_article_page(html: str, url: str) -> Optional[dict]:
    """Extract date, title, and body from a Moneycontrol article page.

    Strategy: JSON-LD (most reliable) -> HTML fallback
    """
    soup = BeautifulSoup(html, "lxml")
    title = ""
    date_val = ""
    news = ""

    # JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string, strict=False)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") in [
                    "NewsArticle", "Article", "VideoObject", "WebPage",
                ] or "@graph" in item:
                    if "@graph" in item:
                        for g_item in item["@graph"]:
                            if g_item.get("@type") in ["NewsArticle", "Article", "VideoObject"]:
                                item = g_item
                                break
                    if item.get("@type") in ["NewsArticle", "Article", "VideoObject"]:
                        title = title or item.get("headline", "") or item.get("name", "")
                        date_val = date_val or item.get("datePublished", "")
                        news = news or item.get("articleBody", "") or item.get("description", "")
        except Exception:
            pass

    # HTML Fallback
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        else:
            og = soup.find("meta", property="og:title")
            if og and og.get("content"):
                title = og["content"].strip()

    if not date_val:
        for prop in ["article:published_time", "article:published"]:
            meta = soup.find("meta", property=prop)
            if meta and meta.get("content"):
                date_val = meta["content"].strip()
                break
    if not date_val:
        time_el = soup.find("time", attrs={"datetime": True})
        if time_el:
            date_val = time_el["datetime"].strip()

    if not news:
        desc_div = soup.select_one(".video-desc, .article_desc, .content_wrapper p")
        if desc_div:
            news = desc_div.get_text(separator=" ", strip=True)
    if not news:
        skip_phrases = ["also read", "subscribe", "click here", "download", "disclaimer", "follow us"]
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text and len(text) > 30:
                if not any(skip in text.lower() for skip in skip_phrases):
                    paragraphs.append(text)
        news = " ".join(paragraphs)
    if not news:
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            news = og_desc["content"].strip()

    if not title or not news:
        return None

    parsed_date = _parse_date_str(date_val) if date_val else ""

    # Normalize whitespace: collapse all newlines/tabs into single spaces
    title = re.sub(r"[\n\r\t]+", " ", title).strip()
    news = re.sub(r"[\n\r\t]+", " ", news).strip()

    return {"date": parsed_date, "title": title, "news": news, "url": url}


class AsyncFetcher:
    def __init__(
        self,
        session: requests.AsyncSession,
        semaphore: asyncio.Semaphore,
        stats: ScraperStats,
    ):
        self._session = session
        self._sem = semaphore
        self._stats = stats

    def _build_headers(self, referer: str | None = None) -> dict[str, str]:
        headers = dict(BASE_HEADERS)
        headers["Referer"] = referer or "https://www.moneycontrol.com/"
        return headers

    async def warmup(self) -> bool:
        log.info("  🌐 Warming up session via curl_cffi…")
        try:
            headers = self._build_headers(referer="https://www.google.com/")
            headers["Sec-Fetch-Site"] = "cross-site"
            resp = await self._session.get(
                "https://www.moneycontrol.com/",
                headers=headers,
                timeout=REQUEST_TIMEOUT_S,
            )
            if resp.status_code == 200:
                log.info("  ✓ Warmup OK — cookies seeded")
                return True
            else:
                log.warning("  ✗ Warmup got HTTP %d", resp.status_code)
                return False
        except Exception as e:
            log.warning("  ✗ Warmup failed: %s", e)
            return False

    async def fetch(self, url: str, referer: str | None = None) -> Optional[str]:
        backoff = INITIAL_BACKOFF_S
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self._sem:
                    delay = random.uniform(*ARTICLE_FETCH_DELAY_S)
                    await asyncio.sleep(delay)
                    headers = self._build_headers(referer=referer)
                    resp = await self._session.get(
                        url, headers=headers, timeout=REQUEST_TIMEOUT_S,
                    )
                    if resp.status_code == 200:
                        return resp.text
                    if resp.status_code == 404:
                        return None
                    if resp.status_code in (403, 429, 503) or resp.status_code >= 500:
                        raise Exception(f"HTTP {resp.status_code}")
                    log.warning("HTTP %d → %s (not retrying)", resp.status_code, url)
                    return None
            except Exception as exc:
                self._stats.retries_total += 1
                jitter = random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
                sleep_time = max(0.5, backoff + jitter)
                if attempt < MAX_RETRIES:
                    log.debug("Retry %d/%d (%.1fs) %s ← %s", attempt, MAX_RETRIES, sleep_time, url, exc)
                    await asyncio.sleep(sleep_time)
                    backoff *= BACKOFF_MULTIPLIER
                else:
                    log.warning("Gave up after %d retries → %s", MAX_RETRIES, url)
                    return None
        return None


class CSVWriter:
    def __init__(self, filepath: Path):
        self._path = filepath
        self._lock = asyncio.Lock()
        self._file: Optional[io.TextIOWrapper] = None
        self._writer: Optional[csv.writer] = None

    async def open(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._path.exists() and self._path.stat().st_size > 0
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


class MoneycontrolScraper:
    def __init__(self):
        self.stop_date: date = datetime.strptime(STOP_DATE, "%Y-%m-%d").date()
        self.stats = ScraperStats()
        self.csv_writer = CSVWriter(OUTPUT_FILE)
        self._shutdown = False
        self.checkpoint = Checkpoint.load(CHECKPOINT_FILE)
        self.scraped_urls: set[str] = load_scraped_urls(OUTPUT_FILE)
        self._section_had_data: bool = False  # track if section actually scraped data

    def _install_signal_handlers(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle_signal)
            except NotImplementedError:
                pass

    def _handle_signal(self, signum, frame):
        log.warning("Signal %s — finishing batch then saving checkpoint…", signum)
        self._shutdown = True

    async def run(self):
        self._install_signal_handlers()
        await self.csv_writer.open()

        if self.checkpoint.last_completed_page > 0 or self.checkpoint.completed_sections:
            log.info(
                "  ↻ RESUMING section=%d, page=%d  (done: %s, %d dedup URLs)",
                self.checkpoint.section_index,
                self.checkpoint.last_completed_page + 1,
                self.checkpoint.completed_sections,
                len(self.scraped_urls),
            )
        else:
            log.info("  ☐ Fresh run (no checkpoint found)")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async with requests.AsyncSession(impersonate=IMPERSONATE_BROWSER) as session:
            fetcher = AsyncFetcher(session, semaphore, self.stats)

            warmup_ok = await fetcher.warmup()
            if not warmup_ok:
                log.warning("  Warmup failed — retrying once…")
                await asyncio.sleep(3)
                await fetcher.warmup()

            with ProcessPoolExecutor(max_workers=PARSER_WORKERS) as pool:
                loop = asyncio.get_running_loop()

                for section_idx, section_url in enumerate(SECTION_URLS):
                    if self._shutdown:
                        break

                    if section_idx in self.checkpoint.completed_sections:
                        log.info("  ⏭ Skipping section %d (done): %s", section_idx, section_url)
                        continue

                    if section_idx == self.checkpoint.section_index:
                        start_page = self.checkpoint.last_completed_page + 1
                    elif section_idx > self.checkpoint.section_index:
                        start_page = 1
                    else:
                        log.info("  ⏭ Skipping section %d (before checkpoint)", section_idx)
                        continue

                    await self._scrape_section(
                        section_idx, section_url, fetcher, pool, loop,
                        start_page=start_page,
                    )

        await self.csv_writer.close()

        self.checkpoint.save(CHECKPOINT_FILE)
        if not self._shutdown:
            log.info("✓ Checkpoint preserved — full run complete")
        else:
            log.info("✓ Checkpoint saved — re-run to resume")

        log.info("═══ DONE ═══  %s", self.stats.summary())

    async def _scrape_section(
        self,
        section_idx: int,
        base_url: str,
        fetcher: AsyncFetcher,
        pool: ProcessPoolExecutor,
        loop: asyncio.AbstractEventLoop,
        start_page: int = 1,
    ):
        section_name = base_url.replace("https://www.moneycontrol.com/", "").strip("/")
        self._section_had_data = False

        if start_page > 1:
            log.info("▶ Resuming section: %s  (from page %d)", section_name, start_page)
        else:
            log.info("▶ Starting section: %s", section_name)

        page_num = start_page
        consecutive_empty = 0

        while not self._shutdown:
            page_url = f"{base_url}/page-{page_num}/"
            log.info("  Page %d → %s", page_num, page_url)

            html = await fetcher.fetch(page_url, referer=base_url + "/")
            if html is None:
                log.warning("  Could not fetch page %d — stopping section", page_num)
                break

            article_urls = await loop.run_in_executor(
                pool, _parse_listing_page, html, page_url,
            )
            self.stats.pages_crawled += 1

            if not article_urls:
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    log.info("  No articles on %d consecutive pages — end of section", consecutive_empty)
                    break
                log.info("  No articles on page %d — trying next", page_num)
                await asyncio.sleep(PAGE_DELAY_S)
                page_num += 1
                continue
            else:
                consecutive_empty = 0
                self._section_had_data = True

            # Dedup
            new_urls = [u for u in article_urls if u not in self.scraped_urls]
            skipped = len(article_urls) - len(new_urls)
            if skipped:
                self.stats.articles_skipped += skipped
                log.info(
                    "  Found %d URLs on page %d (%d new, %d already done)",
                    len(article_urls), page_num, len(new_urls), skipped,
                )
            else:
                log.info("  Found %d article URLs on page %d", len(article_urls), page_num)

            if new_urls:
                should_stop = await self._process_articles(
                    new_urls, fetcher, pool, loop,
                )
            else:
                should_stop = False

            # Checkpoint
            self.checkpoint.section_index = section_idx
            self.checkpoint.last_completed_page = page_num
            self.checkpoint.save(CHECKPOINT_FILE)

            if should_stop:
                log.info(
                    "  Reached STOP_DATE (%s) in '%s' at page %d",
                    STOP_DATE, section_name, page_num,
                )
                break

            await asyncio.sleep(PAGE_DELAY_S)
            page_num += 1

        # Only mark section complete if we actually scraped data
        if not self._shutdown and self._section_had_data:
            if section_idx not in self.checkpoint.completed_sections:
                self.checkpoint.completed_sections.append(section_idx)
            self.checkpoint.section_index = section_idx + 1
            self.checkpoint.last_completed_page = 0
            self.checkpoint.save(CHECKPOINT_FILE)

        log.info("◀ Finished section: %s  %s", section_name, self.stats.summary())

    async def _process_articles(
        self,
        urls: list[str],
        fetcher: AsyncFetcher,
        pool: ProcessPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        total_dated = 0
        old_count = 0

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
                    self.stats.articles_failed += 1
                    continue
                if result is None:
                    self.stats.articles_failed += 1
                    continue

                article: Article = result
                if article.url in self.scraped_urls:
                    self.stats.articles_skipped += 1
                    continue

                articles_to_write.append(article)
                self.scraped_urls.add(article.url)
                self.stats.articles_scraped += 1

                if article.date:
                    try:
                        art_date = datetime.strptime(article.date, "%Y-%m-%d").date()
                        total_dated += 1
                        if art_date < self.stop_date:
                            old_count += 1
                    except ValueError:
                        pass

            if articles_to_write:
                await self.csv_writer.write_many(articles_to_write)

            if self.stats.articles_scraped % 50 == 0:
                log.info("  ├─ %s", self.stats.summary())

        if total_dated > 0:
            old_ratio = old_count / total_dated
            if old_ratio > 0.6:
                log.info(
                    "  ├─ %d/%d articles (%.0f%%) are before %s → stopping section",
                    old_count, total_dated, old_ratio * 100, STOP_DATE,
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
        html = await fetcher.fetch(url, referer="https://www.moneycontrol.com/news/videos/")
        if html is None:
            return None
        parsed = await loop.run_in_executor(pool, _parse_article_page, html, url)
        if parsed is None:
            return None
        return Article(**parsed)


def main():
    log.info("Moneycontrol Scraper (curl_cffi + TLS impersonation)")
    log.info("  STOP_DATE  = %s", STOP_DATE)
    log.info("  OUTPUT     = %s", OUTPUT_FILE)
    log.info("  CHECKPOINT = %s", CHECKPOINT_FILE)
    log.info("  SECTIONS   = %d", len(SECTION_URLS))
    log.info("  WORKERS    = %s", PARSER_WORKERS or os.cpu_count())
    log.info("  SEMAPHORE  = %d", MAX_CONCURRENT_REQUESTS)

    scraper = MoneycontrolScraper()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(scraper.run())


if __name__ == "__main__":
    main()
