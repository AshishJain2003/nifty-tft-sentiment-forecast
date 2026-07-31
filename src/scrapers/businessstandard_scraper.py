#!/usr/bin/env python3
"""
Business Standard scraper.

Collects date, title, news and url from paginated listing pages on
business-standard.com and streams the rows into a CSV.

Listing pages are fetched with curl_cffi (TLS impersonation gets past the bot
protection); article pages are rendered with headless Chrome because the body
copy is injected by JavaScript. Articles are parsed from JSON-LD first, with an
HTML fallback.

A run resumes after a crash: a JSON checkpoint is written after every page
(write-then-rename), and URLs already present in the CSV are loaded at startup
so nothing is scraped twice.

    conda activate MarketFusion
    python src/scrapers/businessstandard_scraper.py
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import queue
import random
import re
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from curl_cffi import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


STOP_DATE: str = "2020-01-01"
WEBSITE_NAME: str = "businessstandard"

SECTION_URLS: list[str] = [
    "https://www.business-standard.com/markets/news",
]

# Concurrency
MAX_CONCURRENT_REQUESTS: int = 15       # curl_cffi semaphore for listing pages
SELENIUM_WORKERS: int = 6              # headless Chrome instances for articles
ARTICLES_PER_BATCH: int = 6            # matches Selenium pool size

# Retry / backoff
MAX_RETRIES: int = 3
INITIAL_BACKOFF_S: float = 2.0
BACKOFF_MULTIPLIER: float = 2.0
BACKOFF_JITTER: float = 1.0

# Delays
PAGE_DELAY_S: float = 1.0
ARTICLE_DELAY_S: float = 0.5           # sleep after driver.get

# Timeout
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
log = logging.getLogger("bs_scraper")

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

    @staticmethod
    def delete(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


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
        log.info("  Loaded %d already-scraped URLs from CSV", len(urls))
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


def _setup_driver() -> webdriver.Chrome:
    """Initializes a headless Chrome driver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_cdp_cmd("Network.setUserAgentOverride", {
        "userAgent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    driver.set_page_load_timeout(30)
    return driver


class SeleniumPool:
    """Thread-safe pool of headless Chrome instances."""

    def __init__(self, size: int):
        self._size = size
        self._queue: queue.Queue[webdriver.Chrome] = queue.Queue()
        self._drivers: list[webdriver.Chrome] = []

    def start(self):
        log.info("  🖥 Initializing %d headless Chrome browsers…", self._size)
        for _ in range(self._size):
            d = _setup_driver()
            self._drivers.append(d)
            self._queue.put(d)
        log.info("  ✓ %d Chrome browsers ready", self._size)

    def acquire(self) -> webdriver.Chrome:
        return self._queue.get()

    def release(self, driver: webdriver.Chrome):
        self._queue.put(driver)

    def shutdown(self):
        for d in self._drivers:
            try:
                d.quit()
            except Exception:
                pass
        log.info("  Chrome browsers shut down")


def _scrape_article_with_selenium(driver: webdriver.Chrome, url: str) -> Optional[dict]:
    """Scrape a single article using Selenium."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            driver.get(url)
            time.sleep(ARTICLE_DELAY_S)
            soup = BeautifulSoup(driver.page_source, "html.parser")

            title = ""
            date_val = ""
            news = ""

            # Method 1: JSON-LD (very resilient)
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string, strict=False)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get("@type") in ["NewsArticle", "Article", "WebPage"] or "@graph" in item:
                            if "@graph" in item:
                                for g_item in item["@graph"]:
                                    if g_item.get("@type") in ["NewsArticle", "Article"]:
                                        item = g_item
                                        break

                            if item.get("@type") in ["NewsArticle", "Article"]:
                                title = title or item.get("headline", "")
                                date_val = date_val or item.get("datePublished", "")
                                news = news or item.get("articleBody", "")
                except Exception:
                    pass

            # Method 2: HTML parsing fallback
            if not title:
                title_tag = soup.find("h1")
                title = title_tag.text.strip() if title_tag else (
                    soup.title.string if soup.title else ""
                )

            if not date_val:
                date_meta = soup.find("meta", property="article:published_time")
                date_val = date_meta["content"] if date_meta else ""

            if not news:
                paragraphs = soup.find_all("p")
                news = "\n".join(
                    [p.text.strip() for p in paragraphs if len(p.text.strip()) > 30]
                )

            if title and news:
                # Clean date to YYYY-MM-DD
                parsed_date = _parse_date(date_val) if date_val else ""
                return {
                    "date": parsed_date,
                    "title": title,
                    "news": news.strip(),
                    "url": url,
                }

            return None

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
            else:
                pass  # Silent fail after max retries
    return None


def _parse_date(date_str: str) -> str:
    """Parse ISO-8601 date string -> YYYY-MM-DD."""
    if not date_str:
        return ""
    try:
        clean = re.sub(r"[+-]\d{2}:\d{2}$", "", date_str)
        clean = re.sub(r"\.\d{3}Z$", "", clean)
        clean = clean.replace("Z", "")
        return datetime.fromisoformat(clean).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _parse_listing_page(html: str, base_url: str) -> list[str]:
    """Extract article URLs from the MAIN content area only.

    Uses `a.smallcard-title` selector to target actual article cards,
    avoiding sidebar/trending links that repeat across every page.
    Falls back to broad regex only if the primary selector finds nothing.
    """
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    urls: list[str] = []

    # Primary: only main article cards (avoids sidebar duplicates)
    for a_tag in soup.select("a.smallcard-title[href]"):
        href = a_tag.get("href", "")
        if isinstance(href, list):
            href = href[0]
        href = href.strip()
        if href.startswith("/"):
            href = "https://www.business-standard.com" + href
        if href and re.search(r"-\d+(_\d+)?\.html$", href) and href not in seen:
            seen.add(href)
            urls.append(href)

    # Fallback: broad scan if primary found nothing
    if not urls:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ("/markets/" in href or "/article/" in href) and re.search(
                r"-\d+(_\d+)?\.html$", href
            ):
                if href.startswith("/"):
                    href = "https://www.business-standard.com" + href
                if href not in seen:
                    urls.append(href)
                    seen.add(href)

    return urls


class AsyncListingFetcher:
    """curl_cffi fetcher for listing pages only (fast, no JS needed)."""

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
        headers["Referer"] = referer or "https://www.business-standard.com/"
        return headers

    async def warmup(self) -> bool:
        log.info("  🌐 Warming up curl_cffi session…")
        try:
            headers = self._build_headers(referer="https://www.google.com/")
            headers["Sec-Fetch-Site"] = "cross-site"
            resp = await self._session.get(
                "https://www.business-standard.com/",
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
                    delay = random.uniform(0.2, 0.8)
                    await asyncio.sleep(delay)
                    headers = self._build_headers(referer=referer)
                    resp = await self._session.get(
                        url, headers=headers, timeout=REQUEST_TIMEOUT_S,
                    )
                    if resp.status_code == 200:
                        return resp.text
                    if resp.status_code == 404:
                        return None
                    raise Exception(f"HTTP {resp.status_code}")
            except Exception as exc:
                self._stats.retries_total += 1
                jitter = random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
                sleep_time = max(0.5, backoff + jitter)
                if attempt < MAX_RETRIES:
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


class BusinessStandardScraper:
    def __init__(self):
        self.stop_date: date = datetime.strptime(STOP_DATE, "%Y-%m-%d").date()
        self.stats = ScraperStats()
        self.csv_writer = CSVWriter(OUTPUT_FILE)
        self._shutdown = False
        self.checkpoint = Checkpoint.load(CHECKPOINT_FILE)
        self.scraped_urls: set[str] = load_scraped_urls(OUTPUT_FILE)

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

        # Initialize Selenium pool
        selenium_pool = SeleniumPool(SELENIUM_WORKERS)
        selenium_pool.start()

        # ThreadPoolExecutor for Selenium calls
        thread_pool = ThreadPoolExecutor(max_workers=SELENIUM_WORKERS)

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        try:
            async with requests.AsyncSession(impersonate=IMPERSONATE_BROWSER) as session:
                fetcher = AsyncListingFetcher(session, semaphore, self.stats)

                warmup_ok = await fetcher.warmup()
                if not warmup_ok:
                    log.warning("  Warmup failed — continuing anyway")
                    await asyncio.sleep(3)
                    await fetcher.warmup()

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
                        section_idx, section_url, fetcher,
                        selenium_pool, thread_pool, loop,
                        start_page=start_page,
                    )
        finally:
            thread_pool.shutdown(wait=False)
            selenium_pool.shutdown()

        await self.csv_writer.close()

        if not self._shutdown:
            log.info("✓ Checkpoint preserved — full run complete")
        else:
            self.checkpoint.save(CHECKPOINT_FILE)
            log.info("✓ Checkpoint saved — re-run to resume")

        log.info("═══ DONE ═══  %s", self.stats.summary())

    async def _scrape_section(
        self,
        section_idx: int,
        base_url: str,
        fetcher: AsyncListingFetcher,
        selenium_pool: SeleniumPool,
        thread_pool: ThreadPoolExecutor,
        loop: asyncio.AbstractEventLoop,
        start_page: int = 1,
    ):
        section_name = base_url.replace("https://www.business-standard.com/", "").strip("/")

        if start_page > 1:
            log.info("▶ Resuming section: %s  (from page %d)", section_name, start_page)
        else:
            log.info("▶ Starting section: %s", section_name)

        page_num = start_page
        consecutive_empty = 0
        section_referer = base_url + "/"

        while not self._shutdown:
            if page_num == 1:
                page_url = base_url
            else:
                page_url = f"{base_url}/page-{page_num}"

            log.info("  Page %d → %s", page_num, page_url)

            html = await fetcher.fetch(page_url, referer=section_referer)
            if html is None:
                log.warning("  Could not fetch page %d — stopping section", page_num)
                break

            # Parse listing page (CPU-light, run in-process)
            article_urls = _parse_listing_page(html, page_url)
            self.stats.pages_crawled += 1

            if not article_urls:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    log.info("  No articles on %d consecutive pages — end of section", consecutive_empty)
                    break
                log.info("  No articles on page %d — trying next", page_num)
                await asyncio.sleep(PAGE_DELAY_S)
                page_num += 1
                continue
            else:
                consecutive_empty = 0

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
                    new_urls, selenium_pool, thread_pool, loop,
                )
            else:
                should_stop = False

            # Save checkpoint
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

        if not self._shutdown:
            if section_idx not in self.checkpoint.completed_sections:
                self.checkpoint.completed_sections.append(section_idx)
            self.checkpoint.section_index = section_idx + 1
            self.checkpoint.last_completed_page = 0
            self.checkpoint.save(CHECKPOINT_FILE)

        log.info("◀ Finished section: %s  %s", section_name, self.stats.summary())

    async def _process_articles(
        self,
        urls: list[str],
        selenium_pool: SeleniumPool,
        thread_pool: ThreadPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """Scrape articles using the Selenium thread pool.

        Returns True only if the MAJORITY (>60%) of dated articles on this
        page are before STOP_DATE - prevents premature stopping caused by
        a stale sidebar link or a single re-published article.
        """
        total_dated = 0
        old_count = 0

        for i in range(0, len(urls), ARTICLES_PER_BATCH):
            if self._shutdown:
                return True

            batch = urls[i : i + ARTICLES_PER_BATCH]
            tasks = [
                self._selenium_scrape_one(url, selenium_pool, thread_pool, loop)
                for url in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            articles_to_write: list[Article] = []
            for result in results:
                if isinstance(result, Exception):
                    log.debug("Article exception: %s", result)
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

                # Track date distribution for stop-date decision
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

            if self.stats.articles_scraped % 20 == 0:
                log.info("  ├─ %s", self.stats.summary())

        # Only stop if majority of dated articles on this page are old
        if total_dated > 0:
            old_ratio = old_count / total_dated
            if old_ratio > 0.6:
                log.info(
                    "  ├─ %d/%d articles (%.0f%%) are before %s → stopping section",
                    old_count, total_dated, old_ratio * 100, STOP_DATE,
                )
                return True

        return False

    async def _selenium_scrape_one(
        self,
        url: str,
        selenium_pool: SeleniumPool,
        thread_pool: ThreadPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> Optional[Article]:
        """Worker: grab a Selenium driver, scrape the article, return it."""

        def _worker(article_url: str) -> Optional[dict]:
            driver = selenium_pool.acquire()
            try:
                return _scrape_article_with_selenium(driver, article_url)
            finally:
                selenium_pool.release(driver)

        parsed = await loop.run_in_executor(thread_pool, _worker, url)
        if parsed is None:
            return None
        return Article(**parsed)


def main():
    log.info("Business Standard Hybrid Scraper (curl_cffi + Selenium)")
    log.info("  STOP_DATE  = %s", STOP_DATE)
    log.info("  OUTPUT     = %s", OUTPUT_FILE)
    log.info("  CHECKPOINT = %s", CHECKPOINT_FILE)
    log.info("  SECTIONS   = %d", len(SECTION_URLS))
    log.info("  SELENIUM   = %d Chrome instances", SELENIUM_WORKERS)
    log.info("  SEMAPHORE  = %d (listing pages)", MAX_CONCURRENT_REQUESTS)

    scraper = BusinessStandardScraper()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(scraper.run())


if __name__ == "__main__":
    main()
