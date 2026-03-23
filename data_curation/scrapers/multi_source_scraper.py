"""Multi-source scraper for building training datasets.

Pulls from arxiv, github, stackoverflow, huggingface, + arbitrary URLs.
Uses async everywhere because scraping is 99% waiting on network.

Usage:
    result = asyncio.run(run_scraping(config))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

import aiohttp
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)


# Data models

@dataclass
class ScrapedDocument:
    """Single scraped document + provenance info."""
    source: str          # arxiv, github, stackoverflow, etc
    source_id: str       # unique within source
    url: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "scraped_at": self.scraped_at,
            "content_hash": self.content_hash,
        }


# Rate limiter

class AdaptiveRateLimiter:
    """
    Token-bucket rate limiter with adaptive backoff on 429s.
    Thread-safe via asyncio locks.
    """

    def __init__(self, requests_per_second: float = 2.0, burst: int = 5):
        self.rate = requests_per_second
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._backoff_until = 0.0
        self._consecutive_429s = 0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()

            # Wait out backoff period
            if now < self._backoff_until:
                wait_time = self._backoff_until - now
                log.debug(f"Rate limiter: backing off for {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                now = time.monotonic()

            # Refill tokens
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
                self._last_refill = time.monotonic()
            else:
                self._tokens -= 1.0

    def report_429(self) -> None:
        """Signal a 429 response to increase backoff."""
        self._consecutive_429s += 1
        backoff = min(2 ** self._consecutive_429s, 60)
        self._backoff_until = time.monotonic() + backoff
        log.warning(f"Rate limited (429). Backing off {backoff}s (consecutive: {self._consecutive_429s})")

    def report_success(self) -> None:
        self._consecutive_429s = 0


# Scraper state (for resumability)

class ScraperState:
    """Persists scraper progress to allow resumption after interruption."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._state: dict = {}
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            with open(self.state_file) as f:
                self._state = json.load(f)

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def get_checkpoint(self, source: str) -> Optional[dict]:
        return self._state.get(source)

    def set_checkpoint(self, source: str, checkpoint: dict) -> None:
        self._state[source] = checkpoint
        self.save()

    def mark_complete(self, source: str) -> None:
        self._state[source] = {"complete": True}
        self.save()

    def is_complete(self, source: str) -> bool:
        return self._state.get(source, {}).get("complete", False)


# Base scraper class

class BaseScraper(ABC):
    """Abstract base for all source-specific scrapers."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        rate_limiter: AdaptiveRateLimiter,
        state: ScraperState,
        output_dir: Path,
    ):
        self.session = session
        self.rate_limiter = rate_limiter
        self.state = state
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def scrape(self, config: dict) -> AsyncIterator[ScrapedDocument]:
        """Yield scraped documents from the source."""
        ...

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _fetch(self, url: str, **kwargs) -> dict | str:
        """Fetch a URL with rate limiting and retry logic."""
        await self.rate_limiter.acquire()

        async with self.session.get(url, **kwargs) as resp:
            if resp.status == 429:
                self.rate_limiter.report_429()
                raise aiohttp.ClientError("Rate limited")
            resp.raise_for_status()
            self.rate_limiter.report_success()

            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type:
                return await resp.json()
            return await resp.text()

    async def _save_batch(self, docs: list[ScrapedDocument], batch_id: int) -> Path:
        """Save a batch of documents to JSONL."""
        outfile = self.output_dir / f"{self.__class__.__name__.lower()}_{batch_id:06d}.jsonl"
        async with aiofiles.open(outfile, "w") as f:
            for doc in docs:
                await f.write(json.dumps(doc.to_dict()) + "\n")
        log.info(f"Saved batch {batch_id}: {len(docs)} docs → {outfile}")
        return outfile


# ArXiv Scraper

class ArXivScraper(BaseScraper):
    """Scrapes papers from ArXiv API with pagination."""

    API_BASE = "http://export.arxiv.org/api/query"

    async def scrape(self, config: dict) -> AsyncIterator[ScrapedDocument]:
        query = config["query"]
        max_papers = config.get("max_papers", 1000)
        date_range = config.get("date_range", [])
        source_name = "arxiv"

        if self.state.is_complete(source_name):
            log.info("ArXiv scraping already complete, skipping")
            return

        checkpoint = self.state.get_checkpoint(source_name) or {"offset": 0, "total": 0}
        offset = checkpoint["offset"]
        batch_size = 100
        total_fetched = checkpoint["total"]

        while total_fetched < max_papers:
            params = {
                "search_query": f"all:{query}",
                "start": offset,
                "max_results": min(batch_size, max_papers - total_fetched),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            try:
                import xml.etree.ElementTree as ET

                text = await self._fetch(self.API_BASE, params=params)
                root = ET.fromstring(text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                entries = root.findall("atom:entry", ns)
                if not entries:
                    break

                for entry in entries:
                    title = entry.find("atom:title", ns).text.strip()
                    summary = entry.find("atom:summary", ns).text.strip()
                    arxiv_id = entry.find("atom:id", ns).text.strip().split("/")[-1]
                    published = entry.find("atom:published", ns).text.strip()
                    authors = [
                        a.find("atom:name", ns).text
                        for a in entry.findall("atom:author", ns)
                    ]
                    categories = [
                        c.attrib["term"]
                        for c in entry.findall("atom:category", ns)
                    ]
                    links = {
                        link.attrib.get("title", "alternate"): link.attrib["href"]
                        for link in entry.findall("atom:link", ns)
                    }

                    # Date filtering
                    if date_range:
                        pub_date = published[:10]
                        if pub_date < date_range[0] or pub_date > date_range[1]:
                            continue

                    content = f"Title: {title}\n\nAbstract: {summary}"

                    yield ScrapedDocument(
                        source="arxiv",
                        source_id=arxiv_id,
                        url=links.get("alternate", f"https://arxiv.org/abs/{arxiv_id}"),
                        title=title,
                        content=content,
                        metadata={
                            "authors": authors,
                            "categories": categories,
                            "published": published,
                            "pdf_url": links.get("pdf", ""),
                        },
                    )
                    total_fetched += 1

                offset += len(entries)
                self.state.set_checkpoint(source_name, {"offset": offset, "total": total_fetched})

                # ArXiv asks for 3s delay between requests
                await asyncio.sleep(3)

            except Exception as e:
                log.error(f"ArXiv scrape error at offset {offset}: {e}")
                raise

        self.state.mark_complete(source_name)
        log.info(f"ArXiv scraping complete: {total_fetched} papers")


# GitHub Scraper

class GitHubScraper(BaseScraper):
    """Scrapes issues, discussions, and README content from GitHub repos."""

    API_BASE = "https://api.github.com"

    async def scrape(self, config: dict) -> AsyncIterator[ScrapedDocument]:
        repos = config.get("repos", [])
        include_issues = config.get("include_issues", True)
        include_discussions = config.get("include_discussions", True)
        max_items = config.get("max_items_per_repo", 1000)

        headers = {}
        token = config.get("token") or __import__("os").environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        headers["Accept"] = "application/vnd.github.v3+json"

        for repo in repos:
            source_key = f"github_{repo.replace('/', '_')}"
            if self.state.is_complete(source_key):
                log.info(f"GitHub {repo} already complete, skipping")
                continue

            # Issues
            if include_issues:
                page = 1
                fetched = 0
                while fetched < max_items:
                    url = f"{self.API_BASE}/repos/{repo}/issues"
                    params = {
                        "state": "all",
                        "per_page": 100,
                        "page": page,
                        "sort": "updated",
                        "direction": "desc",
                    }

                    try:
                        data = await self._fetch(url, params=params, headers=headers)
                        if not data:
                            break

                        for issue in data:
                            if issue.get("pull_request"):
                                continue  # Skip PRs

                            body = issue.get("body", "") or ""
                            comments_text = ""

                            # Fetch top comments
                            if issue.get("comments", 0) > 0:
                                comments_url = issue["comments_url"]
                                comments = await self._fetch(
                                    comments_url,
                                    params={"per_page": 10},
                                    headers=headers,
                                )
                                if isinstance(comments, list):
                                    comments_text = "\n\n---\n\n".join(
                                        f"**{c.get('user', {}).get('login', 'unknown')}**: {c.get('body', '')}"
                                        for c in comments[:10]
                                    )

                            content = (
                                f"Issue: {issue['title']}\n\n"
                                f"Labels: {', '.join(l['name'] for l in issue.get('labels', []))}\n\n"
                                f"Body:\n{body}\n\n"
                                f"Comments:\n{comments_text}"
                            )

                            yield ScrapedDocument(
                                source="github",
                                source_id=f"{repo}/issues/{issue['number']}",
                                url=issue["html_url"],
                                title=issue["title"],
                                content=content,
                                metadata={
                                    "repo": repo,
                                    "type": "issue",
                                    "state": issue["state"],
                                    "labels": [l["name"] for l in issue.get("labels", [])],
                                    "created_at": issue["created_at"],
                                    "updated_at": issue["updated_at"],
                                    "reactions": issue.get("reactions", {}).get("total_count", 0),
                                },
                            )
                            fetched += 1

                        page += 1
                    except Exception as e:
                        log.error(f"GitHub issues error for {repo} page {page}: {e}")
                        break

            self.state.mark_complete(source_key)


# StackOverflow Scraper

class StackOverflowScraper(BaseScraper):
    """Scrapes Q&A pairs from StackOverflow via the API."""

    API_BASE = "https://api.stackexchange.com/2.3"

    async def scrape(self, config: dict) -> AsyncIterator[ScrapedDocument]:
        tags = config.get("tags", [])
        max_questions = config.get("max_questions", 1000)
        min_score = config.get("min_score", 5)
        source_name = "stackoverflow"

        if self.state.is_complete(source_name):
            return

        page = 1
        fetched = 0
        tag_str = ";".join(tags)

        while fetched < max_questions:
            url = f"{self.API_BASE}/questions"
            params = {
                "tagged": tag_str,
                "site": "stackoverflow",
                "sort": "votes",
                "order": "desc",
                "pagesize": 100,
                "page": page,
                "filter": "withbody",
                "min": min_score,
            }
            key = config.get("api_key") or __import__("os").environ.get("SO_API_KEY")
            if key:
                params["key"] = key

            try:
                data = await self._fetch(url, params=params)
                items = data.get("items", [])
                if not items:
                    break

                for q in items:
                    # Fetch accepted/top answers
                    answers_url = f"{self.API_BASE}/questions/{q['question_id']}/answers"
                    answers_data = await self._fetch(
                        answers_url,
                        params={
                            "site": "stackoverflow",
                            "sort": "votes",
                            "order": "desc",
                            "pagesize": 5,
                            "filter": "withbody",
                        },
                    )
                    answers = answers_data.get("items", [])

                    answers_text = "\n\n---\n\n".join(
                        f"[Score: {a['score']}{'  ✓ Accepted' if a.get('is_accepted') else ''}]\n{a.get('body', '')}"
                        for a in answers
                    )

                    content = (
                        f"Question: {q['title']}\n"
                        f"Tags: {', '.join(q.get('tags', []))}\n"
                        f"Score: {q['score']}\n\n"
                        f"{q.get('body', '')}\n\n"
                        f"{'=' * 40}\nAnswers:\n{answers_text}"
                    )

                    yield ScrapedDocument(
                        source="stackoverflow",
                        source_id=str(q["question_id"]),
                        url=q.get("link", f"https://stackoverflow.com/q/{q['question_id']}"),
                        title=q["title"],
                        content=content,
                        metadata={
                            "tags": q.get("tags", []),
                            "score": q["score"],
                            "view_count": q.get("view_count", 0),
                            "answer_count": q.get("answer_count", 0),
                            "is_answered": q.get("is_answered", False),
                        },
                    )
                    fetched += 1

                page += 1
                if not data.get("has_more", False):
                    break

            except Exception as e:
                log.error(f"StackOverflow error page {page}: {e}")
                break

        self.state.mark_complete(source_name)
        log.info(f"StackOverflow scraping complete: {fetched} Q&A pairs")


# HuggingFace Dataset Sampler

class HuggingFaceDatasetScraper(BaseScraper):
    """Samples from existing HuggingFace datasets."""

    async def scrape(self, config: dict) -> AsyncIterator[ScrapedDocument]:
        from datasets import load_dataset

        datasets_list = config.get("datasets", [])
        sample_size = config.get("sample_size_per_dataset", 10000)

        for ds_name in datasets_list:
            source_key = f"hf_{ds_name.replace('/', '_')}"
            if self.state.is_complete(source_key):
                log.info(f"HF dataset {ds_name} already complete, skipping")
                continue

            try:
                log.info(f"Loading HuggingFace dataset: {ds_name}")
                ds = load_dataset(ds_name, split="train", streaming=True)

                count = 0
                for example in ds:
                    if count >= sample_size:
                        break

                    # Adaptive content extraction - handle various dataset formats
                    content_fields = ["text", "instruction", "input", "output", "response",
                                      "question", "answer", "context", "conversations"]
                    content_parts = []
                    for field_name in content_fields:
                        if field_name in example and example[field_name]:
                            val = example[field_name]
                            if isinstance(val, list):
                                # Handle conversation format
                                for turn in val:
                                    if isinstance(turn, dict):
                                        role = turn.get("from", turn.get("role", ""))
                                        text = turn.get("value", turn.get("content", ""))
                                        content_parts.append(f"[{role}]: {text}")
                            else:
                                content_parts.append(f"{field_name}: {val}")

                    if not content_parts:
                        continue

                    content = "\n\n".join(content_parts)

                    yield ScrapedDocument(
                        source="huggingface",
                        source_id=f"{ds_name}/{count}",
                        url=f"https://huggingface.co/datasets/{ds_name}",
                        title=f"{ds_name} - Example {count}",
                        content=content,
                        metadata={
                            "dataset": ds_name,
                            "index": count,
                            "fields": list(example.keys()),
                        },
                    )
                    count += 1

                self.state.mark_complete(source_key)
                log.info(f"Sampled {count} examples from {ds_name}")

            except Exception as e:
                log.error(f"Error loading HF dataset {ds_name}: {e}")


# Custom URL Crawler

class CustomCrawler(BaseScraper):
    """Crawls custom URLs with configurable depth."""

    async def scrape(self, config: dict) -> AsyncIterator[ScrapedDocument]:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse

        urls_file = config.get("urls_file")
        max_depth = config.get("depth", 2)
        respect_robots = config.get("respect_robots", True)

        if not urls_file or not Path(urls_file).exists():
            log.warning(f"URLs file not found: {urls_file}")
            return

        with open(urls_file) as f:
            seed_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        visited = set()
        queue = [(url, 0) for url in seed_urls]  # (url, depth)

        while queue:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue

            visited.add(url)

            try:
                html = await self._fetch(url)
                soup = BeautifulSoup(html, "html.parser")

                # Remove script/style elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)
                title = soup.title.string if soup.title else urlparse(url).path

                if len(text) < 100:
                    continue

                yield ScrapedDocument(
                    source="custom_crawl",
                    source_id=hashlib.md5(url.encode()).hexdigest(),
                    url=url,
                    title=title or url,
                    content=text,
                    metadata={
                        "depth": depth,
                        "domain": urlparse(url).netloc,
                    },
                )

                # Extract links for further crawling
                if depth < max_depth:
                    for link in soup.find_all("a", href=True):
                        next_url = urljoin(url, link["href"])
                        parsed = urlparse(next_url)
                        if parsed.scheme in ("http", "https") and next_url not in visited:
                            queue.append((next_url, depth + 1))

            except Exception as e:
                log.warning(f"Failed to crawl {url}: {e}")


# Orchestrator

SCRAPER_REGISTRY = {
    "arxiv": ArXivScraper,
    "github": GitHubScraper,
    "stackoverflow": StackOverflowScraper,
    "huggingface_datasets": HuggingFaceDatasetScraper,
    "custom_crawl": CustomCrawler,
}


class ScrapingOrchestrator:
    """Coordinates all scrapers, manages batching and persistence."""

    def __init__(self, config: dict, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = ScraperState(self.output_dir / ".scraper_state.json")
        self.rate_limiter = AdaptiveRateLimiter(
            requests_per_second=config.get("rate_limit_per_second", 2.0)
        )
        self.stats = {"total": 0, "by_source": {}}

    async def run(self) -> dict:
        """Execute all configured scrapers and return statistics."""
        connector = aiohttp.TCPConnector(limit=self.config.get("num_workers", 8))
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for source_config in self.config.get("sources", []):
                source_type = source_config["type"]
                scraper_cls = SCRAPER_REGISTRY.get(source_type)

                if not scraper_cls:
                    log.warning(f"Unknown source type: {source_type}")
                    continue

                scraper = scraper_cls(
                    session=session,
                    rate_limiter=self.rate_limiter,
                    state=self.state,
                    output_dir=self.output_dir / source_type,
                )

                log.info(f"Starting scraper: {source_type}")
                batch = []
                batch_id = 0
                source_count = 0

                async for doc in scraper.scrape(source_config):
                    batch.append(doc)
                    source_count += 1

                    if len(batch) >= 500:
                        await scraper._save_batch(batch, batch_id)
                        batch_id += 1
                        batch = []

                # Save remaining
                if batch:
                    await scraper._save_batch(batch, batch_id)

                self.stats["by_source"][source_type] = source_count
                self.stats["total"] += source_count
                log.info(f"Completed {source_type}: {source_count} documents")

        # Save final stats
        stats_file = self.output_dir / "scraping_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)

        return self.stats


async def run_scraping(config: dict) -> dict:
    """Entry point for the scraping stage."""
    orchestrator = ScrapingOrchestrator(
        config=config["data_curation"]["scraping"],
        output_dir=config["data_curation"]["scraping"]["output_dir"],
    )
    return await orchestrator.run()
