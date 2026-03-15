"""Enumerates and downloads all pages from a MediaWiki site via its API.

Uses bulk API calls (50 pages per request) instead of one-at-a-time fetching.
"""

import asyncio
import json
import logging
from pathlib import Path

import httpx
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

# MediaWiki API allows up to 50 titles per query call
BULK_SIZE = 50


class WikiScraper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_url = cfg.wiki_api_url
        self.delay = cfg.scrape_delay
        self.max_pages = cfg.max_pages
        self.namespaces = cfg.namespaces
        self.raw_dir = cfg.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    async def get_total_pages(self, client: httpx.AsyncClient) -> int:
        resp = await client.get(self.api_url, params={
            "action": "query",
            "meta": "siteinfo",
            "siprop": "statistics",
            "format": "json",
        })
        resp.raise_for_status()
        return resp.json()["query"]["statistics"]["articles"]

    async def enumerate_pages(self, client: httpx.AsyncClient) -> list[dict]:
        """Get all page titles + IDs using allpages with continuation."""
        pages = []
        for ns in self.namespaces:
            params = {
                "action": "query",
                "list": "allpages",
                "apnamespace": ns,
                "aplimit": 500,
                "format": "json",
            }
            while True:
                resp = await client.get(self.api_url, params=params)
                resp.raise_for_status()
                data = resp.json()
                for page in data["query"]["allpages"]:
                    pages.append({"title": page["title"], "pageid": page["pageid"]})
                    if self.max_pages and len(pages) >= self.max_pages:
                        return pages
                if "continue" in data:
                    params.update(data["continue"])
                else:
                    break
                await asyncio.sleep(self.delay)
        return pages

    def _cache_path(self, title: str) -> Path:
        safe = title.replace("/", "_SLASH_").replace("\\", "_BSLASH_")
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in safe)
        return self.raw_dir / f"{safe}.json"

    async def fetch_bulk_content(self, client: httpx.AsyncClient, titles: list[str]) -> dict[str, dict]:
        """Fetch wikitext + metadata for up to 50 pages in a single API call."""
        titles_str = "|".join(titles)

        resp = await client.get(self.api_url, params={
            "action": "query",
            "titles": titles_str,
            "prop": "revisions|info|categories|links",
            "rvprop": "content",
            "rvslots": "main",
            "inprop": "url",
            "cllimit": "max",
            "pllimit": "max",
            "format": "json",
            "redirects": 1,
        })
        resp.raise_for_status()
        data = resp.json()

        results = {}
        query_pages = data.get("query", {}).get("pages", {})

        for page_id, page_info in query_pages.items():
            if int(page_id) < 0:  # missing page
                continue

            title = page_info.get("title", "")
            revisions = page_info.get("revisions", [])
            wikitext = ""
            if revisions:
                slots = revisions[0].get("slots", {})
                wikitext = slots.get("main", {}).get("*", "")
                if not wikitext:
                    # Fallback for older API format
                    wikitext = revisions[0].get("*", "")

            categories = [c.get("title", "").replace("Category:", "")
                          for c in page_info.get("categories", [])]

            # Check for disambiguation
            is_disambig = any("disambiguation" in c.lower() for c in categories)
            if is_disambig:
                logger.info("Skipping disambiguation page: %s", title)
                continue

            links = [l.get("title", "") for l in page_info.get("links", [])
                     if l.get("ns", 0) == 0]

            results[title] = {
                "title": title,
                "pageid": page_info.get("pageid"),
                "wikitext": wikitext,
                "html": "",  # not fetched in bulk mode
                "categories": categories,
                "links": links,
                "templates": [],
                "url": page_info.get("fullurl", ""),
                "last_modified": page_info.get("touched", ""),
                "namespace": page_info.get("ns", 0),
            }

        return results

    async def scrape(self, force: bool = False) -> list[Path]:
        """Scrape all pages in bulk, returning paths to cached JSON files."""
        async with httpx.AsyncClient(timeout=60) as client:
            total = await self.get_total_pages(client)
            logger.info("Wiki reports %d articles", total)

            all_pages = await self.enumerate_pages(client)
            logger.info("Enumerated %d page titles", len(all_pages))

            # Filter out already cached pages
            to_fetch = []
            saved = []
            for p in all_pages:
                cache = self._cache_path(p["title"])
                if not force and cache.exists():
                    saved.append(cache)
                else:
                    to_fetch.append(p["title"])

            if saved:
                logger.info("Skipping %d already cached pages", len(saved))

            if not to_fetch:
                logger.info("All pages already cached")
                return saved

            # Fetch in bulk batches
            pbar = tqdm(total=len(to_fetch), desc="Scraping pages")
            for i in range(0, len(to_fetch), BULK_SIZE):
                batch_titles = to_fetch[i:i + BULK_SIZE]
                try:
                    results = await self.fetch_bulk_content(client, batch_titles)
                    for title, page_data in results.items():
                        cache = self._cache_path(title)
                        cache.write_text(
                            json.dumps(page_data, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        saved.append(cache)
                except Exception:
                    logger.exception("Failed to fetch batch starting with '%s'", batch_titles[0])

                pbar.update(len(batch_titles))
                await asyncio.sleep(self.delay)

            pbar.close()
            logger.info("Scraping complete: %d pages saved", len(saved))
            return saved


async def run_scraper(cfg: Config, force: bool = False) -> list[Path]:
    scraper = WikiScraper(cfg)
    return await scraper.scrape(force=force)
