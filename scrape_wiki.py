"""CLI entry point: scrape an entire Fandom wiki using wiki_scraper."""

import asyncio
import logging
import sys

from config import Config
from wiki_scraper import run_scraper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://thehungergames.fandom.com"
    api_url = url.rstrip("/") + "/api.php"

    cfg = Config(wiki_api_url=api_url)
    saved = asyncio.run(run_scraper(cfg))
    print(f"\n{len(saved)} pages saved to {cfg.raw_dir}")


if __name__ == "__main__":
    main()
