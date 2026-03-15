from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    wiki_api_url: str = ""
    raw_dir: Path = Path("raw")
    scrape_delay: float = 0.5
    max_pages: int = 0  # 0 = no limit
    namespaces: list[int] = field(default_factory=lambda: [0])  # 0 = main namespace
