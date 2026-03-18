"""Wiki registry — manages multiple wiki databases under wikis/.

Each wiki gets its own folder with raw data, parsed files, chunks, and ChromaDB:

    wikis/
      hunger_games/
        meta.json
        raw/ parsed/ chunks/ infobox/ summary/ graph/ chromadb_store/
      avatar/
        meta.json
        ...
"""

import json
import re
from pathlib import Path

WIKIS_DIR = Path(__file__).parent / "wikis"

SUBDIRS = ["raw", "parsed", "chunks", "infobox", "summary", "graph", "chromadb_store"]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def list_wikis() -> list[dict]:
    """Return metadata for every registered wiki."""
    wikis = []
    if not WIKIS_DIR.exists():
        return wikis
    for d in sorted(WIKIS_DIR.iterdir()):
        meta_path = d / "meta.json"
        if d.is_dir() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["slug"] = d.name
            # Count docs in chromadb if available
            chromadb_dir = d / "chromadb_store"
            meta["has_embeddings"] = chromadb_dir.exists() and any(chromadb_dir.iterdir()) if chromadb_dir.exists() else False
            wikis.append(meta)
    return wikis


def get_wiki_paths(slug: str) -> dict:
    """Return a dict of all relevant paths for a wiki."""
    base = WIKIS_DIR / slug
    paths = {"base": base}
    for sub in SUBDIRS:
        paths[sub] = base / sub
    paths["meta"] = base / "meta.json"
    return paths


def get_wiki_meta(slug: str) -> dict:
    """Read and return meta.json for a wiki."""
    meta_path = WIKIS_DIR / slug / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Wiki '{slug}' not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def create_wiki(name: str, url: str) -> str:
    """Create a new wiki folder structure and meta.json. Returns the slug."""
    slug = slugify(name)
    base = WIKIS_DIR / slug

    for sub in SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)

    meta = {
        "name": name,
        "slug": slug,
        "url": url.rstrip("/"),
        "collection": f"{slug}_wiki",
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return slug


def delete_wiki(slug: str) -> None:
    """Permanently delete a wiki and all its data."""
    import shutil
    import stat
    base = WIKIS_DIR / slug
    if not base.exists():
        raise FileNotFoundError(f"Wiki '{slug}' not found")

    def _on_error(func, path, exc):
        # Clear read-only flag and retry
        try:
            Path(path).chmod(stat.S_IWRITE)
            func(path)
        except Exception:
            raise exc

    shutil.rmtree(base, onexc=_on_error)


def migrate_existing(
    raw_dir: Path,
    parsed_dir: Path,
    chunks_dir: Path,
    infobox_dir: Path,
    summary_dir: Path,
    graph_dir: Path,
    chromadb_dir: Path,
    name: str,
    url: str,
) -> str:
    """Move existing data folders into the wikis/ structure. Returns slug."""
    import shutil

    slug = create_wiki(name, url)
    paths = get_wiki_paths(slug)

    mapping = {
        raw_dir: paths["raw"],
        parsed_dir: paths["parsed"],
        chunks_dir: paths["chunks"],
        infobox_dir: paths["infobox"],
        summary_dir: paths["summary"],
        graph_dir: paths["graph"],
        chromadb_dir: paths["chromadb_store"],
    }

    for src, dst in mapping.items():
        if src.exists() and src != dst:
            # Copy files into dst (dst already created by create_wiki)
            for f in src.iterdir():
                target = dst / f.name
                if f.is_file():
                    shutil.copy2(f, target)
                elif f.is_dir():
                    shutil.copytree(f, target, dirs_exist_ok=True)

    return slug
