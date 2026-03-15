"""Chunk parsed wiki pages by section and save to chunks/.

Each section becomes its own JSON file: pageslug_sectionname.json
The summary also gets its own file.
"""

import json
import re
import sys
from pathlib import Path

PARSED_DIR = Path(__file__).parent / "parsed"
CHUNKS_DIR = Path(__file__).parent / "chunks"


def slugify(text: str) -> str:
    """Turn a string into a slug for use in chunk IDs and filenames."""
    text = re.sub(r"[''\"]+", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def chunk_page(parsed_path: Path) -> list[dict]:
    """Split a parsed page into one chunk per section + one for summary."""
    data = json.loads(parsed_path.read_text(encoding="utf-8"))
    page_id = data["page_id"]
    title = data["title"]
    chunks = []

    # Summary chunk
    summary = data.get("summary", "").strip()
    if summary:
        chunks.append({
            "chunk_id": f"{page_id}_summary",
            "page_title": title,
            "section": "Summary",
            "content": summary,
        })

    # Section chunks
    for section in data.get("sections", []):
        text = section["text"].strip()
        if not text:
            continue
        section_slug = slugify(section["title"])
        chunks.append({
            "chunk_id": f"{page_id}_{section_slug}",
            "page_title": title,
            "section": section["title"],
            "content": text,
        })

    return chunks


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        files = [PARSED_DIR / sys.argv[1]]
    else:
        files = sorted(PARSED_DIR.glob("*.json"))

    total_chunks = 0
    for path in files:
        if not path.exists():
            print(f"Not found: {path}")
            continue

        chunks = chunk_page(path)
        for chunk in chunks:
            filename = f"{chunk['chunk_id']}.json"
            out_path = CHUNKS_DIR / filename
            out_path.write_text(
                json.dumps(chunk, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            total_chunks += 1

    print(f"Done! {total_chunks} chunks from {len(files)} pages")
    print(f"Output in {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
