"""Extract summaries from parsed wiki pages into summary/ folder.

One JSON file per page: pageslug_summary.json
"""

import json
import sys
from pathlib import Path

PARSED_DIR = Path(__file__).parent / "parsed"
SUMMARY_DIR = Path(__file__).parent / "summary"


def build_summary_chunk(data: dict) -> dict | None:
    """Build a summary chunk dict from parsed page data. Returns None if no summary."""
    summary = data.get("summary", "").strip()
    if not summary:
        return None
    return {
        "chunk_id": data["page_id"] + "_summary",
        "page_title": data.get("title", ""),
        "type": "summary",
        "content": summary,
    }


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        files = [PARSED_DIR / sys.argv[1]]
    else:
        files = sorted(PARSED_DIR.glob("*.json"))

    saved = 0
    skipped = 0

    for path in files:
        if not path.exists():
            print(f"Not found: {path}")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary", "").strip()

        if not summary:
            skipped += 1
            continue

        chunk = {
            "chunk_id": data["page_id"] + "_summary",
            "type": "summary",
            "content": summary,
        }

        out_path = SUMMARY_DIR / f"{data['page_id']}_summary.json"
        out_path.write_text(
            json.dumps(chunk, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved += 1

    print(f"Done! Saved: {saved}, Skipped (no summary): {skipped}")
    print(f"Output in {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
