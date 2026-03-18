"""Extract link graph from parsed wiki pages into graph/ folder.

One JSON file per page: pageslug.json
"""

import json
import sys
from pathlib import Path

PARSED_DIR = Path(__file__).parent / "parsed"
GRAPH_DIR = Path(__file__).parent / "graph"


def build_graph_chunk(data: dict) -> dict | None:
    """Build a graph chunk dict from parsed page data. Returns None if no links."""
    links = data.get("links", [])
    if not links:
        return None
    return {
        "node": data["title"],
        "links": links,
    }


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

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
        links = data.get("links", [])

        if not links:
            skipped += 1
            continue

        graph = {
            "node": data["title"],
            "links": links,
        }

        out_path = GRAPH_DIR / f"{data['page_id']}.json"
        out_path.write_text(
            json.dumps(graph, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved += 1

    print(f"Done! Saved: {saved}, Skipped (no links): {skipped}")
    print(f"Output in {GRAPH_DIR}")


if __name__ == "__main__":
    main()
