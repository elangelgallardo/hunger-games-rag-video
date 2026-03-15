"""Extract infobox data from parsed wiki pages into chunked JSON files.

Reads from parsed/ and writes one JSON file per page into infobox/.
"""

import json
import sys
from pathlib import Path

PARSED_DIR = Path(__file__).parent / "parsed"
INFOBOX_DIR = Path(__file__).parent / "infobox"

# Map raw infobox keys to readable labels
KEY_LABELS = {
    "age": "Age",
    "alias": "Also known as",
    "bookappears": "Appears in",
    "bookmention": "Mentioned in",
    "eyes": "Eyes",
    "fate": "Fate",
    "gender": "Gender",
    "hair": "Hair",
    "height": "Height",
    "home": "Home",
    "inhabitants": "Inhabitants",
    "location": "Location",
    "mentioned": "Mentioned in",
    "movieappears": "Film appearances",
    "moviemention": "Mentioned in films",
    "occupation": "Occupation",
    "portrayer": "Portrayed by",
    "relatives": "Relatives",
    "weapon": "Weapon",
    "appearance": "Appearances",
}


def build_content(title: str, infobox: dict) -> str:
    """Turn infobox key-value pairs into a readable sentence-style string."""
    parts = [title + "."]
    for key, value in infobox.items():
        label = KEY_LABELS.get(key, key.replace("_", " ").title())
        parts.append(f"{label}: {value}.")
    return " ".join(parts)


def main():
    INFOBOX_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        files = [PARSED_DIR / sys.argv[1]]
    else:
        files = sorted(PARSED_DIR.glob("*.json"))

    total = len(files)
    saved = 0
    skipped = 0

    for i, path in enumerate(files, 1):
        if not path.exists():
            print(f"Not found: {path}")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        infobox = data.get("infobox", {})

        if not infobox:
            skipped += 1
            continue

        chunk = {
            "chunk_id": data["page_id"] + "_infobox",
            "type": "infobox",
            "content": build_content(data["title"], infobox),
            "page": data["title"],
        }

        out_path = INFOBOX_DIR / path.name
        out_path.write_text(
            json.dumps(chunk, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved += 1

    print(f"Done! Saved: {saved}, Skipped (no infobox): {skipped}")
    print(f"Output in {INFOBOX_DIR}")


if __name__ == "__main__":
    main()
