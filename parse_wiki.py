"""Parse raw wiki JSON files into structured, clean JSON.

Reads wikitext from raw/ and produces cleaned JSON in parsed/.
"""

import json
import re
import sys
from pathlib import Path

import mwparserfromhell

RAW_DIR = Path(__file__).parent / "raw"
PARSED_DIR = Path(__file__).parent / "parsed"

# Templates that typically hold infobox data
INFOBOX_NAMES = {"character", "infobox", "location", "event", "book", "film"}


def clean_wikitext(text: str) -> str:
    """Strip wikitext markup down to plain text."""
    wikicode = mwparserfromhell.parse(text)

    # Remove file/image links, categories, interlanguage links
    for link in wikicode.filter_wikilinks():
        title = str(link.title).strip()
        if re.match(r"(?i)(File|Image|Category|[a-z]{2}):", title):
            wikicode.remove(link)

    # Remove gallery, ref, nowiki tags entirely
    for tag in list(wikicode.filter_tags()):
        try:
            tag_name = str(tag.tag).strip().lower()
            if tag_name in ("ref", "gallery", "nowiki", "br"):
                wikicode.remove(tag)
        except ValueError:
            pass  # already removed with parent

    # Replace link-like templates ({{W|text}}) with display text, remove the rest.
    # Use list() snapshot since we mutate during iteration; try/except handles
    # nested templates that disappear when their parent is removed.
    for tpl in list(wikicode.filter_templates()):
        try:
            name = str(tpl.name).strip().lower()
            if name in ("w", "wp", "wikipedia") and tpl.params:
                wikicode.replace(tpl, str(tpl.params[0].value).strip())
            else:
                wikicode.remove(tpl)
        except ValueError:
            pass  # already removed with parent template

    # Convert to plain text: resolves [[link|display]] -> display
    text = wikicode.strip_code()

    # Clean up leftover markup
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def parse_infobox(wikicode) -> dict:
    """Extract key-value pairs from the first infobox-like template."""
    for tpl in wikicode.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower()
        if name in INFOBOX_NAMES:
            infobox = {}
            for param in tpl.params:
                key = str(param.name).strip()
                val = clean_wikitext(str(param.value))
                if val and key.lower() not in ("name", "image"):
                    infobox[key] = val.replace("\n", "; ")
            return infobox
    return {}


def parse_sections(wikicode) -> list[dict]:
    """Split wikitext into sections by headings."""
    headings = wikicode.filter_headings()
    if not headings:
        return []

    sections = []
    # Iterate each heading and grab the text that follows it
    for heading in headings:
        title = str(heading.title).strip()

        # Skip meta sections
        if title.lower() in (
            "references", "gallery", "movie stills", "outfits",
            "external links", "see also", "trivia",
        ):
            continue

        # Get text between this heading and the next heading (or end)
        heading_str = str(heading)
        idx = str(wikicode).find(heading_str)
        if idx == -1:
            continue

        after = str(wikicode)[idx + len(heading_str):]
        # Find the next heading at the same or higher level
        next_match = re.search(r"\n={2,}[^=]", after)
        section_raw = after[:next_match.start()] if next_match else after

        text = clean_wikitext(section_raw)
        if text:
            sections.append({"title": title, "text": text})

    return sections


def extract_summary(wikicode) -> str:
    """Extract intro text before the first section heading."""
    text = str(wikicode)
    # Find the first == heading
    match = re.search(r"\n==[^=]", text)
    intro = text[:match.start()] if match else text

    # Remove the infobox template at the top
    intro_parsed = mwparserfromhell.parse(intro)
    for tpl in intro_parsed.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower()
        if name in INFOBOX_NAMES or name in ("quote", "dialogue"):
            intro_parsed.remove(tpl)

    return clean_wikitext(str(intro_parsed))


def parse_page(raw_path: Path) -> dict:
    """Parse a single raw JSON file into the clean format."""
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    title = raw["title"]
    wikitext = raw.get("wikitext", "")
    wikicode = mwparserfromhell.parse(wikitext)

    return {
        "page_id": re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_"),
        "title": title,
        "summary": extract_summary(wikicode),
        "infobox": parse_infobox(wikicode),
        "sections": parse_sections(wikicode),
        "links": raw.get("links", []),
        "categories": raw.get("categories", []),
    }


def main():
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    # If a filename is given, only parse that one
    if len(sys.argv) > 1:
        target = sys.argv[1]
        files = [RAW_DIR / target]
    else:
        files = sorted(RAW_DIR.glob("*.json"))

    total = len(files)
    for i, path in enumerate(files, 1):
        if not path.exists():
            print(f"Not found: {path}")
            continue
        try:
            result = parse_page(path)
            out_path = PARSED_DIR / path.name
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[{i}/{total}] Parsed: {result['title']}")
        except Exception as e:
            print(f"[{i}/{total}] Error on {path.name}: {e}")

    print(f"\nDone! Output in {PARSED_DIR}")


if __name__ == "__main__":
    main()
