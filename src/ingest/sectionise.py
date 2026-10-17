
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HEADING_RE = re.compile(r"^(={1,6})\s+(.+?)\s*$")
INCLUDE_RE = re.compile(r"^\s*include::")
IMAGE_BLOCK_RE = re.compile(r"^\s*image::")
COMMENT_START_RE = re.compile(r"^\s*////\s*$")
COMMENT_END_RE = re.compile(r"^\s*////\s*$")
LINE_COMMENT_RE = re.compile(r"^\s*//(?!/)")
BLOCK_ATTRIBUTE_RE = re.compile(
    r"^\s*\[(source(?:,[^\]]*)?|NOTE|TIP|WARNING|CAUTION|IMPORTANT|"
    r"quote(?:,[^\]]*)?|example|listing|sidebar|verse)\]\s*$",
    re.IGNORECASE,
)
ANCHOR_RE = re.compile(r"^\s*\[\[.*\]\]\s*$")
BLOCK_DELIMITER_RE = re.compile(r"^\s*([.\-_*]{4,}|={4,})\s*$")


def slugify(value: str) -> str:
    """Create a stable filesystem- and citation-friendly slug."""
    value = value.lower().strip()
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"[*_+#]", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "untitled"


def clean_inline_markup(line: str) -> str:
    """Remove common inline AsciiDoc noise while retaining readable text."""
    line = re.sub(r"link:([^\[]+)\[([^\]]*)\]", r"\2", line)
    line = re.sub(r"https?://[^\[]+\[([^\]]* )\]", r"\1", line)
    line = re.sub(r"image:[^\[]+\[([^\]]*)\]", r"\1", line)

    # Keep visible text from common AsciiDoc formatting.
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"\*([^*]+)\*", r"\1", line)
    line = re.sub(r"__([^_]+)__", r"\1", line)
    line = re.sub(r"_([^_]+)_", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)

    # Remove formatting attributes such as {counter:...}.
    line = re.sub(r"\{[^{}\n]+\}", "", line)

    return line


def normalise_text(lines: list[str]) -> str:
    cleaned: list[str] = []
    previous_blank = False

    for raw_line in lines:
        line = raw_line.rstrip()

        if not line.strip():
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue

        line = clean_inline_markup(line)
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            cleaned.append(line)
            previous_blank = False

    while cleaned and not cleaned[-1]:
        cleaned.pop()

    return "\n".join(cleaned)


def clean_file(path: Path) -> list[str]:
    """Remove includes, images, comments and non-content AsciiDoc directives."""
    output: list[str] = []
    in_comment_block = False
    in_literal_block = False

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()

        if COMMENT_START_RE.match(raw_line):
            in_comment_block = not in_comment_block
            continue

        if in_comment_block:
            continue

        if INCLUDE_RE.match(raw_line):
            continue

        if IMAGE_BLOCK_RE.match(raw_line):
            continue

        if LINE_COMMENT_RE.match(raw_line):
            continue

        if ANCHOR_RE.match(raw_line):
            continue

        if BLOCK_ATTRIBUTE_RE.match(raw_line):
            continue

        if BLOCK_DELIMITER_RE.match(raw_line):
            # Preserve source-code block contents but remove AsciiDoc delimiters.
            in_literal_block = not in_literal_block
            continue

        # Remove standalone document attributes such as :toc: or :imagesdir:
        if re.match(r"^\s*:[^:]+:.*$", raw_line):
            continue

        output.append(raw_line)

    return output


def build_sections(source_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    source_files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".adoc", ".asciidoc", ".asc"}
    )

    for source_path in source_files:
        relative_path = source_path.relative_to(source_root).as_posix()
        file_slug = slugify(source_path.stem)

        lines = clean_file(source_path)
        current_heading: str | None = None
        current_level: int | None = None
        current_lines: list[str] = []
        heading_path: list[str] = []
        counters = [0] * 7
        section_number = 0

        def flush_section() -> None:
            nonlocal current_heading, current_level
            nonlocal current_lines, section_number

            text = normalise_text(current_lines)

            if not current_heading and not text:
                return

            section_number += 1

            if current_heading:
                title_slug = slugify(current_heading)
                section_id = (
                    f"{file_slug}."
                    f"{section_number:03d}-"
                    f"{title_slug}"
                )
            else:
                section_id = f"{file_slug}.000-preamble"

            records.append(
                {
                    "section_id": section_id,
                    "chapter": file_slug,
                    "heading": current_heading or "Preamble",
                    "heading_level": current_level or 0,
                    "heading_path": heading_path.copy(),
                    "source_file": relative_path,
                    "text": text,
                }
            )

            current_heading = None
            current_level = None
            current_lines = []

        for line in lines:
            heading_match = HEADING_RE.match(line)

            if heading_match:
                flush_section()

                level = len(heading_match.group(1))
                heading = clean_inline_markup(heading_match.group(2)).strip()

                # Update counters to create deterministic hierarchy metadata.
                counters[level] += 1
                for index in range(level + 1, len(counters)):
                    counters[index] = 0

                heading_path = heading_path[: max(0, level - 1)]
                heading_path.append(heading)

                current_heading = heading
                current_level = level
                continue

            current_lines.append(line)

        flush_section()

    return records


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Pro Git AsciiDoc files and create section JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/book"),
        help="Raw AsciiDoc directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sections.jsonl"),
        help="Output JSONL file.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input directory does not exist: {args.input}")

    records = build_sections(args.input)
    write_jsonl(records, args.output)

    print(f"Source directory: {args.input}")
    print(f"Sections written: {len(records)}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()

