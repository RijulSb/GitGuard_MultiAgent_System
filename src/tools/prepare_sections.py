from __future__ import annotations

import json
from pathlib import Path


def prepare_section_files() -> int:
    out_dir = Path("data/processed/sections")
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open("data/processed/sections.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sec = json.loads(line)
            sid = sec["section_id"]
            heading = sec.get("heading", "")
            text = sec.get("text", "")
            fpath = out_dir / f"{sid}.txt"
            fpath.write_text(f"[{sid}] {heading}\n\n{text}", encoding="utf-8")
            count += 1
    return count


if __name__ == "__main__":
    c = prepare_section_files()
    print(f"Prepared {c} individual section files in data/processed/sections")
