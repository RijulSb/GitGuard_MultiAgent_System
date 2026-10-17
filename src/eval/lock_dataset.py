from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> None:
    with open("configs/settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    path = Path(cfg["evaluation"]["dataset_path"])
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 60 or not all(r.get("reviewed") for r in rows):
        raise SystemExit(f"Review all records first; expected at least 60 records and reviewed=true on every record, got {len(rows)}.")
    meta = Path("data/eval/dataset_metadata.json")
    data = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    data.update({"status": "locked", "locked": True, "records": len(rows), "answerable": sum(1 for r in rows if r["answerable"]), "unanswerable": sum(1 for r in rows if not r["answerable"])})
    meta.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Locked {len(rows)} records: {path}")


if __name__ == "__main__":
    main()
