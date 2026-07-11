#!/usr/bin/env python3
"""
Flatten validated Gemini outputs into original-style datasets.

Reads all JSON files under datav2/validated_responses_g3/valuepair_*/ and
writes consolidated arrays into datav2/originalAB/{AB,BC,CA}.json, mirroring
data/originalAB/* layout.
"""

import json
from pathlib import Path
from typing import Dict, List

BASE_IN = Path("datav2/validated_responses_g3")
BASE_OUT = Path("datav2/originalAB")

PAIR_TO_NAME: Dict[str, str] = {
    "valuepair_0": "AB",  # Honesty vs Justice
    "valuepair_1": "BC",  # Justice vs Autonomy
    "valuepair_2": "CA",  # Autonomy vs Honesty
}


def load_validated_files(pair_dir: Path) -> List[dict]:
    """Load and flatten all fixed dilemmas in a valuepair directory."""
    merged: List[dict] = []
    for fp in sorted(pair_dir.glob("*.json")):
        base = fp.stem  # e.g., scenario_0_value_0_run_0
        with open(fp, "r", encoding="utf-8") as f:
            items = json.load(f)
            for idx_in_file, item in enumerate(items, start=1):
                dilemm = item.get("fixed_dilemma", item)
                # Strip validation metadata if present
                dilemm.pop("_validation", None)
                # Rebuild index to match original style: filename stem + _i<index>
                raw_idx = dilemm.get("index", idx_in_file)
                dilemm["index"] = f"{base}_i{raw_idx}"
                merged.append(dilemm)
    return merged


def main():
    if not BASE_IN.exists():
        raise SystemExit(f"Input directory not found: {BASE_IN}")

    BASE_OUT.mkdir(parents=True, exist_ok=True)

    for pair, out_name in PAIR_TO_NAME.items():
        pair_dir = BASE_IN / pair
        if not pair_dir.exists():
            print(f"Skipping {pair}: directory missing")
            continue

        merged = load_validated_files(pair_dir)
        out_path = BASE_OUT / f"{out_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(merged)} dilemmas to {out_path}")


if __name__ == "__main__":
    main()
