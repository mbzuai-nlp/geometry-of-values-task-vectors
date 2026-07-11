"""Reorder translated response files to match the original ordering.

For each language under ``datav2/translated_responses/{lang}``, this script
reorders the JSON objects in ``*.json`` files to follow the index ordering from
the corresponding file in ``datav2/originalAB``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_original_order(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["index"] for item in data]


def load_translation(path: Path) -> Tuple[List[dict], bool, List[str]]:
    """Load a translation file.

    Returns (objects, is_array_format, duplicate_indices)
    """
    text = path.read_text(encoding="utf-8")
    first_char = next((ch for ch in text if not ch.isspace()), "")
    is_array = first_char == "["

    objs: List[dict]
    if is_array:
        objs = json.loads(text)
    else:
        objs = [json.loads(line) for line in text.splitlines() if line.strip()]

    dupes: List[str] = []
    seen: Dict[str, int] = {}
    for obj in objs:
        idx = obj.get("index")
        if idx in seen:
            dupes.append(idx)
        seen[idx] = seen.get(idx, 0) + 1
    return objs, is_array, dupes


def reorder_objects(
    original_order: Iterable[str], translations: List[dict]
) -> Tuple[List[dict], List[str], List[str]]:
    by_index = {obj["index"]: obj for obj in translations}
    missing = [idx for idx in original_order if idx not in by_index]
    reordered = [by_index[idx] for idx in original_order if idx in by_index]
    extras = [idx for idx in by_index if idx not in set(original_order)]
    return reordered, missing, extras


def write_translation(path: Path, objs: List[dict], is_array: bool) -> None:
    if is_array:
        with path.open("w", encoding="utf-8") as f:
            json.dump(objs, f, ensure_ascii=False, indent=2)
            f.write("\n")
    else:
        lines = [json.dumps(obj, ensure_ascii=False) for obj in objs]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_languages(translated_root: Path, only: List[str] | None) -> List[str]:
    if only:
        return only
    return sorted(
        d.name for d in translated_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def iter_original_files(original_root: Path, only_files: List[str] | None) -> List[Path]:
    if only_files:
        return [original_root / name for name in only_files]
    return sorted(original_root.glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorder translated response JSON files to match original ordering."
    )
    parser.add_argument(
        "--original-dir",
        default="datav2/originalAB",
        type=Path,
        help="Directory with original JSON arrays (default: datav2/originalAB)",
    )
    parser.add_argument(
        "--translated-dir",
        default="datav2/translated_responses",
        type=Path,
        help="Root directory containing language subfolders (default: datav2/translated_responses)",
    )
    parser.add_argument(
        "--langs",
        nargs="*",
        help="Specific languages to process (subdirectories of translated-dir). Defaults to all.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific JSON filenames to process (e.g., AB.json). Defaults to all in original-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without writing files.",
    )

    args = parser.parse_args()

    original_dir: Path = args.original_dir
    translated_dir: Path = args.translated_dir

    langs = iter_languages(translated_dir, args.langs)
    original_files = iter_original_files(original_dir, args.files)

    for orig_path in original_files:
        if not orig_path.exists():
            print(f"[skip] original missing: {orig_path}")
            continue

        original_order = load_original_order(orig_path)
        base_name = orig_path.name

        for lang in langs:
            lang_dir = translated_dir / lang
            trans_path = lang_dir / base_name
            if not trans_path.exists():
                print(f"[skip] {lang} missing {base_name}")
                continue

            translations, is_array, duplicates = load_translation(trans_path)
            reordered, missing, extras = reorder_objects(original_order, translations)

            print(
                f"[{lang}/{base_name}] total={len(translations)} reordered={len(reordered)} "
                f"missing={len(missing)} extras={len(extras)} dupes={len(duplicates)}"
            )
            if missing:
                print(f"  missing indices: {', '.join(missing)}")
            if extras:
                print(f"  extra indices: {', '.join(extras)}")
            if duplicates:
                print(f"  duplicate indices: {', '.join(duplicates)}")

            if args.dry_run:
                continue

            write_translation(trans_path, reordered, is_array)


if __name__ == "__main__":
    main()
