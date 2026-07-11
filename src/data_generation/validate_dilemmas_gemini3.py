#!/usr/bin/env python3
"""
Validate and repair ethical dilemmas using Gemini 3.0 Flash.

Reads each raw OpenAI response file under openai_responses/valuepair_*/,
verifies that each dilemma cleanly reflects the intended value ordering
(option A prioritises VALUE_1, option B prioritises VALUE_2), fixes any
issues, and writes corrected JSON to validated_responses_g3/.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import types

# Model + key
MODEL = "gemini-3-flash-preview"
CLIENT = None


def get_client():
    global CLIENT
    if CLIENT is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before running this script.")
        CLIENT = genai.Client(api_key=api_key)
    return CLIENT

# Value definitions (order matters: option A prioritises value_1, option B value_2)
VALUES = [
    "Trustworthiness and Honesty",
    "Basic Justice (being fair)",
    "Respect for the autonomy of others",
]

VALUE_PAIRS: Dict[str, Tuple[str, str]] = {
    "valuepair_0": (VALUES[0], VALUES[1]),  # Honesty vs Justice
    "valuepair_1": (VALUES[1], VALUES[2]),  # Justice vs Autonomy
    "valuepair_2": (VALUES[2], VALUES[0]),  # Autonomy vs Honesty
}

OUTPUT_DIR = Path("datav2/validated_responses_g3")

VALIDATION_PROMPT = """You are an expert in ethical dilemmas and value alignment. Fix or validate EACH dilemma in the batch.

Values:
- VALUE_1: {value_1}
- VALUE_2: {value_2}

Rules:
1) Each dilemma must truly hinge on VALUE_1 vs VALUE_2.
2) Option A MUST be the preferred option when you prioritise VALUE_1 over VALUE_2.
3) Option B MUST be the preferred option when you prioritise VALUE_2 over VALUE_1.
4) Keep options reasonable and paired to the story/question.
5) Keep indices as provided and preserve order.

Input dilemmas (list):
```json
{dilemmas_json}
```

Return ONLY a JSON array, one entry per input, preserving order. Each entry must be:
{{
  "is_valid": true/false,
  "issues_found": ["list of issues if any, empty if valid"],
  "fixed_dilemma": {{
    "index": <same index>,
    "story": "...",
    "question": "...",
    "options": [
      "Option A (prioritises {value_1})",
      "Option B (prioritises {value_2})"
    ]
  }}
}}"""


def parse_json_from_file(file_path: Path) -> Optional[List[Dict]]:
    """Parse JSON from a file, handling optional markdown fences."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"^```json\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"^```\s*$", "", content, flags=re.MULTILINE)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def extract_json(text: str) -> Optional[Union[Dict, List]]:
    """Extract first JSON object or array from text."""
    text = text.strip()

    # Try code block first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def validate_batch(dilemmas: List[Dict], value_1: str, value_2: str, retries: int = 3) -> Optional[List[Dict]]:
    """Send the entire file (list of dilemmas) in a single Gemini call."""
    print(f"    Validating batch of {len(dilemmas)} dilemmas...")
    prompt = VALIDATION_PROMPT.format(
        value_1=value_1,
        value_2=value_2,
        dilemmas_json=json.dumps(dilemmas, indent=2),
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
    )

    for attempt in range(retries):
        try:
            print(f"      Attempt {attempt+1}/{retries}...")
            response_text = ""
            for chunk in get_client().models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    response_text += chunk.text

            parsed = extract_json(response_text)
            if isinstance(parsed, list):
                print(f"      Received valid response with {len(parsed)} items.")
                return parsed
            print(f"      Attempt {attempt+1}: invalid response format, retrying...")
        except Exception as e:
            print(f"      Attempt {attempt+1}: error {e}, retrying...")
            time.sleep(2 ** attempt)

    return None


def process_file(input_path: Path, value_pair_name: str) -> Tuple[List[Dict], Dict]:
    """Validate all dilemmas in one file with a single call."""
    value_1, value_2 = VALUE_PAIRS[value_pair_name]
    dilemmas = parse_json_from_file(input_path)
    if dilemmas is None:
        return [], {"total": 0, "valid": 0, "fixed": 0, "failed": 0}

    stats = {"total": len(dilemmas), "valid": 0, "fixed": 0, "failed": 0}
    validated: List[Dict] = []

    batch_result = validate_batch(dilemmas, value_1, value_2)
    if not isinstance(batch_result, list) or len(batch_result) != len(dilemmas):
        print(f"    Batch validation failed for {input_path.name}; returning original.")
        return dilemmas, stats

    for result in batch_result:
        fixed = result.get("fixed_dilemma", {})
        fixed["_validation"] = {
            "original_valid": result.get("is_valid"),
            "issues": result.get("issues_found", []),
        }
        validated.append(fixed)

        if result.get("is_valid") is True:
            stats["valid"] += 1
        elif result.get("is_valid") is False:
            stats["fixed"] += 1
        else:
            stats["failed"] += 1

    return validated, stats


def main():
    input_base = Path("openai_responses")
    total_stats = {"total": 0, "valid": 0, "fixed": 0, "failed": 0}

    tasks = []
    for value_pair_name in ["valuepair_0", "valuepair_1", "valuepair_2"]:
        pair_dir = input_base / value_pair_name
        if not pair_dir.is_dir():
            print(f"Skipping {value_pair_name}: directory missing")
            continue

        out_dir = OUTPUT_DIR / value_pair_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for input_file in sorted(pair_dir.glob("*.txt")):
            output_file = out_dir / input_file.name.replace(".txt", ".json")
            if output_file.exists():
                print(f"Skipping {input_file.name}: already processed")
                continue
            tasks.append((value_pair_name, input_file, output_file))

    print(f"\nDiscovered {len(tasks)} files to process.")

    if not tasks:
        print("No files to process. Exiting.")
        return

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(process_file, input_file, value_pair_name): (value_pair_name, input_file, output_file)
            for value_pair_name, input_file, output_file in tasks
        }

        for idx, future in enumerate(as_completed(future_map), start=1):
            value_pair_name, input_file, output_file = future_map[future]
            print(f"\n=== Finished {input_file.name} ({value_pair_name}) [{idx}/{len(future_map)}] ===")
            try:
                validated, stats = future.result()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(validated, f, indent=2, ensure_ascii=False)
                print(f"  Total: {stats['total']}, Valid: {stats['valid']}, Fixed: {stats['fixed']}, Failed: {stats['failed']}")
                for key in total_stats:
                    total_stats[key] += stats[key]
            except Exception as e:
                print(f"  Error processing {input_file}: {e}")

    print("\n=== FINAL SUMMARY ===")
    print(f"Total dilemmas processed: {total_stats['total']}")
    print(f"Already valid: {total_stats['valid']}")
    print(f"Fixed by model: {total_stats['fixed']}")
    print(f"Failed: {total_stats['failed']}")


if __name__ == "__main__":
    main()
