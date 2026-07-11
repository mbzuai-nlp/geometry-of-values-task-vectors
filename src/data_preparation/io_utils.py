import json
from pathlib import Path


def load_json_or_jsonl(path):
    """Load either a JSON array or newline-delimited JSON records."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped[0] == "[":
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return data
