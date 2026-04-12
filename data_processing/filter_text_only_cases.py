#!/usr/bin/env python3

"""Filter cases that appear solvable without images.

This script keeps the GitHub release lightweight: it does not run any model by
itself. Instead, it consumes one or more text-only / options-only probe outputs
and removes benchmark items that those probes solved correctly.

Supported probe formats:
- `{"data": [...]}`
- plain JSON list
- JSONL

Each probe item is matched back to the benchmark item using the first available
stable key among:
- `source_row_index`
- `source_link`
- `title`
- `(title, clinical_history)`

Probe correctness is read from:
- `is_correct` when present, or
- predicted letter vs. ground-truth letter when both are present.

Example:
```bash
python data_processing/filter_text_only_cases.py \
  --input_json data/final_test.json \
  --probe_eval probes/options_only_eval.json \
  --probe_eval probes/history_plus_options_eval.json \
  --output_json data/final_test_filtered.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def _clinical_history(item: Dict[str, Any]) -> str:
    return str(
        item.get("CLINICAL_HISTORY")
        or item.get("clinical_history")
        or item.get("clinical history")
        or ""
    ).strip()


def _iter_match_keys(item: Dict[str, Any]) -> Iterable[str]:
    for field in ("source_row_index", "source_link", "title"):
        value = item.get(field)
        if value is None:
            continue
        norm = str(value).strip()
        if norm:
            yield f"{field}:{norm}"
    title = _norm_text(item.get("title"))
    history = _norm_text(_clinical_history(item))
    if title or history:
        yield f"title_history:{title}||{history}"


def _primary_key(item: Dict[str, Any]) -> Optional[str]:
    for key in _iter_match_keys(item):
        return key
    return None


def _load_json_like(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
        return items, [x for x in items if isinstance(x, dict)]

    raw = json.loads(text)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw, [x for x in raw["data"] if isinstance(x, dict)]
    if isinstance(raw, list):
        return raw, [x for x in raw if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _norm_text(value)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _pick_first(item: Dict[str, Any], fields: Sequence[str]) -> Optional[Any]:
    for field in fields:
        if field in item and item[field] not in (None, ""):
            return item[field]
    return None


def _predicted_letter(item: Dict[str, Any]) -> Optional[str]:
    fields = (
        "parsed_answer_letter",
        "prediction",
        "answer",
        "pred",
        "predicted_answer",
    )
    value = _pick_first(item, fields)
    if value is None:
        return None
    text = str(value).strip().upper()
    return text[:1] if text else None


def _ground_truth_letter(item: Dict[str, Any]) -> Optional[str]:
    fields = (
        "gt_letter",
        "correct_answer",
        "answer_key",
        "label",
    )
    value = _pick_first(item, fields)
    if value is None:
        return None
    text = str(value).strip().upper()
    return text[:1] if text else None


def _probe_is_correct(item: Dict[str, Any]) -> Optional[bool]:
    if "is_correct" in item:
        return _to_bool(item.get("is_correct"))

    pred = _predicted_letter(item)
    gt = _ground_truth_letter(item)
    if pred and gt:
        return pred == gt
    return None


def _build_probe_index(items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    ambiguous = set()
    for item in items:
        for key in _iter_match_keys(item):
            if key in index:
                ambiguous.add(key)
            else:
                index[key] = item
    for key in ambiguous:
        index.pop(key, None)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove cases that text-only or options-only probes solved correctly."
    )
    parser.add_argument("--input_json", type=str, required=True, help="Benchmark JSON/JSONL to filter.")
    parser.add_argument(
        "--probe_eval",
        type=str,
        action="append",
        required=True,
        help="Probe eval JSON/JSONL. Pass multiple times for multiple probes.",
    )
    parser.add_argument("--output_json", type=str, required=True, help="Filtered dataset output path.")
    parser.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="Optional report path. Defaults to <output_json>.report.json",
    )
    parser.add_argument(
        "--remove_if",
        type=str,
        choices=("any", "all"),
        default="any",
        help="Remove a case if any probe is correct, or only if all matched probes are correct.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    report_path = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else Path(str(output_path) + ".report.json")
    )

    raw_input, input_items = _load_json_like(input_path)
    probe_payloads: List[Tuple[Path, Dict[str, Dict[str, Any]]]] = []
    for probe_path_raw in args.probe_eval:
        probe_path = Path(probe_path_raw).expanduser().resolve()
        _raw_probe, probe_items = _load_json_like(probe_path)
        probe_payloads.append((probe_path, _build_probe_index(probe_items)))

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for item in input_items:
        key = _primary_key(item)
        matched_results: List[bool] = []
        matched_probe_names: List[str] = []

        if key is not None:
            for probe_path, probe_index in probe_payloads:
                probe_item = probe_index.get(key)
                if probe_item is None:
                    continue
                is_correct = _probe_is_correct(probe_item)
                if is_correct is None:
                    continue
                matched_results.append(bool(is_correct))
                matched_probe_names.append(probe_path.name)

        should_remove = False
        if matched_results:
            if args.remove_if == "any":
                should_remove = any(matched_results)
            else:
                should_remove = all(matched_results)

        if should_remove:
            removed.append(
                {
                    "key": key,
                    "title": item.get("title"),
                    "source_row_index": item.get("source_row_index"),
                    "source_link": item.get("source_link"),
                    "matched_probes": matched_probe_names,
                    "matched_results": matched_results,
                }
            )
        else:
            kept.append(item)

    if isinstance(raw_input, dict) and isinstance(raw_input.get("data"), list):
        output_payload = dict(raw_input)
        output_payload["data"] = kept
    else:
        output_payload = kept

    report_payload = {
        "input_json": str(input_path),
        "output_json": str(output_path),
        "probe_eval": [str(path) for path, _ in probe_payloads],
        "remove_if": args.remove_if,
        "input_count": len(input_items),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "removed": removed,
    }

    _json_dump(output_path, output_payload)
    _json_dump(report_path, report_payload)

    print(
        json.dumps(
            {
                "input_count": len(input_items),
                "kept_count": len(kept),
                "removed_count": len(removed),
                "output_json": str(output_path),
                "report_json": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
