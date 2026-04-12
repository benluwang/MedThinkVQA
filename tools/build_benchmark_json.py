#!/usr/bin/env python3
"""Build docs/benchmark.json from selected_model_newacc_updated.csv."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

RATIO_RE = re.compile(r"ratio(\d+)", re.IGNORECASE)
EFFORT_WORD_RE = re.compile(r"(xhigh|high|medium|low|minimal)", re.IGNORECASE)


def should_include(model_id: str) -> bool:
    """Keep rows with ratio100 or without any ratio token."""
    return bool(re.search(r"ratio100", model_id, re.IGNORECASE) or not RATIO_RE.search(model_id))


def parse_mode(model_id: str) -> str:
    lower = model_id.lower()
    if re.search(r"nonthinking|nothinking", lower):
        return "Non-thinking"
    if re.search(r"(?:^|[-_])thinking(?:$|[-_])", lower) or re.search(
        r"(xhigh|high|medium|low|minimal)thinking", lower
    ):
        return "Thinking"
    if re.search(r"(?:^|[-_])instruct(?:$|[-_])", lower):
        return "Non-thinking"
    return "Unspecified"


def parse_effort(model_id: str, mode: str) -> str:
    lower = model_id.lower()
    camel_match = re.search(r"(xhigh|high|medium|low|minimal)thinking", lower)
    if camel_match:
        return camel_match.group(1)
    effort_match = re.search(r"effort-(xhigh|high|medium|low|minimal)", lower)
    if effort_match:
        return effort_match.group(1)
    if mode == "Thinking":
        return "standard"
    return "N/A"


def clean_model_name(model_id: str) -> str:
    name = model_id
    name = re.sub(r"^(api|vllm)-", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-ratio\d+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-acc$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-qmode(?=-|$)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-effort-(xhigh|high|medium|low|minimal)(?=-|$)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-(xhigh|high|medium|low|minimal)thinking(?=-|$)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-(nonthinking|nothinking|thinking)(?=-|$)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-_ ")
    return name


def parse_ratio_bucket(model_id: str) -> str:
    if re.search(r"ratio100", model_id, re.IGNORECASE):
        return "ratio100"
    return "no-ratio"


def build_entries(csv_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            model_id = (row.get("model") or "").strip()
            if not model_id or not should_include(model_id):
                continue

            try:
                acc = float((row.get("newacc") or "").strip())
            except ValueError:
                continue

            mode = parse_mode(model_id)
            entries.append(
                {
                    "model": clean_model_name(model_id),
                    "acc": acc,
                    "mode": mode,
                    "effort": parse_effort(model_id, mode),
                    "ratio_bucket": parse_ratio_bucket(model_id),
                }
            )

    entries.sort(key=lambda item: (-item["acc"], item["model"].lower()))
    return entries


def build_payload(csv_path_for_meta: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": "MedThinkVQA Benchmark Leaderboard",
        "source_csv": csv_path_for_meta,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate docs/benchmark.json from CSV")
    parser.add_argument("--input", required=True, help="Path to selected_model_newacc_updated.csv")
    parser.add_argument("--output", required=True, help="Path to docs/benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    input_path_for_meta = Path(args.input).as_posix()
    output_path = Path(args.output).resolve()

    entries = build_entries(input_path)
    payload = build_payload(input_path_for_meta, entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")

    print(f"wrote {len(entries)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
