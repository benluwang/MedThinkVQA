#!/usr/bin/env python3

"""Step 3 helper: split free-form reasoning into atomic steps.

This is a lightweight public snapshot of the step-splitting logic used for
MedThinkVQA's Step 3 analysis. The goal is to preserve the core algorithm and
prompt design without exposing the original internal experiment layout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.gpt import APIModel


class StepList(BaseModel):
    """Structured output for step splitting."""

    model_config = ConfigDict(extra="forbid")

    steps: List[str] = Field(
        ...,
        min_length=1,
        max_length=15,
        description=(
            "Ordered list of concise atomic steps that preserve the original "
            "meaning of the explanation without adding new facts."
        ),
    )


def _load_payload(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw, [x for x in raw["data"] if isinstance(x, dict)]
    if isinstance(raw, list):
        return raw, [x for x in raw if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _save_payload(path: Path, original: Any, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(original, dict) and isinstance(original.get("data"), list):
        payload = dict(original)
        payload["data"] = items
    else:
        payload = items
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_fallback(text: str, max_steps: int = 15) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+|;\s+|\n+", text)
    steps = [re.sub(r"\s+", " ", p).strip(" \t-•") for p in parts if p.strip()]
    if not steps:
        steps = [re.sub(r"\s+", " ", text)]
    return steps[:max_steps]


def _extract_steps(obj: Any, fallback_text: str) -> List[str]:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
        steps = [re.sub(r"\s+", " ", str(s)).strip() for s in obj["steps"] if str(s).strip()]
        if steps:
            return steps
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
            steps = [re.sub(r"\s+", " ", str(s)).strip() for s in parsed["steps"] if str(s).strip()]
            if steps:
                return steps
    return _split_fallback(fallback_text)


def _system_prompt() -> str:
    return (
        "You are a meticulous clinical reasoning editor. Convert a diagnostic explanation into "
        "an ordered list of atomic reasoning steps.\n"
        "Rules:\n"
        "1) Preserve meaning and evidence from the explanation.\n"
        "2) Do not introduce facts that are not already present.\n"
        "3) Each step should be one concise sentence.\n"
        "4) Separate observations, interpretations, and elimination logic whenever possible.\n"
        "5) Return only JSON following the provided schema."
    )


def _user_prompt(item: Dict[str, Any], explanation: str) -> str:
    title = str(item.get("title") or "").strip()
    history = str(item.get("CLINICAL_HISTORY") or item.get("clinical_history") or "").strip()
    findings = str(item.get("IMAGING_FINDINGS") or item.get("imaging_findings") or "").strip()
    return (
        "Task: rewrite the explanation below into ordered atomic steps.\n\n"
        "Context is for referent clarity only. Do not add facts beyond the explanation.\n"
        f"- Title: {title}\n"
        f"- Clinical history: {history}\n"
        f"- Imaging findings: {findings}\n\n"
        "Explanation:\n"
        "<<<\n"
        f"{explanation.strip()}\n"
        ">>>\n\n"
        "Return only JSON that matches the schema."
    )


def split_reasoning_steps(
    input_json: str,
    output_json: Optional[str] = None,
    *,
    model_name: str = "gpt-5-mini",
    reasoning_effort: Optional[str] = None,
    input_field: str = "parsed_explanation",
    output_field: str = "explanation_steps",
    count_field: str = "explanation_step_count",
    max_concurrency: int = 32,
) -> str:
    input_path = Path(input_json).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve() if output_json else input_path

    original, items = _load_payload(input_path)

    prompts: List[Tuple[str, str, Optional[List[str]]]] = []
    item_indices: List[int] = []
    for idx, item in enumerate(items):
        explanation = str(item.get(input_field) or "").strip()
        if not explanation:
            item[output_field] = []
            item[count_field] = 0
            continue
        prompts.append((_system_prompt(), _user_prompt(item, explanation), None))
        item_indices.append(idx)

    if prompts:
        llm = APIModel(
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        results = llm.generate(
            prompts=prompts,
            schema=StepList,
            show_progress=True,
            max_concurrency=max_concurrency,
        )
    else:
        results = []

    for local_idx, result in enumerate(results):
        obj = result[0] if isinstance(result, (list, tuple)) and result else result
        item = items[item_indices[local_idx]]
        explanation = str(item.get(input_field) or "").strip()
        steps = _extract_steps(obj, explanation)
        item[output_field] = steps
        item[count_field] = len(steps)

    _save_payload(output_path, original, items)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split free-form reasoning into atomic steps.")
    parser.add_argument("--input_json", required=True, help="Input JSON file.")
    parser.add_argument("--output_json", default=None, help="Output JSON file. Defaults to in-place.")
    parser.add_argument("--model", default="gpt-5-mini", help="Judge model for step splitting.")
    parser.add_argument("--reasoning_effort", default=None, help="Optional reasoning effort for APIModel.")
    parser.add_argument("--input_field", default="parsed_explanation", help="Field that stores the free-form explanation.")
    parser.add_argument("--output_field", default="explanation_steps", help="Field to write the step list into.")
    parser.add_argument("--count_field", default="explanation_step_count", help="Field to write the step count into.")
    parser.add_argument("--max_concurrency", type=int, default=32, help="Max concurrent LLM requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = split_reasoning_steps(
        input_json=args.input_json,
        output_json=args.output_json,
        model_name=args.model,
        reasoning_effort=args.reasoning_effort,
        input_field=args.input_field,
        output_field=args.output_field,
        count_field=args.count_field,
        max_concurrency=max(1, int(args.max_concurrency)),
    )
    print(json.dumps({"output_json": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
