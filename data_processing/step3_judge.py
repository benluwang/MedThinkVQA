#!/usr/bin/env python3

"""Step 3 helper: judge stepwise reasoning with factuality and error taxonomy.

This file is a cleaned public snapshot of the LLM-judge logic used for Step 3
analysis in MedThinkVQA. It preserves the core taxonomy and prompt structure
while omitting internal experiment-specific layout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.gpt import APIModel


ErrorType = Literal[
    "Reasoning Err",
    "Image Understanding Err",
    "Clinical Scenario Err",
    "Medical Knowledge Err",
]


class StepVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_factual: bool = Field(..., description="Whether the step is supported by the provided case context.")
    is_critical: bool = Field(..., description="Whether the step is essential for the final diagnosis.")
    explanation: str = Field(..., min_length=2, max_length=300, description="Brief justification.")
    error_types: Optional[List[ErrorType]] = Field(
        default=None,
        min_length=1,
        description="Only include when is_factual is false.",
    )


class StepJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: List[StepVerdict] = Field(..., min_length=1)


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


def _collect_captions(item: Dict[str, Any]) -> List[str]:
    captions: List[str] = []
    seen = set()
    for image in item.get("img", []) or []:
        if not isinstance(image, dict):
            continue
        for field in ("img_alt2", "img_alt"):
            value = str(image.get(field) or "").strip()
            if value and value not in seen:
                seen.add(value)
                captions.append(value)
    return captions


def _guess_final_diagnosis(item: Dict[str, Any]) -> str:
    for field in ("correct_answer_text", "final_diagnosis", "diagnosis", "dx"):
        value = str(item.get(field) or "").strip()
        if value:
            return value
    discussion = str(item.get("DISCUSSION") or item.get("discussion") or "").strip()
    if discussion:
        match = re.search(r"(?:final\s+diagnosis|diagnosis)\s*:\s*(.+)", discussion, flags=re.I)
        if match:
            return match.group(1).strip()
    return "(not provided)"


def _system_prompt() -> str:
    return (
        "You are an evaluator for radiology case reasoning.\n"
        "For each reasoning step, judge whether it is correct or reasonably supported by the provided case context.\n"
        "Rules:\n"
        "1) Mark True when the step is explicitly supported, correctly implied, or medically reasonable from the case.\n"
        "2) Mark False only when it is clearly wrong, contradictory, or unsupported.\n"
        "3) Also mark whether the step is critical for reaching the final diagnosis.\n"
        "4) If a step is incorrect, assign one or more error types from exactly:\n"
        "   ['Reasoning Err', 'Image Understanding Err', 'Clinical Scenario Err', 'Medical Knowledge Err'].\n"
        "5) Return only JSON following the schema."
    )


def _user_prompt(item: Dict[str, Any], steps: List[str]) -> str:
    title = str(item.get("title") or "").strip()
    history = str(item.get("CLINICAL_HISTORY") or item.get("clinical_history") or "").strip()
    findings = str(item.get("IMAGING_FINDINGS") or item.get("imaging_findings") or "").strip()
    discussion = str(item.get("DISCUSSION") or item.get("discussion") or "").strip()
    final_dx = _guess_final_diagnosis(item)
    captions = _collect_captions(item)

    steps_block = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
    captions_block = "\n".join(f"- {caption}" for caption in captions) if captions else "(none)"

    return (
        "Task: judge each reasoning step against the case context.\n\n"
        f"- Title: {title}\n"
        f"- Clinical history: {history}\n"
        f"- Imaging findings: {findings}\n"
        f"- Discussion: {discussion}\n"
        f"- Final diagnosis: {final_dx}\n"
        f"- Captions:\n{captions_block}\n\n"
        "Steps to judge:\n"
        f"{steps_block}\n\n"
        "Return one verdict per step, in order, using the schema only."
    )


def _extract_verdicts(obj: Any) -> List[Dict[str, Any]]:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, dict) and isinstance(obj.get("verdicts"), list):
        verdicts = []
        for raw in obj["verdicts"]:
            if not isinstance(raw, dict):
                continue
            verdict: Dict[str, Any] = {
                "is_factual": bool(raw.get("is_factual")),
                "is_critical": bool(raw.get("is_critical")),
                "explanation": re.sub(r"\s+", " ", str(raw.get("explanation") or "")).strip(),
            }
            if verdict["is_factual"] is False and isinstance(raw.get("error_types"), list):
                verdict["error_types"] = [str(x).strip() for x in raw["error_types"] if str(x).strip()]
            verdicts.append(verdict)
        return verdicts
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except Exception:
            return []
        return _extract_verdicts(parsed)
    return []


def _default_verdicts(n_steps: int) -> List[Dict[str, Any]]:
    return [
        {
            "is_factual": False,
            "is_critical": False,
            "explanation": "No valid structured verdict was returned.",
        }
        for _ in range(n_steps)
    ]


def _pad_or_trim(verdicts: List[Dict[str, Any]], n_steps: int) -> List[Dict[str, Any]]:
    verdicts = verdicts[:n_steps]
    while len(verdicts) < n_steps:
        verdicts.append(
            {
                "is_factual": False,
                "is_critical": False,
                "explanation": "Missing verdict; treated as unsupported.",
            }
        )
    return verdicts


def judge_reasoning_steps(
    input_json: str,
    output_json: Optional[str] = None,
    *,
    model_name: str = "gpt-5",
    reasoning_effort: Optional[str] = None,
    steps_field: str = "explanation_steps",
    output_field: str = "explanation_steps_checks",
    max_concurrency: int = 16,
) -> str:
    input_path = Path(input_json).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve() if output_json else input_path

    original, items = _load_payload(input_path)
    prompts: List[Tuple[str, str, Optional[List[str]]]] = []
    item_indices: List[int] = []

    for idx, item in enumerate(items):
        steps = item.get(steps_field) or []
        if not isinstance(steps, list):
            steps = []
        steps = [re.sub(r"\s+", " ", str(step)).strip() for step in steps if str(step).strip()]
        if not steps:
            item[output_field] = []
            continue
        prompts.append((_system_prompt(), _user_prompt(item, steps), None))
        item_indices.append(idx)

    if prompts:
        llm = APIModel(
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        results = llm.generate(
            prompts=prompts,
            schema=StepJudgeOutput,
            show_progress=True,
            max_concurrency=max_concurrency,
        )
    else:
        results = []

    for local_idx, result in enumerate(results):
        obj = result[0] if isinstance(result, (list, tuple)) and result else result
        item = items[item_indices[local_idx]]
        steps = item.get(steps_field) or []
        n_steps = len(steps)
        verdicts = _extract_verdicts(obj)
        if not verdicts:
            verdicts = _default_verdicts(n_steps)
        item[output_field] = _pad_or_trim(verdicts, n_steps)

    _save_payload(output_path, original, items)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge stepwise reasoning with factuality and error taxonomy.")
    parser.add_argument("--input_json", required=True, help="Input JSON file.")
    parser.add_argument("--output_json", default=None, help="Output JSON file. Defaults to in-place.")
    parser.add_argument("--model", default="gpt-5", help="Judge model.")
    parser.add_argument("--reasoning_effort", default=None, help="Optional reasoning effort for APIModel.")
    parser.add_argument("--steps_field", default="explanation_steps", help="Field that stores reasoning steps.")
    parser.add_argument("--output_field", default="explanation_steps_checks", help="Field to write verdicts into.")
    parser.add_argument("--max_concurrency", type=int, default=16, help="Max concurrent LLM requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = judge_reasoning_steps(
        input_json=args.input_json,
        output_json=args.output_json,
        model_name=args.model,
        reasoning_effort=args.reasoning_effort,
        steps_field=args.steps_field,
        output_field=args.output_field,
        max_concurrency=max(1, int(args.max_concurrency)),
    )
    print(json.dumps({"output_json": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
