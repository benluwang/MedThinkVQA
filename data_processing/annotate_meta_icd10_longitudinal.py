#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.gpt import APIModel


DEFAULT_APPENDIX = "data_processing/files/icd10_who_2019_chapter_block_category_appendix.txt"

CHAPTER_RE = re.compile(r"^Chapter\s+([IVXLC]+)\s+(.+?)\s+\(([A-Z0-9]{3}-[A-Z0-9]{3})\)\s*$")
BLOCK_RE = re.compile(r"^Block \|\s+([A-Z0-9]{3}-[A-Z0-9]{3})\s+(.+?)\s*$")
CATEGORY_RE = re.compile(r"^\s*([A-Z][0-9][0-9A-Z])\s+(.+?)\s*$")


@dataclass(frozen=True)
class ICD10Entry:
    chapter_label: str
    chapter_roman: str
    chapter_title: str
    chapter_range: str
    block_range: str
    block_title: str
    category_code: str
    category_title: str


class ICD10RawOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chapter: str = Field(..., description="Chapter label, e.g., Chapter I")
    block: str = Field(..., description="Block range, e.g., A20-A28")
    category: str = Field(..., description="3-character ICD-10 category code, e.g., A20")
    evidence: str = Field(..., description="One short evidence sentence from provided text")


class LongitudinalRawOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_longitudinal: Literal["yes", "no"] = Field(..., description="yes or no only")
    timepoint_count: int = Field(..., ge=0, description="Count of distinct timepoints/visit nodes")
    timepoint_evidence: List[str] = Field(
        ..., description="Time expressions/visit markers quoted from provided text"
    )
    interval_text: Optional[str] = Field(
        ...,
        description=(
            "Human-readable time interval from first to last timepoint; "
            "'unknown' if the interval is not explicitly stated; null when no"
        ),
    )
    evidence: str = Field(..., description="One short evidence sentence from provided text")


class CaseRawAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    icd10: ICD10RawOutput
    longitudinal: LongitudinalRawOutput


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s).strip())


def _strip_text(x: Any) -> str:
    return str(x or "").strip()


def _case_title(row: Dict[str, str]) -> str:
    return _strip_text(row.get("\ufeffcase_title") or row.get("case_title"))


def _count_csv_rows(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def _iter_selected_rows(
    csv_path: str,
    start: int = 0,
    limit: Optional[int] = None,
) -> Iterator[Tuple[int, Dict[str, str]]]:
    seen = 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if row_idx < start:
                continue
            if limit is not None and seen >= limit:
                break
            seen += 1
            yield row_idx, row


def _extract_image_captions(row: Dict[str, str]) -> List[str]:
    captions: List[str] = []
    for i in range(1, 50):
        c = _strip_text(row.get(f"image{i}_caption"))
        if c:
            captions.append(f"image{i}: {c}")
    return captions


def _build_case_payload(row_idx: int, row: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], str]:
    history = _strip_text(row.get("clinical_history"))
    findings = _strip_text(row.get("imaging_findings"))
    final_dx = _strip_text(row.get("final_diagnosis"))
    captions = _extract_image_captions(row)

    if not final_dx:
        return None, "missing_final_diagnosis"
    if not any([history, findings, final_dx, captions]):
        return None, "empty_case"

    payload = {
        "source_row_index": int(row_idx),
        "case_title": _case_title(row),
        "link": _strip_text(row.get("link")),
        "case_date": _strip_text(row.get("case_date")),
        "clinical_history": history,
        "imaging_findings": findings,
        "image_captions": captions,
        "final_diagnosis": final_dx,
    }
    return payload, "ok"


def _load_appendix_and_mapping(path: str) -> Tuple[str, Dict[str, ICD10Entry]]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"ICD appendix not found: {p}")

    appendix_text = p.read_text(encoding="utf-8")
    mapping: Dict[str, ICD10Entry] = {}

    chapter_label = ""
    chapter_roman = ""
    chapter_title = ""
    chapter_range = ""
    block_range = ""
    block_title = ""

    for line_no, raw_line in enumerate(appendix_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        m_ch = CHAPTER_RE.match(stripped)
        if m_ch:
            chapter_roman = m_ch.group(1).strip()
            chapter_title = m_ch.group(2).strip()
            chapter_range = m_ch.group(3).strip()
            chapter_label = f"Chapter {chapter_roman}"
            block_range = ""
            block_title = ""
            continue

        m_blk = BLOCK_RE.match(stripped)
        if m_blk:
            if not chapter_label:
                raise ValueError(f"Block found before chapter at line {line_no}: {raw_line}")
            block_range = m_blk.group(1).strip()
            block_title = m_blk.group(2).strip()
            continue

        m_cat = CATEGORY_RE.match(raw_line)
        if m_cat:
            if not chapter_label or not block_range:
                raise ValueError(f"Category found before chapter/block at line {line_no}: {raw_line}")
            code = m_cat.group(1).strip().upper()
            cat_title = m_cat.group(2).strip()
            if code in mapping:
                raise ValueError(f"Duplicate category code in appendix: {code}")
            mapping[code] = ICD10Entry(
                chapter_label=chapter_label,
                chapter_roman=chapter_roman,
                chapter_title=chapter_title,
                chapter_range=chapter_range,
                block_range=block_range,
                block_title=block_title,
                category_code=code,
                category_title=cat_title,
            )
            continue

    if not mapping:
        raise RuntimeError("Failed to parse ICD appendix: no 3-character categories found.")

    return appendix_text, mapping


def _build_system_prompt() -> str:
    return (
        "You are a rigorous medical annotation assistant. "
        "Use only the provided fields. Do not use external knowledge and do not invent facts. "
        "You must return strict JSON that matches the schema exactly."
    )


def _build_user_prompt(case_payload: Dict[str, Any], appendix_text: str) -> str:
    captions: List[str] = case_payload.get("image_captions") or []
    if captions:
        captions_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(captions))
    else:
        captions_block = "(none)"

    history = case_payload.get("clinical_history") or ""
    findings = case_payload.get("imaging_findings") or ""
    final_dx = case_payload.get("final_diagnosis") or ""

    return (
        "Please annotate the case using the following text fields.\n\n"
        "[Clinical history]\n"
        f"{history}\n\n"
        "[All image captions]\n"
        f"{captions_block}\n\n"
        "Note: prefixes like image1/image2/... are only image identifiers, not chronological order, "
        "and must not be used as time evidence for longitudinal judgment. "
        "Also, do not infer multiple timepoints from having multiple images/captions; "
        "count timepoints only when the text explicitly indicates different visits/exams over time.\n\n"
        "[Image findings]\n"
        f"{findings}\n\n"
        "[Final diagnosis]\n"
        f"{final_dx}\n\n"

        "Task 1: ICD-10\n"
        "1) Output chapter, block, and exactly one 3-character category code (for example, Chapter I, A20-A28, A20).\n"
        "2) chapter and block must exactly match that category in the appendix.\n"
        "3) Make the best classification choice by jointly considering final diagnosis and other provided content.\n"

        "Task 2: Longitudinal case\n"
        "1) This label is meant for time-study / follow-up cases. "
        "Set is_longitudinal to yes only if the text indicates >=2 distinct timepoints for the same patient, "
        "typically baseline vs follow-up / repeat imaging / post-treatment / post-operative reassessment.\n"
        "2) Do NOT label as longitudinal solely because the narrative describes multiple tests, modalities, "
        "or a multi-step evaluation/treatment workflow. Treat such sequences as a single timepoint unless the text "
        "explicitly indicates different visits/exams separated in time.\n"
        "3) timepoint_evidence must quote explicit temporal markers (e.g., \"follow-up\", \"6 weeks later\", "
        "\"after 1 year\", \"post-op day 3\", \"repeat CT\", \"compared to prior\"). "
        "Do not use modality/procedure names alone as time evidence.\n"
        "4) If yes, return interval_text for the first-to-last timepoint span. "
        "If the interval is not explicitly stated, return \"unknown\".\n"
        "5) If no, interval_text must be null.\n\n"
        "Return structured JSON only, with no extra text.\n\n"
        "=== WHO ICD-10 Appendix Reference (must follow this exact chapter/block/category indexing) ===\n"
        f"{appendix_text}"
    )


def _normalize_icd_category(raw: Any) -> str:
    s = str(raw or "").upper()
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^([A-Z][0-9][0-9A-Z])(?:[.\-][0-9A-Z]+)?$", s)
    if m:
        return m.group(1)
    m = re.search(r"([A-Z][0-9][0-9A-Z])", s)
    return m.group(1) if m else ""


def _normalize_block(raw: Any) -> str:
    s = str(raw or "").upper()
    m = re.search(r"([A-Z0-9]{3}-[A-Z0-9]{3})", s)
    return m.group(1) if m else ""


def _normalize_chapter(raw: Any) -> str:
    s = str(raw or "").strip()
    up = s.upper()
    m = re.search(r"CHAPTER\s*([IVXLC]+)", up)
    if m:
        return f"Chapter {m.group(1)}"
    if re.fullmatch(r"[IVXLC]+", up):
        return f"Chapter {up}"
    return ""


def _postprocess_annotation(
    raw_obj: Any,
    icd_mapping: Dict[str, ICD10Entry],
) -> Dict[str, Any]:
    if not isinstance(raw_obj, dict):
        return {
            "error": f"non_dict_response: {type(raw_obj).__name__}",
            "raw_response": raw_obj,
        }

    icd_raw = raw_obj.get("icd10") or {}
    long_raw = raw_obj.get("longitudinal") or {}

    raw_category = icd_raw.get("category", "")
    norm_category = _normalize_icd_category(raw_category)
    icd_entry = icd_mapping.get(norm_category)

    raw_chapter = icd_raw.get("chapter", "")
    raw_block = icd_raw.get("block", "")

    icd_out: Dict[str, Any] = {
        "category_code": norm_category or None,
        "is_valid_category_in_appendix": bool(icd_entry),
        "model_raw": {
            "chapter": raw_chapter,
            "block": raw_block,
            "category": raw_category,
            "evidence": icd_raw.get("evidence"),
        },
    }

    if icd_entry:
        chapter_match = _normalize_chapter(raw_chapter) == icd_entry.chapter_label
        block_match = _normalize_block(raw_block) == icd_entry.block_range
        icd_out.update(
            {
                "chapter": icd_entry.chapter_label,
                "chapter_roman": icd_entry.chapter_roman,
                "chapter_title": icd_entry.chapter_title,
                "chapter_range": icd_entry.chapter_range,
                "block": icd_entry.block_range,
                "block_title": icd_entry.block_title,
                "category": icd_entry.category_code,
                "category_title": icd_entry.category_title,
                "model_chapter_match_appendix": chapter_match,
                "model_block_match_appendix": block_match,
            }
        )
    else:
        icd_out.update(
            {
                "chapter": None,
                "chapter_roman": None,
                "chapter_title": None,
                "chapter_range": None,
                "block": None,
                "block_title": None,
                "category": None,
                "category_title": None,
                "model_chapter_match_appendix": False,
                "model_block_match_appendix": False,
            }
        )

    long_yes_no = str(long_raw.get("is_longitudinal", "")).strip().lower()
    is_longitudinal = "yes" if long_yes_no == "yes" else "no"
    timepoint_count_raw = long_raw.get("timepoint_count", 0)
    try:
        timepoint_count = max(0, int(timepoint_count_raw))
    except Exception:
        timepoint_count = 0

    timepoint_evidence = long_raw.get("timepoint_evidence")
    if not isinstance(timepoint_evidence, list):
        timepoint_evidence = []
    timepoint_evidence = [str(x).strip() for x in timepoint_evidence if str(x).strip()]

    interval_text = long_raw.get("interval_text")

    if is_longitudinal == "no":
        interval_text = None
    else:
        if interval_text is None or (isinstance(interval_text, str) and not interval_text.strip()):
            interval_text = "unknown"

    long_out = {
        "is_longitudinal": is_longitudinal,
        "timepoint_count": timepoint_count,
        "timepoint_evidence": timepoint_evidence,
        "interval_text": interval_text,
        "meets_minimum_timepoint_rule": (timepoint_count >= 2) if is_longitudinal == "yes" else True,
        "model_raw": {
            "is_longitudinal": long_raw.get("is_longitudinal"),
            "timepoint_count": long_raw.get("timepoint_count"),
            "timepoint_evidence": long_raw.get("timepoint_evidence"),
            "interval_text": long_raw.get("interval_text"),
            "evidence": long_raw.get("evidence"),
        },
    }

    return {
        "icd10": icd_out,
        "longitudinal": long_out,
    }


def _default_out_jsonl(model_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = _sanitize_name(str(model_name).split("/")[-1])
    return os.path.join("data_processing", "files", f"meta_icd10_longitudinal_{safe_model}_{ts}.jsonl")


def _default_summary_json(out_jsonl: str) -> str:
    p = Path(out_jsonl).expanduser().resolve()
    return str(p.with_suffix(".summary.json"))


def _ensure_parent(path: str) -> str:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Use gpt-5.2 to annotate ICD-10 (chapter/block/3-char category) and "
            "longitudinal-case status from a case metadata CSV without images or differential diagnosis."
        )
    )
    p.add_argument("--csv_path", type=str, required=True, help="Input case metadata CSV")
    p.add_argument(
        "--appendix_path",
        type=str,
        default=DEFAULT_APPENDIX,
        help="WHO ICD-10 appendix text file to append at end of every prompt",
    )
    p.add_argument("--out_jsonl", type=str, default=None, help="Output JSONL path")
    p.add_argument("--summary_json", type=str, default=None, help="Output summary JSON path")
    p.add_argument("--model", type=str, default="gpt-5.2", help="Model name for APIModel")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--start", type=int, default=0, help="Start row index (inclusive)")
    p.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    p.add_argument(
        "--prepare_only",
        action="store_true",
        help="Validate input/appendix and print one sample prompt; skip model inference",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = str(Path(args.csv_path).expanduser().resolve())
    appendix_path = str(Path(args.appendix_path).expanduser().resolve())
    out_jsonl = _ensure_parent(args.out_jsonl or _default_out_jsonl(args.model))
    summary_json = _ensure_parent(args.summary_json or _default_summary_json(out_jsonl))

    appendix_text, icd_mapping = _load_appendix_and_mapping(appendix_path)

    total_rows = _count_csv_rows(csv_path)
    selected_total = max(0, total_rows - max(0, int(args.start)))
    if args.limit is not None:
        selected_total = min(selected_total, max(0, int(args.limit)))

    print(
        f"[DATA] csv={csv_path} total_rows={total_rows} selected_rows={selected_total} "
        f"range=[{max(0, int(args.start))}, {max(0, int(args.start)) + selected_total})"
    )
    print(
        f"[ICD] appendix={appendix_path} parsed_categories={len(icd_mapping)} "
        f"appendix_chars={len(appendix_text)}"
    )
    print(
        f"[CONFIG] model={args.model} temperature={args.temperature}"
    )
    print(f"[OUT] jsonl={out_jsonl}")
    print(f"[OUT] summary={summary_json}")

    if selected_total <= 0:
        raise RuntimeError("No rows selected. Check --start/--limit.")

    first_row_iter = _iter_selected_rows(csv_path, start=max(0, int(args.start)), limit=1)
    first_item = next(first_row_iter, None)
    if first_item is None:
        raise RuntimeError("Could not read selected rows from CSV.")
    _, first_row = first_item
    first_payload, first_reason = _build_case_payload(max(0, int(args.start)), first_row)
    if first_payload is None:
        raise RuntimeError(f"First selected row is not valid for prompting: {first_reason}")
    first_prompt = _build_user_prompt(first_payload, appendix_text)
    print(f"[PROMPT] first_prompt_chars={len(first_prompt)}")

    if args.prepare_only:
        print("[PREPARE_ONLY] Input and prompt validation finished. No model calls were made.")
        return

    llm = APIModel(
        model_name=args.model,
        temperature=float(args.temperature),
    )

    stats: Dict[str, int] = {
        "selected_rows": int(selected_total),
        "processed_cases": 0,
        "successful_cases": 0,
        "valid_icd10_cases": 0,
        "longitudinal_yes_cases": 0,
        "model_error_cases": 0,
        "parse_error_cases": 0,
        "skipped_missing_final_diagnosis": 0,
        "skipped_empty_case": 0,
    }

    prompts: List[Tuple[str, str, Optional[List[str]]]] = []
    payloads: List[Dict[str, Any]] = []
    system_prompt = _build_system_prompt()

    for row_idx, row in _iter_selected_rows(
        csv_path=csv_path,
        start=max(0, int(args.start)),
        limit=args.limit,
    ):
        payload, reason = _build_case_payload(row_idx, row)
        if payload is None:
            if reason == "missing_final_diagnosis":
                stats["skipped_missing_final_diagnosis"] += 1
            else:
                stats["skipped_empty_case"] += 1
            continue

        user_prompt = _build_user_prompt(payload, appendix_text)
        prompts.append((system_prompt, user_prompt, None))
        payloads.append(payload)

    if not prompts:
        raise RuntimeError("No valid rows available for model inference after filtering.")

    outputs = llm.generate(
        prompts=prompts,
        schema=CaseRawAnnotation,
        show_progress=(not args.no_progress),
    )
    if len(outputs) != len(payloads):
        raise RuntimeError(f"Output length mismatch: {len(outputs)} vs {len(payloads)}")

    with open(out_jsonl, "w", encoding="utf-8") as writer:
        for payload, (resp, in_tok, out_tok) in zip(payloads, outputs):
            stats["processed_cases"] += 1

            record: Dict[str, Any] = {
                "source_row_index": payload["source_row_index"],
                "case_title": payload["case_title"],
                "link": payload["link"],
                "case_date": payload["case_date"],
                "input_used": {
                    "clinical_history": payload["clinical_history"],
                    "image_captions": payload["image_captions"],
                    "imaging_findings": payload["imaging_findings"],
                    "final_diagnosis": payload["final_diagnosis"],
                },
                "tokens": {
                    "input_tokens": int(in_tok or 0),
                    "output_tokens": int(out_tok or 0),
                },
            }

            if isinstance(resp, dict) and "__error__" in resp:
                stats["model_error_cases"] += 1
                record["error"] = resp.get("__error__")
            else:
                annotation = _postprocess_annotation(raw_obj=resp, icd_mapping=icd_mapping)
                if "error" in annotation:
                    stats["parse_error_cases"] += 1
                else:
                    stats["successful_cases"] += 1
                    if annotation["icd10"].get("is_valid_category_in_appendix"):
                        stats["valid_icd10_cases"] += 1
                    if annotation["longitudinal"].get("is_longitudinal") == "yes":
                        stats["longitudinal_yes_cases"] += 1
                record["annotation"] = annotation

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "csv_path": csv_path,
        "appendix_path": appendix_path,
        "out_jsonl": out_jsonl,
        "model": args.model,
        "temperature": float(args.temperature),
        "start": int(args.start),
        "limit": args.limit if args.limit is not None else None,
        "stats": stats,
        "token_usage": {
            "requests_made": int(getattr(llm, "requests_made", 0)),
            "tokens_used": int(getattr(llm, "tokens_used", 0)),
            "cached_input_tokens_used": int(getattr(llm, "cached_input_tokens_used", 0)),
            "reasoning_output_tokens_used": int(getattr(llm, "reasoning_output_tokens_used", 0)),
            "estimated_total_cost_usd": float(getattr(llm, "total_cost_usd", 0.0)),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[DONE] processed={stats['processed_cases']} success={stats['successful_cases']} "
        f"valid_icd10={stats['valid_icd10_cases']} longitudinal_yes={stats['longitudinal_yes_cases']} "
        f"model_error={stats['model_error_cases']} parse_error={stats['parse_error_cases']}"
    )
    print(f"[SAVED] jsonl={out_jsonl}")
    print(f"[SAVED] summary={summary_json}")


if __name__ == "__main__":
    main()
