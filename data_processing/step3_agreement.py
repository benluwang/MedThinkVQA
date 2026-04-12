#!/usr/bin/env python3

"""Step 3 helper: compare human step labels with LLM-judge outputs.

This public snapshot keeps only the core post-hoc metrics used in Step 3:
- step-level factual agreement
- MCC / Cohen's kappa
- critical-step ratio
- human error-type counts

The human annotation table is intentionally strict and expects the following
column names:
- `title`
- `clinical_history`
- `step1`, `step2`, ...
- `human_factual1`, `human_factual2`, ...
- optional `human_critical1`, `human_critical2`, ...
- optional `human_error_types1`, `human_error_types2`, ...
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\u200b", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text.strip().lower())


def _match_key(title: Any, history: Any) -> str:
    return f"{_norm_text(title)}||{_norm_text(history)}"


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = _norm_text(value)
    if text in {"true", "t", "1", "yes", "y", "correct", "critical"}:
        return True
    if text in {"false", "f", "0", "no", "n", "incorrect"}:
        return False
    return None


def _parse_err_types(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[;,/|]+", text)
    out = set()
    for part in parts:
        token = _norm_text(part)
        if not token:
            continue
        if "reason" in token:
            out.add("Reasoning Err")
        elif "image" in token:
            out.add("Image Understanding Err")
        elif "clinic" in token or "scenario" in token or "senario" in token:
            out.add("Clinical Scenario Err")
        elif "medical" in token or "knowledge" in token:
            out.add("Medical Knowledge Err")
        else:
            out.add(part.strip())
    return sorted(out)


def _load_judge_json(path: Path) -> Dict[str, Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        items = raw["data"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    index: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _match_key(item.get("title"), item.get("CLINICAL_HISTORY") or item.get("clinical_history"))
        if key:
            index.setdefault(key, item)
    return index


def _column_lookup(columns: List[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for col in columns:
        key = str(col).strip().lower()
        if key:
            lookup[key] = col
    return lookup


def _collect_step_numbers(columns: List[str]) -> List[int]:
    numbers = {
        int(match.group(1))
        for col in columns
        for match in [re.fullmatch(r"step(\d+)", str(col).strip().lower())]
        if match
    }
    if not numbers:
        raise ValueError("Human table must contain `step1`, `step2`, ... columns.")
    return sorted(numbers)


def _required_column(lookup: Dict[str, str], name: str) -> str:
    col = lookup.get(name)
    if col is None:
        raise ValueError(f"Missing required human-table column: `{name}`")
    return col


def _confusion_counts(y_true: List[bool], y_pred: List[bool]) -> Dict[str, int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for gold, pred in zip(y_true, y_pred):
        if gold and pred:
            counts["tp"] += 1
        elif (not gold) and (not pred):
            counts["tn"] += 1
        elif (not gold) and pred:
            counts["fp"] += 1
        else:
            counts["fn"] += 1
    return counts


def _agreement(y_true: List[bool], y_pred: List[bool]) -> float:
    if not y_true:
        return float("nan")
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def _phi_mcc(y_true: List[bool], y_pred: List[bool]) -> float:
    counts = _confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return float("nan")
    return (tp * tn - fp * fn) / math.sqrt(denom)


def _cohens_kappa(y_true: List[bool], y_pred: List[bool]) -> float:
    if not y_true:
        return float("nan")
    po = _agreement(y_true, y_pred)
    n = len(y_true)
    p_true = sum(1 for x in y_true if x) / n
    p_pred = sum(1 for x in y_pred if x) / n
    pe = p_true * p_pred + (1 - p_true) * (1 - p_pred)
    if 1 - pe == 0:
        return float("nan")
    return (po - pe) / (1 - pe)


def compare_human_and_judge(
    human_table: str,
    judge_json: str,
    *,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    table_path = Path(human_table).expanduser().resolve()
    judge_path = Path(judge_json).expanduser().resolve()

    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(table_path)
    else:
        df = pd.read_csv(table_path)
    if limit is not None and limit > 0:
        df = df.iloc[:limit].copy()

    columns = list(df.columns)
    lookup = _column_lookup(columns)
    title_col = _required_column(lookup, "title")
    history_col = _required_column(lookup, "clinical_history")
    step_numbers = _collect_step_numbers(columns)
    for step_no in step_numbers:
        _required_column(lookup, f"human_factual{step_no}")

    judge_index = _load_judge_json(judge_path)
    gold: List[bool] = []
    pred: List[bool] = []
    debug_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        key = _match_key(row.get(title_col), row.get(history_col))
        judged_item = judge_index.get(key)
        if judged_item is None:
            continue
        checks = judged_item.get("explanation_steps_checks") or []
        if not isinstance(checks, list):
            continue

        for step_no in step_numbers:
            human_step_col = lookup[f"step{step_no}"]
            factual_col = lookup[f"human_factual{step_no}"]
            critical_col = lookup.get(f"human_critical{step_no}")
            err_col = lookup.get(f"human_error_types{step_no}")

            gold_bool = _coerce_bool(row.get(factual_col))
            if gold_bool is None:
                continue

            judge_idx = step_no - 1
            if judge_idx < 0 or judge_idx >= len(checks):
                continue
            judge_step = checks[judge_idx]
            if not isinstance(judge_step, dict):
                continue

            pred_bool = _coerce_bool(judge_step.get("is_factual"))
            if pred_bool is None:
                continue

            human_critical = _coerce_bool(row.get(critical_col)) if critical_col else False
            human_err_types = _parse_err_types(row.get(err_col)) if err_col else []

            gold.append(gold_bool)
            pred.append(pred_bool)
            debug_rows.append(
                {
                    "title": row.get(title_col),
                    "clinical_history": row.get(history_col),
                    "step": step_no,
                    "step_text": row.get(human_step_col),
                    "human_factual": gold_bool,
                    "judge_factual": pred_bool,
                    "human_critical": bool(human_critical),
                    "human_error_types": "|".join(human_err_types),
                }
            )

    critical_rows = [row for row in debug_rows if row["human_critical"]]
    err_counts: Dict[str, int] = {}
    crit_err_counts: Dict[str, int] = {}
    for row in debug_rows:
        for err in {x for x in row["human_error_types"].split("|") if x}:
            err_counts[err] = err_counts.get(err, 0) + 1
    for row in critical_rows:
        for err in {x for x in row["human_error_types"].split("|") if x}:
            crit_err_counts[err] = crit_err_counts.get(err, 0) + 1

    summary = {
        "matched_step_pairs": len(gold),
        "agreement": _agreement(gold, pred),
        "phi_mcc": _phi_mcc(gold, pred),
        "cohens_kappa": _cohens_kappa(gold, pred),
        "confusion": _confusion_counts(gold, pred),
        "critical_step_ratio": (len(critical_rows) / len(debug_rows)) if debug_rows else float("nan"),
        "error_type_counts": err_counts,
        "critical_error_type_counts": crit_err_counts,
    }

    if output_csv:
        out_csv = Path(output_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(debug_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    if output_json:
        out_json = Path(output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare human Step 3 labels with LLM-judge outputs. "
            "Expected human columns: title, clinical_history, stepN, human_factualN, "
            "and optional human_criticalN / human_error_typesN."
        )
    )
    parser.add_argument("--human_table", required=True, help="CSV or XLSX file with human step labels.")
    parser.add_argument("--judge_json", required=True, help="JSON or JSONL file with `explanation_steps_checks`.")
    parser.add_argument("--output_json", default=None, help="Optional JSON summary path.")
    parser.add_argument("--output_csv", default=None, help="Optional step-level debug CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for the human table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare_human_and_judge(
        human_table=args.human_table,
        judge_json=args.judge_json,
        output_json=args.output_json,
        output_csv=args.output_csv,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
