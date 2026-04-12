# -*- coding: utf-8 -*-
"""Detect direct diagnosis leakage from text-only case metadata.

This public snapshot keeps the core leakage check used during MedThinkVQA
curation. The script supports JSON, JSONL, and top-level `{"data": [...]}` or
mapping-style payloads, normalizes options into `A`-`E` style labels, and asks
an LLM judge whether the history text directly reveals any candidate option.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple, Optional, Dict, Any, Iterable, Union

from pydantic import BaseModel, Field, ValidationError
from pydantic import ConfigDict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.gpt import APIModel


# =========================
# 1) Structured output schema (Pydantic v2, minimal & API-friendly)
# =========================

class LeakResult(BaseModel):
    """
    Minimal stable structure: only whether there is leakage and which options are leaked (uppercase A/B/C/...).
    """
    model_config = ConfigDict(extra="ignore")  # Ignore extra fields

    has_leak: bool = Field(
        ...,
        description="True if any option label (or its synonym) explicitly appears in the history text."
    )
    leaked_options: List[str] = Field(
        default_factory=list,
        description='Uppercase option letters that are directly revealed, e.g., ["A","C"].'
    )
    # Optional: short reason used when has_leak=false (not required)
    non_leak_reason: Optional[str] = Field(
        default=None,
        description="Optional explanation when has_leak=false."
    )


# =========================
# 2) Prompt construction
# =========================
SYSTEM_PROMPT = """You are a meticulous clinical QA auditor for multiple-choice diagnosis questions. Your job:
Given ONLY the CLINICAL HISTORY text and the list of candidate diagnosis OPTIONS, decide
whether the history text DIRECTLY REVEALS any option(s).

Definition of DIRECT REVEAL (diagnosis label appears in the text itself, not inferred):
• L3 Explicit label: the exact diagnosis name or ICD/standard label appears, or patterns like
  "Diagnosis: X", "biopsy-proven X".
• L2 Explicit synonym/acronym/eponym/foreign-language variant of the diagnosis label appears
  (e.g., "MI" for myocardial infarction; "Osler-Weber-Rendu" for HHT).
• L1 Explicit but uncertain mention of the diagnosis label (or its synonym/acronym/eponym):
  e.g., "?X", "r/o X", "rule out X", "query X", "suspected X", "possible/probable X",
  "consistent with X", "concern for X", "Hx of/known case of X".

NOT a leak: symptoms, signs, risk factors, imaging descriptors, or lab patterns that merely
SUGGEST a diagnosis. Only mark a leak if the diagnosis LABEL itself (or its standard
synonym/acronym/eponym) occurs in the text.

Use the OPTIONS solely as a dictionary of candidate labels and their widely-used
synonyms/acronyms/eponyms to search for DIRECT textual mentions. Do NOT infer diagnoses
from context. Do NOT mark based on reasoning.

For each leaked option, return:
- option_id, option_text
- overall leak_level (max severity across its evidences; L3>L2>L1)
- evidences: verbatim snippet(s) with [start,end) character indices into the EXACT Clinical
  history string
- a brief justification

If no option is leaked, set has_leak=false and provide non_leak_reason.

Return ONLY valid JSON following the required schema. No extra prose.
"""


def build_user_prompt(history: str,
                      options_dict: Dict[str, str],
                      letters_order: List[str]) -> str:
    """
    Build the user content according to the LaTeX appendix template.
    """
    opt_lines = [f"{k}) {options_dict.get(k, '')}" for k in letters_order]

    prompt = (
        "CLINICAL HISTORY (use this exact string when computing char spans):\n"
        "<<<<HISTORY>>>>\n"
        f"{history}\n"
        "<<<<END_HISTORY>>>>\n\n"
        "OPTIONS (candidate diagnoses; DO NOT infer—use only as label dictionary):\n"
        + "\n".join(opt_lines) +
        "\n\n"
        "Task: Identify ALL options (if any) that are directly revealed by the HISTORY text "
        "under L1/L2/L3 definitions. Extract verbatim evidence snippet(s) and 0-based [start,end) "
        "char spans into the exact HISTORY string above. If none, set has_leak=false."
    )
    return prompt


# =========================
# 3) Data loading & normalization utilities
# =========================

_HISTORY_KEYS = ("CLINICAL_HISTORY", "clinical_history", "ClinicalHistory", "history", "History")


def _first_non_empty(*vals: Any, default: str = "") -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
    return default


def _extract_history(sample: Dict[str, Any]) -> str:
    return _first_non_empty(*(sample.get(k) for k in _HISTORY_KEYS), default="")


def _letters(n: int) -> List[str]:
    """Generate A..Z..AA.. (MCQs rarely exceed 26; here we go to Z and extend if needed)."""
    # MCQs rarely exceed 26 options; extend to AA, AB... only if ever needed
    base = [chr(ord('A') + i) for i in range(min(n, 26))]
    if n <= 26:
        return base
    # If >26 (rare): continue with AA, AB, ...
    extra = []
    count = n - 26
    a_to_z = [chr(ord('A') + i) for i in range(26)]
    idx = 0
    while count > 0:
        extra.append('A' + a_to_z[idx % 26])
        idx += 1
        count -= 1
    return base + extra


def _coerce_options(any_obj: Any) -> Tuple[Dict[str, str], List[str], Dict[str, str], Dict[str, str]]:
    """
    Normalize various option shapes to {'A': text, 'B': text, ...}
    Returns: options_dict, letters_order, letter2orig, orig2letter

    Supports:
      - dict: may be {'A': '...','B':'...'} or {'1': '...'} or arbitrary keys
      - list[str]: ['...', '...']
      - list[dict]: [{'label':'A','text':'...'}] / [{'id':'1','value':'...'}] / [{'option':'...'}], etc.
    """
    options_dict: Dict[str, str] = {}
    letter2orig: Dict[str, str] = {}
    orig2letter: Dict[str, str] = {}

    if isinstance(any_obj, dict):
        items = list(any_obj.items())
        # Try to preserve original iteration order (Python 3.7+ dicts are ordered)
        # If keys are single letters, uppercase them; otherwise map to A, B, C, ...
        keys = [str(k) for k, _ in items]

        def is_alpha_letter(s: str) -> bool:
            return len(s) == 1 and s.isalpha()

        if all(is_alpha_letter(k) for k in keys):
            letters_order = [k.upper() for k in keys]
            for (k, v) in items:
                L = str(k).upper()
                options_dict[L] = str(v) if v is not None else ""
                letter2orig[L] = str(k)
                orig2letter[str(k)] = L
        else:
            letters_order = _letters(len(items))
            for (idx, (ok, v)) in enumerate(items):
                L = letters_order[idx]
                options_dict[L] = str(v) if v is not None else ""
                letter2orig[L] = str(ok)
                orig2letter[str(ok)] = L
        return options_dict, letters_order, letter2orig, orig2letter

    if isinstance(any_obj, list):
        # List may contain pure strings or dicts
        if all(isinstance(x, str) for x in any_obj):
            letters_order = _letters(len(any_obj))
            for i, txt in enumerate(any_obj):
                L = letters_order[i]
                options_dict[L] = txt
                letter2orig[L] = str(i)
                orig2letter[str(i)] = L
            return options_dict, letters_order, letter2orig, orig2letter

        if all(isinstance(x, dict) for x in any_obj):
            # Prefer recognizable fields
            letters_order: List[str] = []
            tmp: List[Tuple[str, str]] = []  # (orig_key, text)
            for i, d in enumerate(any_obj):
                # Possible field names
                cand_label = d.get("label") or d.get("id") or d.get("key") or d.get("option_id")
                cand_text = d.get("text") or d.get("value") or d.get("option") or d.get("desc") or d.get("content")
                if cand_text is None:
                    # Fallback: use the first non-empty string field
                    for vv in d.values():
                        if isinstance(vv, str) and vv.strip():
                            cand_text = vv
                            break
                if cand_label is None:
                    cand_label = str(i)
                tmp.append((str(cand_label), str(cand_text) if cand_text is not None else ""))

            # Map to letters
            letters_order = _letters(len(tmp))
            for i, (ok, txt) in enumerate(tmp):
                L = letters_order[i]
                options_dict[L] = txt
                letter2orig[L] = ok
                orig2letter[ok] = L
            return options_dict, letters_order, letter2orig, orig2letter

    # Fallback: empty or unknown shape
    return {}, [], {}, {}


def _normalize_correct_id(raw: Any,
                          letters_order: List[str],
                          orig2letter: Dict[str, str]) -> Optional[str]:
    """
    Normalize correct_answer into an uppercase letter; if it is an original key or a numeric index,
    convert via the provided mapping.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Already a letter
    if len(s) == 1 and s.isalpha():
        u = s.upper()
        return u if u in letters_order else u

    # Try original key -> letter
    if s in orig2letter:
        return orig2letter[s]

    # Common variants (e.g., "option_A", "A)", "A.")
    su = s.upper()
    if su.endswith(")") or su.endswith("."):
        su = su[:-1]
    if su.startswith("OPTION_") and len(su) >= 8 and su[-1].isalpha():
        su = su[-1]
    if len(su) == 1 and su.isalpha():
        return su

    # Numeric index strings: '0','1',... (may come from list index or numeric dict keys)
    if su.isdigit():
        if su in orig2letter:
            return orig2letter[su]
        try:
            idx = int(su)
            if 0 <= idx < len(letters_order):
                return letters_order[idx]
        except Exception:
            pass

    return None


def _looks_like_item(d: Any) -> bool:
    """Heuristic: looks like an item dict if it has history/options (either suffices)."""
    if not isinstance(d, dict):
        return False
    has_hist = any(k in d for k in _HISTORY_KEYS)
    has_opts = "options" in d
    # In practice, having either history or options is enough to treat it as an item
    return has_hist or has_opts


def _flatten_top_mapping(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    For top-level {key: item_dict, ...} objects, flatten into a list and store the key in item['src_key'].
    Keep only values that look like items.
    """
    out: List[Dict[str, Any]] = []
    for k, v in obj.items():
        if isinstance(v, dict) and _looks_like_item(v):
            vv = dict(v)  # shallow copy to avoid mutation
            # If there is no link/title, keep the outer key for traceability
            if "src_key" not in vv:
                vv["src_key"] = str(k)
            out.append(vv)
    return out


def load_items(path: str) -> List[Dict[str, Any]]:
    """
    Supports JSONL / JSON(list) / JSON(single object) / JSON(top-level mapping).
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return []

    # Try JSON first
    try:
        obj = json.loads(content)
    except Exception:
        obj = None

    if obj is not None:
        # Parsed as JSON
        if isinstance(obj, list):
            # Keep entries that look like items
            return [x for x in obj if _looks_like_item(x)]
        if isinstance(obj, dict):
            # Single item dict
            if _looks_like_item(obj):
                return [obj]
            # Otherwise treat as top-level mapping and flatten
            flat = _flatten_top_mapping(obj)
            if flat:
                return flat
            # Fallback: common container keys like "data"
            for cand_key in ("data", "items", "records"):
                if cand_key in obj and isinstance(obj[cand_key], list):
                    return [x for x in obj[cand_key] if _looks_like_item(x)]
        # If structure is unexpected, fall back to JSONL parsing

    # JSON failed or unexpected: try JSONL
    items: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            one = json.loads(line)
        except Exception:
            continue
        if _looks_like_item(one):
            items.append(one)
    return items


def _normalize_option_ids(raw_ids: List[Any], valid_ids: List[str]) -> List[str]:
    """
    Normalize leaked_options:
      - Uppercase
      - Strip common trailing punctuation, such as ')' or '.'
      - Keep only letters present in valid_ids (the real option set)
      - Deduplicate and keep order consistent with valid_ids
    """
    seen = set()
    ordered = []
    valid_set = set(valid_ids)

    def canon_one(x: Any) -> Optional[str]:
        s = str(x).strip().upper()
        if s.endswith(")") or s.endswith("."):
            s = s[:-1]
        if len(s) == 1 and s in valid_set:
            return s
        return None

    for x in (raw_ids or []):
        c = canon_one(x)
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    # Sort by the order in valid_ids
    order_map = {k: i for i, k in enumerate(valid_ids)}
    ordered.sort(key=lambda k: order_map.get(k, 1_000_000))
    return ordered


def _unpack_generate_result(r: Any) -> Tuple[Dict[str, Any], Optional[int], Optional[int]]:
    """
    Try to unpack (obj, in_tok, out_tok) from various generate return shapes.
    Allows:
      - obj
      - (obj, in_tok, out_tok)
      - (obj, in_tok, out_tok, extras)
      - (obj, stats_dict) where token counts may live in stats_dict (not required here)
    """
    obj: Dict[str, Any]
    in_tok: Optional[int] = None
    out_tok: Optional[int] = None

    # Sequence-like
    if isinstance(r, (list, tuple)):
        if len(r) >= 3:
            obj = r[0]
            if isinstance(r[1], (int, float)):
                in_tok = int(r[1])
            if isinstance(r[2], (int, float)):
                out_tok = int(r[2])
            return _ensure_obj_dict(obj), in_tok, out_tok
        if len(r) == 2:
            obj = r[0]
            # r[1] may be a stats dict; we do not parse it here
            return _ensure_obj_dict(obj), None, None
        if len(r) == 1:
            return _ensure_obj_dict(r[0]), None, None
        return {}, None, None

    # Dict / string
    if isinstance(r, dict):
        return _ensure_obj_dict(r), None, None
    if isinstance(r, str):
        try:
            parsed = json.loads(r)
            if isinstance(parsed, dict):
                return parsed, None, None
        except Exception:
            pass
        return {}, None, None

    return {}, None, None


def _ensure_obj_dict(x: Any) -> Dict[str, Any]:
    """Ensure the returned object is a dict; if it's a JSON string, deserialize it."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else {}
        except Exception:
            return {}
    return {}


# =========================
# 4) Run inference
# =========================

def run_judgement(
    api: APIModel,
    items: List[Dict[str, Any]],
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    prompts: List[Tuple[str, str, Optional[List[str]]]] = []
    letters_orders: List[List[str]] = []
    normalized_options_per_item: List[Dict[str, str]] = []
    orig_maps: List[Dict[str, Dict[str, str]]] = []  # [{'letter2orig':..., 'orig2letter':...}, ...]

    # Preprocess: normalize history and options
    histories: List[str] = []
    for it in items:
        history = _extract_history(it)
        options_raw = it.get("options", {})
        options_dict, letters_order, letter2orig, orig2letter = _coerce_options(options_raw)

        histories.append(history)
        normalized_options_per_item.append(options_dict)
        letters_orders.append(letters_order)
        orig_maps.append({"letter2orig": letter2orig, "orig2letter": orig2letter})

        user_msg = build_user_prompt(history, options_dict, letters_order)
        prompts.append((SYSTEM_PROMPT, user_msg, None))

    # Generate
    results = api.generate(prompts=prompts, schema=LeakResult, show_progress=show_progress)

    out_rows: List[Dict[str, Any]] = []
    for it, options_dict, letters_order, maps, history, r in zip(
        items, normalized_options_per_item, letters_orders, orig_maps, histories, results
    ):
        obj, in_tok, out_tok = _unpack_generate_result(r)
        # Robustness: if the SDK did not enforce structured output, validate manually
        try:
            judgement = LeakResult.model_validate(obj if isinstance(obj, dict) else {})
        except ValidationError:
            judgement = LeakResult(has_leak=False, leaked_options=[], non_leak_reason="validation_failed")

        # Normalize leaked_options to keep only real option letters
        leaked_ids = _normalize_option_ids(judgement.leaked_options, letters_order)

        # Normalize correct answer id
        correct_raw = it.get("correct_answer")
        correct_id = _normalize_correct_id(correct_raw, letters_order, maps["orig2letter"])

        # Whether leaked_ids include the correct answer
        includes_correct = (correct_id in leaked_ids) if correct_id else False

        # Backfill correct answer text when missing
        correct_text = it.get("correct_answer_text")
        if (not isinstance(correct_text, str) or not correct_text.strip()) and correct_id:
            correct_text = options_dict.get(correct_id)

        leaked_texts = [options_dict[k] for k in leaked_ids if k in options_dict]

        row = {
            "title": it.get("title"),
            "src_key": it.get("src_key"),  # if from a top-level mapping, aids traceability (e.g., URL)
            "has_leak": bool(judgement.has_leak and len(leaked_ids) > 0),
            "leaked_options": leaked_ids,            # e.g., ["B","E"]
            "leaked_option_texts": leaked_texts,     # e.g., ["Abscess", "Seroma"]
            "non_leak_reason": getattr(judgement, "non_leak_reason", None),
            "includes_correct_answer": includes_correct,
            "correct_answer": correct_id if correct_id is not None else correct_raw,
            "correct_answer_text": correct_text,
            "tokens": {"prompt": in_tok, "completion": out_tok},
        }
        out_rows.append(row)

    return out_rows


# =========================
# 5) CLI entry point
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Leakage (direct diagnosis reveal) detector via APIModel.")
    ap.add_argument("--data", type=str, default="leakage.json", help="Input data path (json/jsonl).")
    ap.add_argument("--out", type=str, default="leakage_out.json", help="Output path (JSONL).")
    ap.add_argument("--model", type=str, default="gpt-5-mini", help="Model name for APIModel.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    return ap.parse_args()


def main():
    args = parse_args()
    data_path = args.data
    out_path = args.out

    items = load_items(data_path)
    if not items:
        print("No items loaded from:", data_path)
        sys.exit(1)

    # Build APIModel (your class)
    api = APIModel(
        model_name=args.model,
    )

    results = run_judgement(api, items, show_progress=(not args.no_progress))

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Short summary
    total = len(results)
    leak_cnt = sum(1 for r in results if r["has_leak"])
    corr_leak = sum(1 for r in results if r["includes_correct_answer"])
    print(f"[Done] items={total}  has_leak={leak_cnt}  includes_correct_answer={corr_leak}")
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
