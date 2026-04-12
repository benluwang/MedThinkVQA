#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trim MCQ options to 5 according to rules:

- If an item has exactly 5 options: keep as is.
- If an item has >5 options:
  * If the model answered correctly: keep the correct option + random others to 5.
  * If the model answered incorrectly: keep BOTH the correct option and the model's chosen (wrong) option,
    then add random others to 5.

Other details:
- Works whether `options` is a dict like {"A": "...", "B": "..."} or a list of strings.
- Robustly infers correct answer letter from fields: gt_letter, correct_answer (letter or text), correct_answer_text.
- Robustly infers model answer letter from: parsed_answer_letter, raw_output.answer (letter or text), etc.
- Reindexes the kept options to letters A..E and updates `correct_answer` and `correct_answer_text` accordingly.
- Updates `n_options` to the new count.
- Removes evaluation fields: parsed_answer_letter, parsed_explanation, raw_output, gt_letter, is_eval_eligible,
  is_correct, answer_token_topk, answer_letter_probs, answer_letter_probs_norm, answer_confidence, model_answer, model_explanation, answer (top-level).

Usage:
    python trim_to_5_options.py /path/to/input.json [--seed 42]
Output:
    Saves to /path/to/input_5option.json
"""

import json
import os
import re
import sys
import random
from typing import Any, Dict, List, Tuple, Optional

LETTER_ORDER = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Fields to drop at the end (from "parsed_answer_letter and below")
DROP_EXACT = {
    "parsed_answer_letter",
    "parsed_explanation",
    "raw_output",
    "gt_letter",
    "is_eval_eligible",
    "is_correct",
    "answer_token_topk",
    "answer_letter_probs",
    "answer_letter_probs_norm",
    "answer_confidence",
    "model_answer",
    "model_explanation",
    "answer",  # note: we only drop top-level 'answer' used by some pipelines to store model letter/text (not the options)
}

def normalize_text(s: Any) -> str:
    """Lowercase + collapse whitespace for robust text matching."""
    s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s).strip().lower()

def options_to_pairs(options: Any) -> List[Tuple[str, str]]:
    """
    Normalize options to a list of (key, text) pairs, preserving original order if dict.
    If it's a list, assign letters A.. as keys.
    """
    if isinstance(options, dict):
        # Python 3.7+ dict preserves insertion order
        return [(k, str(v)) for k, v in options.items()]
    elif isinstance(options, list):
        pairs = []
        for i, v in enumerate(options):
            if i >= len(LETTER_ORDER):
                break
            pairs.append((LETTER_ORDER[i], str(v)))
        return pairs
    else:
        raise ValueError("Unsupported options type. Expect dict or list.")

def find_key_by_text(pairs: List[Tuple[str, str]], text: Any) -> Optional[str]:
    """Try to find the option key whose text matches `text` (robust string matching)."""
    target = normalize_text(text)
    for k, v in pairs:
        if normalize_text(v) == target:
            return k
    return None

def parse_correct_key(item: Dict[str, Any], pairs: List[Tuple[str, str]]) -> Optional[str]:
    """Infer correct option key (letter) using several possible fields."""
    keys_present = {k for k, _ in pairs}
    # 1) gt_letter (preferred if present)
    gt = item.get("gt_letter")
    if isinstance(gt, str) and len(gt.strip()) == 1:
        cand = gt.strip().upper()
        if cand in keys_present:
            return cand
    # 2) correct_answer_letter
    ca_letter = item.get("correct_answer_letter")
    if isinstance(ca_letter, str) and len(ca_letter.strip()) == 1:
        cand = ca_letter.strip().upper()
        if cand in keys_present:
            return cand
    # 3) correct_answer (letter or text)
    ca = item.get("correct_answer")
    if isinstance(ca, str) and ca.strip():
        s = ca.strip()
        if len(s) == 1 and s.upper() in keys_present:
            return s.upper()
        # maybe it's the full text
        key = find_key_by_text(pairs, s)
        if key:
            return key
    # 4) correct_answer_text
    cat = item.get("correct_answer_text")
    if isinstance(cat, str) and cat.strip():
        key = find_key_by_text(pairs, cat)
        if key:
            return key
    return None

def parse_pred_key(item: Dict[str, Any], pairs: List[Tuple[str, str]]) -> Optional[str]:
    """Infer model predicted option key (letter) from several fields."""
    keys_present = {k for k, _ in pairs}
    for f in ["parsed_answer_letter", "answer_letter", "pred_letter", "model_answer_letter"]:
        v = item.get(f)
        if isinstance(v, str) and len(v.strip()) == 1:
            vv = v.strip().upper()
            if vv in keys_present:
                return vv
    # raw_output.answer
    ro = item.get("raw_output")
    if isinstance(ro, dict):
        a = ro.get("answer")
        if isinstance(a, str) and a.strip():
            s = a.strip()
            if len(s) == 1 and s.upper() in keys_present:
                return s.upper()
            key = find_key_by_text(pairs, s)
            if key:
                return key
    # top-level 'answer' (some pipelines store it there)
    ans = item.get("answer")
    if isinstance(ans, str) and ans.strip():
        s = ans.strip()
        if len(s) == 1 and s.upper() in keys_present:
            return s.upper()
        key = find_key_by_text(pairs, s)
        if key:
            return key
    return None

def compute_is_correct(item: Dict[str, Any], correct_key: Optional[str], pred_key: Optional[str]) -> Optional[bool]:
    """Prefer provided is_correct, else compute from keys if available."""
    if isinstance(item.get("is_correct"), bool):
        return item["is_correct"]
    if correct_key and pred_key:
        return correct_key == pred_key
    return None  # unknown

def reindex_to_letters(pairs_kept: List[Tuple[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Reindex kept pairs to consecutive letters A.., returning:
    - new_options: { 'A': textA, ... }
    - old_to_new: mapping old_key -> new_letter
    """
    new_options: Dict[str, str] = {}
    old_to_new: Dict[str, str] = {}
    for i, (old_k, text) in enumerate(pairs_kept):
        new_k = LETTER_ORDER[i]
        new_options[new_k] = text
        old_to_new[old_k] = new_k
    return new_options, old_to_new

def drop_eval_fields(item: Dict[str, Any]) -> None:
    """Remove evaluation/diagnostic fields from an item."""
    for k in list(DROP_EXACT):
        if k in item:
            item.pop(k, None)

def process_item(item: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """
    Apply trimming rules to a single item. Returns the modified item (in-place safe).
    """
    # Use 'options' if present; else fall back to 'gpt_options'
    options_raw = item.get("options", item.get("gpt_options"))
    if options_raw is None:
        # No options → just strip eval fields and return
        drop_eval_fields(item)
        return item

    pairs = options_to_pairs(options_raw)
    opt_dict = dict(pairs)
    n = len(pairs)

    # Find correct/predicted keys and correctness
    correct_key = parse_correct_key(item, pairs)
    pred_key = parse_pred_key(item, pairs)
    is_corr = compute_is_correct(item, correct_key, pred_key)

    # Case 1: <= 5 options → keep as is
    if n <= 5:
        # Ensure dict form
        if isinstance(options_raw, dict):
            new_options = options_raw
        else:
            new_options = {LETTER_ORDER[i]: txt for i, txt in enumerate([t for _, t in pairs])}
        item["options"] = new_options
        item["n_options"] = len(new_options)

        # Try to normalize correct_answer letter if we can
        if correct_key and correct_key in new_options:
            # If correct_answer is letter-like, keep it consistent
            ca = item.get("correct_answer")
            if isinstance(ca, str) and len(ca.strip()) == 1:
                item["correct_answer"] = correct_key
            # Update/ensure correct_answer_text too
            item["correct_answer_text"] = new_options[correct_key]

        drop_eval_fields(item)
        return item

    # Case 2: > 5 options → trim to 5
    must_keep = set()
    if correct_key in opt_dict:
        must_keep.add(correct_key)
    # If incorrect prediction, keep the wrong predicted key too
    if is_corr is False and pred_key in opt_dict and pred_key != correct_key:
        must_keep.add(pred_key)

    # Fill up to 5 with random others, preserving original order afterwards
    remaining_keys_in_order = [k for k, _ in pairs if k not in must_keep]
    need = max(0, 5 - len(must_keep))
    sampled = rng.sample(remaining_keys_in_order, k=min(need, len(remaining_keys_in_order)))
    kept_keys = [k for k, _ in pairs if (k in must_keep or k in sampled)]

    # Safety: if somehow >5 (shouldn't happen), randomly drop non-must_keep to exactly 5
    if len(kept_keys) > 5:
        optional = [k for k in kept_keys if k not in must_keep]
        excess = len(kept_keys) - 5
        to_drop = set(rng.sample(optional, k=min(excess, len(optional))))
        kept_keys = [k for k in kept_keys if k not in to_drop]

    kept_pairs = [(k, opt_dict[k]) for k in [k for k, _ in pairs if k in kept_keys]]

    # Reindex to A..E
    new_options, old_to_new = reindex_to_letters(kept_pairs)
    item["options"] = new_options
    item["n_options"] = len(new_options)

    # Update correct_answer and correct_answer_text
    if correct_key and correct_key in old_to_new:
        new_ca = old_to_new[correct_key]
        item["correct_answer"] = new_ca
        item["correct_answer_text"] = new_options[new_ca]
    else:
        # Fallback by text if needed
        cat = item.get("correct_answer_text")
        if isinstance(cat, str):
            norm_cat = normalize_text(cat)
            for nk, txt in new_options.items():
                if normalize_text(txt) == norm_cat:
                    item["correct_answer"] = nk
                    item["correct_answer_text"] = new_options[nk]
                    break

    # Finally remove evaluation fields
    drop_eval_fields(item)
    return item

def derive_output_path(in_path: str) -> str:
    base, ext = os.path.splitext(in_path)
    if not ext:
        return base + "_5option.json"
    return base + "_5option.json"

def main():
    if len(sys.argv) < 2:
        print("Usage: python trim_to_5_options.py /path/to/input.json [--seed 42]")
        sys.exit(1)

    in_path = sys.argv[1]
    seed = None
    if len(sys.argv) >= 4 and sys.argv[2] == "--seed":
        try:
            seed = int(sys.argv[3])
        except Exception:
            seed = None

    rng = random.Random(seed)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect structure and process
    processed = None
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        items = data["data"]
        out_items = [process_item(dict(item), rng) for item in items]
        processed = dict(data)
        processed["data"] = out_items
    elif isinstance(data, list):
        processed = [process_item(dict(item), rng) if isinstance(item, dict) else item for item in data]
    elif isinstance(data, dict):
        # Maybe a mapping id->item; keep any 'summary' key untouched if present
        processed = {}
        for k, v in data.items():
            if k == "summary" and isinstance(v, dict):
                processed[k] = v
            elif isinstance(v, dict):
                processed[k] = process_item(dict(v), rng)
            else:
                processed[k] = v
    else:
        print("Unsupported JSON top-level structure.")
        sys.exit(2)

    out_path = derive_output_path(in_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to: {out_path}")

if __name__ == "__main__":
    main()
