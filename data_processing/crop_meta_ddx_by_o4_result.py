#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LETTERS = ["A", "B", "C", "D", "E"]


def _norm_text(s: Any) -> str:
    s = ("" if s is None else str(s)).casefold().strip()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _split_ddx_text(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []
    parts = [p.strip(" ;\t\r\n") for p in raw.split(";;")]
    return [p for p in parts if p]


def _find_answer_text(candidates: List[str], final_dx: str) -> str:
    final_dx = (final_dx or "").strip()
    if not final_dx:
        return ""

    key_final = _norm_text(final_dx)
    if not key_final:
        return final_dx

    for c in candidates:
        if _norm_text(c) == key_final:
            return c

    for c in candidates:
        kc = _norm_text(c)
        if not kc:
            continue
        if key_final in kc or kc in key_final:
            return c

    return final_dx


def _norm_letter(x: Any) -> Optional[str]:
    if x is None:
        return None
    m = re.search(r"([A-Za-z])", str(x))
    if not m:
        return None
    u = m.group(1).upper()
    return u


def _match_text_to_letter(text: str, options: Dict[str, str]) -> Optional[str]:
    t = _norm_text(text)
    if not t:
        return None
    for k, v in options.items():
        if _norm_text(v) == t:
            return k
    return None


def _ordered_option_pairs(options_obj: Any) -> List[Tuple[str, str]]:
    if not isinstance(options_obj, dict) or not options_obj:
        return []

    keys = [str(k).strip() for k in options_obj.keys()]
    if all(len(k) == 1 and k.isalpha() for k in keys):
        sorted_keys = sorted(keys, key=lambda x: ord(x.upper()))
    else:
        sorted_keys = keys

    out: List[Tuple[str, str]] = []
    for k in sorted_keys:
        v = options_obj.get(k)
        if v is None:
            continue
        out.append((k.upper(), str(v).strip()))
    return out


def _load_eval_map(eval_json_path: str) -> Dict[int, Dict[str, Any]]:
    with open(eval_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        items = obj["data"]
    elif isinstance(obj, list):
        items = obj
    else:
        raise ValueError(f"Unsupported eval JSON format: {eval_json_path}")

    out: Dict[int, Dict[str, Any]] = {}
    for rec in items:
        if not isinstance(rec, dict):
            continue
        idx_raw = rec.get("source_row_index")
        try:
            idx = int(idx_raw)
        except Exception:
            continue
        out[idx] = rec
    return out


def _fallback_pick_five(
    ddx: List[str],
    final_dx: str,
    rng: random.Random,
) -> Tuple[List[str], str]:
    correct_text = _find_answer_text(ddx, final_dx)
    correct_key = _norm_text(correct_text)

    distractors = [x for x in ddx if _norm_text(x) != correct_key]
    if len(distractors) >= 4:
        chosen_distractors = rng.sample(distractors, 4)
    else:
        chosen_distractors = list(distractors)
        for x in ddx:
            if len(chosen_distractors) >= 4:
                break
            if _norm_text(x) == correct_key:
                continue
            chosen_distractors.append(x)
    picked = [correct_text] + chosen_distractors[:4]

    if len(picked) < 5:
        for x in ddx:
            if len(picked) >= 5:
                break
            if x not in picked:
                picked.append(x)
    if len(picked) < 5:
        while len(picked) < 5:
            picked.append(correct_text)
    return picked[:5], correct_text


def _pick_five_for_gt5(
    ddx: List[str],
    final_dx: str,
    eval_rec: Optional[Dict[str, Any]],
    rng: random.Random,
) -> Tuple[List[str], str, bool, bool]:
    """
    Returns:
      selected_5_texts,
      correct_text,
      used_eval_record,
      model_was_wrong
    """
    if not eval_rec:
        picked, correct_text = _fallback_pick_five(ddx, final_dx, rng)
        return picked, correct_text, False, False

    pairs = _ordered_option_pairs(eval_rec.get("options"))
    if len(pairs) < 5:
        picked, correct_text = _fallback_pick_five(ddx, final_dx, rng)
        return picked, correct_text, False, False

    opt = {k: v for k, v in pairs}
    letters_in_order = [k for k, _ in pairs]

    corr = _norm_letter(eval_rec.get("gt_letter")) or _norm_letter(eval_rec.get("correct_answer"))
    if corr not in opt:
        corr = _match_text_to_letter(str(eval_rec.get("correct_answer_text") or ""), opt)
    if corr not in opt:
        corr = _match_text_to_letter(final_dx, opt)
    if corr not in opt:
        picked, correct_text = _fallback_pick_five(ddx, final_dx, rng)
        return picked, correct_text, False, False

    pred = _norm_letter(eval_rec.get("parsed_answer_letter"))
    is_correct = eval_rec.get("is_correct")
    model_was_wrong = bool(is_correct is False)

    must_keep: List[str] = [corr]
    if model_was_wrong and pred in opt and pred != corr:
        must_keep.append(pred)

    remaining = [x for x in letters_in_order if x not in must_keep]
    need = 5 - len(must_keep)
    if need > len(remaining):
        need = len(remaining)
    sampled = rng.sample(remaining, need) if need > 0 else []
    chosen_letters = must_keep + sampled

    # Preserve original option order in final picked set.
    chosen_set = set(chosen_letters)
    picked = [opt[k] for k in letters_in_order if k in chosen_set][:5]
    if len(picked) < 5:
        for k in letters_in_order:
            if len(picked) >= 5:
                break
            v = opt[k]
            if v not in picked:
                picked.append(v)
    if len(picked) < 5:
        picked_fb, corr_fb = _fallback_pick_five(ddx, final_dx, rng)
        return picked_fb, corr_fb, False, model_was_wrong

    correct_text = opt[corr]
    if _norm_text(correct_text) not in {_norm_text(x) for x in picked}:
        picked[-1] = correct_text

    return picked[:5], correct_text, True, model_was_wrong


def _split_correct_distractors(selected: List[str], correct_text: str) -> Tuple[str, List[str]]:
    key = _norm_text(correct_text)
    idx = -1
    for i, x in enumerate(selected):
        if _norm_text(x) == key:
            idx = i
            break
    if idx < 0:
        idx = 0
    correct = selected[idx]
    distractors = [selected[i] for i in range(len(selected)) if i != idx]
    return correct, distractors


def _balanced_targets(n: int, rng: random.Random) -> List[str]:
    base = n // 5
    rem = n % 5
    out: List[str] = []
    for i, letter in enumerate(LETTERS):
        cnt = base + (1 if i < rem else 0)
        out.extend([letter] * cnt)
    rng.shuffle(out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Crop a differential-diagnosis CSV into 5 options using evaluator outputs. "
            "Rules: keep 5-diagnosis rows unchanged; for >5 rows keep correct + wrong-pred (if wrong) + random others. "
            "Write OptionA~OptionE and correct Answer with balanced A-E distribution on >5 rows."
        )
    )
    p.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Input CSV path with differential diagnosis candidates.",
    )
    p.add_argument(
        "--eval_json",
        type=str,
        required=True,
        help="Evaluator output JSON path used to preserve hard distractors.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Default: <csv_stem>_cropped_o4.csv",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for option sampling and target-letter balancing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    csv_path = str(Path(args.csv_path).expanduser().resolve())
    eval_path = str(Path(args.eval_json).expanduser().resolve())

    if args.out:
        out_path = str(Path(args.out).expanduser().resolve())
    else:
        p = Path(csv_path)
        out_path = str(p.with_name(f"{p.stem}_cropped_o4.csv"))

    eval_map = _load_eval_map(eval_path)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Stage 1: prepare 5-option candidates for each row.
    gt5_bucket: List[Dict[str, Any]] = []
    exact5_count = 0
    gt5_count = 0
    gt5_used_eval = 0
    gt5_wrong_rows = 0

    for row_idx, row in enumerate(rows):
        ddx = _split_ddx_text((row.get("differential_diagnosis") or "").strip())
        final_dx = (row.get("final_diagnosis") or "").strip()

        if len(ddx) == 5:
            exact5_count += 1
            selected = ddx[:5]
            correct_text = _find_answer_text(selected, final_dx)
            correct_key = _norm_text(correct_text)
            correct_letter = ""
            for i, x in enumerate(selected):
                if _norm_text(x) == correct_key:
                    correct_letter = LETTERS[i]
                    break
            row["OptionA"] = selected[0]
            row["OptionB"] = selected[1]
            row["OptionC"] = selected[2]
            row["OptionD"] = selected[3]
            row["OptionE"] = selected[4]
            row["correct Answer"] = correct_letter
            continue

        if len(ddx) > 5:
            gt5_count += 1
            selected, correct_text, used_eval, model_was_wrong = _pick_five_for_gt5(
                ddx=ddx,
                final_dx=final_dx,
                eval_rec=eval_map.get(row_idx),
                rng=rng,
            )
            if used_eval:
                gt5_used_eval += 1
            if model_was_wrong:
                gt5_wrong_rows += 1

            corr, distractors = _split_correct_distractors(selected, correct_text)
            while len(distractors) < 4:
                distractors.append(corr)
            gt5_bucket.append(
                {
                    "row_idx": row_idx,
                    "correct": corr,
                    "distractors": distractors[:4],
                }
            )
            continue

        # ddx < 5: leave added columns empty.
        row["OptionA"] = ""
        row["OptionB"] = ""
        row["OptionC"] = ""
        row["OptionD"] = ""
        row["OptionE"] = ""
        row["correct Answer"] = ""

    # Stage 2: assign balanced correct letter targets for >5 rows.
    targets = _balanced_targets(len(gt5_bucket), rng)
    gt5_correct_dist = {k: 0 for k in LETTERS}
    for item, target in zip(gt5_bucket, targets):
        row_idx = int(item["row_idx"])
        correct_text = str(item["correct"])
        distractors = list(item["distractors"])
        rng.shuffle(distractors)

        option_map: Dict[str, str] = {}
        di = 0
        for letter in LETTERS:
            if letter == target:
                option_map[letter] = correct_text
            else:
                option_map[letter] = distractors[di]
                di += 1

        rows[row_idx]["OptionA"] = option_map["A"]
        rows[row_idx]["OptionB"] = option_map["B"]
        rows[row_idx]["OptionC"] = option_map["C"]
        rows[row_idx]["OptionD"] = option_map["D"]
        rows[row_idx]["OptionE"] = option_map["E"]
        rows[row_idx]["correct Answer"] = target
        gt5_correct_dist[target] += 1

    # Ensure output field order keeps existing columns and appends new ones.
    for col in ("OptionA", "OptionB", "OptionC", "OptionD", "OptionE", "correct Answer"):
        if col not in fieldnames:
            fieldnames.append(col)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    all_dist = {k: 0 for k in LETTERS}
    for row in rows:
        l = _norm_letter(row.get("correct Answer"))
        if l in all_dist:
            all_dist[l] += 1

    print(f"[SAVE] wrote: {out_path}")
    print(
        f"[STATS] total_rows={len(rows)} exact5_rows={exact5_count} gt5_rows={gt5_count} "
        f"gt5_used_eval={gt5_used_eval} gt5_wrong_rows={gt5_wrong_rows}"
    )
    print(f"[DIST][gt5_only] {gt5_correct_dist}")
    print(f"[DIST][all_filled] {all_dist}")


if __name__ == "__main__":
    main()
