#!/usr/bin/env python3

"""Prune discussion text so it only refers to the released five options."""

import argparse
import os
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple, Optional, Union
from pydantic import BaseModel, Field

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.gpt import APIModel




def _ensure_list_from_dif(dif_raw: Union[str, List[str], None]) -> List[str]:

    if dif_raw is None:
        return []
    if isinstance(dif_raw, list):
        return [str(x).strip() for x in dif_raw if str(x).strip()]
    if isinstance(dif_raw, str):
        parts = [p.strip() for p in dif_raw.split(",")]
        return [p for p in parts if p]
    return []


def _ensure_options_list(options_raw: Union[Dict[str, Any], List[str], None]) -> List[str]:

    if options_raw is None:
        return []
    if isinstance(options_raw, dict):
        vals = [str(v).strip() for v in options_raw.values()]
        return [v for v in vals if v]
    if isinstance(options_raw, list):
        return [str(x).strip() for x in options_raw if str(x).strip()]
    return []


def _compute_extra_to_remove(dif_list: List[str], option_texts: List[str]) -> List[str]:
    opt_set = set(option_texts)
    return [x for x in dif_list if x not in opt_set]



class DiscussionOut(BaseModel):
    discussion_new: str = Field(..., description="Edited discussion text after pruning extra differential diagnoses")



def build_prune_prompt(
    discussion: str,
    dif_list_clean: List[str],
    option_texts: List[str],
    extra_to_remove: List[str],
) -> Tuple[str, str]:

    system_msg = (
        "You are a careful clinical editor. Your job is to MINIMALLY edit a medical DISCUSSION.\n"
        "Goal: remove references to extra differential diagnoses that appear in DIF_DIAGNOSIS_LIST "
        "but are NOT among the five ALLOWED OPTIONS. Preserve all content related to ALLOWED OPTIONS. "
        "Keep the original clinical reasoning flow, tone, and meaning. Do not add new facts.\n\n"
        "Rules:\n"
        "1) NEVER delete information that relates to any ALLOWED_OPTIONS (even if an EXTRA item partially overlaps).\n"
        "2) Remove sentences/clauses whose main role is to introduce, justify, or list items in EXTRA_TO_REMOVE.\n"
        "   If a sentence mixes allowed and extra diagnoses, keep the allowed part and delete only the extra part, "
        "   then fix grammar to remain fluent.\n"
        "3) Keep general disease definitions, imaging/lab reasoning, and conclusions that support ALLOWED_OPTIONS.\n"
        "4) Maintain coherence and clinical correctness; do NOT invent new claims.\n"
        "5) Output strictly as JSON with one key: discussion_new.\n"
        "6) If EXTRA_TO_REMOVE is empty, return the original discussion as discussion_new.\n"
    )


    user_msg = (
        "Edit the DISCUSSION by deleting only the parts about the extra differentials.\n\n"
        f"ALLOWED_OPTIONS (keep anything related to these):\n{json.dumps(option_texts, ensure_ascii=False)}\n\n"
        f"DIF_DIAGNOSIS_LIST_CLEAN:\n{json.dumps(dif_list_clean, ensure_ascii=False)}\n\n"
        f"EXTRA_TO_REMOVE (delete content only about these):\n{json.dumps(extra_to_remove, ensure_ascii=False)}\n\n"
        "DISCUSSION:\n"
        "```text\n"
        f"{discussion}\n"
        "```\n\n"
        "Return JSON: {\"discussion_new\": \"...\"}"
    )

    return system_msg, user_msg




def _load_json_any(path: Union[str, os.PathLike]) -> List[Dict[str, Any]]:

    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8").strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    items: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            continue
    if items:
        return items

    raise ValueError(f"Unrecognized JSON structure in {path}")


def _dump_json(items: List[Dict[str, Any]], out_path: Union[str, os.PathLike]) -> None:
    outp = pathlib.Path(out_path)
    outp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def prune_discussion_file(
    input_json_path: Union[str, os.PathLike],
    output_json_path: Optional[Union[str, os.PathLike]] = None,
    model_name: str = "gpt-5",
    temperature: float = 1.0,
    show_progress: bool = True,
) -> str:

    items_all = _load_json_any(input_json_path)


    selected: List[Tuple[int, Dict[str, Any], List[str], List[str], List[str]]] = []
    for idx, obj in enumerate(items_all):
        dif_list = _ensure_list_from_dif(obj.get("DIF_DIAGNOSIS_LIST"))
        if len(dif_list) > 5:
            option_texts = _ensure_options_list(obj.get("options"))
            extra = _compute_extra_to_remove(dif_list, option_texts)
            selected.append((idx, obj, dif_list, option_texts, extra))

    out_path = pathlib.Path(output_json_path) if output_json_path else _derive_out_path(input_json_path)

    if not selected:
        _dump_json(items_all, out_path)
        return str(out_path)

    prompts: List[Tuple[str, str, Optional[List[str]]]] = []
    for _, obj, dif_list, option_texts, extra in selected:
        discussion = str(obj.get("DISCUSSION", "")).strip()
        sys_msg, user_msg = build_prune_prompt(
            discussion=discussion,
            dif_list_clean=dif_list,
            option_texts=option_texts,
            extra_to_remove=extra,
        )
        prompts.append((sys_msg, user_msg, None))

    api = APIModel(model_name=model_name, temperature=temperature)
    results = api.generate(prompts, schema=DiscussionOut, show_progress=show_progress)

    processed_items: List[Dict[str, Any]] = [dict(item) for item in items_all]
    for (idx, obj, _dif, _opts, _extra), (resp, _in_tok, _out_tok) in zip(selected, results):
        discussion_new: Optional[str] = None
        if isinstance(resp, dict) and "discussion_new" in resp:
            discussion_new = str(resp["discussion_new"])
        elif isinstance(resp, str):
            try:
                maybe = json.loads(resp)
                if isinstance(maybe, dict) and "discussion_new" in maybe:
                    discussion_new = str(maybe["discussion_new"])
            except json.JSONDecodeError:
                pass

        if not discussion_new:
            discussion_new = str(obj.get("DISCUSSION", ""))

        processed_items[idx]["DISCUSSION"] = discussion_new

    _dump_json(processed_items, out_path)
    return str(out_path)


def _derive_out_path(input_json_path: Union[str, os.PathLike]) -> pathlib.Path:
    p = pathlib.Path(input_json_path)
    stem = p.stem
    suffix = p.suffix if p.suffix else ".json"
    out_name = f"{stem}_gptdiscussion{suffix}"
    return p.with_name(out_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune discussion text to match the released five-option DDx set.")
    parser.add_argument("--input_json", required=True, help="Input JSON or JSONL file.")
    parser.add_argument("--output_json", default=None, help="Output JSON path. Defaults to <input>_gptdiscussion.json.")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name for APIModel.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = prune_discussion_file(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        model_name=args.model,
        temperature=float(args.temperature),
        show_progress=(not args.no_progress),
    )
    print(f"Wrote: {out}")
