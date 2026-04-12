#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_SOURCE_URL = "https://icdcdn.who.int/icd10/meta/icd102019enMeta.zip"
DEFAULT_OUTPUT = "data_processing/files/icd10_who_2019_chapter_block_category_appendix.txt"
DEFAULT_TIMEOUT_SECONDS = 60

EXPECTED_CHAPTERS = 22
EXPECTED_BLOCKS = 263
EXPECTED_CATEGORIES = 2050

ZIP_FILES = {
    "chapters": "icd102019syst_chapters.txt",
    "groups": "icd102019syst_groups.txt",
    "codes": "icd102019syst_codes.txt",
}

SPOT_CHECKS = {
    "A00": "Cholera",
    "A01": "Typhoid and paratyphoid fevers",
    "U07": "Emergency use of U07",
}

CODE_PATTERN = re.compile(r"^[A-Z][0-9][0-9A-Z]$")

ROMAN_1_TO_22 = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
    13: "XIII",
    14: "XIV",
    15: "XV",
    16: "XVI",
    17: "XVII",
    18: "XVIII",
    19: "XIX",
    20: "XX",
    21: "XXI",
    22: "XXII",
}


class DownloadError(RuntimeError):
    pass


class SchemaChangedError(RuntimeError):
    pass


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Chapter:
    chapter_no: str
    title: str


@dataclass(frozen=True)
class Block:
    start: str
    end: str
    chapter_no: str
    title: str


@dataclass(frozen=True)
class Category:
    chapter_no: str
    block_start: str
    code: str
    title: str


def _read_zip_text_rows(zf: zipfile.ZipFile, member_name: str, expected_cols: int) -> List[List[str]]:
    try:
        raw = zf.read(member_name)
    except KeyError as exc:
        available = ", ".join(sorted(zf.namelist()))
        raise SchemaChangedError(
            f"missing expected file '{member_name}'. available files: {available}"
        ) from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaChangedError(f"failed to decode '{member_name}' as UTF-8") from exc

    rows: List[List[str]] = []
    reader = csv.reader(text.splitlines(), delimiter=";")
    for idx, row in enumerate(reader, start=1):
        if len(row) != expected_cols:
            raise SchemaChangedError(
                f"unexpected column count in '{member_name}' line {idx}: "
                f"expected {expected_cols}, got {len(row)}"
            )
        rows.append(row)
    return rows


def _download_zip(source_url: str, timeout_seconds: int) -> zipfile.ZipFile:
    try:
        with urlopen(source_url, timeout=timeout_seconds) as resp:
            payload = resp.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise DownloadError(f"could not fetch '{source_url}': {exc}") from exc

    try:
        return zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"payload from '{source_url}' is not a valid zip file") from exc


def _parse_chapters(rows: List[List[str]]) -> List[Chapter]:
    chapters: List[Chapter] = []
    for row in rows:
        chapter_no, title = row[0].strip(), row[1].strip()
        if not chapter_no or not title:
            raise SchemaChangedError("empty chapter number/title in chapters file")
        chapters.append(Chapter(chapter_no=chapter_no, title=title))
    return chapters


def _parse_blocks(rows: List[List[str]]) -> List[Block]:
    blocks: List[Block] = []
    for row in rows:
        start, end, chapter_no, title = (v.strip() for v in row)
        if not start or not end or not chapter_no or not title:
            raise SchemaChangedError("empty required field in groups file")
        blocks.append(Block(start=start, end=end, chapter_no=chapter_no, title=title))
    return blocks


def _parse_categories(rows: List[List[str]]) -> List[Category]:
    categories: List[Category] = []
    for row in rows:
        if row[0] != "3":
            continue
        chapter_no = row[3].strip()
        block_start = row[4].strip()
        code = row[7].strip()
        title = row[8].strip()
        if not chapter_no or not block_start or not code or not title:
            raise SchemaChangedError("empty required field in three-character code row")
        if not CODE_PATTERN.fullmatch(code):
            raise SchemaChangedError(f"unexpected three-character code format: '{code}'")
        categories.append(
            Category(chapter_no=chapter_no, block_start=block_start, code=code, title=title)
        )
    return categories


def _build_hierarchy(
    chapters: List[Chapter], blocks: List[Block], categories: List[Category]
) -> Tuple[Dict[str, List[Block]], Dict[Tuple[str, str], List[Category]]]:
    chapter_to_blocks: Dict[str, List[Block]] = {c.chapter_no: [] for c in chapters}
    block_to_categories: Dict[Tuple[str, str], List[Category]] = {}

    for block in blocks:
        if block.chapter_no not in chapter_to_blocks:
            raise ValidationError(
                f"group references unknown chapter '{block.chapter_no}' ({block.start}-{block.end})"
            )
        chapter_to_blocks[block.chapter_no].append(block)
        block_to_categories[(block.chapter_no, block.start)] = []

    for cat in categories:
        key = (cat.chapter_no, cat.block_start)
        if key not in block_to_categories:
            raise ValidationError(
                f"category '{cat.code}' points to unknown block key {cat.chapter_no}/{cat.block_start}"
            )
        block_to_categories[key].append(cat)

    return chapter_to_blocks, block_to_categories


def _run_hard_validations(
    chapters: List[Chapter],
    blocks: List[Block],
    categories: List[Category],
    chapter_to_blocks: Dict[str, List[Block]],
    block_to_categories: Dict[Tuple[str, str], List[Category]],
) -> None:
    if len(chapters) != EXPECTED_CHAPTERS:
        raise ValidationError(
            f"chapter count mismatch: expected {EXPECTED_CHAPTERS}, got {len(chapters)}"
        )
    if len(blocks) != EXPECTED_BLOCKS:
        raise ValidationError(f"block count mismatch: expected {EXPECTED_BLOCKS}, got {len(blocks)}")
    if len(categories) != EXPECTED_CATEGORIES:
        raise ValidationError(
            f"3-char category count mismatch: expected {EXPECTED_CATEGORIES}, got {len(categories)}"
        )

    codes = [c.code for c in categories]
    if len(set(codes)) != len(codes):
        raise ValidationError("duplicate three-character categories detected")

    for block in blocks:
        key = (block.chapter_no, block.start)
        cats = block_to_categories.get(key, [])
        if not cats:
            raise ValidationError(
                f"block has no three-character categories: {block.start}-{block.end} ({block.title})"
            )

    for ch in chapters:
        if not chapter_to_blocks.get(ch.chapter_no):
            raise ValidationError(f"chapter has no groups: {ch.chapter_no} ({ch.title})")

    code_to_title = {c.code: c.title for c in categories}
    for code, expected_title in SPOT_CHECKS.items():
        actual_title = code_to_title.get(code)
        if actual_title != expected_title:
            raise ValidationError(
                f"spot-check failed for {code}: expected '{expected_title}', got '{actual_title}'"
            )


def _render_appendix(
    source_url: str,
    retrieved_utc: str,
    chapters: List[Chapter],
    chapter_to_blocks: Dict[str, List[Block]],
    block_to_categories: Dict[Tuple[str, str], List[Category]],
    categories_count: int,
) -> str:
    lines: List[str] = []
    lines.append("# WHO ICD-10 Appendix (Chapter-Block-Category, 3-character)")
    lines.append(f"# Source: {source_url}")
    lines.append(f"# Retrieved UTC: {retrieved_utc}")
    lines.append(
        f"# Counts: chapters={len(chapters)}, blocks={sum(len(v) for v in chapter_to_blocks.values())}, "
        f"categories={categories_count}"
    )
    lines.append("")

    for chapter in chapters:
        chapter_blocks = chapter_to_blocks[chapter.chapter_no]
        chapter_start = chapter_blocks[0].start
        chapter_end = chapter_blocks[-1].end
        chapter_idx = int(chapter.chapter_no)
        roman = ROMAN_1_TO_22.get(chapter_idx)
        if roman is None:
            raise ValidationError(f"unsupported chapter index for Roman numeral: {chapter.chapter_no}")

        lines.append(f"Chapter {roman}  {chapter.title} ({chapter_start}-{chapter_end})")
        for block in chapter_blocks:
            lines.append(f"Block | {block.start}-{block.end}  {block.title}")
            for cat in block_to_categories[(block.chapter_no, block.start)]:
                lines.append(f"    {cat.code}  {cat.title}")
            lines.append("")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_appendix(source_url: str, output_path: Path, timeout_seconds: int) -> None:
    retrieved_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    zf = _download_zip(source_url=source_url, timeout_seconds=timeout_seconds)

    expected_members = set(ZIP_FILES.values())
    actual_members = set(zf.namelist())
    missing_members = sorted(expected_members - actual_members)
    if missing_members:
        available = ", ".join(sorted(actual_members))
        raise SchemaChangedError(
            f"missing required zip members: {', '.join(missing_members)}. available files: {available}"
        )

    chapters_rows = _read_zip_text_rows(zf, ZIP_FILES["chapters"], expected_cols=2)
    groups_rows = _read_zip_text_rows(zf, ZIP_FILES["groups"], expected_cols=4)
    codes_rows = _read_zip_text_rows(zf, ZIP_FILES["codes"], expected_cols=17)

    chapters = _parse_chapters(chapters_rows)
    blocks = _parse_blocks(groups_rows)
    categories = _parse_categories(codes_rows)

    chapter_to_blocks, block_to_categories = _build_hierarchy(chapters, blocks, categories)
    _run_hard_validations(chapters, blocks, categories, chapter_to_blocks, block_to_categories)

    output = _render_appendix(
        source_url=source_url,
        retrieved_utc=retrieved_utc,
        chapters=chapters,
        chapter_to_blocks=chapter_to_blocks,
        block_to_categories=block_to_categories,
        categories_count=len(categories),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WHO ICD-10 Appendix (chapter -> block -> 3-character category) as text."
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="WHO ICD-10 meta zip URL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .txt file path")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)

    try:
        build_appendix(
            source_url=args.source_url,
            output_path=output_path,
            timeout_seconds=args.timeout_seconds,
        )
    except DownloadError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    except SchemaChangedError as exc:
        print(f"schema changed: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote WHO ICD-10 appendix to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
