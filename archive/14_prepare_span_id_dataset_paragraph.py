#!/usr/bin/env python3
"""
Prepare PARAGRAPH-level Span Identification dataset from section-level JSONL files.

What it does:
1) Loads per-page section JSONL files (each file contains multiple sections).
2) Concatenates section texts into a single page text (same as page-level pipeline).
3) Converts section-local link offsets into page-level span offsets.
4) Splits the page text into paragraphs (by paragraph_separator, default: "\\n\\n").
5) For each paragraph, keeps spans fully inside it and remaps to paragraph-local offsets.
6) Splits pages deterministically into train/dev/test using page_key (so no leakage).
7) Writes train/dev/test JSONL and stats.

Intended as a controlled ablation vs page-level training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.span_identifier.prep.io_jsonl import write_jsonl, write_json


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_yaml(path: str) -> Dict:
    """Load a YAML configuration file. Returns empty dict if file is empty."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_bucket(name: str) -> str:
    """
    Deterministically assign a document to train/dev/test split based on a hash of page key.

    Distribution:
      - 80% train
      - 10% dev
      - 10% test

    NOTE: Uses Python hash(); reproducibility across processes depends on PYTHONHASHSEED.
    """
    h = abs(hash(name)) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "dev"
    return "test"


def load_page_sections_jsonl(page_jsonl: Path) -> List[Dict[str, Any]]:
    """Load all section records from a per-page JSONL file (one JSON object per line)."""
    rows: List[Dict[str, Any]] = []
    with open(page_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _norm(x: Any) -> str:
    """Normalize string-ish fields for robust comparisons."""
    return str(x).strip().lower()


def _is_external_section(rec: Dict[str, Any]) -> bool:
    """
    Decide whether a section record corresponds to an 'external' section.

    Works if your section JSON objects contain keys like:
      - 'section', 'section_name', 'title', etc.
    If none exist, returns False.
    """
    candidates = []
    for k in ("section", "section_name", "section_title", "title", "heading"):
        if k in rec and rec.get(k) is not None:
            candidates.append(_norm(rec.get(k)))

    if not candidates:
        return False

    external_like = {
        "external", "external links", "external_links", "external-links",
        "references", "reference", "citations", "notes", "footnotes",
        "see also", "see_also", "bibliography",
    }

    for c in candidates:
        if c in external_like:
            return True
        if "external" in c and "link" in c:
            return True
        if c.startswith("references"):
            return True

    return False


# ---------------------------------------------------------------------
# Core: build page text + page-level spans (same as your page-level script)
# ---------------------------------------------------------------------

def build_page_text_and_spans_from_sections(
    section_rows: List[Dict[str, Any]],
    section_sep: str = "\n\n",
    validate_anchor_text: bool = True,
    keep_internal_only: bool = True,
    drop_external_sections: bool = True,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Concatenate section texts into page text and remap section-local spans to page-level offsets.

    keep_internal_only=True: keep only link_type == 'internal' (case-insensitive)
    drop_external_sections=True: skip entire sections like External links/References (if metadata exists)
    """
    texts: List[str] = []
    spans: List[Dict[str, Any]] = []

    total_links = 0
    kept_links = 0
    bad_offsets = 0
    bad_anchor_mismatch = 0
    skipped_external_sections = 0
    skipped_non_internal_links = 0

    cursor = 0

    for rec in section_rows:
        if drop_external_sections and _is_external_section(rec):
            skipped_external_sections += 1
            continue

        sec_text = rec.get("text") or ""
        sec_links = rec.get("links") or []

        texts.append(sec_text)
        sec_base = cursor

        for link in sec_links:
            total_links += 1

            lt_raw = link.get("link_type")
            lt = _norm(lt_raw) if lt_raw is not None else ""

            if keep_internal_only and lt != "internal":
                skipped_non_internal_links += 1
                continue

            s = link.get("start")
            e = link.get("end")
            if s is None or e is None:
                bad_offsets += 1
                continue

            try:
                s = int(s)
                e = int(e)
            except Exception:
                bad_offsets += 1
                continue

            if s < 0 or e <= s or e > len(sec_text):
                bad_offsets += 1
                continue

            if validate_anchor_text:
                anchor = (link.get("anchor_text") or "")
                if anchor:
                    sub = sec_text[s:e]
                    if sub != anchor:
                        bad_anchor_mismatch += 1
                        # keep span anyway

            kept_links += 1
            spans.append(
                {
                    "start": sec_base + s,
                    "end": sec_base + e,
                    "type": "internal",
                }
            )

        cursor += len(sec_text)
        cursor += len(section_sep)

    page_text = section_sep.join(texts)

    clipped: List[Dict[str, Any]] = []
    for sp in spans:
        if 0 <= sp["start"] < sp["end"] <= len(page_text):
            clipped.append(sp)

    stats = {
        "total_links": total_links,
        "kept_links": kept_links,
        "skipped_non_internal_links": skipped_non_internal_links,
        "bad_offsets": bad_offsets,
        "bad_anchor_mismatch": bad_anchor_mismatch,
        "skipped_external_sections": skipped_external_sections,
        "num_spans": len(clipped),
        "num_sections_in_file": len(section_rows),
        "num_sections_used": len(texts),
        "text_len": len(page_text),
    }
    return page_text, clipped, stats


# ---------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------

def split_page_into_paragraph_examples(
    doc_id_prefix: str,
    page_key: str,
    page_text: str,
    page_spans: List[Dict[str, Any]],
    paragraph_separator: str = "\n\n",
    min_chars: int = 1,
    require_spans: bool = True,
) -> List[Dict[str, Any]]:
    """
    Split page_text into paragraphs and produce paragraph-level examples.

    - Keeps only spans fully contained within each paragraph.
    - Remaps spans to paragraph-local offsets.
    """
    examples: List[Dict[str, Any]] = []

    cursor = 0
    paras = page_text.split(paragraph_separator)

    for i, para in enumerate(paras):
        para_start = cursor
        para_end = para_start + len(para)

        cursor = para_end + len(paragraph_separator)

        if len(para.strip()) < min_chars:
            continue

        para_spans: List[Dict[str, Any]] = []
        for sp in page_spans:
            if para_start <= sp["start"] and sp["end"] <= para_end:
                para_spans.append(
                    {
                        "start": sp["start"] - para_start,
                        "end": sp["end"] - para_start,
                        "type": sp.get("type", "internal"),
                    }
                )

        if require_spans and not para_spans:
            continue

        examples.append(
            {
                "doc_id": f"{doc_id_prefix}||{page_key}||para_{i}",
                "text": para,
                "spans": para_spans,
                "meta": {
                    "page_key": page_key,
                    "paragraph_index": i,
                },
            }
        )

    return examples


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/span_id_prep.yaml")
    ap.add_argument("--domain", required=True, help="money-heist or marvel")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    sections_root = Path(cfg["sections_dir"]).expanduser()
    out_root = Path(cfg["out_dir"]).expanduser()

    validate_anchor_text = bool(cfg.get("validate_anchor_text", True))
    section_sep = cfg.get("section_separator", "\n\n")

    keep_internal_only = bool(cfg.get("keep_internal_only", True))
    drop_external_sections = bool(cfg.get("drop_external_sections", True))

    paragraph_separator = cfg.get("paragraph_separator", "\n\n")
    min_paragraph_chars = int(cfg.get("min_paragraph_chars", 1))
    require_spans = bool(cfg.get("paragraph_require_spans", False))

    sections_dir = sections_root / f"sections_parsed_{args.domain}_by_page"
    if not sections_dir.exists():
        raise FileNotFoundError(f"Sections dir not found: {sections_dir}")

    # Write into a separate subfolder so you don't overwrite page-level
    out_dir = out_root / f"{args.domain}_paragraph"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_train: List[Dict] = []
    rows_dev: List[Dict] = []
    rows_test: List[Dict] = []
    per_page_stats: List[Dict] = []

    page_files = sorted(sections_dir.glob("*.jsonl"))
    if not page_files:
        raise FileNotFoundError(f"No per-page section JSONLs found in {sections_dir}")

    pages_processed = 0
    paras_written = 0

    for page_jsonl in page_files:
        page_key = page_jsonl.stem

        section_rows = load_page_sections_jsonl(page_jsonl)
        if not section_rows:
            continue

        page_text, spans, st = build_page_text_and_spans_from_sections(
            section_rows=section_rows,
            section_sep=section_sep,
            validate_anchor_text=validate_anchor_text,
            keep_internal_only=keep_internal_only,
            drop_external_sections=drop_external_sections,
        )

        para_examples = split_page_into_paragraph_examples(
            doc_id_prefix=args.domain,
            page_key=page_key,
            page_text=page_text,
            page_spans=spans,
            paragraph_separator=paragraph_separator,
            min_chars=min_paragraph_chars,
            require_spans=require_spans,
        )

        bucket = split_bucket(page_key)  # IMPORTANT: split by page_key to avoid leakage across splits
        if bucket == "train":
            rows_train.extend(para_examples)
        elif bucket == "dev":
            rows_dev.extend(para_examples)
        else:
            rows_test.extend(para_examples)

        st.update({"page_key": page_key, "sections_file": str(page_jsonl), "num_paragraphs_written": len(para_examples)})
        per_page_stats.append(st)

        pages_processed += 1
        paras_written += len(para_examples)

    write_jsonl(str(out_dir / "train.jsonl"), rows_train)
    write_jsonl(str(out_dir / "dev.jsonl"), rows_dev)
    write_jsonl(str(out_dir / "test.jsonl"), rows_test)

    summary = {
        "domain": args.domain,
        "mode": "paragraph",
        "sections_dir": str(sections_dir),
        "out_dir": str(out_dir),
        "pages_total_sections_files": len(page_files),
        "pages_processed": pages_processed,
        "paragraphs_written": paras_written,
        "train_docs": len(rows_train),
        "dev_docs": len(rows_dev),
        "test_docs": len(rows_test),
        "validate_anchor_text": validate_anchor_text,
        "section_separator": section_sep,
        "keep_internal_only": keep_internal_only,
        "drop_external_sections": drop_external_sections,
        "paragraph_separator": paragraph_separator,
        "min_paragraph_chars": min_paragraph_chars,
        "paragraph_require_spans": require_spans,
    }

    write_json(str(out_dir / "stats.json"), {"summary": summary, "per_page": per_page_stats})

    print("Wrote:", out_dir)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
