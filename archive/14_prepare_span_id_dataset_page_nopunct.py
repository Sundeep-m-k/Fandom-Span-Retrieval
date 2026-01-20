#!/usr/bin/env python3
"""
Prepare PAGE-level Span Identification dataset (internal links only),
with OPTIONAL punctuation removal while tracking original boundaries.

This script:
1) Loads per-page section JSONL files (each contains multiple sections).
2) Concatenates section texts into a single page-level document.
3) Converts section-local link offsets into page-level offsets.
4) Filters to internal links only (optional).
5) Optionally drops "External links"/"References" sections (best-effort).
6) Optionally removes punctuation from the final page text AND remaps spans.
7) Splits pages deterministically into train/dev/test.
8) Writes JSONL datasets and statistics.

IMPORTANT:
- When punctuation removal is enabled, we:
   - produce "text" = punctuation-stripped text
   - produce "spans" = remapped spans in stripped text coordinates
   - store original spans in meta["orig_spans"] for traceability
"""

from __future__ import annotations

import argparse
import json
import unicodedata
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
    Deterministically assign a document to train/dev/test split
    based on a hash of its page key.

    Distribution:
      - 80% train
      - 10% dev
      - 10% test
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
    Best-effort detection of sections like External links / References / etc.
    Only works if section metadata exists in section JSON objects.
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
# Punctuation removal + span remapping
# ---------------------------------------------------------------------

def is_punctuation(ch: str) -> bool:
    """
    Return True if a character is Unicode punctuation.
    This covers ASCII punctuation AND Unicode punctuation like “ ” — – ’ etc.
    """
    # Unicode category starting with 'P' means punctuation
    return unicodedata.category(ch).startswith("P")


def strip_punct_with_mapping(text: str) -> Tuple[str, List[int]]:
    """
    Remove punctuation characters from `text` and build an index mapping.

    Returns:
      - new_text: punctuation-removed text
      - old2new: list of length len(text)+1 where:
          old2new[i] = length of new_text after processing text[:i]
        So a span [start,end] in old text maps to:
          new_start = old2new[start]
          new_end   = old2new[end]
    """
    out_chars: List[str] = []
    old2new: List[int] = [0] * (len(text) + 1)

    new_i = 0
    for i, ch in enumerate(text):
        old2new[i] = new_i
        if not is_punctuation(ch):
            out_chars.append(ch)
            new_i += 1

    old2new[len(text)] = new_i
    return "".join(out_chars), old2new


def remap_spans_remove_punct(
    spans: List[Dict[str, Any]],
    old2new: List[int],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remap spans from original text indices -> punctuation-removed text indices.
    Drops spans that collapse to empty after punctuation removal.

    Returns:
      - new_spans
      - dropped_count
    """
    new_spans: List[Dict[str, Any]] = []
    dropped = 0

    for sp in spans:
        s = int(sp["start"])
        e = int(sp["end"])
        ns = old2new[s]
        ne = old2new[e]
        if ne <= ns:
            dropped += 1
            continue
        new_spans.append({"start": ns, "end": ne, "type": sp.get("type", "internal")})

    return new_spans, dropped


# ---------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------

def build_page_text_and_spans_from_sections(
    section_rows: List[Dict[str, Any]],
    section_sep: str = "\n\n",
    validate_anchor_text: bool = True,
    keep_internal_only: bool = True,
    drop_external_sections: bool = True,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a single page-level document by concatenating section texts
    and remapping section-level link spans into page-level offsets.
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
                        # Keep span anyway; offsets are authoritative

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
# Main entry point
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/span_id_prep.yaml")
    ap.add_argument("--domain", required=True, help="money-heist or marvel")
    ap.add_argument(
        "--remove_punct",
        action="store_true",
        help="If set, remove punctuation from the final page text and remap spans.",
    )
    ap.add_argument(
        "--out_suffix",
        default="",
        help="Optional suffix appended to output domain dir (e.g., '_nopunct').",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    sections_root = Path(cfg["sections_dir"]).expanduser()
    out_root = Path(cfg["out_dir"]).expanduser()

    validate_anchor_text = bool(cfg.get("validate_anchor_text", True))
    section_sep = cfg.get("section_separator", "\n\n")

    keep_internal_only = bool(cfg.get("keep_internal_only", True))
    drop_external_sections = bool(cfg.get("drop_external_sections", True))

    sections_dir = sections_root / f"sections_parsed_{args.domain}_by_page"
    if not sections_dir.exists():
        raise FileNotFoundError(f"Sections dir not found: {sections_dir}")

    # Output directory naming:
    # - original: <out_root>/<domain>
    # - with suffix: <out_root>/<domain><suffix>  (e.g., money-heist_nopunct)
    out_dir = out_root / f"{args.domain}{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_train: List[Dict] = []
    rows_dev: List[Dict] = []
    rows_test: List[Dict] = []
    per_page_stats: List[Dict] = []

    page_files = sorted(sections_dir.glob("*.jsonl"))
    if not page_files:
        raise FileNotFoundError(f"No per-page section JSONLs found in {sections_dir}")

    pages_processed = 0
    total_spans_dropped_after_punct = 0

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

        orig_text = page_text
        orig_spans = [{"start": s["start"], "end": s["end"], "type": s.get("type", "internal")} for s in spans]

        # Optional punctuation removal + span remapping
        if args.remove_punct:
            new_text, old2new = strip_punct_with_mapping(orig_text)
            new_spans, dropped = remap_spans_remove_punct(orig_spans, old2new)
            total_spans_dropped_after_punct += dropped

            page_text = new_text
            spans = new_spans

        example = {
            "doc_id": f"{args.domain}||{page_key}",
            "text": page_text,
            "spans": spans,
            "meta": {
                "sections_file": str(page_jsonl),
                "num_sections_used": st["num_sections_used"],
                "num_sections_in_file": st["num_sections_in_file"],
                # Track original boundaries for debugging/traceability
                "remove_punct": bool(args.remove_punct),
                "orig_text_len": len(orig_text),
                "orig_spans": orig_spans,  # original positions in original text
            },
        }

        bucket = split_bucket(page_key)
        if bucket == "train":
            rows_train.append(example)
        elif bucket == "dev":
            rows_dev.append(example)
        else:
            rows_test.append(example)

        st.update(
            {
                "doc_id": example["doc_id"],
                "page_key": page_key,
                "sections_file": str(page_jsonl),
                "remove_punct": bool(args.remove_punct),
                "spans_dropped_after_punct": int(total_spans_dropped_after_punct) if args.remove_punct else 0,
            }
        )
        per_page_stats.append(st)
        pages_processed += 1

    write_jsonl(str(out_dir / "train.jsonl"), rows_train)
    write_jsonl(str(out_dir / "dev.jsonl"), rows_dev)
    write_jsonl(str(out_dir / "test.jsonl"), rows_test)

    summary = {
        "domain": args.domain,
        "sections_dir": str(sections_dir),
        "out_dir": str(out_dir),
        "pages_total_sections_files": len(page_files),
        "pages_processed": pages_processed,
        "train_docs": len(rows_train),
        "dev_docs": len(rows_dev),
        "test_docs": len(rows_test),
        "validate_anchor_text": validate_anchor_text,
        "section_separator": section_sep,
        "keep_internal_only": keep_internal_only,
        "drop_external_sections": drop_external_sections,
        "remove_punct": bool(args.remove_punct),
        "total_spans_dropped_after_punct": int(total_spans_dropped_after_punct),
    }

    write_json(str(out_dir / "stats.json"), {"summary": summary, "per_page": per_page_stats})

    print("Wrote:", out_dir)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
