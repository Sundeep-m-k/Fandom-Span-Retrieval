#!/usr/bin/env python3
# scripts/Span_Identification/14_prepare_span_id_dataset_paragraph_nopunct.py

"""
Prepare PARAGRAPH-level Span Identification dataset from section-level per-page JSONLs,
keeping ONLY internal links, optionally dropping external sections, and optionally removing
punctuation from paragraph text while remapping span boundaries.

Key points:
- We first build a page-level text by concatenating sections (like your page script).
- Then we split that page text into paragraphs using a separator (default: "\n\n").
- We keep spans that are fully contained inside a paragraph and remap them to paragraph-local offsets.
- Optional: remove punctuation in paragraph text and remap paragraph spans again.
- Splits (train/dev/test) are deterministic by page_key, so paragraphs from a page stay in one split.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.span_identifier.prep.io_jsonl import write_jsonl, write_json


# ----------------------------
# Basic utilities
# ----------------------------

def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_bucket(name: str) -> str:
    """Stable-ish split by page key (all paragraphs of a page go to same split)."""
    h = abs(hash(name)) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "dev"
    return "test"


def load_page_sections_jsonl(page_jsonl: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(page_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _norm(x: Any) -> str:
    return str(x).strip().lower()


def _is_external_section(rec: Dict[str, Any]) -> bool:
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


# ----------------------------
# Punctuation removal helpers
# ----------------------------

def is_punctuation(ch: str) -> bool:
    """Unicode punctuation check (covers ASCII + fancy punctuation)."""
    return unicodedata.category(ch).startswith("P")


def strip_punct_with_mapping(text: str) -> Tuple[str, List[int]]:
    """
    Remove punctuation from text and build an old->new index mapping.

    Returns:
      new_text
      old2new: length len(text)+1; old2new[i] = new length after processing text[:i]
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


def remap_spans(old_spans: List[Dict[str, Any]], old2new: List[int]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remap spans from old text indices to new text indices using old2new.
    Drops spans that collapse to empty.
    """
    new_spans: List[Dict[str, Any]] = []
    dropped = 0
    for sp in old_spans:
        s = int(sp["start"])
        e = int(sp["end"])
        ns = old2new[s]
        ne = old2new[e]
        if ne <= ns:
            dropped += 1
            continue
        new_spans.append({"start": ns, "end": ne, "type": sp.get("type", "internal")})
    return new_spans, dropped


# ----------------------------
# Page build (same idea as your page-level script)
# ----------------------------

def build_page_text_and_internal_spans(
    section_rows: List[Dict[str, Any]],
    section_sep: str = "\n\n",
    validate_anchor_text: bool = True,
    keep_internal_only: bool = True,
    drop_external_sections: bool = True,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
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

            kept_links += 1
            spans.append({"start": sec_base + s, "end": sec_base + e, "type": "internal"})

        cursor += len(sec_text)
        cursor += len(section_sep)

    page_text = section_sep.join(texts)

    clipped = [sp for sp in spans if 0 <= sp["start"] < sp["end"] <= len(page_text)]

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


# ----------------------------
# Paragraph splitting
# ----------------------------

def split_paragraphs_with_offsets(text: str, para_sep: str = "\n\n") -> List[Tuple[int, int, str]]:
    """
    Split text into paragraphs and return list of (start, end, para_text) in PAGE coordinates.
    Keeps empty paragraphs out.
    """
    out: List[Tuple[int, int, str]] = []
    n = len(text)
    i = 0
    sep_len = len(para_sep)

    while i <= n:
        j = text.find(para_sep, i)
        if j == -1:
            j = n
        para = text[i:j]
        if para.strip():
            out.append((i, j, para))
        i = j + sep_len
        if j == n:
            break
    return out


def spans_in_range(spans: List[Dict[str, Any]], lo: int, hi: int) -> List[Dict[str, Any]]:
    """
    Keep spans fully contained in [lo, hi) and shift them to be relative to lo.
    We drop spans that cross paragraph boundaries (rare but possible).
    """
    kept: List[Dict[str, Any]] = []
    for sp in spans:
        s = int(sp["start"])
        e = int(sp["end"])
        if lo <= s and e <= hi:
            kept.append({"start": s - lo, "end": e - lo, "type": sp.get("type", "internal")})
    return kept


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/span_id_prep.yaml")
    ap.add_argument("--domain", required=True, help="money-heist or marvel")
    ap.add_argument("--out_suffix", default="_paragraph_nopunct", help="suffix for out dir")
    ap.add_argument("--remove_punct", action="store_true", help="remove punctuation and remap spans")
    ap.add_argument("--para_sep", default="\n\n", help="paragraph separator used for splitting")
    ap.add_argument("--require_spans", action="store_true",
                    help="if set, only keep paragraphs that have at least one span (positives only)")
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
    total_paras = 0
    total_paras_kept = 0
    total_spans_dropped_after_punct = 0

    for page_jsonl in page_files:
        page_key = page_jsonl.stem
        section_rows = load_page_sections_jsonl(page_jsonl)
        if not section_rows:
            continue

        page_text, page_spans, st = build_page_text_and_internal_spans(
            section_rows=section_rows,
            section_sep=section_sep,
            validate_anchor_text=validate_anchor_text,
            keep_internal_only=keep_internal_only,
            drop_external_sections=drop_external_sections,
        )

        paras = split_paragraphs_with_offsets(page_text, para_sep=args.para_sep)
        total_paras += len(paras)

        bucket = split_bucket(page_key)

        kept_here = 0
        for pi, (p_start, p_end, p_text) in enumerate(paras):
            para_spans = spans_in_range(page_spans, p_start, p_end)

            if args.require_spans and len(para_spans) == 0:
                continue

            orig_para_text = p_text
            orig_para_spans = [{"start": s["start"], "end": s["end"], "type": s.get("type", "internal")}
                               for s in para_spans]

            # Optional punctuation removal at PARAGRAPH level
            if args.remove_punct:
                new_text, old2new = strip_punct_with_mapping(orig_para_text)
                new_spans, dropped = remap_spans(orig_para_spans, old2new)
                total_spans_dropped_after_punct += dropped
                p_text = new_text
                para_spans = new_spans

            ex = {
                "doc_id": f"{args.domain}||{page_key}||para_{pi}",
                "text": p_text,
                "spans": para_spans,
                "meta": {
                    "page_key": page_key,
                    "para_index": pi,
                    "sections_file": str(page_jsonl),
                    "page_text_len": len(page_text),
                    "para_start_in_page": p_start,
                    "para_end_in_page": p_end,
                    "remove_punct": bool(args.remove_punct),
                    # traceability for professor's requirement
                    "orig_para_text_len": len(orig_para_text),
                    "orig_para_spans": orig_para_spans,
                },
            }

            if bucket == "train":
                rows_train.append(ex)
            elif bucket == "dev":
                rows_dev.append(ex)
            else:
                rows_test.append(ex)

            kept_here += 1

        total_paras_kept += kept_here

        st.update(
            {
                "doc_id": f"{args.domain}||{page_key}",
                "page_key": page_key,
                "sections_file": str(page_jsonl),
                "num_paras_total": len(paras),
                "num_paras_kept": kept_here,
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
        "paragraphs_total": total_paras,
        "paragraphs_kept": total_paras_kept,
        "require_spans": bool(args.require_spans),
        "para_sep": args.para_sep,
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