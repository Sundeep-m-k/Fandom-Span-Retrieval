#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

# ---------------------------------------------------
# Ensure project root on PYTHONPATH
# ---------------------------------------------------
PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
sys.path.insert(0, str(PROJECT_ROOT))

"""
14_prepare_span_id_dataset_paragraph_nopunct.py (PIPELINE)

NEW ENGINEERING kept:
- Input:  data/interim/<domain>/sections_parsed_<domain>_by_page/*.jsonl
- Output: data/span_identification/<domain>/train<suffix>.jsonl etc.
- Logs:   data/logs/span_identification/<domain>/14_prepare_span_id_dataset_<domain>.log
- Reads domain from pipeline config.

LEGACY BEHAVIOR adapted (to reproduce "old" dataset behavior/metrics):
(A) Split bucket: old-style hash(page_key)%100
    - Option 1: set env PYTHONHASHSEED=0 and use real Python hash() (closest match)
    - Option 2: stable "hash-like" bucket using md5 -> int -> %100 (reproducible across machines)
(B) Page offset cursor: legacy cursor update that adds section_sep after every kept section,
    which can create the same offset behavior as the old script.

Use flags:
  --split_mode stable  (default)  : reproducible "hash-like" via md5
  --split_mode legacy_hash        : uses Python hash(); set PYTHONHASHSEED=0 for reproducibility
  --offset_mode new               : correct separator accounting (recommended for rigor)
  --offset_mode legacy            : emulate old cursor behavior (recommended to match old numbers)
"""

import argparse
import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import yaml

from src.utils.logging_utils import create_logger
from src.span_identifier.prep.io_jsonl import write_jsonl, write_json


# ----------------------------
# Deterministic / legacy split
# ----------------------------

def bucket_stable_hashlike(page_key: str) -> str:
    """
    Stable replacement that mimics 'hash()%100' distribution but is deterministic.
    """
    h = hashlib.md5(page_key.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) % 100
    if v < 80:
        return "train"
    if v < 90:
        return "dev"
    return "test"


def bucket_legacy_python_hash(page_key: str) -> str:
    """
    Old behavior: abs(hash(page_key)) % 100
    NOTE: This is ONLY reproducible if PYTHONHASHSEED is fixed (e.g., PYTHONHASHSEED=0).
    """
    v = abs(hash(page_key)) % 100
    if v < 80:
        return "train"
    if v < 90:
        return "dev"
    return "test"


# ----------------------------
# Stats
# ----------------------------

@dataclass
class PageStats:
    page_key: str
    sections_file: str
    num_sections_in_file: int
    num_sections_used: int
    text_len: int

    total_links: int
    kept_links: int
    skipped_non_internal_links: int
    bad_offsets: int
    bad_anchor_mismatch: int
    skipped_external_sections: int

    num_spans: int
    num_paras_total: int
    num_paras_kept: int

    spans_dropped_after_punct: int


@dataclass
class GlobalStats:
    domain: str
    sections_dir: str
    out_dir: str
    log_dir: str

    pages_total_sections_files: int
    pages_processed: int

    train_docs: int
    dev_docs: int
    test_docs: int

    paragraphs_total: int
    paragraphs_kept: int

    require_spans: bool
    para_sep: str

    validate_anchor_text: bool
    section_separator: str
    keep_internal_only: bool
    drop_external_sections: bool

    remove_punct: bool
    total_spans_dropped_after_punct: int

    output_suffix: str

    split_mode: str
    offset_mode: str


# ----------------------------
# Config helpers
# ----------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_domain_from_pipeline_config(pipeline_cfg_path: Path) -> str:
    if not pipeline_cfg_path.exists():
        raise FileNotFoundError(f"Missing pipeline config: {pipeline_cfg_path}")

    cfg = load_yaml(pipeline_cfg_path)
    domain = cfg.get("domain")
    if not domain:
        raise ValueError(f"Missing required key 'domain' in {pipeline_cfg_path}")
    return str(domain).strip()


# ----------------------------
# IO helpers
# ----------------------------

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
# Punctuation removal + mapping
# ----------------------------

def is_punctuation(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def strip_punct_with_mapping(text: str) -> Tuple[str, List[int]]:
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
# Page build (two modes)
# ----------------------------

def build_page_text_and_internal_spans(
    section_rows: List[Dict[str, Any]],
    section_sep: str,
    validate_anchor_text: bool,
    keep_internal_only: bool,
    drop_external_sections: bool,
    logger,
    page_key: str,
    offset_mode: str,  # "new" or "legacy"
) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
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
        sec_len = len(sec_text)

        # -----------------------------
        # OFFSET MODE SWITCH
        # -----------------------------
        if offset_mode == "new":
            # Correct: add sep length only between kept sections
            if texts:
                cursor += len(section_sep)
            texts.append(sec_text)
            sec_base = cursor
            cursor += sec_len
        elif offset_mode == "legacy":
            # Legacy: emulate old cursor logic (adds sep after every kept section)
            texts.append(sec_text)
            sec_base = cursor
            cursor += sec_len
            cursor += len(section_sep)
        else:
            raise ValueError(f"Unknown offset_mode: {offset_mode}")

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

            if s < 0 or e <= s or e > sec_len:
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

    page_text = section_sep.join(texts)
    page_len = len(page_text)

    clipped: List[Dict[str, Any]] = []
    for sp in spans:
        s = int(sp["start"])
        e = int(sp["end"])
        if 0 <= s < e <= page_len:
            clipped.append(sp)
        else:
            # In legacy mode this can happen more often; keep warning for traceability
            logger.warning(f"Span out of range | page={page_key} | span=({s},{e}) | page_len={page_len}")

    st = {
        "total_links": total_links,
        "kept_links": kept_links,
        "skipped_non_internal_links": skipped_non_internal_links,
        "bad_offsets": bad_offsets,
        "bad_anchor_mismatch": bad_anchor_mismatch,
        "skipped_external_sections": skipped_external_sections,
        "num_spans": len(clipped),
        "num_sections_in_file": len(section_rows),
        "num_sections_used": len(texts),
        "text_len": page_len,
    }
    return page_text, clipped, st


# ----------------------------
# Paragraph splitting
# ----------------------------

def split_paragraphs_with_offsets(text: str, para_sep: str) -> List[Tuple[int, int, str]]:
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
    t0 = time.perf_counter()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "span_id_prep.yaml"))
    ap.add_argument("--pipeline_config", default=str(PROJECT_ROOT / "configs" / "pipeline_span_id.yaml"))
    ap.add_argument("--out_suffix", default="_paragraph_nopunct")
    ap.add_argument("--remove_punct", action="store_true")
    ap.add_argument("--para_sep", default="\n\n")
    ap.add_argument("--require_spans", action="store_true")

    # NEW knobs to emulate old behavior
    ap.add_argument("--split_mode", choices=["stable", "legacy_hash"], default="stable",
                    help="stable=md5-based; legacy_hash=python hash (set PYTHONHASHSEED=0 to reproduce).")
    ap.add_argument("--offset_mode", choices=["new", "legacy"], default="legacy",
                    help="legacy emulates old cursor/sep behavior; new is offset-correct.")

    args = ap.parse_args()

    PIPELINE_CFG = Path(args.pipeline_config)
    domain = load_domain_from_pipeline_config(PIPELINE_CFG)

    SECTIONS_DIR = PROJECT_ROOT / "data" / "interim" / domain / f"sections_parsed_{domain}_by_page"
    OUT_DIR = PROJECT_ROOT / "data" / "span_identification" / domain
    LOG_DIR = PROJECT_ROOT / "data" / "logs" / "span_identification" / domain
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger, log_file = create_logger(LOG_DIR, f"14_prepare_span_id_dataset_{domain}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"DOMAIN: {domain}")
    logger.info(f"PIPELINE_CFG: {PIPELINE_CFG}")
    logger.info(f"SECTIONS_DIR: {SECTIONS_DIR}")
    logger.info(f"OUT_DIR: {OUT_DIR}")
    logger.info(f"split_mode={args.split_mode} | offset_mode={args.offset_mode}")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing span prep config: {cfg_path}")
    cfg = load_yaml(cfg_path)

    validate_anchor_text = bool(cfg.get("validate_anchor_text", True))
    section_sep = str(cfg.get("section_separator", "\n\n"))
    keep_internal_only = bool(cfg.get("keep_internal_only", True))
    drop_external_sections = bool(cfg.get("drop_external_sections", True))

    if not SECTIONS_DIR.exists():
        raise FileNotFoundError(f"Sections dir not found: {SECTIONS_DIR}")

    page_files = sorted(SECTIONS_DIR.glob("*.jsonl"))
    if not page_files:
        raise FileNotFoundError(f"No per-page section JSONLs found in {SECTIONS_DIR}")

    suf = args.out_suffix.strip()
    train_path = OUT_DIR / f"train{suf}.jsonl"
    dev_path = OUT_DIR / f"dev{suf}.jsonl"
    test_path = OUT_DIR / f"test{suf}.jsonl"
    stats_path = OUT_DIR / f"stats{suf}.json"

    rows_train: List[Dict[str, Any]] = []
    rows_dev: List[Dict[str, Any]] = []
    rows_test: List[Dict[str, Any]] = []
    per_page_stats: List[Dict[str, Any]] = []

    pages_processed = 0
    total_paras = 0
    total_paras_kept = 0
    total_spans_dropped_after_punct = 0

    logger.info(f"Found per-page section files: {len(page_files)}")
    logger.info("Starting processing...")

    for idx, page_jsonl in enumerate(page_files, start=1):
        page_key = page_jsonl.stem

        section_rows = load_page_sections_jsonl(page_jsonl)
        if not section_rows:
            logger.warning(f"Empty page file | page={page_key} | file={page_jsonl}")
            continue

        page_text, page_spans, st = build_page_text_and_internal_spans(
            section_rows=section_rows,
            section_sep=section_sep,
            validate_anchor_text=validate_anchor_text,
            keep_internal_only=keep_internal_only,
            drop_external_sections=drop_external_sections,
            logger=logger,
            page_key=page_key,
            offset_mode=args.offset_mode,
        )

        paras = split_paragraphs_with_offsets(page_text, para_sep=args.para_sep)
        total_paras += len(paras)

        if args.split_mode == "stable":
            bucket = bucket_stable_hashlike(page_key)
        else:
            bucket = bucket_legacy_python_hash(page_key)

        kept_here = 0
        dropped_here_punct = 0

        for pi, (p_start, p_end, p_text) in enumerate(paras):
            para_spans = spans_in_range(page_spans, p_start, p_end)

            if args.require_spans and len(para_spans) == 0:
                continue

            orig_para_text = p_text
            orig_para_spans = [{"start": s["start"], "end": s["end"], "type": s.get("type", "internal")}
                               for s in para_spans]

            if args.remove_punct:
                new_text, old2new = strip_punct_with_mapping(orig_para_text)
                new_spans, dropped = remap_spans(orig_para_spans, old2new)
                dropped_here_punct += dropped
                total_spans_dropped_after_punct += dropped
                p_text = new_text
                para_spans = new_spans

            ex = {
                "doc_id": f"{domain}||{page_key}||para_{pi}",
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

        ps = PageStats(
            page_key=page_key,
            sections_file=str(page_jsonl),
            num_sections_in_file=int(st["num_sections_in_file"]),
            num_sections_used=int(st["num_sections_used"]),
            text_len=int(st["text_len"]),
            total_links=int(st["total_links"]),
            kept_links=int(st["kept_links"]),
            skipped_non_internal_links=int(st["skipped_non_internal_links"]),
            bad_offsets=int(st["bad_offsets"]),
            bad_anchor_mismatch=int(st["bad_anchor_mismatch"]),
            skipped_external_sections=int(st["skipped_external_sections"]),
            num_spans=int(st["num_spans"]),
            num_paras_total=len(paras),
            num_paras_kept=kept_here,
            spans_dropped_after_punct=int(dropped_here_punct),
        )
        per_page_stats.append(asdict(ps))
        pages_processed += 1

        if idx % 50 == 0:
            logger.info(
                f"Progress | pages={idx}/{len(page_files)} | "
                f"docs(train/dev/test)={len(rows_train)}/{len(rows_dev)}/{len(rows_test)} | "
                f"paras_kept={total_paras_kept}"
            )

    logger.info("Writing outputs...")
    write_jsonl(str(train_path), rows_train)
    write_jsonl(str(dev_path), rows_dev)
    write_jsonl(str(test_path), rows_test)

    gs = GlobalStats(
        domain=domain,
        sections_dir=str(SECTIONS_DIR),
        out_dir=str(OUT_DIR),
        log_dir=str(LOG_DIR),
        pages_total_sections_files=len(page_files),
        pages_processed=pages_processed,
        train_docs=len(rows_train),
        dev_docs=len(rows_dev),
        test_docs=len(rows_test),
        paragraphs_total=total_paras,
        paragraphs_kept=total_paras_kept,
        require_spans=bool(args.require_spans),
        para_sep=args.para_sep,
        validate_anchor_text=validate_anchor_text,
        section_separator=section_sep,
        keep_internal_only=keep_internal_only,
        drop_external_sections=drop_external_sections,
        remove_punct=bool(args.remove_punct),
        total_spans_dropped_after_punct=int(total_spans_dropped_after_punct),
        output_suffix=suf,
        split_mode=args.split_mode,
        offset_mode=args.offset_mode,
    )
    write_json(str(stats_path), {"summary": asdict(gs), "per_page": per_page_stats})

    t1 = time.perf_counter()
    logger.info("✅ 14_prepare_span_id_dataset_paragraph_nopunct completed successfully.")
    logger.info("Outputs:")
    logger.info(f"  {train_path}")
    logger.info(f"  {dev_path}")
    logger.info(f"  {test_path}")
    logger.info(f"  {stats_path}")
    logger.info(f"Total runtime: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
