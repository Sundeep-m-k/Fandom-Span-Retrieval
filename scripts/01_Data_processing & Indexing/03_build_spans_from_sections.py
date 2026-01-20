#!/usr/bin/env python3
"""
03_build_spans_from_sections.py

Build spans from 02_parse_html_to_sections.py output and align links to spans.

Critical rule (for correctness):
- Do NOT strip/normalize `rec["text"]` here. 02 already produced the canonical
  text and link offsets are relative to that exact string.

Outputs (UPDATED: under data/processed/<domain>/):
  data/processed/<domain>/spans_<domain>.csv
  data/processed/<domain>/span_links_<domain>.csv
  data/processed/<domain>/spans_<domain>_by_page/<article_id|page_name|stem>.csv
  data/processed/<domain>/span_links_<domain>_by_page/<article_id|page_name|stem>.csv
"""

import sys
import re
import json
import csv
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


# ---------------------- CONFIG ----------------------


def load_scraping_config():
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is not installed. Install it with: pip install pyyaml")
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "scraping.yaml"
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "base_url" not in cfg:
        print("ERROR: 'base_url' must be defined in configs/scraping.yaml")
        sys.exit(1)

    return cfg, project_root


# ---------------------- LOGGING ----------------------


def create_logger(project_root: Path, script_name: str = "03_build_spans_from_sections"):
    log_dir = project_root / "data" / "logs" / "preprocessing"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{ts}_{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger initialized for {script_name}")
    logger.info(f"Log file: {log_path}")
    return logger, log_path


# ---------------------- SPAN HELPERS ----------------------

_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_NL_BOUNDARY_RE = re.compile(r"\n+")


def iter_sentence_spans(text: str):
    """
    Return sentence spans as (start, end) indices over the ORIGINAL text.
    Keeps offsets stable; no .strip() anywhere.
    """
    if not text:
        return []

    cuts = [0]
    for m in _SENT_BOUNDARY_RE.finditer(text):
        cuts.append(m.end())
    cuts.append(len(text))

    spans = []
    for a, b in zip(cuts, cuts[1:]):
        if text[a:b].strip():
            spans.append((a, b))

    if len(spans) <= 1:
        cuts = [0]
        for m in _NL_BOUNDARY_RE.finditer(text):
            cuts.append(m.end())
        cuts.append(len(text))
        spans = []
        for a, b in zip(cuts, cuts[1:]):
            if text[a:b].strip():
                spans.append((a, b))

    return spans


def build_spans_for_section(
    text: str,
    max_chars: int = 512,
    max_sentences: int = 4,
    min_chars: int = 30,
    hard_max_chars: int = 1024,
):
    """
    Build spans as absolute char ranges over `text` (the exact string from 02).
    """
    if not text or not text.strip():
        return []

    sent_spans = iter_sentence_spans(text)
    if not sent_spans:
        return []

    spans = []
    cur_start = None
    cur_end = None
    cur_sent_count = 0

    for s_start, s_end in sent_spans:
        if cur_start is None:
            cur_start, cur_end, cur_sent_count = s_start, s_end, 1
            continue

        tentative_len = s_end - cur_start
        tentative_sent_count = cur_sent_count + 1

        if tentative_len > max_chars or tentative_sent_count > max_sentences:
            span_text = text[cur_start:cur_end]
            span_len = len(span_text)
            if span_len >= min_chars and span_text.strip():
                spans.append({
                    "start": cur_start,
                    "end": cur_end,
                    "text": span_text,
                    "len_chars": span_len,
                    "num_sents": cur_sent_count,
                })
            cur_start, cur_end, cur_sent_count = s_start, s_end, 1
        else:
            cur_end = s_end
            cur_sent_count = tentative_sent_count

    if cur_start is not None and cur_end is not None:
        span_text = text[cur_start:cur_end]
        span_len = len(span_text)
        if span_len >= min_chars and span_text.strip():
            spans.append({
                "start": cur_start,
                "end": cur_end,
                "text": span_text,
                "len_chars": span_len,
                "num_sents": cur_sent_count,
            })

    final_spans = []
    for sp in spans:
        if sp["len_chars"] <= hard_max_chars:
            final_spans.append(sp)
            continue

        start_base = sp["start"]
        span_text = sp["text"]

        offset = 0
        while offset < len(span_text):
            chunk = span_text[offset: offset + hard_max_chars]
            if not chunk.strip():
                break
            chunk_start = start_base + offset
            chunk_end = chunk_start + len(chunk)
            if len(chunk) >= min_chars:
                final_spans.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": chunk,
                    "len_chars": len(chunk),
                    "num_sents": None,
                })
            offset += hard_max_chars

    return final_spans


# ---------------------- CSV HELPERS ----------------------


def safe_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def append_rows(path: Path, fieldnames, rows):
    """
    Append rows to a CSV. Write header if file is new/empty.
    """
    if not rows:
        return
    is_new = (not path.exists()) or (path.stat().st_size == 0)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------- CORE ----------------------


def build_spans_and_links(
    cfg,
    project_root: Path,
    logger: logging.Logger,
    max_chars: int = 512,
    max_sentences: int = 4,
    min_chars: int = 30,
    hard_max_chars: int = 1024,
):
    base_url = cfg["base_url"].rstrip("/")
    domain = urlparse(base_url).netloc.split(".")[0]

    interim_dir = project_root / "data" / "interim" / domain

    # UPDATED: everything for this domain goes under data/processed/<domain>/
    processed_dir = project_root / "data" / "processed" / domain
    processed_dir.mkdir(parents=True, exist_ok=True)

    sections_path = interim_dir / f"sections_parsed_{domain}.jsonl"
    if not sections_path.exists():
        logger.error(f"Sections file not found: {sections_path}")
        sys.exit(1)

    spans_master = processed_dir / f"spans_{domain}.csv"
    links_master = processed_dir / f"span_links_{domain}.csv"

    spans_by_page_dir = processed_dir / f"spans_{domain}_by_page"
    links_by_page_dir = processed_dir / f"span_links_{domain}_by_page"
    spans_by_page_dir.mkdir(parents=True, exist_ok=True)
    links_by_page_dir.mkdir(parents=True, exist_ok=True)

    span_fieldnames = [
        "span_id",
        "article_id",
        "page_name",
        "title",
        "section",
        "span_index",
        "start_char",
        "end_char",
        "len_chars",
        "num_sents",
        "text",
        "url",
        "source_path",
    ]
    link_fieldnames = [
        "span_id",
        "span_index",
        "article_id",
        "page_name",
        "title",
        "section",
        "anchor_index",
        "anchor_text",
        "link_type",
        "link_abs_start",
        "link_abs_end",
        "link_rel_start",
        "link_rel_end",
        "target_page_name",
        "article_id_of_internal_link",
        "resolved_url",
        "url",
        "source_path",
    ]

    # ensure masters start fresh
    spans_master.unlink(missing_ok=True)
    links_master.unlink(missing_ok=True)

    logger.info(f"Sections JSONL: {sections_path}")
    logger.info(f"Master spans:   {spans_master}")
    logger.info(f"Master links:   {links_master}")
    logger.info(f"Per-page spans dir: {spans_by_page_dir}")
    logger.info(f"Per-page links dir: {links_by_page_dir}")
    logger.info(
        "Span settings: max_chars=%d, max_sentences=%d, min_chars=%d, hard_max_chars=%d",
        max_chars, max_sentences, min_chars, hard_max_chars,
    )

    total_section_rows = 0
    sections_with_spans = 0
    total_spans = 0
    total_links = 0
    pages_seen = set()
    span_id_counter = 0

    with open(sections_path, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_idx}: JSON decode error: {e}")
                continue

            article_id = rec.get("article_id")
            page_name = rec.get("page_name")
            title = rec.get("title")
            section = rec.get("section") or "Unknown"
            text = rec.get("text") or ""   # DO NOT strip; offsets depend on this exact string
            url = rec.get("url")
            source_path = rec.get("source_path")
            links = rec.get("links") or []

            if not text.strip():
                continue

            total_section_rows += 1
            pages_seen.add(article_id or page_name or source_path)

            spans = build_spans_for_section(
                text,
                max_chars=max_chars,
                max_sentences=max_sentences,
                min_chars=min_chars,
                hard_max_chars=hard_max_chars,
            )
            if not spans:
                continue

            sections_with_spans += 1

            if article_id is not None:
                page_key = str(article_id)
            elif page_name:
                page_key = page_name
            elif source_path:
                page_key = Path(source_path).stem
            else:
                page_key = f"line_{line_idx}"

            page_key_safe = safe_key(page_key)
            spans_page_path = spans_by_page_dir / f"{page_key_safe}.csv"
            links_page_path = links_by_page_dir / f"{page_key_safe}.csv"

            span_rows = []
            link_rows = []

            for span_index, sp in enumerate(spans):
                span_id_counter += 1
                span_id = f"{domain}_span_{span_id_counter:07d}"

                span_rows.append({
                    "span_id": span_id,
                    "article_id": article_id,
                    "page_name": page_name,
                    "title": title,
                    "section": section,
                    "span_index": span_index,
                    "start_char": sp["start"],
                    "end_char": sp["end"],
                    "len_chars": sp["len_chars"],
                    "num_sents": sp.get("num_sents"),
                    "text": sp["text"],
                    "url": url,
                    "source_path": source_path,
                })
                total_spans += 1

                s_start, s_end = sp["start"], sp["end"]
                anchor_index = 0

                for lk in links:
                    l_start = lk.get("start")
                    l_end = lk.get("end")
                    if l_start is None or l_end is None:
                        continue

                    if not (s_start <= l_start < s_end):
                        continue

                    if l_end > s_end:
                        l_end = s_end

                    anchor_index += 1
                    link_rows.append({
                        "span_id": span_id,
                        "span_index": span_index,
                        "article_id": article_id,
                        "page_name": page_name,
                        "title": title,
                        "section": section,
                        "anchor_index": anchor_index,
                        "anchor_text": lk.get("anchor_text") or "",
                        "link_type": lk.get("link_type"),
                        "link_abs_start": l_start,
                        "link_abs_end": l_end,
                        "link_rel_start": l_start - s_start,
                        "link_rel_end": l_end - s_start,
                        "target_page_name": lk.get("target_page_name"),
                        "article_id_of_internal_link": lk.get("target_article_id"),
                        "resolved_url": lk.get("resolved_url"),
                        "url": url,
                        "source_path": source_path,
                    })
                    total_links += 1

            append_rows(spans_master, span_fieldnames, span_rows)
            append_rows(links_master, link_fieldnames, link_rows)
            append_rows(spans_page_path, span_fieldnames, span_rows)
            append_rows(links_page_path, link_fieldnames, link_rows)

            if line_idx % 500 == 0:
                logger.info(
                    "Processed %d records | sections_with_spans=%d | spans=%d | links=%d",
                    line_idx, sections_with_spans, total_spans, total_links
                )

    logger.info("=== Span build summary ===")
    logger.info(f"Pages seen (unique):     {len(pages_seen)}")
    logger.info(f"Section rows processed:  {total_section_rows}")
    logger.info(f"Sections with spans:     {sections_with_spans}")
    logger.info(f"Spans generated:         {total_spans}")
    logger.info(f"Span-links generated:    {total_links}")
    logger.info(f"Master spans:            {spans_master}")
    logger.info(f"Master links:            {links_master}")
    logger.info(f"Per-page spans dir:      {spans_by_page_dir}")
    logger.info(f"Per-page links dir:      {links_by_page_dir}")

    return spans_master, links_master, spans_by_page_dir, links_by_page_dir


# ---------------------- ENTRYPOINT ----------------------


if __name__ == "__main__":
    cfg, project_root = load_scraping_config()
    logger, log_path = create_logger(project_root)

    start = time.time()
    logger.info("=== Starting 03_build_spans_from_sections ===")

    try:
        spans_path, links_path, spans_by_page_dir, links_by_page_dir = build_spans_and_links(
            cfg, project_root, logger
        )
    except Exception as e:
        logger.error("Unhandled error in 03_build_spans_from_sections: %s", e, exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info("Finished 03_build_spans_from_sections in %.2f seconds", elapsed)
    logger.info("Log file: %s", log_path)

    print(f"📄 Spans (master) written to: {spans_path}")
    print(f"🔗 Span-links (master) written to: {links_path}")
    print(f"📂 Per-page spans under: {spans_by_page_dir}")
    print(f"📂 Per-page span-links under: {links_by_page_dir}")
    print(f"📝 Log saved to: {log_path}")