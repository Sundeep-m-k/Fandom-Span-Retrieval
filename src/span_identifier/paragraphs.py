# src/span_identifier/paragraphs.py

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.utils.logging_utils import create_logger


def split_section_into_paragraphs(text: str) -> List[Dict[str, Any]]:
    """
    Split the section-level text into paragraphs and track absolute offsets.
    Here, 'absolute' means offsets inside this section's text string.
    """
    para_list = []
    offset = 0
    pid = 0

    # Paragraphs separated by blank lines at section level
    raw_paras = [p for p in text.split("\n\n") if p.strip() != ""]

    for p in raw_paras:
        idx = text.find(p, offset)
        if idx == -1:
            abs_start = offset
            abs_end = offset + len(p)
        else:
            abs_start = idx
            abs_end = idx + len(p)

        para_list.append(
            {
                "paragraph_id": pid,
                "abs_start": abs_start,  # offset in section text
                "abs_end": abs_end,
                "text": p,
            }
        )
        pid += 1
        offset = abs_end

    return para_list


def build_paragraph_master_from_cfg(span_cfg: dict):
    """
    Build paragraph-level span master CSV using section-level text from
    sections_parsed_<domain>.jsonl and span_links_<domain>.csv.

    Each row in the output represents one paragraph within a section,
    with spans whose link_abs_start/end fall inside that paragraph.
    """

    # ----- logger -----
    log_dir = Path(span_cfg["log_dir"])
    logger, log_file = create_logger(log_dir, script_name="01_build_paragraph_master")
    logger.info("Step 1: Building paragraph-level master CSV from section text")

    # Config
    project_root = Path(span_cfg.get("project_root", "."))  # optional; default to cwd
    domain = span_cfg["domain"]

    # Sections JSONL path (from 02_parse_html_to_sections)
    sections_path = project_root / "data" / "interim" / domain / f"sections_parsed_{domain}.jsonl"

    # Master links CSV from 03_build_spans_from_sections
    spans_csv_path = Path(span_cfg["spans_csv_path"])

    out_csv_path = Path(span_cfg["paragraph_master_csv"])
    min_spans_per_paragraph = int(span_cfg.get("min_spans_per_paragraph", 1))

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not sections_path.exists():
        logger.error(f"Sections JSONL not found: {sections_path}")
        raise FileNotFoundError(f"Sections JSONL not found: {sections_path}")

    logger.info(f"Reading sections from: {sections_path}")
    logger.info(f"Reading link spans from: {spans_csv_path}")

    # Load link spans
    spans_df = pd.read_csv(spans_csv_path)

    required_cols = {"article_id", "page_name", "section", "link_abs_start", "link_abs_end", "anchor_text"}
    missing = required_cols - set(spans_df.columns)
    if missing:
        logger.error(f"Missing required columns in spans CSV: {missing}")
        raise ValueError(f"Missing required columns in spans CSV: {missing}")

    num_articles = spans_df["article_id"].nunique()
    num_spans = len(spans_df)
    logger.info(f"Stats: unique articles in spans CSV = {num_articles}")
    logger.info(f"Stats: total spans in spans CSV   = {num_spans}")

    # Group spans by (article_id, page_name, section) to match sections_parsed
    group_cols = ["article_id", "page_name", "section"]
    spans_df["article_id"] = spans_df["article_id"].fillna(-1).astype(int)
    grouped_spans = spans_df.groupby(group_cols, sort=False)

    # Map for quick lookup: key -> DataFrame
    span_groups = {key: group for key, group in grouped_spans}

    rows = []

    # Stats counters
    total_sections_seen = 0          # non-empty text sections
    sections_with_links = 0
    sections_without_links = 0
    total_paragraphs_all = 0         # all paragraphs (before filter)
    total_paragraphs_kept = 0        # with >= min_spans_per_paragraph
    spans_per_paragraph = []         # number of spans per paragraph (before filter)
    paragraphs_per_article = {}      # article_id -> count
    articles_with_spans = set()
    assigned_spans = 0               # spans that end up inside some paragraph

    # Iterate over sections_parsed_<domain>.jsonl
    with open(sections_path, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Line {line_idx}: JSON decode error, skipping")
                continue

            article_id = rec.get("article_id")
            page_name = rec.get("page_name")
            section = rec.get("section") or "Unknown"
            text = rec.get("text") or ""
            url = rec.get("url")
            source_path = rec.get("source_path")

            # Normalize article_id to int or sentinel
            if article_id is None:
                article_key = -1
            else:
                article_key = int(article_id)

            key = (article_key, page_name, section)

            if not text.strip():
                continue

            total_sections_seen += 1

            # Check if we have spans for this section
            if key not in span_groups:
                sections_without_links += 1
                continue

            sections_with_links += 1

            sec_spans_df = span_groups[key]
            sec_spans = sec_spans_df.to_dict(orient="records")

            paragraphs = split_section_into_paragraphs(text)
            if not paragraphs:
                continue

            for para in paragraphs:
                p_start = para["abs_start"]
                p_end = para["abs_end"]
                para_spans = []

                # count paragraph even if we later drop it
                total_paragraphs_all += 1

                for span in sec_spans:
                    s_start = int(span["link_abs_start"])
                    s_end = int(span["link_abs_end"])
                    anchor = str(span.get("anchor_text", ""))

                    # Span fully inside paragraph (section-level coordinates)
                    if p_start <= s_start and s_end <= p_end:
                        rel_start = s_start - p_start
                        rel_end = s_end - p_start

                        para_spans.append(
                            {
                                "start": rel_start,
                                "end": rel_end,
                                "anchor_text": anchor,
                                "link_type": span.get("link_type", ""),
                                "target_page_name": span.get("target_page_name", ""),
                                "article_id_of_internal_link": span.get(
                                    "article_id_of_internal_link", ""
                                ),
                                "span_id": span.get("span_id", ""),
                                "span_index": span.get("span_index", ""),
                            }
                        )

                spans_per_paragraph.append(len(para_spans))
                assigned_spans += len(para_spans)

                if len(para_spans) < min_spans_per_paragraph:
                    continue

                total_paragraphs_kept += 1
                articles_with_spans.add(article_id)
                paragraphs_per_article[article_id] = paragraphs_per_article.get(article_id, 0) + 1

                rows.append(
                    {
                        "article_id": article_id,
                        "page_name": page_name,
                        "section": section,
                        "paragraph_id": para["paragraph_id"],
                        "paragraph_abs_start": p_start,
                        "paragraph_abs_end": p_end,
                        "paragraph_text": para["text"],
                        "url": url,
                        "source_path": source_path,
                        "spans_json": json.dumps(para_spans, ensure_ascii=False),
                    }
                )

    if not rows:
        logger.warning("No paragraphs with spans found; nothing written.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv_path, index=False)
    logger.info(f"Wrote paragraph master to: {out_csv_path}")

    # ----- Summary stats -----
    logger.info(f"Total sections (non-empty text):           {total_sections_seen}")
    logger.info(f"Sections with link spans:                  {sections_with_links}")
    logger.info(f"Sections without link spans:               {sections_without_links}")
    logger.info(f"Total paragraphs (all, before span filter): {total_paragraphs_all}")
    logger.info(
        f"Total paragraph rows (>= {min_spans_per_paragraph} spans): {total_paragraphs_kept}"
    )

    # Paragraph-level span distribution
    if spans_per_paragraph:
        vals = spans_per_paragraph
        vals_nonzero = [v for v in vals if v > 0]
        logger.info(
            "Spans per paragraph (all): min=%d, max=%d, mean=%.2f",
            min(vals),
            max(vals),
            sum(vals) / len(vals),
        )
        if vals_nonzero:
            logger.info(
                "Spans per paragraph (non-empty): min=%d, max=%d, mean=%.2f",
                min(vals_nonzero),
                max(vals_nonzero),
                sum(vals_nonzero) / len(vals_nonzero),
            )

    # Article-level stats
    if paragraphs_per_article:
        counts = list(paragraphs_per_article.values())
        logger.info(
            "Paragraphs per article (with spans): min=%d, max=%d, mean=%.2f",
            min(counts),
            max(counts),
            sum(counts) / len(counts),
        )
    logger.info(f"Articles with at least one span-paragraph: {len(articles_with_spans)}")

    # Span coverage
    if num_spans > 0:
        unassigned_spans = num_spans - assigned_spans
        coverage = 100.0 * assigned_spans / num_spans
        logger.info(f"Spans assigned to paragraphs: {assigned_spans}")
        logger.info(f"Spans NOT assigned:          {unassigned_spans}")
        logger.info(f"Span coverage:               {coverage:.2f}%")