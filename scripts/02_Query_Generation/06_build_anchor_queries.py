#!/usr/bin/env python3
"""
06_build_anchor_queries.py

Build article-level retrieval queries from span_links_<domain>.csv.

For each internal link:
    query = "Retrieve the correct article for the term '{anchor}' given this context: {span_text}"
    label = target article_id (not the source article!)

Input:
    data/processed/<domain>/spans_<domain>.csv
    data/processed/<domain>/span_links_<domain>.csv

Output:
    data/processed/<domain>/anchor_queries_<domain>.csv
    data/processed/<domain>/anchor_queries_<domain>.json
"""

import sys
import csv
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


# ---------------------- CONFIG LOADING ----------------------

def load_scraping_config():
    """
    Load configs/scraping.yaml relative to project root.

    Needs at least:
      base_url: "https://money-heist.fandom.com"
    """
    try:
        import yaml
    except ImportError:
        print(
            "ERROR: PyYAML is not installed.\n"
            "Install it with: pip install pyyaml"
        )
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


# ---------------------- LOGGING SETUP -----------------------

def create_logger(project_root: Path, script_name: str = "06_build_anchor_queries"):
    log_dir = project_root / "data" / "logs" / "anchor_queries"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{ts}_{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()

    fmt = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger initialized for {script_name}")
    logger.info(f"Log file: {log_path}")
    return logger, log_path


# ---------------------- CORE -------------------------------

def build_anchor_queries(cfg, project_root: Path, logger: logging.Logger):
    base_url = cfg["base_url"].rstrip("/")
    domain = urlparse(base_url).netloc.split(".")[0]  # e.g. "money-heist"

    # UPDATED: processed/<domain>/
    processed_dir = project_root / "data" / "processed" / domain
    processed_dir.mkdir(parents=True, exist_ok=True)

    spans_csv = processed_dir / f"spans_{domain}.csv"
    links_csv = processed_dir / f"span_links_{domain}.csv"

    out_csv = processed_dir / f"anchor_queries_{domain}.csv"
    out_json = processed_dir / f"anchor_queries_{domain}.json"

    logger.info(f"Base URL:  {base_url}")
    logger.info(f"DOMAIN:    {domain}")
    logger.info(f"SPANS_CSV: {spans_csv}")
    logger.info(f"LINKS_CSV: {links_csv}")
    logger.info(f"OUT_CSV:   {out_csv}")
    logger.info(f"OUT_JSON:  {out_json}")

    if not spans_csv.exists():
        logger.error(f"Span CSV not found: {spans_csv}")
        sys.exit(1)
    if not links_csv.exists():
        logger.error(f"Links CSV not found: {links_csv}")
        sys.exit(1)

    df_spans = pd.read_csv(spans_csv)
    df_links = pd.read_csv(links_csv)

    logger.info(f"Loaded spans: {df_spans.shape[0]} rows, columns={list(df_spans.columns)}")
    logger.info(f"Loaded links: {df_links.shape[0]} rows, columns={list(df_links.columns)}")

    if "span_id" not in df_spans.columns or "text" not in df_spans.columns:
        logger.error("Spans CSV must contain 'span_id' and 'text' columns.")
        sys.exit(1)

    if "span_id" not in df_links.columns:
        logger.error("Links CSV must contain 'span_id' column.")
        sys.exit(1)

    required_links_col = "article_id_of_internal_link"
    if required_links_col not in df_links.columns:
        logger.error(
            f"Expected '{required_links_col}' in {links_csv}. "
            "Ensure your span-link builder writes it."
        )
        sys.exit(1)

    df_spans = df_spans.set_index("span_id", drop=False)
    logger.info("Indexed df_spans by 'span_id'.")

    rows = []
    q_id = 1
    skipped_no_anchor = 0
    skipped_missing_span = 0
    skipped_missing_target = 0

    MAX_CONTEXT_CHARS = 500  # keep queries sane + stable

    for _, link_row in df_links.iterrows():
        span_id = link_row["span_id"]
        anchor = str(link_row.get("anchor_text", "")).strip()

        if not anchor:
            skipped_no_anchor += 1
            continue

        if span_id not in df_spans.index:
            skipped_missing_span += 1
            continue

        target_article_id = link_row.get(required_links_col)
        if pd.isna(target_article_id):
            skipped_missing_target += 1
            continue

        try:
            target_article_id = int(target_article_id)
        except Exception:
            skipped_missing_target += 1
            continue

        span_row = df_spans.loc[span_id]
        para = str(span_row.get("text", "")).strip()[:MAX_CONTEXT_CHARS]

        try:
            source_article_id = int(span_row.get("article_id"))
        except Exception:
            source_article_id = None

        query = (
            f"Retrieve the correct article for the term '{anchor}' "
            f"given this context: {para}"
        )

        rows.append(
            {
                "q_id": q_id,
                "query": query,
                "linked_word": anchor,
                "paragraph_text": para,
                "correct_article_id": target_article_id,
                "source_article_id": source_article_id,
            }
        )
        q_id += 1

        if q_id % 1000 == 0:
            logger.info(f"Checkpoint: built {q_id-1} queries so far...")

    logger.info(f"Total queries built: {len(rows)}")
    logger.info(f"Skipped (empty anchor_text): {skipped_no_anchor}")
    logger.info(f"Skipped (missing span_id in spans): {skipped_missing_span}")
    logger.info(f"Skipped (missing/invalid target id): {skipped_missing_target}")

    if not rows:
        logger.warning("No anchor queries generated. Nothing to write.")
        return out_csv, out_json, 0

    logger.info("Writing CSV output...")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Writing JSON output...")
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(rows, jf, ensure_ascii=False, indent=2)

    logger.info(f"[done] wrote {len(rows)} anchor queries → {out_csv}")
    logger.info(f"[json] wrote {len(rows)} objects → {out_json}")

    return out_csv, out_json, len(rows)


# ---------------------- ENTRYPOINT --------------------------

if __name__ == "__main__":
    cfg, project_root = load_scraping_config()
    logger, log_path = create_logger(project_root)

    logger.info("=== Starting 06_build_anchor_queries ===")
    start = time.time()

    try:
        out_csv, out_json, n = build_anchor_queries(cfg, project_root, logger)
    except Exception as e:
        logger.error("Unhandled error in 06_build_anchor_queries: %s", e, exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info(f"Finished 06_build_anchor_queries in {elapsed:.2f} seconds")
    logger.info(f"Anchor queries CSV:  {out_csv}")
    logger.info(f"Anchor queries JSON: {out_json}")
    logger.info(f"Total queries:       {n}")
    logger.info(f"Log file:            {log_path}")

    print(f"[done] wrote {n} anchor queries → {out_csv}")
    print(f"[json] wrote {n} objects → {out_json}")
    print(f"[log]  saved to → {log_path}")