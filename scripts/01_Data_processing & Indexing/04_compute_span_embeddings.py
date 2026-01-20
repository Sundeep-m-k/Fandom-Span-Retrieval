#!/usr/bin/env python3
"""
04_compute_span_embeddings.py

Compute dense embeddings for spans and save them for retrieval.

Input:
    data/processed/spans_<domain>.csv

Output (UPDATED: in data/embeddings/<domain>/):
    spans_<domain>.npy              # (N, D) float32
    spans_<domain>.index_ids.npy    # (N,) span_id strings
    model_info_<domain>.json        # metadata about model + run
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


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


def create_logger(project_root: Path, script_name: str = "04_compute_span_embeddings"):
    log_dir = project_root / "data" / "logs" / "embeddings"
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


def compute_span_embeddings(
    cfg,
    project_root: Path,
    logger: logging.Logger,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
):
    base_url = cfg["base_url"].rstrip("/")
    domain = urlparse(base_url).netloc.split(".")[0]  # e.g. "money-heist"

    processed_dir = project_root / "data" / "processed" / domain
    # UPDATED: store embeddings under data/embeddings/<domain>/
    embeddings_dir = project_root / "data" / "embeddings" / domain
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    spans_path = processed_dir / f"spans_{domain}.csv"
    if not spans_path.exists():
        logger.error(f"Spans file not found: {spans_path}")
        sys.exit(1)

    logger.info(f"Base URL: {base_url}")
    logger.info(f"Domain: {domain}")
    logger.info(f"Spans CSV: {spans_path}")
    logger.info(f"Embeddings dir: {embeddings_dir}")
    logger.info(f"Model: {model_name}")

    # 1) Load spans
    df = pd.read_csv(spans_path)
    if "span_id" not in df.columns or "text" not in df.columns:
        logger.error("Spans CSV must contain 'span_id' and 'text' columns.")
        sys.exit(1)

    span_ids = df["span_id"].astype(str).tolist()
    texts = df["text"].fillna("").astype(str).tolist()
    num_spans = len(span_ids)

    logger.info(f"Loaded {num_spans} spans")

    # 2) Load embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)

    # 3) Encode in batches
    logger.info("Starting encoding...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # good for cosine similarity
    ).astype("float32")

    elapsed = time.time() - start
    logger.info(f"Finished encoding {num_spans} spans in {elapsed:.2f} seconds")

    dim = embeddings.shape[1]
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # 4) Save to disk
    spans_npy_path = embeddings_dir / f"spans_{domain}.npy"
    spans_ids_path = embeddings_dir / f"spans_{domain}.index_ids.npy"
    model_info_path = embeddings_dir / f"model_info_{domain}.json"

    np.save(spans_npy_path, embeddings)
    np.save(spans_ids_path, np.array(span_ids, dtype=object))

    model_info = {
        "domain": domain,
        "model_name": model_name,
        "embedding_dim": int(dim),
        "num_spans": int(num_spans),
        "normalize_embeddings": True,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "spans_file": str(spans_path.relative_to(project_root)),
        "embeddings_file": str(spans_npy_path.relative_to(project_root)),
        "index_ids_file": str(spans_ids_path.relative_to(project_root)),
    }

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved embeddings to: {spans_npy_path}")
    logger.info(f"Saved index IDs to:  {spans_ids_path}")
    logger.info(f"Saved model info to: {model_info_path}")

    return spans_npy_path, spans_ids_path, model_info_path


# ---------------------- ENTRYPOINT --------------------------


if __name__ == "__main__":
    cfg, project_root = load_scraping_config()
    logger, log_path = create_logger(project_root)

    logger.info("=== Starting 04_compute_span_embeddings ===")
    start = time.time()

    try:
        spans_npy_path, spans_ids_path, model_info_path = compute_span_embeddings(
            cfg,
            project_root,
            logger,
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # or all-mpnet-base-v2
            batch_size=64,
        )
    except Exception as e:
        logger.error("Unhandled error in 04_compute_span_embeddings: %s", e, exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info(f"Finished 04_compute_span_embeddings in {elapsed:.2f} seconds")
    logger.info(f"Log file: {log_path}")

    print(f"💾 Embeddings:   {spans_npy_path}")
    print(f"🆔 Index IDs:    {spans_ids_path}")
    print(f"ℹ️ Model info:   {model_info_path}")
    print(f"📝 Log saved to: {log_path}")