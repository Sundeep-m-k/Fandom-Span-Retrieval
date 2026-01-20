#!/usr/bin/env python3
"""
05_build_faiss_index.py

Build a FAISS index over span embeddings for fast retrieval.

Input (from 04_compute_span_embeddings.py):
    data/embeddings/<domain>/spans_<domain>.npy
    data/embeddings/<domain>/spans_<domain>.index_ids.npy
    data/embeddings/<domain>/model_info_<domain>.json

Output (UPDATED: in data/indexes/<domain>/):
    faiss_flat_<domain>.index
    faiss_index_info_<domain>.json

We use:
    - IndexFlatIP (inner product)
    - Embeddings are assumed L2-normalized, so IP ~ cosine similarity
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

try:
    import faiss  # faiss-cpu or faiss-gpu, but env uses faiss-cpu
except ImportError:
    print("ERROR: faiss is not installed. Install faiss-cpu via conda:")
    print("       conda install -c conda-forge faiss-cpu")
    sys.exit(1)


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


def create_logger(project_root: Path, script_name: str = "05_build_faiss_index"):
    log_dir = project_root / "data" / "logs" / "retrieval"
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


def build_faiss_index(cfg, project_root: Path, logger: logging.Logger):
    base_url = cfg["base_url"].rstrip("/")
    domain = urlparse(base_url).netloc.split(".")[0]  # e.g. "money-heist"

    # UPDATED: read embeddings from data/embeddings/<domain>/
    embeddings_dir = project_root / "data" / "embeddings" / domain

    # UPDATED: write indexes under data/indexes/<domain>/
    indexes_dir = project_root / "data" / "indexes" / domain
    indexes_dir.mkdir(parents=True, exist_ok=True)

    spans_npy_path = embeddings_dir / f"spans_{domain}.npy"
    spans_ids_path = embeddings_dir / f"spans_{domain}.index_ids.npy"
    model_info_path = embeddings_dir / f"model_info_{domain}.json"

    if not spans_npy_path.exists():
        logger.error(f"Embeddings file not found: {spans_npy_path}")
        sys.exit(1)
    if not spans_ids_path.exists():
        logger.error(f"Index IDs file not found: {spans_ids_path}")
        sys.exit(1)

    logger.info(f"Base URL: {base_url}")
    logger.info(f"Embeddings file: {spans_npy_path}")
    logger.info(f"Index IDs file:  {spans_ids_path}")
    logger.info(f"Model info file: {model_info_path} (optional)")
    logger.info(f"Indexes dir:     {indexes_dir}")

    # 1) Load embeddings and IDs
    logger.info("Loading embeddings...")
    embeddings = np.load(spans_npy_path).astype("float32")
    span_ids = np.load(spans_ids_path, allow_pickle=True).astype(str)

    if embeddings.shape[0] != span_ids.shape[0]:
        logger.error(
            f"Mismatch: embeddings N={embeddings.shape[0]} vs "
            f"span_ids N={span_ids.shape[0]}"
        )
        sys.exit(1)

    num_spans, dim = embeddings.shape
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # 2) Build FAISS IndexFlatIP (inner product)
    logger.info("Building FAISS IndexFlatIP (inner product)...")
    index = faiss.IndexFlatIP(dim)

    # If embeddings are not normalized, uncomment this:
    # faiss.normalize_L2(embeddings)

    logger.info("Adding embeddings to index...")
    index.add(embeddings)
    logger.info(f"Index ntotal: {index.ntotal}")

    # 3) Save FAISS index + index metadata
    faiss_index_path = indexes_dir / f"faiss_flat_{domain}.index"
    faiss.write_index(index, str(faiss_index_path))

    index_info_path = indexes_dir / f"faiss_index_info_{domain}.json"

    model_info = {}
    if model_info_path.exists():
        try:
            with open(model_info_path, "r", encoding="utf-8") as f:
                model_info = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read model_info file: {e}")

    index_info = {
        "domain": domain,
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "num_spans": int(num_spans),
        "dim": int(dim),
        "embeddings_file": str(spans_npy_path.relative_to(project_root)),
        "index_ids_file": str(spans_ids_path.relative_to(project_root)),
        "faiss_index_file": str(faiss_index_path.relative_to(project_root)),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_info": model_info,
    }

    with open(index_info_path, "w", encoding="utf-8") as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved FAISS index to: {faiss_index_path}")
    logger.info(f"Saved FAISS index info to: {index_info_path}")

    return faiss_index_path, index_info_path


# ---------------------- ENTRYPOINT --------------------------


if __name__ == "__main__":
    cfg, project_root = load_scraping_config()
    logger, log_path = create_logger(project_root)

    logger.info("=== Starting 05_build_faiss_index ===")
    start = time.time()

    try:
        faiss_index_path, index_info_path = build_faiss_index(cfg, project_root, logger)
    except Exception as e:
        logger.error("Unhandled error in 05_build_faiss_index: %s", e, exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info(f"Finished 05_build_faiss_index in {elapsed:.2f} seconds")
    logger.info(f"FAISS index file:      {faiss_index_path}")
    logger.info(f"FAISS index info file: {index_info_path}")
    logger.info(f"Log file:              {log_path}")

    print(f"📦 FAISS index:      {faiss_index_path}")
    print(f"ℹ️ Index metadata:   {index_info_path}")
    print(f"📝 Log saved to:     {log_path}")