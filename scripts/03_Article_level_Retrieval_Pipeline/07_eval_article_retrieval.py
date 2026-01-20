#!/usr/bin/env python3
"""
07_eval_article_retrieval.py

Evaluate *article-level* retrieval using anchor-based queries.

Assumes:
    - data/processed/<domain>/spans_<domain>.csv
        contains at least: span_id, article_id, title, text
    - data/embeddings/<domain>/spans_<domain>.npy
        span embeddings aligned with spans_<domain>.index_ids.npy
    - data/embeddings/<domain>/spans_<domain>.index_ids.npy
        span_ids in the same order as span embeddings
    - data/embeddings/<domain>/model_info_<domain>.json
        {"model_name": "..."} for SentenceTransformer
    - data/processed/<domain>/anchor_queries_<domain>.csv
        columns: q_id, query, linked_word, paragraph_text, correct_article_id

Outputs (domain-scoped):
    - metrics CSV:  data/logs/article_retrieval/<domain>/article_retrieval_metrics_<domain>.csv
    - per-query top-k CSV: data/logs/article_retrieval/<domain>/article_retrieval_topk_<domain>.csv
    - log:          data/logs/article_retrieval/<domain>/article_retrieval_<domain>.log
"""

import csv
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import faiss
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Paths / Config (domain-independent)
# ---------------------------------------------------------------------

# Explicit project root
PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")

CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping.yaml"

# UPDATED: these are now base dirs; domain subdir will be appended in main()
PROCESSED_BASE_DIR = PROJECT_ROOT / "data" / "processed"
EMB_BASE_DIR = PROJECT_ROOT / "data" / "embeddings"

# Base log dir (domain-specific dir created in main)
LOG_BASE_DIR = PROJECT_ROOT / "data" / "logs" / "article_retrieval"
LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 50  # for MRR@TOP_K and FAISS search

# Will be filled in main() after we know DOMAIN
MODEL_INFO = None
SPAN_CSV = None
SPAN_EMB_PATH = None
SPAN_IDS_PATH = None
ANCHOR_QUERIES_CSV = None
METRICS_CSV = None
TOPK_RESULTS_CSV = None
LOG_FILE = None


# ---------------------------------------------------------------------
# Helpers: domain + logging
# ---------------------------------------------------------------------

def load_domain_from_config() -> str:
    """
    Load DOMAIN from configs/scraping.yaml.

    Priority:
      1. cfg['domain'] or cfg['domain_slug'] if present
      2. Infer from cfg['base_url'] host: e.g., "https://money-heist.fandom.com"
         -> "money-heist"
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    domain = cfg.get("domain") or cfg.get("domain_slug")

    if not domain:
        base_url = cfg.get("base_url", "").rstrip("/")
        if not base_url:
            raise ValueError(
                "Could not determine DOMAIN. "
                "Please set 'domain' (or 'domain_slug') or 'base_url' in configs/scraping.yaml"
            )
        parsed = urlparse(base_url)
        host = parsed.netloc  # e.g. "money-heist.fandom.com"
        domain = host.split(".")[0]  # "money-heist"

    return domain


def setup_logging(log_file: Path) -> None:
    """
    Initialize logging to both file and stdout.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    root.addHandler(fh)
    root.addHandler(sh)

    logging.info("Logger initialized.")
    logging.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    logging.info(f"CONFIG_PATH  = {CONFIG_PATH}")
    logging.info(f"LOG_FILE     = {log_file}")


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_model_name() -> str:
    """
    Load the SentenceTransformer model name used for span embeddings.
    Falls back to MiniLM if metadata is missing.
    """
    global MODEL_INFO
    default_model = "sentence-transformers/all-MiniLM-L6-v2"

    if MODEL_INFO is None:
        logging.warning("MODEL_INFO path is not set. Falling back to default model.")
        return default_model

    if not MODEL_INFO.exists():
        logging.warning(f"{MODEL_INFO} missing. Falling back to {default_model}")
        return default_model

    with open(MODEL_INFO, "r", encoding="utf-8") as f:
        info = json.load(f)

    model_name = info.get("model_name", default_model)
    logging.info(f"Using model for query embeddings: {model_name}")
    return model_name


def load_spans_and_embeddings():
    """
    Load spans_<domain>.csv and span embeddings.

    Returns:
        df_spans: DataFrame indexed by span_id
        span_ids: np.ndarray of span_id strings aligned with embeddings
        span_embs: np.ndarray of shape (N_spans, D)
    """
    global SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH

    logging.info(f"Loading spans from {SPAN_CSV} ...")
    df_spans = pd.read_csv(SPAN_CSV)

    if "span_id" not in df_spans.columns:
        raise ValueError(f"{SPAN_CSV} must contain a 'span_id' column.")
    if "article_id" not in df_spans.columns:
        raise ValueError(f"{SPAN_CSV} must contain an 'article_id' column.")

    df_spans = df_spans.set_index("span_id", drop=False)

    logging.info(f"Loading span embeddings from {SPAN_EMB_PATH} ...")
    span_embs = np.load(SPAN_EMB_PATH).astype("float32")
    span_ids = np.load(SPAN_IDS_PATH, allow_pickle=True).astype(str)

    if span_embs.shape[0] != len(span_ids):
        logging.warning(
            f"Span embeddings count ({span_embs.shape[0]}) != span_ids length ({len(span_ids)}). "
            "Assuming order still matches FAISS index."
        )

    logging.info(f"Loaded {len(df_spans)} spans; embeddings shape = {span_embs.shape}")
    return df_spans, span_ids, span_embs


def build_article_embeddings(df_spans, span_ids, span_embs):
    """
    Aggregate span embeddings into article embeddings by mean pooling.
    """
    logging.info("Building article embeddings via mean over span embeddings...")

    spanid_to_embidx = {sid: i for i, sid in enumerate(span_ids)}

    article_to_embs = {}
    article_meta = {}

    for sid in df_spans["span_id"]:
        if sid not in spanid_to_embidx:
            continue

        row = df_spans.loc[sid]
        article_id = int(row["article_id"])
        emb_idx = spanid_to_embidx[sid]
        vec = span_embs[emb_idx]

        article_to_embs.setdefault(article_id, []).append(vec)

        if article_id not in article_meta:
            article_meta[article_id] = {
                "title": str(row.get("title", "")),
                "page_name": str(row.get("page_name", "")),
            }

    article_ids_sorted = np.array(sorted(article_to_embs.keys()), dtype=int)
    article_emb_list = []
    for aid in article_ids_sorted:
        vecs = np.stack(article_to_embs[aid], axis=0)
        article_emb_list.append(vecs.mean(axis=0))

    article_embs = np.stack(article_emb_list, axis=0).astype("float32")
    logging.info(
        f"Built {len(article_ids_sorted)} article embeddings; dim = {article_embs.shape[1]}"
    )

    return article_ids_sorted, article_embs, article_meta


def build_faiss_article_index(article_embs):
    logging.info("Normalizing article embeddings and building FAISS index...")
    faiss.normalize_L2(article_embs)
    index = faiss.IndexFlatIP(article_embs.shape[1])
    index.add(article_embs)
    logging.info(f"FAISS article index: ntotal = {index.ntotal}, dim = {index.d}")
    return index


def load_anchor_queries():
    global ANCHOR_QUERIES_CSV

    logging.info(f"Loading anchor queries from {ANCHOR_QUERIES_CSV} ...")
    df_q = pd.read_csv(ANCHOR_QUERIES_CSV)

    required_cols = {"query", "correct_article_id"}
    missing = required_cols - set(df_q.columns)
    if missing:
        raise ValueError(f"{ANCHOR_QUERIES_CSV} missing columns: {missing}")

    queries = df_q["query"].astype(str).tolist()
    correct_article_ids = [int(x) for x in df_q["correct_article_id"].tolist()]

    linked_words = (
        df_q["linked_word"].astype(str).tolist()
        if "linked_word" in df_q.columns
        else [""] * len(queries)
    )
    para_texts = (
        df_q["paragraph_text"].astype(str).tolist()
        if "paragraph_text" in df_q.columns
        else [""] * len(queries)
    )
    q_ids = df_q["q_id"].tolist() if "q_id" in df_q.columns else list(range(1, len(queries) + 1))

    logging.info(f"Loaded {len(queries)} anchor queries.")
    return q_ids, queries, correct_article_ids, linked_words, para_texts


def encode_queries(model_name: str, queries):
    logging.info(f"Encoding {len(queries)} queries with {model_name} ...")
    model = SentenceTransformer(model_name)
    q_emb = model.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    logging.info(f"Query embeddings shape: {q_emb.shape}")
    return q_emb


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_article_retrieval(
    index,
    article_ids_sorted,
    article_embs,
    article_meta,
    q_ids,
    queries,
    correct_article_ids,
    linked_words,
    para_texts,
):
    global METRICS_CSV, TOPK_RESULTS_CSV

    logging.info(f"Running FAISS search over articles with TOP_K={TOP_K} ...")

    model_name = load_model_name()
    query_embs = encode_queries(model_name, queries)

    logging.info("Performing FAISS search for all queries...")
    distances, indices = index.search(query_embs, TOP_K)

    recall_at = {1: [], 3: [], 5: [], 10: [], 50: [], 100: [], 1000: []}
    reciprocal_ranks = []

    logging.info(f"Writing per-query top-k results to {TOPK_RESULTS_CSV}")
    with open(TOPK_RESULTS_CSV, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            [
                "q_id",
                "query",
                "linked_word",
                "paragraph_text",
                "correct_article_id",
                "correct_title",
                "found_rank",
                "retrieved_article_ids",
                "retrieved_scores",
            ]
        )

        for i, (retrieved_idxs, scores, gold_aid) in enumerate(
            zip(indices, distances, correct_article_ids)
        ):
            retrieved_article_ids = [int(article_ids_sorted[j]) for j in retrieved_idxs]
            scores_list = [float(s) for s in scores]

            for k in recall_at.keys():
                topk = retrieved_article_ids[:k]
                recall_at[k].append(1 if gold_aid in topk else 0)

            found_rank = 0
            for rank, aid in enumerate(retrieved_article_ids, start=1):
                if aid == gold_aid:
                    found_rank = rank
                    break

            reciprocal_ranks.append(1.0 / found_rank if found_rank > 0 else 0.0)

            gold_meta = article_meta.get(gold_aid, {})
            gold_title = gold_meta.get("title", "N/A")

            writer.writerow(
                [
                    q_ids[i],
                    queries[i],
                    linked_words[i],
                    para_texts[i],
                    gold_aid,
                    gold_title,
                    found_rank,
                    "|".join(map(str, retrieved_article_ids)),
                    "|".join(f"{s:.6f}" for s in scores_list),
                ]
            )

            if (i + 1) % 10000 == 0:
                logging.info(f"Processed {i + 1} / {len(queries)} queries...")

    logging.info("\n--- Article-level Retrieval Metrics ---")
    metrics = {}

    for k, vals in recall_at.items():
        avg = float(np.mean(vals)) if vals else 0.0
        metrics[f"Recall@{k}"] = avg
        logging.info(f"Recall@{k:<3}: {avg:.4f}")

    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics["MRR@TOP_K"] = mrr
    logging.info(f"MRR@TOP_K (K={TOP_K}): {mrr:.4f}")

    logging.info(f"Saving metrics to {METRICS_CSV}")
    with open(METRICS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "faiss"])
        for name, val in metrics.items():
            writer.writerow([name, f"{val:.6f}"])

    logging.info(f"Saved per-query top-k results to {TOPK_RESULTS_CSV}")
    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global MODEL_INFO, SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH
    global ANCHOR_QUERIES_CSV, METRICS_CSV, TOPK_RESULTS_CSV, LOG_FILE

    DOMAIN = load_domain_from_config()

    DOMAIN_LOG_DIR = LOG_BASE_DIR / DOMAIN
    DOMAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # UPDATED: domain-scoped dirs
    PROCESSED_DIR = PROCESSED_BASE_DIR / DOMAIN
    EMB_DIR = EMB_BASE_DIR / DOMAIN

    SPAN_CSV = PROCESSED_DIR / f"spans_{DOMAIN}.csv"
    SPAN_EMB_PATH = EMB_DIR / f"spans_{DOMAIN}.npy"
    SPAN_IDS_PATH = EMB_DIR / f"spans_{DOMAIN}.index_ids.npy"
    MODEL_INFO = EMB_DIR / f"model_info_{DOMAIN}.json"
    ANCHOR_QUERIES_CSV = PROCESSED_DIR / f"anchor_queries_{DOMAIN}.csv"

    METRICS_CSV = DOMAIN_LOG_DIR / f"article_retrieval_metrics_{DOMAIN}.csv"
    TOPK_RESULTS_CSV = DOMAIN_LOG_DIR / f"article_retrieval_topk_{DOMAIN}.csv"
    LOG_FILE = DOMAIN_LOG_DIR / f"article_retrieval_{DOMAIN}.log"

    setup_logging(LOG_FILE)

    logging.info(f"DOMAIN                = {DOMAIN}")
    logging.info(f"SPAN_CSV              = {SPAN_CSV}")
    logging.info(f"SPAN_EMB_PATH         = {SPAN_EMB_PATH}")
    logging.info(f"SPAN_IDS_PATH         = {SPAN_IDS_PATH}")
    logging.info(f"MODEL_INFO            = {MODEL_INFO}")
    logging.info(f"ANCHOR_QUERIES_CSV    = {ANCHOR_QUERIES_CSV}")
    logging.info(f"METRICS_CSV           = {METRICS_CSV}")
    logging.info(f"TOPK_RESULTS_CSV      = {TOPK_RESULTS_CSV}")

    for p in [SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH, ANCHOR_QUERIES_CSV]:
        if not p.exists():
            logging.error(f"Required file missing: {p}")
            raise FileNotFoundError(f"Required file missing: {p}")

    df_spans, span_ids, span_embs = load_spans_and_embeddings()
    article_ids_sorted, article_embs, article_meta = build_article_embeddings(
        df_spans, span_ids, span_embs
    )
    index = build_faiss_article_index(article_embs)
    q_ids, queries, correct_article_ids, linked_words, para_texts = load_anchor_queries()

    logging.info("Starting article-level retrieval evaluation ...")
    metrics = evaluate_article_retrieval(
        index,
        article_ids_sorted,
        article_embs,
        article_meta,
        q_ids,
        queries,
        correct_article_ids,
        linked_words,
        para_texts,
    )

    logging.info("\n✅ Article-level retrieval evaluation finished.")
    logging.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()