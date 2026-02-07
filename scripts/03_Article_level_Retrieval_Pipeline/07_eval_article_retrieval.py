#!/usr/bin/env python3
"""
07_eval_article_retrieval.py

Evaluate article-level retrieval using anchor-based queries.

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
    - per-query top-k CSVs (one per variant)
    - log:          data/logs/article_retrieval/<domain>/article_retrieval_<domain>.log

Variants evaluated:
    - mean_chunks: mean of all chunk embeddings per article (current)
    - mean_paragraphs: mean of all paragraph embeddings per article (raw text)
    - first_paragraph: first paragraph embedding per article (raw text)
"""

import csv
import json
import logging
import re
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

PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping.yaml"

PROCESSED_BASE_DIR = PROJECT_ROOT / "data" / "processed"
EMB_BASE_DIR = PROJECT_ROOT / "data" / "embeddings"

LOG_BASE_DIR = PROJECT_ROOT / "data" / "logs" / "article_retrieval"
LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 50

MODEL_INFO = None
SPAN_CSV = None
SPAN_EMB_PATH = None
SPAN_IDS_PATH = None
ANCHOR_QUERIES_CSV = None
METRICS_CSV = None
LOG_FILE = None


# ---------------------------------------------------------------------
# Helpers: domain + logging
# ---------------------------------------------------------------------

def load_domain_from_config() -> str:
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
        domain = parsed.netloc.split(".")[0]

    return domain


def setup_logging(log_file: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
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


def load_raw_article_paragraphs(domain: str):
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fandom_html" / domain
    if not raw_dir.exists():
        logging.warning(f"Raw text dir not found: {raw_dir}")
        return {}

    paragraphs_by_article = {}
    txt_files = sorted(raw_dir.glob("*.txt"))

    for path in txt_files:
        stem = path.stem
        if not stem.isdigit():
            continue
        article_id = int(stem)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logging.warning(f"Failed to read {path}: {e}")
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if paragraphs:
            paragraphs_by_article[article_id] = paragraphs

    logging.info(
        "Loaded raw paragraphs: articles=%d, files=%d",
        len(paragraphs_by_article),
        len(txt_files),
    )
    return paragraphs_by_article


def encode_texts(model, texts, label: str):
    logging.info(f"Encoding {len(texts)} {label} ...")
    embs = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    logging.info(f"{label} embeddings shape: {embs.shape}")
    return embs


def build_article_embeddings_from_paragraphs(paragraphs_by_article, model, mode: str):
    if not paragraphs_by_article:
        return np.array([], dtype=int), np.zeros((0, 1), dtype="float32")

    if mode == "mean":
        texts = []
        article_ids = []
        for aid, paras in paragraphs_by_article.items():
            for p in paras:
                texts.append(p)
                article_ids.append(aid)

        if not texts:
            return np.array([], dtype=int), np.zeros((0, 1), dtype="float32")

        embs = encode_texts(model, texts, "paragraphs (mean)")
        sum_vecs = {}
        counts = {}
        for emb, aid in zip(embs, article_ids):
            if aid not in sum_vecs:
                sum_vecs[aid] = emb.copy()
                counts[aid] = 1
            else:
                sum_vecs[aid] += emb
                counts[aid] += 1

        article_ids_sorted = np.array(sorted(sum_vecs.keys()), dtype=int)
        article_embs = np.stack(
            [sum_vecs[aid] / counts[aid] for aid in article_ids_sorted],
            axis=0,
        ).astype("float32")
        return article_ids_sorted, article_embs

    if mode == "first":
        pairs = []
        for aid, paras in paragraphs_by_article.items():
            if paras:
                pairs.append((aid, paras[0]))
        if not pairs:
            return np.array([], dtype=int), np.zeros((0, 1), dtype="float32")

        pairs.sort(key=lambda x: x[0])
        article_ids_sorted = np.array([p[0] for p in pairs], dtype=int)
        texts = [p[1] for p in pairs]
        embs = encode_texts(model, texts, "first paragraphs")
        return article_ids_sorted, embs

    raise ValueError(f"Unknown paragraph embedding mode: {mode}")


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


def encode_queries(model, queries):
    return encode_texts(model, queries, "queries")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_article_retrieval(
    index,
    article_ids_sorted,
    article_meta,
    q_ids,
    queries,
    correct_article_ids,
    linked_words,
    para_texts,
    query_embs,
    topk_results_csv: Path,
    variant_name: str,
):
    logging.info(f"[{variant_name}] Running FAISS search with TOP_K={TOP_K} ...")

    logging.info("Performing FAISS search for all queries...")
    distances, indices = index.search(query_embs, TOP_K)

    recall_at = {1: [], 3: [], 5: [], 10: [], 50: [], 100: [], 1000: []}
    reciprocal_ranks = []

    logging.info(f"[{variant_name}] Writing per-query top-k results to {topk_results_csv}")
    with open(topk_results_csv, "w", encoding="utf-8", newline="") as f_out:
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

    logging.info(f"\n--- Article-level Retrieval Metrics ({variant_name}) ---")
    metrics = {}

    for k, vals in recall_at.items():
        avg = float(np.mean(vals)) if vals else 0.0
        metrics[f"Recall@{k}"] = avg
        logging.info(f"Recall@{k:<3}: {avg:.4f}")

    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics["MRR@TOP_K"] = mrr
    logging.info(f"MRR@TOP_K (K={TOP_K}): {mrr:.4f}")

    logging.info(f"[{variant_name}] Saved per-query top-k results to {topk_results_csv}")
    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    global MODEL_INFO, SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH
    global ANCHOR_QUERIES_CSV, METRICS_CSV, LOG_FILE

    DOMAIN = load_domain_from_config()

    DOMAIN_LOG_DIR = LOG_BASE_DIR / DOMAIN
    DOMAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    PROCESSED_DIR = PROCESSED_BASE_DIR / DOMAIN
    EMB_DIR = EMB_BASE_DIR / DOMAIN

    SPAN_CSV = PROCESSED_DIR / f"spans_{DOMAIN}.csv"
    SPAN_EMB_PATH = EMB_DIR / f"spans_{DOMAIN}.npy"
    SPAN_IDS_PATH = EMB_DIR / f"spans_{DOMAIN}.index_ids.npy"
    MODEL_INFO = EMB_DIR / f"model_info_{DOMAIN}.json"
    ANCHOR_QUERIES_CSV = PROCESSED_DIR / f"anchor_queries_{DOMAIN}.csv"

    METRICS_CSV = DOMAIN_LOG_DIR / f"article_retrieval_metrics_{DOMAIN}.csv"
    LOG_FILE = DOMAIN_LOG_DIR / f"article_retrieval_{DOMAIN}.log"

    setup_logging(LOG_FILE)

    logging.info(f"DOMAIN                = {DOMAIN}")
    logging.info(f"SPAN_CSV              = {SPAN_CSV}")
    logging.info(f"SPAN_EMB_PATH         = {SPAN_EMB_PATH}")
    logging.info(f"SPAN_IDS_PATH         = {SPAN_IDS_PATH}")
    logging.info(f"MODEL_INFO            = {MODEL_INFO}")
    logging.info(f"ANCHOR_QUERIES_CSV    = {ANCHOR_QUERIES_CSV}")
    logging.info(f"METRICS_CSV           = {METRICS_CSV}")

    for p in [SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH, ANCHOR_QUERIES_CSV]:
        if not p.exists():
            logging.error(f"Required file missing: {p}")
            raise FileNotFoundError(f"Required file missing: {p}")

    df_spans, span_ids, span_embs = load_spans_and_embeddings()
    article_ids_sorted, article_embs, article_meta = build_article_embeddings(
        df_spans, span_ids, span_embs
    )
    q_ids, queries, correct_article_ids, linked_words, para_texts = load_anchor_queries()

    model_name = load_model_name()
    bi_model = SentenceTransformer(model_name)
    query_embs = encode_queries(bi_model, queries)

    metrics_by_variant = {}

    # Variant 1: mean of all chunks
    topk_mean_chunks = DOMAIN_LOG_DIR / f"article_retrieval_topk_{DOMAIN}_mean_chunks.csv"
    index_chunks = build_faiss_article_index(article_embs)
    metrics_by_variant["mean_chunks"] = evaluate_article_retrieval(
        index_chunks,
        article_ids_sorted,
        article_meta,
        q_ids,
        queries,
        correct_article_ids,
        linked_words,
        para_texts,
        query_embs,
        topk_mean_chunks,
        "mean_chunks",
    )

    # Variants 2/3: paragraphs from raw text
    paragraphs_by_article = load_raw_article_paragraphs(DOMAIN)

    if paragraphs_by_article:
        # Variant 2: mean of all paragraphs per article
        art_ids_para, art_embs_para = build_article_embeddings_from_paragraphs(
            paragraphs_by_article, bi_model, mode="mean"
        )
        if len(art_ids_para) > 0:
            topk_mean_paras = DOMAIN_LOG_DIR / f"article_retrieval_topk_{DOMAIN}_mean_paragraphs.csv"
            index_paras = build_faiss_article_index(art_embs_para)
            metrics_by_variant["mean_paragraphs"] = evaluate_article_retrieval(
                index_paras,
                art_ids_para,
                article_meta,
                q_ids,
                queries,
                correct_article_ids,
                linked_words,
                para_texts,
                query_embs,
                topk_mean_paras,
                "mean_paragraphs",
            )

        # Variant 3: first paragraph per article
        art_ids_first, art_embs_first = build_article_embeddings_from_paragraphs(
            paragraphs_by_article, bi_model, mode="first"
        )
        if len(art_ids_first) > 0:
            topk_first_para = DOMAIN_LOG_DIR / f"article_retrieval_topk_{DOMAIN}_first_paragraph.csv"
            index_first = build_faiss_article_index(art_embs_first)
            metrics_by_variant["first_paragraph"] = evaluate_article_retrieval(
                index_first,
                art_ids_first,
                article_meta,
                q_ids,
                queries,
                correct_article_ids,
                linked_words,
                para_texts,
                query_embs,
                topk_first_para,
                "first_paragraph",
            )
    else:
        logging.warning("No raw paragraphs found; skipping paragraph-based variants.")

    # Write combined metrics table
    metrics_order = [
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "Recall@50",
        "Recall@100",
        "Recall@1000",
        "MRR@TOP_K",
    ]
    variant_order = ["mean_chunks", "mean_paragraphs", "first_paragraph"]

    logging.info(f"Saving metrics to {METRICS_CSV}")
    with open(METRICS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + variant_order)
        for name in metrics_order:
            row = [name]
            for v in variant_order:
                val = metrics_by_variant.get(v, {}).get(name)
                row.append(f"{val:.6f}" if val is not None else "")
            writer.writerow(row)

    logging.info("Article-level retrieval evaluation finished.")
    logging.info(f"Final metrics: {metrics_by_variant}")


if __name__ == "__main__":
    main()