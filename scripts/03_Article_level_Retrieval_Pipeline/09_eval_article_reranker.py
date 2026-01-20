#!/usr/bin/env python3
"""
09_eval_article_reranker.py

Compare FAISS-only article-level retrieval vs
article-level cross-encoder reranker.

UPDATED PATH CONVENTION:
    - data/processed/<domain>/spans_<domain>.csv
    - data/processed/<domain>/anchor_queries_<domain>.csv
    - data/embeddings/<domain>/spans_<domain>.npy
    - data/embeddings/<domain>/spans_<domain>.index_ids.npy
    - data/embeddings/<domain>/model_info_<domain>.json
    - models/article_reranker/<domain>/best

Outputs (domain-scoped):
    - data/logs/article_reranker/<domain>/article_reranker_vs_faiss_metrics_<domain>.csv
    - data/logs/article_reranker/<domain>/article_reranker_topk_<domain>.csv
    - logs in data/logs/article_reranker/<domain>/eval_article_reranker_<domain>.log
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
import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------------
# Paths / Config (domain-independent)
# -------------------------------------------------------

PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping.yaml"

# UPDATED: base dirs; we will append /<domain> in main()
PROCESSED_BASE_DIR = PROJECT_ROOT / "data" / "processed"
EMB_BASE_DIR = PROJECT_ROOT / "data" / "embeddings"

LOG_BASE_DIR = PROJECT_ROOT / "data" / "logs" / "article_reranker"
MODEL_BASE_DIR = PROJECT_ROOT / "models" / "article_reranker"

LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 50

# These will be set in main() after we know DOMAIN
DOMAIN = None
LOG_DIR = None
MODEL_DIR = None
SPAN_CSV = None
SPAN_EMB_PATH = None
SPAN_IDS_PATH = None
MODEL_INFO_PATH = None
ANCHOR_QUERIES_CSV = None
METRICS_CSV = None
TOPK_CSV = None
LOG_FILE = None


# -------------------------------------------------------
# Domain + logging helpers
# -------------------------------------------------------

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
        host = parsed.netloc
        domain = host.split(".")[0]

    return domain


def setup_logging(log_file: Path) -> None:
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


# -------------------------------------------------------
# Loading helpers
# -------------------------------------------------------

def load_bi_encoder_name() -> str:
    global MODEL_INFO_PATH
    default_model = "sentence-transformers/all-MiniLM-L6-v2"

    if MODEL_INFO_PATH is None or not MODEL_INFO_PATH.exists():
        logging.warning(
            f"MODEL_INFO_PATH missing ({MODEL_INFO_PATH}). "
            f"Falling back to default bi-encoder: {default_model}"
        )
        return default_model

    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    model_name = info.get("model_name", default_model)
    logging.info(f"Bi-encoder model from MODEL_INFO: {model_name}")
    return model_name


def load_spans_and_embeddings():
    global SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH

    logging.info(f"Loading spans from {SPAN_CSV} ...")
    df_spans = pd.read_csv(SPAN_CSV)
    if "span_id" not in df_spans.columns or "article_id" not in df_spans.columns:
        raise ValueError("spans CSV must contain 'span_id' and 'article_id'.")

    df_spans["article_id"] = df_spans["article_id"].astype(int)
    df_spans = df_spans.set_index("span_id", drop=False)

    logging.info(f"Loading span embeddings from {SPAN_EMB_PATH} ...")
    span_embs = np.load(SPAN_EMB_PATH).astype("float32")
    span_ids = np.load(SPAN_IDS_PATH, allow_pickle=True).astype(str)

    if len(span_ids) != span_embs.shape[0]:
        logging.warning(
            f"span_ids length ({len(span_ids)}) != span_embs rows ({span_embs.shape[0]}). "
            "Assuming order still matches."
        )

    logging.info(f"Loaded {len(df_spans)} spans, emb shape = {span_embs.shape}")
    return df_spans, span_ids, span_embs


def build_article_embeddings(df_spans, span_ids, span_embs):
    logging.info("Building article embeddings by mean over span embeddings...")
    spanid_to_embidx = {sid: i for i, sid in enumerate(span_ids)}

    article_to_vecs = {}
    article_meta = {}

    for sid in df_spans["span_id"]:
        if sid not in spanid_to_embidx:
            continue
        row = df_spans.loc[sid]
        aid = int(row["article_id"])
        vec = span_embs[spanid_to_embidx[sid]]

        article_to_vecs.setdefault(aid, []).append(vec)
        if aid not in article_meta:
            article_meta[aid] = {
                "title": str(row.get("title", "")),
                "page_name": str(row.get("page_name", "")),
            }

    article_ids_sorted = np.array(sorted(article_to_vecs.keys()), dtype=int)
    article_embs = []
    for aid in article_ids_sorted:
        vecs = np.stack(article_to_vecs[aid], axis=0)
        article_embs.append(vecs.mean(axis=0))

    article_embs = np.stack(article_embs, axis=0).astype("float32")
    logging.info(
        f"Built {len(article_ids_sorted)} article embeddings, "
        f"dim = {article_embs.shape[1]}"
    )
    return article_ids_sorted, article_embs, article_meta


def build_faiss_article_index(article_embs):
    logging.info("Building FAISS article index...")
    faiss.normalize_L2(article_embs)
    index = faiss.IndexFlatIP(article_embs.shape[1])
    index.add(article_embs)
    logging.info(f"FAISS ntotal = {index.ntotal}, dim = {index.d}")
    return index


def load_anchor_queries():
    global ANCHOR_QUERIES_CSV

    logging.info(f"Loading anchor queries from {ANCHOR_QUERIES_CSV} ...")
    df_q = pd.read_csv(ANCHOR_QUERIES_CSV)

    for col in ["query", "correct_article_id"]:
        if col not in df_q.columns:
            raise ValueError(f"{ANCHOR_QUERIES_CSV} must contain '{col}'")

    df_q["correct_article_id"] = df_q["correct_article_id"].astype(int)
    if "q_id" not in df_q.columns:
        df_q["q_id"] = np.arange(1, len(df_q) + 1)

    queries = df_q["query"].astype(str).tolist()
    gold_ids = df_q["correct_article_id"].tolist()
    q_ids = df_q["q_id"].tolist()

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

    logging.info(f"Loaded {len(queries)} queries.")
    return q_ids, queries, gold_ids, linked_words, para_texts


def build_article_texts(df_spans):
    logging.info("Building article_texts from spans...")
    df = df_spans.reset_index(drop=True)
    df["article_id"] = df["article_id"].astype(int)
    df = df.sort_values(["article_id", "span_id"])

    article_texts = {}
    for aid, group in df.groupby("article_id"):
        spans_text = [str(t).strip() for t in group["text"].tolist() if str(t).strip()]
        article_text = "\n".join(spans_text)
        article_texts[aid] = article_text

    logging.info(f"Built article_texts for {len(article_texts)} articles.")
    return article_texts


# -------------------------------------------------------
# Reranker scoring
# -------------------------------------------------------

def load_reranker():
    global MODEL_DIR

    if MODEL_DIR is None or not MODEL_DIR.exists():
        raise FileNotFoundError(f"Reranker model dir not found: {MODEL_DIR}")

    logging.info(f"Loading article reranker from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logging.info(f"Reranker loaded on device: {device}")
    return tokenizer, model, device


def rerank_candidates_for_query(
    query,
    candidate_article_ids,
    article_texts,
    tokenizer,
    model,
    device,
    batch_size=8,
    max_length=512,
):
    pairs = []
    for aid in candidate_article_ids:
        text = article_texts.get(aid, "")
        pairs.append((query, text))

    all_scores = []

    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_queries = [p[0] for p in batch_pairs]
            batch_docs = [p[1] for p in batch_pairs]

            enc = tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**enc)
            logits = outputs.logits

            if logits.shape[1] == 1:
                batch_scores = logits.squeeze(-1).detach().cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1)
                batch_scores = probs[:, 1].detach().cpu().numpy()

            all_scores.extend(batch_scores.tolist())

    return all_scores


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def evaluate_faiss_and_reranker():
    global METRICS_CSV, TOPK_CSV

    df_spans, span_ids, span_embs = load_spans_and_embeddings()
    article_ids_sorted, article_embs, article_meta = build_article_embeddings(
        df_spans, span_ids, span_embs
    )
    index = build_faiss_article_index(article_embs)
    article_texts = build_article_texts(df_spans)

    q_ids, queries, gold_ids, linked_words, para_texts = load_anchor_queries()

    bi_model_name = load_bi_encoder_name()
    logging.info(f"Loading bi-encoder for queries: {bi_model_name}")
    bi_model = SentenceTransformer(bi_model_name)
    tokenizer_rerank, model_rerank, device = load_reranker()

    recall_faiss = {1: [], 3: [], 5: [], 10: [], 50: []}
    recall_rerank = {1: [], 3: [], 5: [], 10: [], 50: []}
    mrr_faiss = []
    mrr_rerank = []

    logging.info(f"Writing per-query results to {TOPK_CSV}")
    with open(TOPK_CSV, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            [
                "q_id",
                "query",
                "linked_word",
                "paragraph_text",
                "gold_article_id",
                "faiss_rank",
                "rerank_rank",
                "faiss_article_ids",
                "rerank_article_ids",
            ]
        )

        logging.info("Encoding queries with bi-encoder...")
        q_embs = bi_model.encode(
            queries,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        logging.info(f"Running FAISS + reranker for {len(queries)} queries...")
        for i, (q_vec, gold_aid) in enumerate(zip(q_embs, gold_ids)):
            q_vec = q_vec.reshape(1, -1)

            scores, idxs = index.search(q_vec, TOP_K)
            idxs = idxs[0]
            faiss_article_ids = [int(article_ids_sorted[j]) for j in idxs]

            faiss_rank = 0
            for rank, aid in enumerate(faiss_article_ids, start=1):
                if aid == gold_aid:
                    faiss_rank = rank
                    break
            mrr_faiss.append(1.0 / faiss_rank if faiss_rank > 0 else 0.0)
            for k in recall_faiss.keys():
                recall_faiss[k].append(1 if gold_aid in faiss_article_ids[:k] else 0)

            rerank_scores = rerank_candidates_for_query(
                queries[i],
                faiss_article_ids,
                article_texts,
                tokenizer_rerank,
                model_rerank,
                device,
                batch_size=8,
                max_length=512,
            )

            rerank_pairs = list(zip(faiss_article_ids, rerank_scores))
            rerank_pairs.sort(key=lambda x: x[1], reverse=True)
            rerank_article_ids = [p[0] for p in rerank_pairs]

            rerank_rank = 0
            for rank, aid in enumerate(rerank_article_ids, start=1):
                if aid == gold_aid:
                    rerank_rank = rank
                    break
            mrr_rerank.append(1.0 / rerank_rank if rerank_rank > 0 else 0.0)
            for k in recall_rerank.keys():
                recall_rerank[k].append(1 if gold_aid in rerank_article_ids[:k] else 0)

            writer.writerow(
                [
                    q_ids[i],
                    queries[i],
                    linked_words[i],
                    para_texts[i],
                    gold_aid,
                    faiss_rank,
                    rerank_rank,
                    "|".join(map(str, faiss_article_ids)),
                    "|".join(map(str, rerank_article_ids)),
                ]
            )

            if (i + 1) % 10000 == 0:
                logging.info(f"Processed {i + 1} / {len(queries)} queries...")

    metrics = {}
    logging.info("\n--- FAISS vs Reranker (Article-level) ---")
    for k in recall_faiss.keys():
        faiss_val = float(np.mean(recall_faiss[k])) if recall_faiss[k] else 0.0
        rerank_val = float(np.mean(recall_rerank[k])) if recall_rerank[k] else 0.0
        metrics[f"Recall@{k}_faiss"] = faiss_val
        metrics[f"Recall@{k}_rerank"] = rerank_val
        logging.info(f"Recall@{k:<3}: FAISS={faiss_val:.4f} | Rerank={rerank_val:.4f}")

    mrr_f = float(np.mean(mrr_faiss)) if mrr_faiss else 0.0
    mrr_r = float(np.mean(mrr_rerank)) if mrr_rerank else 0.0
    metrics["MRR@TOP_K_faiss"] = mrr_f
    metrics["MRR@TOP_K_rerank"] = mrr_r
    logging.info(f"MRR@TOP_K (K={TOP_K}): FAISS={mrr_f:.4f} | Rerank={mrr_r:.4f}")

    logging.info(f"Saving metrics to {METRICS_CSV}")
    with open(METRICS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "faiss", "rerank"])
        for k in [1, 3, 5, 10, 50]:
            writer.writerow(
                [
                    f"Recall@{k}",
                    f"{metrics[f'Recall@{k}_faiss']:.6f}",
                    f"{metrics[f'Recall@{k}_rerank']:.6f}",
                ]
            )
        writer.writerow(
            [
                "MRR@TOP_K",
                f"{metrics['MRR@TOP_K_faiss']:.6f}",
                f"{metrics['MRR@TOP_K_rerank']:.6f}",
            ]
        )

    logging.info(f"Saved per-query ranks to {TOPK_CSV}")
    return metrics


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    global DOMAIN, LOG_DIR, MODEL_DIR
    global SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH, MODEL_INFO_PATH, ANCHOR_QUERIES_CSV
    global METRICS_CSV, TOPK_CSV, LOG_FILE

    DOMAIN = load_domain_from_config()

    # Domain-scoped dirs
    PROCESSED_DIR = PROCESSED_BASE_DIR / DOMAIN
    EMB_DIR = EMB_BASE_DIR / DOMAIN
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    LOG_DIR = LOG_BASE_DIR / DOMAIN
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_DIR = MODEL_BASE_DIR / DOMAIN / "best"

    SPAN_CSV = PROCESSED_DIR / f"spans_{DOMAIN}.csv"
    SPAN_EMB_PATH = EMB_DIR / f"spans_{DOMAIN}.npy"
    SPAN_IDS_PATH = EMB_DIR / f"spans_{DOMAIN}.index_ids.npy"
    MODEL_INFO_PATH = EMB_DIR / f"model_info_{DOMAIN}.json"
    ANCHOR_QUERIES_CSV = PROCESSED_DIR / f"anchor_queries_{DOMAIN}.csv"

    METRICS_CSV = LOG_DIR / f"article_reranker_vs_faiss_metrics_{DOMAIN}.csv"
    TOPK_CSV = LOG_DIR / f"article_reranker_topk_{DOMAIN}.csv"
    LOG_FILE = LOG_DIR / f"eval_article_reranker_{DOMAIN}.log"

    setup_logging(LOG_FILE)

    logging.info(f"DOMAIN              = {DOMAIN}")
    logging.info(f"SPAN_CSV            = {SPAN_CSV}")
    logging.info(f"SPAN_EMB_PATH       = {SPAN_EMB_PATH}")
    logging.info(f"SPAN_IDS_PATH       = {SPAN_IDS_PATH}")
    logging.info(f"MODEL_INFO_PATH     = {MODEL_INFO_PATH}")
    logging.info(f"ANCHOR_QUERIES_CSV  = {ANCHOR_QUERIES_CSV}")
    logging.info(f"MODEL_DIR           = {MODEL_DIR}")
    logging.info(f"METRICS_CSV         = {METRICS_CSV}")
    logging.info(f"TOPK_CSV            = {TOPK_CSV}")

    for p in [SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH, ANCHOR_QUERIES_CSV]:
        if not p.exists():
            logging.error(f"Required file missing: {p}")
            raise FileNotFoundError(f"Required file missing: {p}")

    if not MODEL_DIR.exists():
        logging.error(f"Reranker model dir not found: {MODEL_DIR}")
        raise FileNotFoundError(f"Reranker model dir not found: {MODEL_DIR}")

    if not MODEL_INFO_PATH.exists():
        logging.warning(
            f"MODEL_INFO file missing: {MODEL_INFO_PATH} "
            "(will fallback to default bi-encoder)."
        )

    metrics = evaluate_faiss_and_reranker()
    logging.info("\n✅ Article reranker evaluation finished.")
    logging.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()