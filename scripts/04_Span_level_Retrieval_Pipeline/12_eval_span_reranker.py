#!/usr/bin/env python3
import sys
from pathlib import Path

# ---------------------------------------------------
# Ensure project root on PYTHONPATH
# ---------------------------------------------------
PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
sys.path.insert(0, str(PROJECT_ROOT))

"""
12_eval_span_reranker.py (FINAL)

One script that produces THREE comparable evaluations:

(A) Pool-only reranker ranking evaluation (uses retrieval_eval_<domain>.jsonl):
    - Scores the pre-constructed candidate pool per query
    - Measures ranking metrics: Recall@K, MRR@10, NDCG@10

(B) True retrieval: FAISS baseline (bi-encoder over FULL span index):
    - Builds/loads FAISS on span embeddings
    - Encodes queries with bi-encoder model used for embeddings (model_info_<domain>.json)
    - Retrieves top-K spans for each query
    - Evaluates BOTH:
        - Span-hit metrics (gold span_id appears in top-K)
        - Article-hit metrics (any retrieved span from gold article_id appears in top-K)

(C) True retrieval: FAISS + Cross-encoder reranking:
    - Reranks the FAISS top-K candidates with trained cross-encoder reranker
    - Evaluates the same span-hit and article-hit metrics

Assumptions / Inputs (domain-scoped):
  - data/processed/<domain>/retrieval_eval_<domain>.jsonl
      each line has at least: {"query": str, "text": str, "label": 0/1}
      optionally contains: "span_id", "article_id"
  - data/processed/<domain>/spans_<domain>.csv
      must contain: span_id, article_id, text
  - data/embeddings/spans_<domain>.npy
  - data/embeddings/spans_<domain>.index_ids.npy  (span_ids aligned with embeddings)
  - data/embeddings/model_info_<domain>.json       {"model_name": "..."} for bi-encoder
  - models/reranker/<domain>/best/                 trained cross-encoder reranker

Outputs (domain-scoped):
  - data/logs/reranker_eval/<domain>/eval12_metrics_<domain>.csv
  - data/logs/reranker_eval/<domain>/eval12_per_query_<domain>.csv
  - data/logs/reranker_eval/<domain>/12_eval_span_reranker_<domain>.log
"""

import csv
import json
import math
import time
import logging
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.logging_utils import create_logger


# ----------------------------
# Metrics helpers
# ----------------------------
KS = [1, 3, 5, 10, 100, 1000]


def dcg_at_rank(rank_1based: int) -> float:
    # binary relevance, single relevant item
    return 1.0 / math.log2(rank_1based + 1.0)


def compute_ranking_metrics_from_first_pos_rank(ranks_1based, ks=KS):
    """
    ranks_1based: list[int|None], rank of first positive; None means not found.
    Returns dict with Recall@K, MRR@10, NDCG@10
    """
    n = 0
    recall_hits = {k: 0 for k in ks}
    mrr10_sum = 0.0
    ndcg10_sum = 0.0

    for r in ranks_1based:
        if r is None:
            continue
        n += 1

        for k in ks:
            if r <= k:
                recall_hits[k] += 1

        if r <= 10:
            mrr10_sum += 1.0 / r
            ndcg10_sum += dcg_at_rank(r)
        else:
            # MRR@10 adds 0, NDCG@10 adds 0
            pass

    if n == 0:
        return {"n_queries": 0, **{f"Recall@{k}": 0.0 for k in ks}, "MRR@10": 0.0, "NDCG@10": 0.0}

    out = {"n_queries": n}
    for k in ks:
        out[f"Recall@{k}"] = recall_hits[k] / n
    out["MRR@10"] = mrr10_sum / n
    out["NDCG@10"] = ndcg10_sum / n  # IDCG = 1.0 for single relevant
    return out


# ----------------------------
# IO helpers
# ----------------------------
def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_domain_from_config(config_path: Path) -> str:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    domain = cfg.get("domain") or cfg.get("domain_slug")
    if not domain:
        base_url = (cfg.get("base_url") or "").rstrip("/")
        if not base_url:
            raise ValueError(
                "Could not determine domain. Please set 'domain' (or 'domain_slug') "
                "or 'base_url' in configs/scraping.yaml"
            )
        domain = urlparse(base_url).netloc.split(".")[0]
    return domain


def load_bi_encoder_name(model_info_path: Path) -> str:
    default_model = "sentence-transformers/all-MiniLM-L6-v2"
    if not model_info_path.exists():
        logging.warning(f"MODEL_INFO missing: {model_info_path}. Using default: {default_model}")
        return default_model
    with open(model_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info.get("model_name", default_model)


# ----------------------------
# Core: Pool evaluation (your old 12)
# ----------------------------
def eval_pool_only_reranker(by_query, tokenizer, model, device):
    """
    by_query: dict[str, list[rows]] from retrieval_eval jsonl (candidate pool)
    returns: metrics dict, per_query list of dicts
    """
    ranks = []
    per_query = []

    for q, candidates in tqdm(by_query.items(), desc="(A) Pool-only reranker eval"):
        # require at least one positive label in pool
        if not any(int(c.get("label", 0)) == 1 for c in candidates):
            continue

        texts = [str(c.get("text", "")) for c in candidates]
        with torch.no_grad():
            batch = tokenizer(
                [q] * len(texts),
                texts,
                padding=True,
                truncation=True,
                max_length=320,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            # class-1 logit preferred (consistent with your training head)
            if logits.size(-1) == 1:
                scores_tensor = logits.squeeze(-1)
            else:
                scores_tensor = logits[:, 1]
            scores = scores_tensor.detach().cpu().tolist()

        scored = []
        for c, s in zip(candidates, scores):
            scored.append(
                {
                    "span_id": c.get("span_id"),
                    "article_id": c.get("article_id"),
                    "label": int(c.get("label", 0)),
                    "score": float(s),
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)

        # first positive rank
        r = None
        for i, item in enumerate(scored):
            if item["label"] == 1:
                r = i + 1
                break
        ranks.append(r)

        # record top-10 ids for debugging
        per_query.append(
            {
                "eval_mode": "POOL_ONLY_RERANKER",
                "query": q,
                "n_candidates": len(candidates),
                "first_pos_rank": r if r is not None else 0,
                "top10_span_ids": "|".join(str(x.get("span_id")) for x in scored[:10]),
                "top10_article_ids": "|".join(str(x.get("article_id")) for x in scored[:10]),
            }
        )

    metrics = compute_ranking_metrics_from_first_pos_rank(ranks, ks=KS)
    return metrics, per_query


# ----------------------------
# Core: True retrieval via FAISS (baseline + rerank)
# ----------------------------
def build_faiss_index(span_embs: np.ndarray):
    # We assume cosine similarity: L2-normalize then inner product
    span_embs = span_embs.astype("float32", copy=False)
    faiss.normalize_L2(span_embs)
    index = faiss.IndexFlatIP(span_embs.shape[1])
    index.add(span_embs)
    return index


def rerank_cross_encoder(query, cand_texts, tokenizer, model, device, batch_size=32, max_length=320):
    """
    returns: list[float] scores aligned with cand_texts
    """
    scores_all = []
    with torch.no_grad():
        for i in range(0, len(cand_texts), batch_size):
            chunk = cand_texts[i:i + batch_size]
            batch = tokenizer(
                [query] * len(chunk),
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            if logits.size(-1) == 1:
                s = logits.squeeze(-1)
            else:
                s = logits[:, 1]
            scores_all.extend(s.detach().cpu().tolist())
    return scores_all


def derive_gold_from_pool(by_query):
    """
    From retrieval_eval jsonl pool, derive per-query gold span_id and gold article_id.
    Uses first positive row.
    Returns:
      queries: list[str]
      gold_span_ids: list[str|None]
      gold_article_ids: list[int|None]
    """
    queries = []
    gold_span_ids = []
    gold_article_ids = []

    for q, candidates in by_query.items():
        pos = None
        for c in candidates:
            if int(c.get("label", 0)) == 1:
                pos = c
                break
        if pos is None:
            continue
        queries.append(q)
        gold_span_ids.append(str(pos.get("span_id")) if pos.get("span_id") is not None else None)
        ga = pos.get("article_id")
        gold_article_ids.append(int(ga) if ga is not None and str(ga).strip() != "" else None)

    return queries, gold_span_ids, gold_article_ids


def eval_true_retrieval(
    queries,
    gold_span_ids,
    gold_article_ids,
    bi_encoder,
    faiss_index,
    span_ids,
    spanid_to_articleid,
    spanid_to_text,
    rerank_tokenizer,
    rerank_model,
    device,
    top_k=50,
):
    """
    Returns:
      metrics_faiss_span, metrics_faiss_article,
      metrics_rerank_span, metrics_rerank_article,
      per_query_rows (for csv)
    """
    # Encode queries
    q_embs = bi_encoder.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    ranks_span_faiss = []
    ranks_article_faiss = []
    ranks_span_rerank = []
    ranks_article_rerank = []

    per_query = []

    for i, q in enumerate(tqdm(queries, desc="(B/C) True retrieval eval")):
        qv = q_embs[i].reshape(1, -1)

        # --- FAISS retrieve spans ---
        scores, idxs = faiss_index.search(qv, top_k)
        idxs = idxs[0].tolist()

        retrieved_span_ids = [str(span_ids[j]) for j in idxs if j >= 0]
        retrieved_article_ids = [spanid_to_articleid.get(sid) for sid in retrieved_span_ids]

        gsid = gold_span_ids[i]
        gaid = gold_article_ids[i]

        # Span-hit rank (FAISS)
        r_span = None
        if gsid is not None:
            for rank, sid in enumerate(retrieved_span_ids, start=1):
                if sid == gsid:
                    r_span = rank
                    break
        ranks_span_faiss.append(r_span)

        # Article-hit rank (FAISS) : first rank where article_id matches
        r_art = None
        if gaid is not None:
            for rank, aid in enumerate(retrieved_article_ids, start=1):
                if aid == gaid:
                    r_art = rank
                    break
        ranks_article_faiss.append(r_art)

        # --- Cross-encoder rerank the retrieved spans ---
        cand_texts = [spanid_to_text.get(sid, "") for sid in retrieved_span_ids]
        rerank_scores = rerank_cross_encoder(
            q, cand_texts, rerank_tokenizer, rerank_model, device, batch_size=32, max_length=320
        )
        rerank_pairs = list(zip(retrieved_span_ids, retrieved_article_ids, rerank_scores))
        rerank_pairs.sort(key=lambda x: x[2], reverse=True)

        reranked_span_ids = [p[0] for p in rerank_pairs]
        reranked_article_ids = [p[1] for p in rerank_pairs]

        # Span-hit rank (Reranked)
        rr_span = None
        if gsid is not None:
            for rank, sid in enumerate(reranked_span_ids, start=1):
                if sid == gsid:
                    rr_span = rank
                    break
        ranks_span_rerank.append(rr_span)

        # Article-hit rank (Reranked)
        rr_art = None
        if gaid is not None:
            for rank, aid in enumerate(reranked_article_ids, start=1):
                if aid == gaid:
                    rr_art = rank
                    break
        ranks_article_rerank.append(rr_art)

        per_query.append(
            {
                "eval_mode": "TRUE_RETRIEVAL",
                "query": q,
                "gold_span_id": gsid or "",
                "gold_article_id": gaid if gaid is not None else "",
                "faiss_span_rank": r_span if r_span is not None else 0,
                "faiss_article_rank": r_art if r_art is not None else 0,
                "rerank_span_rank": rr_span if rr_span is not None else 0,
                "rerank_article_rank": rr_art if rr_art is not None else 0,
                "faiss_top10_span_ids": "|".join(retrieved_span_ids[:10]),
                "rerank_top10_span_ids": "|".join(reranked_span_ids[:10]),
                "faiss_top10_article_ids": "|".join("" if a is None else str(a) for a in retrieved_article_ids[:10]),
                "rerank_top10_article_ids": "|".join("" if a is None else str(a) for a in reranked_article_ids[:10]),
            }
        )

    metrics_faiss_span = compute_ranking_metrics_from_first_pos_rank(ranks_span_faiss, ks=KS)
    metrics_faiss_article = compute_ranking_metrics_from_first_pos_rank(ranks_article_faiss, ks=KS)
    metrics_rerank_span = compute_ranking_metrics_from_first_pos_rank(ranks_span_rerank, ks=KS)
    metrics_rerank_article = compute_ranking_metrics_from_first_pos_rank(ranks_article_rerank, ks=KS)

    return (
        metrics_faiss_span,
        metrics_faiss_article,
        metrics_rerank_span,
        metrics_rerank_article,
        per_query,
    )


# ----------------------------
# Main
# ----------------------------
def main():
    start_time = time.perf_counter()

    CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping.yaml"
    domain = load_domain_from_config(CONFIG_PATH)

    # domain-scoped dirs
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / domain
    EMB_DIR = PROJECT_ROOT / "data" / "embeddings" / domain
    LOG_BASE_DIR = PROJECT_ROOT / "data" / "logs" / "reranker_eval"
    MODEL_BASE = PROJECT_ROOT / "models" / "reranker"

    LOG_DIR = LOG_BASE_DIR / domain
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger, log_file = create_logger(LOG_DIR, f"12_eval_span_reranker_{domain}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"DOMAIN: {domain}")
    logger.info(f"PROCESSED_DIR: {PROCESSED_DIR}")

    # Inputs
    EVAL_JSONL = PROCESSED_DIR / f"retrieval_eval_{domain}.jsonl"
    SPANS_CSV = PROCESSED_DIR / f"spans_{domain}.csv"

    SPAN_EMB_PATH = EMB_DIR / f"spans_{domain}.npy"
    SPAN_IDS_PATH = EMB_DIR / f"spans_{domain}.index_ids.npy"
    MODEL_INFO_PATH = EMB_DIR / f"model_info_{domain}.json"

    RERANKER_DIR = MODEL_BASE / domain / "best"

    # Outputs
    METRICS_CSV = LOG_DIR / f"eval12_metrics_{domain}.csv"
    PER_QUERY_CSV = LOG_DIR / f"eval12_per_query_{domain}.csv"

    # Sanity checks
    required = [EVAL_JSONL, SPANS_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH, RERANKER_DIR]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    # Load pool jsonl
    rows = load_jsonl(EVAL_JSONL)
    by_query = defaultdict(list)
    for r in rows:
        q = r.get("query")
        if q is None:
            continue
        by_query[str(q)].append(r)

    logger.info(f"Loaded eval JSONL rows: {len(rows)}")
    logger.info(f"Unique queries (pool): {len(by_query)}")

    # Load reranker model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading reranker from {RERANKER_DIR} on device={device}")
    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_DIR)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_DIR)
    rerank_model.to(device)
    rerank_model.eval()

    # ---------------------------------------------------------
    # (A) Pool-only reranker evaluation
    # ---------------------------------------------------------
    logger.info("=== (A) Pool-only reranker evaluation ===")
    metrics_pool, per_query_pool = eval_pool_only_reranker(by_query, rerank_tokenizer, rerank_model, device)

    # ---------------------------------------------------------
    # (B/C) True retrieval evaluation: FAISS baseline + rerank
    # ---------------------------------------------------------
    logger.info("=== (B/C) True retrieval evaluation (FAISS -> optional rerank) ===")

    # Load spans metadata
    df_spans = pd.read_csv(SPANS_CSV)
    needed_cols = {"span_id", "article_id", "text"}
    missing = needed_cols - set(df_spans.columns)
    if missing:
        raise ValueError(f"{SPANS_CSV} missing required columns: {missing}")

    df_spans["span_id"] = df_spans["span_id"].astype(str)
    df_spans["article_id"] = df_spans["article_id"].astype(int)

    spanid_to_articleid = dict(zip(df_spans["span_id"].tolist(), df_spans["article_id"].tolist()))
    spanid_to_text = dict(zip(df_spans["span_id"].tolist(), df_spans["text"].astype(str).tolist()))

    # Load embeddings + ids
    span_embs = np.load(SPAN_EMB_PATH).astype("float32")
    span_ids = np.load(SPAN_IDS_PATH, allow_pickle=True).astype(str)
    if span_embs.shape[0] != len(span_ids):
        raise ValueError(f"Embeddings rows ({span_embs.shape[0]}) != span_ids ({len(span_ids)})")

    # Build FAISS index
    logger.info("Building FAISS index over ALL spans...")
    faiss_index = build_faiss_index(span_embs)
    logger.info(f"FAISS ntotal={faiss_index.ntotal}, dim={faiss_index.d}")

    # Bi-encoder for query embeddings
    bi_model_name = load_bi_encoder_name(MODEL_INFO_PATH)
    logger.info(f"Loading bi-encoder for queries: {bi_model_name}")
    bi_encoder = SentenceTransformer(bi_model_name)

    # Derive gold from pool (first positive per query)
    queries, gold_span_ids, gold_article_ids = derive_gold_from_pool(by_query)
    logger.info(f"Queries with positives (derived gold): {len(queries)}")

    # Use the same TOP_K as your earlier scripts unless you change here
    TOP_K = 50

    (
        metrics_faiss_span,
        metrics_faiss_article,
        metrics_rerank_span,
        metrics_rerank_article,
        per_query_true,
    ) = eval_true_retrieval(
        queries=queries,
        gold_span_ids=gold_span_ids,
        gold_article_ids=gold_article_ids,
        bi_encoder=bi_encoder,
        faiss_index=faiss_index,
        span_ids=span_ids,
        spanid_to_articleid=spanid_to_articleid,
        spanid_to_text=spanid_to_text,
        rerank_tokenizer=rerank_tokenizer,
        rerank_model=rerank_model,
        device=device,
        top_k=TOP_K,
    )

    # ---------------------------------------------------------
    # Logging: print metrics (human readable)
    # ---------------------------------------------------------
    def log_metrics_block(title, m):
        logger.info(title)
        logger.info(f"Evaluated queries: {m.get('n_queries', 0)}")
        for k in KS:
            logger.info(f"Recall@{k}: {m.get(f'Recall@{k}', 0.0):.4f}")
        logger.info(f"MRR@10:  {m.get('MRR@10', 0.0):.4f}")
        logger.info(f"NDCG@10: {m.get('NDCG@10', 0.0):.4f}")

    logger.info("\n==================== FINAL METRICS ====================")
    log_metrics_block("---- (A) Pool-only reranker ----", metrics_pool)

    log_metrics_block("---- (B) True retrieval (FAISS) [SPAN-HIT] ----", metrics_faiss_span)
    log_metrics_block("---- (B) True retrieval (FAISS) [ARTICLE-HIT] ----", metrics_faiss_article)

    log_metrics_block("---- (C) True retrieval (FAISS+RERANK) [SPAN-HIT] ----", metrics_rerank_span)
    log_metrics_block("---- (C) True retrieval (FAISS+RERANK) [ARTICLE-HIT] ----", metrics_rerank_article)
    logger.info("=======================================================\n")

    # ---------------------------------------------------------
    # Write CSV outputs
    # ---------------------------------------------------------
    # 1) Per-query CSV
    logger.info(f"Writing per-query debug CSV: {PER_QUERY_CSV}")
    per_query_all = per_query_pool + per_query_true
    if per_query_all:
        # union of keys
        all_keys = sorted({k for row in per_query_all for k in row.keys()})
        with open(PER_QUERY_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in per_query_all:
                w.writerow(r)

    # 2) Metrics CSV (single file, easy to paste into paper)
    logger.info(f"Writing metrics CSV: {METRICS_CSV}")
    with open(METRICS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "metric", "value"])

        def write_metrics(section, m):
            w.writerow([section, "n_queries", m.get("n_queries", 0)])
            for k in KS:
                w.writerow([section, f"Recall@{k}", f"{m.get(f'Recall@{k}', 0.0):.6f}"])
            w.writerow([section, "MRR@10", f"{m.get('MRR@10', 0.0):.6f}"])
            w.writerow([section, "NDCG@10", f"{m.get('NDCG@10', 0.0):.6f}"])

        write_metrics("(A) Pool-only reranker", metrics_pool)
        write_metrics("(B) True retrieval FAISS [SPAN-HIT]", metrics_faiss_span)
        write_metrics("(B) True retrieval FAISS [ARTICLE-HIT]", metrics_faiss_article)
        write_metrics("(C) True retrieval FAISS+RERANK [SPAN-HIT]", metrics_rerank_span)
        write_metrics("(C) True retrieval FAISS+RERANK [ARTICLE-HIT]", metrics_rerank_article)

    end_time = time.perf_counter()
    logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")
    logger.info("✅ 12_eval_span_reranker (FINAL) completed successfully.")


if __name__ == "__main__":
    main()