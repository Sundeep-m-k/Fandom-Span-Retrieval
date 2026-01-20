#!/usr/bin/env python3
"""
13_run_demo_span_server.py

Interactive demo for the Fandom_SI span-level retrieval pipeline.

For each user query, this script:
    1. Uses the bi-encoder + FAISS index to retrieve top-K spans.
    2. Uses the span-level cross-encoder reranker to re-score those K spans.
    3. Prints:
        - FAISS baseline top-K spans
        - Reranked top-K spans

Assumes the following files exist for the given DOMAIN:
    - data/processed/spans_<domain>.csv
    - data/embeddings/spans_<domain>.npy
    - data/embeddings/spans_<domain>.index_ids.npy
    - data/embeddings/model_info_<domain>.json
    - data/indexes/faiss_flat_<domain>.index
    - models/reranker/<domain>/best/         (trained span reranker)

Usage:
    cd /data/sundeep/Fandom_SI
    python scripts/13_run_demo_span_server.py
"""

import json
import sys
from pathlib import Path
from textwrap import shorten

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DOMAIN = "money-heist"
TOP_K = 10            # how many candidates from FAISS
SHOW_K = 5            # how many to display in the console

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
IDX_DIR = PROJECT_ROOT / "data" / "indexes"
MODEL_DIR = PROJECT_ROOT / "models" / "reranker" / DOMAIN / "best"

SPAN_CSV = PROCESSED_DIR / f"spans_{DOMAIN}.csv"
SPAN_EMB_PATH = EMB_DIR / f"spans_{DOMAIN}.npy"
SPAN_IDS_PATH = EMB_DIR / f"spans_{DOMAIN}.index_ids.npy"
MODEL_INFO_PATH = EMB_DIR / f"model_info_{DOMAIN}.json"
FAISS_PATH = IDX_DIR / f"faiss_flat_{DOMAIN}.index"

for p in [SPAN_CSV, SPAN_EMB_PATH, SPAN_IDS_PATH, MODEL_INFO_PATH, FAISS_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Required file missing: {p}")

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Span reranker not found at: {MODEL_DIR}")

# ---------------------------------------------------------------------
# Loading: spans, embeddings, FAISS, models
# ---------------------------------------------------------------------


def load_spans():
    df = pd.read_csv(SPAN_CSV)
    if "span_id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"{SPAN_CSV} must contain at least 'span_id' and 'text' columns.")
    df = df.set_index("span_id", drop=False)
    return df


def load_span_embeddings_and_index():
    span_embs = np.load(SPAN_EMB_PATH).astype("float32")
    span_ids = np.load(SPAN_IDS_PATH, allow_pickle=True).astype(str)

    if span_embs.shape[0] != len(span_ids):
        print(
            f"[WARN] span_embs rows ({span_embs.shape[0]}) != span_ids length ({len(span_ids)}). "
            "Assuming order still matches FAISS index."
        )

    index = faiss.read_index(str(FAISS_PATH))
    return span_embs, span_ids, index


def load_bi_encoder():
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    model_name = info.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"[INFO] Loading bi-encoder: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def load_span_reranker():
    print(f"[INFO] Loading span reranker from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


# ---------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------


def faiss_retrieve_spans(query, bi_encoder, faiss_index, span_ids, top_k=TOP_K):
    """
    Given a query string, return top_k (span_id, score) from FAISS.
    """
    q_emb = bi_encoder.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, idxs = faiss_index.search(q_emb, top_k)
    scores = scores[0]
    idxs = idxs[0]

    results = []
    for score, idx in zip(scores, idxs):
        if idx < 0:
            continue
        sid = span_ids[idx]
        results.append((sid, float(score)))
    return results


def rerank_spans(query, candidate_spans, df_spans, tokenizer, model, device, batch_size=8, max_length=256):
    """
    candidate_spans: list of (span_id, faiss_score)
    Returns list of (span_id, rerank_score) sorted desc.
    """
    texts = [str(df_spans.loc[sid]["text"]) for sid, _ in candidate_spans]
    pairs = list(zip([query] * len(texts), texts))

    all_scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            q_batch = [p[0] for p in batch_pairs]
            t_batch = [p[1] for p in batch_pairs]

            enc = tokenizer(
                q_batch,
                t_batch,
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

    reranked = list(zip([sid for sid, _ in candidate_spans], all_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def pretty_print_results(title, results, df_spans, k=SHOW_K):
    print(f"\n=== {title} (top-{k}) ===")
    for rank, (sid, score) in enumerate(results[:k], start=1):
        row = df_spans.loc[sid]
        article_id = row.get("article_id", "NA")
        page_name = row.get("page_name", "NA")
        section = row.get("section", "NA")
        text = shorten(str(row.get("text", "")).replace("\n", " "), width=160, placeholder=" ...")
        print(
            f"[{rank:2d}] span_id={sid} | article_id={article_id} | page={page_name} | section={section} | "
            f"score={score:.4f}\n     {text}"
        )


# ---------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------


def main():
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] DOMAIN       = {DOMAIN}")

    df_spans = load_spans()
    span_embs, span_ids, faiss_index = load_span_embeddings_and_index()
    bi_encoder = load_bi_encoder()
    tokenizer_rerank, model_rerank, device = load_span_reranker()

    print("\n[READY] Span demo is live!")
    print("Type a query and press Enter.")
    print("Type 'quit' or 'exit' or just press Enter on empty line to stop.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if not query or query.lower() in {"quit", "exit"}:
            print("[INFO] Bye!")
            break

        # 1) FAISS retrieval
        faiss_results = faiss_retrieve_spans(query, bi_encoder, faiss_index, span_ids, top_k=TOP_K)

        if not faiss_results:
            print("No FAISS results found.")
            continue

        # 2) Rerank with cross-encoder
        reranked_results = rerank_spans(
            query,
            faiss_results,
            df_spans,
            tokenizer_rerank,
            model_rerank,
            device,
            batch_size=8,
            max_length=256,
        )

        # 3) Pretty print
        pretty_print_results("FAISS baseline", faiss_results, df_spans, k=SHOW_K)
        pretty_print_results("Reranked (FAISS + cross-encoder)", reranked_results, df_spans, k=SHOW_K)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()