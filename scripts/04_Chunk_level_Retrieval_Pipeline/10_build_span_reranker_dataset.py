#!/usr/bin/env python3
"""
10_build_span_reranker_dataset.py

Generic reranker dataset builder for ANY fandom wiki.
Generates:
    - retrieval_train_<domain>.jsonl
    - retrieval_dev_<domain>.jsonl
    - retrieval_eval_<domain>.jsonl

UPDATED PATH CONVENTION:
    - data/processed/<domain>/spans_<domain>.csv
    - data/processed/<domain>/span_links_<domain>.csv
    - data/embeddings/<domain>/spans_<domain>.npy
    - data/embeddings/<domain>/spans_<domain>.index_ids.npy
    - data/indexes/<domain>/faiss_flat_<domain>.index
    - outputs written to data/processed/<domain>/

Sources for POSITIVES:
    1. Title-based templates
    2. Section-based templates
    3. First-sentence queries
    4. Anchor-text based queries (internal links)

NEGATIVES (FIXED):
    - Hard negatives from FAISS top-k for the *query embedding* (excluding positive span)
    - Easy random negatives

IMPORTANT:
    - Train/dev/test are split at the **article level**, not row-level,
      to avoid data leakage.
"""

import json
import random
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import yaml
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------
# Paths / Config (domain-independent)
# --------------------------------------------------------

PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping.yaml"

PROCESSED_BASE_DIR = PROJECT_ROOT / "data" / "processed"
EMB_BASE_DIR = PROJECT_ROOT / "data" / "embeddings"
IDX_BASE_DIR = PROJECT_ROOT / "data" / "indexes"

LOG_BASE_DIR = PROJECT_ROOT / "data" / "logs" / "span_reranker"
LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Bi-encoder to embed queries (MUST match your span embedding model for best negatives)
QUERY_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Will be filled in main()
DOMAIN = None
LOG_DIR = None
LOG_FILE = None

SPAN_CSV = None
LINKS_CSV = None
EMB_PATH = None
IDS_PATH = None
FAISS_PATH = None
OUT_TRAIN = None
OUT_DEV = None
OUT_TEST = None

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------------
# Helpers: domain + logging
# --------------------------------------------------------

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


# --------------------------------------------------------
# Generic query templates
# --------------------------------------------------------

def title_queries(title):
    t = str(title).strip()
    if not t:
        return []
    return [
        f"What is {t}?",
        f"Who is {t}?",
        f"Tell me about {t}",
        f"Information about {t}",
        f"Describe {t}",
        f"{t} details",
        f"{t} summary",
        f"{t} overview",
    ]


def section_queries(title, section):
    t = str(title).strip()
    sec = str(section).replace("[ ]", "").strip()
    if not t or not sec:
        return []
    return [
        f"{t} {sec}",
        f"{sec} of {t}",
        f"Describe the {sec} of {t}",
        f"What does the {sec} section say about {t}?",
    ]


def first_sentence_query(text):
    s = str(text).split(".")[0].strip()
    if len(s) > 5:
        return [f"{s}?"]
    return []


def anchor_queries(anchor):
    a = str(anchor).strip()
    if not a:
        return []
    return [
        a,
        f"Information about {a}",
        f"Tell me about {a}",
        f"Describe {a}",
    ]


# --------------------------------------------------------
# Hard negative helper (FIXED)
# --------------------------------------------------------

def get_hard_negatives_for_query(query_emb, pos_span_id: str, k=50, max_hard=5):
    """
    Return hard negatives from FAISS top-k for the query embedding.

    query_emb: (1, D) L2-normalized np.array (float32)
    """
    scores, idxs = faiss_index.search(query_emb, k)
    idxs = idxs[0]

    hard = []
    for idx in idxs:
        if idx < 0:
            continue
        sid = span_ids[idx]
        if sid == pos_span_id:
            continue
        hard.append(sid)
        if len(hard) >= max_hard:
            break
    return hard


# --------------------------------------------------------
# JSONL writer
# --------------------------------------------------------

def write_jsonl(path: Path, rows):
    logging.info(f"Writing {len(rows)} rows to {path} ...")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info(f"Wrote {len(rows)} rows to {path}")


# --------------------------------------------------------
# Main pipeline
# --------------------------------------------------------

def main():
    global DOMAIN, LOG_DIR, LOG_FILE
    global SPAN_CSV, LINKS_CSV, EMB_PATH, IDS_PATH, FAISS_PATH
    global OUT_TRAIN, OUT_DEV, OUT_TEST
    global faiss_index, span_embeddings, span_ids, span_id_to_row

    # 1) Domain
    DOMAIN = load_domain_from_config()

    # 2) Domain-scoped dirs
    PROCESSED_DIR = PROCESSED_BASE_DIR / DOMAIN
    EMB_DIR = EMB_BASE_DIR / DOMAIN
    IDX_DIR = IDX_BASE_DIR / DOMAIN

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    IDX_DIR.mkdir(parents=True, exist_ok=True)

    # Domain-scoped log dir + file
    LOG_DIR = LOG_BASE_DIR / DOMAIN
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / f"10_build_span_reranker_dataset_{DOMAIN}.log"

    # Inputs
    SPAN_CSV = PROCESSED_DIR / f"spans_{DOMAIN}.csv"
    LINKS_CSV = PROCESSED_DIR / f"span_links_{DOMAIN}.csv"
    EMB_PATH = EMB_DIR / f"spans_{DOMAIN}.npy"
    IDS_PATH = EMB_DIR / f"spans_{DOMAIN}.index_ids.npy"
    FAISS_PATH = IDX_DIR / f"faiss_flat_{DOMAIN}.index"

    # Outputs
    OUT_TRAIN = PROCESSED_DIR / f"retrieval_train_{DOMAIN}.jsonl"
    OUT_DEV = PROCESSED_DIR / f"retrieval_dev_{DOMAIN}.jsonl"
    OUT_TEST = PROCESSED_DIR / f"retrieval_eval_{DOMAIN}.jsonl"

    # 3) Logging
    setup_logging(LOG_FILE)

    logging.info(f"DOMAIN     = {DOMAIN}")
    logging.info(f"SPAN_CSV   = {SPAN_CSV}")
    logging.info(f"LINKS_CSV  = {LINKS_CSV}")
    logging.info(f"EMB_PATH   = {EMB_PATH}")
    logging.info(f"IDS_PATH   = {IDS_PATH}")
    logging.info(f"FAISS_PATH = {FAISS_PATH}")
    logging.info(f"OUT_TRAIN  = {OUT_TRAIN}")
    logging.info(f"OUT_DEV    = {OUT_DEV}")
    logging.info(f"OUT_TEST   = {OUT_TEST}")

    # 4) Sanity-check required files
    for p in [SPAN_CSV, LINKS_CSV, EMB_PATH, IDS_PATH, FAISS_PATH]:
        if not p.exists():
            logging.error(f"Required file missing: {p}")
            raise FileNotFoundError(f"Required file missing: {p}")

    # 5) Load everything
    logging.info("Loading spans, links, embeddings, FAISS index...")
    df = pd.read_csv(SPAN_CSV)
    df_links = pd.read_csv(LINKS_CSV)
    span_embeddings = np.load(EMB_PATH).astype("float32")
    span_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)
    faiss_index = faiss.read_index(str(FAISS_PATH))

    if len(span_ids) != span_embeddings.shape[0]:
        logging.error("span_ids and span_embeddings size mismatch")
        raise ValueError("span_ids and span_embeddings size mismatch")

    df = df.set_index("span_id", drop=False)
    span_id_to_row = df.to_dict("index")

    logging.info("Loaded:")
    logging.info(f" - {len(df)} spans")
    logging.info(f" - {len(df_links)} internal links")
    logging.info(f" - embeddings: {span_embeddings.shape}")
    logging.info(f" - FAISS ntotal: {faiss_index.ntotal}")

    # Query encoder (FIX)
    logging.info(f"Loading query encoder: {QUERY_ENCODER_NAME}")
    q_encoder = SentenceTransformer(QUERY_ENCODER_NAME)

    dataset = []
    logging.info("\nGenerating reranker pairs...\n")

    for span_id, row in tqdm(span_id_to_row.items(), total=len(span_id_to_row)):
        title = row.get("title", "")
        section = row.get("section", "")
        text = row.get("text", "")
        article_id = int(row["article_id"])
        page_name = row.get("page_name", "")

        positive_span = span_id

        queries = set()
        for q in title_queries(title):
            queries.add(q)
        for q in section_queries(title, section):
            queries.add(q)
        for q in first_sentence_query(text):
            queries.add(q)

        anchors = df_links[df_links["span_id"] == span_id]["anchor_text"].unique()
        for a in anchors:
            for q in anchor_queries(a):
                queries.add(q)

        if not queries:
            continue

        queries = list(queries)

        # Encode all queries for this span in a batch (efficient)
        q_embs = q_encoder.encode(
            queries,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        for qi, q in enumerate(queries):
            dataset.append(
                {
                    "query": q,
                    "span_id": positive_span,
                    "text": text,
                    "label": 1,
                    "source": "positive",
                    "article_id": article_id,
                    "page_name": page_name,
                }
            )

            q_emb = q_embs[qi].reshape(1, -1)

            # FIXED hard negatives: FAISS top-k for query embedding
            hard_negs = get_hard_negatives_for_query(
                q_emb, pos_span_id=positive_span, k=50, max_hard=5
            )

            for hn in hard_negs:
                hn_row = span_id_to_row.get(hn)
                if hn_row is None:
                    continue
                dataset.append(
                    {
                        "query": q,
                        "span_id": hn,
                        "text": hn_row["text"],
                        "label": 0,
                        "source": "hard_negative",
                        "article_id": int(hn_row["article_id"]),
                        "page_name": hn_row.get("page_name", ""),
                    }
                )

            # Easy negative (random span)
            random_neg = random.choice(span_ids)
            if random_neg != positive_span:
                rn_row = span_id_to_row.get(random_neg)
                if rn_row is not None:
                    dataset.append(
                        {
                            "query": q,
                            "span_id": random_neg,
                            "text": rn_row["text"],
                            "label": 0,
                            "source": "easy_negative",
                            "article_id": int(rn_row["article_id"]),
                            "page_name": rn_row.get("page_name", ""),
                        }
                    )

    logging.info(f"\nTotal rows (before split): {len(dataset)}")

    # --------------------------------------------------------
    # Article-level split into train/dev/test
    # --------------------------------------------------------
    logging.info("Performing article-level split into train/dev/test...")

    by_article = {}
    for row in dataset:
        aid = row["article_id"]
        by_article.setdefault(aid, []).append(row)

    article_ids = list(by_article.keys())
    random.shuffle(article_ids)

    n_articles = len(article_ids)
    n_test = max(1, int(0.10 * n_articles))
    n_dev = max(1, int(0.10 * n_articles))

    test_articles = set(article_ids[:n_test])
    dev_articles = set(article_ids[n_test:n_test + n_dev])
    train_articles = set(article_ids[n_test + n_dev:])

    train, dev, test = [], [], []
    for aid, rows in by_article.items():
        if aid in test_articles:
            test.extend(rows)
        elif aid in dev_articles:
            dev.extend(rows)
        else:
            train.extend(rows)

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    logging.info("Article-level split:")
    logging.info(f" - #articles total: {n_articles}")
    logging.info(f" - #articles train: {len(train_articles)}")
    logging.info(f" - #articles dev:   {len(dev_articles)}")
    logging.info(f" - #articles test:  {len(test_articles)}")
    logging.info(f" - rows train: {len(train)}")
    logging.info(f" - rows dev:   {len(dev)}")
    logging.info(f" - rows test:  {len(test)}")

    write_jsonl(OUT_TRAIN, train)
    write_jsonl(OUT_DEV, dev)
    write_jsonl(OUT_TEST, test)

    logging.info("✅ Finished building span reranker dataset.")


if __name__ == "__main__":
    main()