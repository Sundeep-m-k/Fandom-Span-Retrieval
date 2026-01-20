#!/usr/bin/env python3
"""
08_train_article_reranker.py

Train an *article-level* cross-encoder reranker for the Fandom_SI project.

Uses:
    - data/processed/<domain>/spans_<domain>.csv
        (must contain: span_id, article_id, title, page_name, text)
    - data/processed/<domain>/anchor_queries_<domain>.csv
        (q_id, query, linked_word, paragraph_text, correct_article_id)

For each query:
    - Positive pair:  (query, article_text[correct_article_id], label=1)
    - Negative pairs: (query, article_text[neg_article_id],  label=0)
      where neg_article_id is sampled from other articles.

Trains:
    - HuggingFace AutoModelForSequenceClassification as cross-encoder
      on (query, article_text) → {0,1}.

Saves:
    - models/article_reranker/<domain>/best
    - training logs in data/logs/article_reranker/<domain>/
      (domain-scoped log file: train_article_reranker_<domain>.log)
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# -------------------------------------------------------
# Paths / Config (domain-independent)
# -------------------------------------------------------

PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")

CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping.yaml"

# UPDATED: base processed dir; domain subdir chosen in main()
PROCESSED_BASE_DIR = PROJECT_ROOT / "data" / "processed"

LOG_BASE_DIR = PROJECT_ROOT / "data" / "logs" / "article_reranker"
MODEL_BASE_DIR = PROJECT_ROOT / "models" / "article_reranker"

# Ensure base dirs exist
LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)

# These will be filled in main() once DOMAIN is known
DOMAIN = None
LOG_DIR = None
MODEL_DIR = None
SPAN_CSV = None
ANCHOR_QUERIES_CSV = None
LOG_FILE = None

# Model (you can change this if you want a different base)
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Training hyperparameters
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LR = 2e-5
NUM_NEGATIVES_PER_QUERY = 4  # random negatives per positive

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -------------------------------------------------------
# Domain + Logging helpers
# -------------------------------------------------------

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

    We forcibly clear existing root handlers so repeated runs in the
    same Python process (notebooks, multi-domain loops) behave correctly.
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


# -------------------------------------------------------
# Data prep: articles & queries
# -------------------------------------------------------

def build_article_texts():
    """
    Build article-level texts by concatenating spans belonging to each article_id.

    Returns:
        article_texts: dict[article_id] -> article_text (str)
        article_meta:  dict[article_id] -> {"title": ..., "page_name": ...}
    """
    logging.info(f"Loading spans from {SPAN_CSV} ...")
    df = pd.read_csv(SPAN_CSV)

    required = {"span_id", "article_id", "title", "page_name", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{SPAN_CSV} is missing required columns {missing}. "
            "Expected at least span_id, article_id, title, page_name, text."
        )

    df["article_id"] = df["article_id"].astype(int)

    df = df.sort_values(["article_id", "span_id"])

    article_texts = {}
    article_meta = {}

    for article_id, group in df.groupby("article_id"):
        spans_text = [str(t).strip() for t in group["text"].tolist() if str(t).strip()]
        article_text = "\n".join(spans_text)

        first = group.iloc[0]
        article_meta[article_id] = {
            "title": str(first.get("title", "")),
            "page_name": str(first.get("page_name", "")),
        }
        article_texts[article_id] = article_text

    logging.info(f"Built article_texts for {len(article_texts)} articles.")
    return article_texts, article_meta


def load_anchor_queries():
    """
    Load anchor-based queries:
        columns: q_id, query, linked_word, paragraph_text, correct_article_id
    """
    logging.info(f"Loading anchor queries from {ANCHOR_QUERIES_CSV} ...")
    df_q = pd.read_csv(ANCHOR_QUERIES_CSV)

    if "query" not in df_q.columns or "correct_article_id" not in df_q.columns:
        raise ValueError(
            f"{ANCHOR_QUERIES_CSV} must contain 'query' and 'correct_article_id' columns."
        )

    df_q["correct_article_id"] = df_q["correct_article_id"].astype(int)

    if "q_id" not in df_q.columns:
        df_q["q_id"] = np.arange(1, len(df_q) + 1)

    logging.info(f"Loaded {len(df_q)} queries.")
    return df_q


def build_reranker_examples(article_texts, df_q, num_negatives=4):
    logging.info("Building reranker examples (positives + random negatives)...")

    all_article_ids = list(article_texts.keys())
    examples = []
    num_skipped_no_article_text = 0

    for _, row in tqdm(df_q.iterrows(), total=len(df_q)):
        query = str(row["query"])
        gold_aid = int(row["correct_article_id"])

        if gold_aid not in article_texts:
            num_skipped_no_article_text += 1
            continue

        examples.append(
            {
                "query": query,
                "article_id": gold_aid,
                "article_text": article_texts[gold_aid],
                "label": 1,
            }
        )

        neg_pool = [aid for aid in all_article_ids if aid != gold_aid]
        if not neg_pool:
            continue

        neg_sample = random.sample(neg_pool, min(num_negatives, len(neg_pool)))
        for neg_aid in neg_sample:
            examples.append(
                {
                    "query": query,
                    "article_id": neg_aid,
                    "article_text": article_texts[neg_aid],
                    "label": 0,
                }
            )

    logging.info(
        f"Built {len(examples)} examples "
        f"({len(examples) / max(len(df_q), 1):.2f} examples/query on avg)."
    )
    logging.info(f"Skipped queries with missing article_text: {num_skipped_no_article_text}")
    return examples


def train_val_split(examples, val_ratio=0.1):
    logging.info("Splitting examples into train/val at query level...")

    by_query = {}
    for idx, ex in enumerate(examples):
        q = ex["query"]
        by_query.setdefault(q, []).append(idx)

    all_queries = list(by_query.keys())
    random.shuffle(all_queries)

    n_val_q = max(1, int(len(all_queries) * val_ratio))
    val_qs = set(all_queries[:n_val_q])

    train_indices, val_indices = [], []
    for q, idxs in by_query.items():
        if q in val_qs:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    train_ds = [examples[i] for i in train_indices]
    val_ds = [examples[i] for i in val_indices]

    logging.info(
        f"Train examples: {len(train_ds)}, "
        f"Val examples: {len(val_ds)}, "
        f"Val queries: {len(val_qs)}/{len(all_queries)}"
    )
    return train_ds, val_ds


# -------------------------------------------------------
# Dataset & collator
# -------------------------------------------------------

class ArticleRerankerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        q = ex["query"]
        a_text = ex["article_text"]
        label = int(ex["label"])

        encoded = self.tokenizer(
            q,
            a_text,
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = label
        return encoded


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# -------------------------------------------------------
# Main training entry
# -------------------------------------------------------

def main():
    global DOMAIN, LOG_DIR, MODEL_DIR, SPAN_CSV, ANCHOR_QUERIES_CSV, LOG_FILE

    DOMAIN = load_domain_from_config()

    LOG_DIR = LOG_BASE_DIR / DOMAIN
    MODEL_DIR = MODEL_BASE_DIR / DOMAIN

    # UPDATED: domain-scoped processed dir
    PROCESSED_DIR = PROCESSED_BASE_DIR / DOMAIN
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    SPAN_CSV = PROCESSED_DIR / f"spans_{DOMAIN}.csv"
    ANCHOR_QUERIES_CSV = PROCESSED_DIR / f"anchor_queries_{DOMAIN}.csv"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    LOG_FILE = LOG_DIR / f"train_article_reranker_{DOMAIN}.log"

    setup_logging(LOG_FILE)

    logging.info(f"DOMAIN                = {DOMAIN}")
    logging.info(f"SPAN_CSV              = {SPAN_CSV}")
    logging.info(f"ANCHOR_QUERIES_CSV    = {ANCHOR_QUERIES_CSV}")
    logging.info(f"MODEL_NAME            = {MODEL_NAME}")
    logging.info(f"MODEL_DIR             = {MODEL_DIR}")
    logging.info(f"LOG_DIR               = {LOG_DIR}")

    for p in [SPAN_CSV, ANCHOR_QUERIES_CSV]:
        if not p.exists():
            logging.error(f"Required file missing: {p}")
            raise FileNotFoundError(f"Required file missing: {p}")

    article_texts, article_meta = build_article_texts()
    logging.info(f"Article meta example (first 3): {list(article_meta.items())[:3]}")

    df_q = load_anchor_queries()

    examples = build_reranker_examples(
        article_texts, df_q, num_negatives=NUM_NEGATIVES_PER_QUERY
    )
    if not examples:
        logging.error("No training examples were built. Aborting.")
        raise RuntimeError("No examples for training reranker.")

    train_examples, val_examples = train_val_split(examples, val_ratio=0.1)

    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    train_dataset = ArticleRerankerDataset(train_examples, tokenizer, max_length=512)
    eval_dataset = ArticleRerankerDataset(val_examples, tokenizer, max_length=512)
    logging.info(
        f"Final dataset sizes → train: {len(train_dataset)}, val: {len(eval_dataset)}"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    hf_log_dir = LOG_DIR / "hf_logs"
    hf_log_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(hf_log_dir),
        logging_steps=LOGGING_STEPS,
        report_to="none",
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training article-level reranker...")
    trainer.train()

    best_dir = MODEL_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    info = {
        "model_name": MODEL_NAME,
        "domain": DOMAIN,
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "num_negatives_per_query": NUM_NEGATIVES_PER_QUERY,
        "random_seed": RANDOM_SEED,
    }
    with open(best_dir / "article_reranker_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ Training finished. Best model saved to: {best_dir}")


if __name__ == "__main__":
    main()