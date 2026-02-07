#!/usr/bin/env python3

"""
11_train_reranker.py (PAIR-BASED, L6 MODEL)

Train a Cross-Encoder reranker for Fandom span retrieval using
independent (query, span, label) pairs.

Expected input JSONL files in:
    data/processed/<domain>/
        - retrieval_train_<domain>.jsonl
        - retrieval_dev_<domain>.jsonl
        - retrieval_eval_<domain>.jsonl

Each line:
    {
      "query": "...",
      "text":  "...",   # span text
      "label": 0 or 1,
      ... (extra fields ignored)
    }

Model:
    - cross-encoder/ms-marco-MiniLM-L-6-v2  (L6 cross-encoder)

Outputs:
    - models/reranker/<domain>/best/        (saved model + tokenizer)
    - logs in data/logs/reranker_train/<domain>/
"""

import sys
import json
import random
import time
from pathlib import Path
from urllib.parse import urlparse
from inspect import signature

import yaml
import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ---------------------------------------------------
# Project root & PYTHONPATH
# ---------------------------------------------------
PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
sys.path.insert(0, str(PROJECT_ROOT))

# Uses your utility logger (assumed stable)
from src.utils.logging_utils import create_logger

# ============================================================
# Paths & Config
# ============================================================

CONFIG_PATH = PROJECT_ROOT / "configs" / "scraping.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

# Prefer explicit domain/domain_slug; fall back to base_url host
DOMAIN = cfg.get("domain") or cfg.get("domain_slug")
if not DOMAIN:
    base_url = cfg.get("base_url", "").rstrip("/")
    if not base_url:
        raise ValueError(
            "Could not determine DOMAIN. Please set 'domain' (or 'domain_slug') "
            "or 'base_url' in configs/scraping.yaml"
        )
    DOMAIN = urlparse(base_url).netloc.split(".")[0]

# IMPORTANT: new convention uses processed/<domain>/
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / DOMAIN
MODEL_OUT_BASE = PROJECT_ROOT / "models" / "reranker"
LOG_BASE_DIR = PROJECT_ROOT / "data" / "logs" / "reranker_train"

MODEL_OUT_BASE.mkdir(parents=True, exist_ok=True)
LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Domain-specific log directory
LOG_DIR = LOG_BASE_DIR / DOMAIN
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Logging: domain-scoped directory + domain in log filename
# ------------------------------------------------------------
logger, log_file = create_logger(LOG_DIR, f"11_train_span_reranker_pairs_{DOMAIN}")

start_time = time.perf_counter()
logger.info(f"Log file: {log_file}")
logger.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
logger.info(f"CONFIG_PATH  = {CONFIG_PATH}")
logger.info(f"DOMAIN       = {DOMAIN}")
logger.info(f"PROCESSED_DIR= {PROCESSED_DIR}")

# Match filenames from 10_build_span_reranker_dataset.py
TRAIN_PATH = PROCESSED_DIR / f"retrieval_train_{DOMAIN}.jsonl"
DEV_PATH = PROCESSED_DIR / f"retrieval_dev_{DOMAIN}.jsonl"
TEST_PATH = PROCESSED_DIR / f"retrieval_eval_{DOMAIN}.jsonl"

for p in [TRAIN_PATH, DEV_PATH, TEST_PATH]:
    if not p.exists():
        logger.error(f"Missing file: {p}")
        raise FileNotFoundError(f"Missing file: {p}")

logger.info(f"Train file: {TRAIN_PATH}")
logger.info(f"Dev file:   {DEV_PATH}")
logger.info(f"Test file:  {TEST_PATH}")

# ============================================================
# Reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
logger.info(f"Random seed set to {SEED}")

# ============================================================
# Load pair-wise JSONL data
# ============================================================

def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

logger.info("Loading JSONL datasets...")
raw_train = load_jsonl(TRAIN_PATH)
raw_dev = load_jsonl(DEV_PATH)
raw_test = load_jsonl(TEST_PATH)

logger.info(
    f"Loaded PAIR dataset sizes: train={len(raw_train)}, dev={len(raw_dev)}, test={len(raw_test)}"
)

# Optional subsample for speed
MAX_TRAIN_PAIRS = 100_000
if len(raw_train) > MAX_TRAIN_PAIRS:
    logger.info(f"Subsampling train PAIRS from {len(raw_train)} to {MAX_TRAIN_PAIRS}")
    raw_train = random.sample(raw_train, MAX_TRAIN_PAIRS)
else:
    logger.info(f"Using full train pairs (<= {MAX_TRAIN_PAIRS})")

# ============================================================
# Dataset (classic pair-wise reranker)
# ============================================================

class PairRerankerDataset(Dataset):
    """
    Each item is a single (query, span_text, label) pair.
    Required keys: query, text, label
    """

    def __init__(self, rows, tokenizer, max_len=320):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        q = str(r.get("query", ""))
        t = str(r.get("text", ""))
        y = int(r.get("label", 0))

        enc = self.tok(
            q,
            t,
            truncation=True,
            max_length=self.max_len,
        )
        enc["labels"] = y
        return enc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if logits.shape[-1] == 1:
        preds = (logits.reshape(-1) > 0).astype(int)
    else:
        preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# ============================================================
# Model & Tokenizer (L6 cross-encoder)
# ============================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
logger.info(f"Loading tokenizer and model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

train_dataset = PairRerankerDataset(raw_train, tokenizer)
dev_dataset = PairRerankerDataset(raw_dev, tokenizer)
test_dataset = PairRerankerDataset(raw_test, tokenizer)

logger.info(
    "Final PAIR datasets:\n"
    f"  - train: {len(train_dataset)}\n"
    f"  - dev:   {len(dev_dataset)}\n"
    f"  - test:  {len(test_dataset)}"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# Version-safe TrainingArguments
# ============================================================

SAVE_DIR = MODEL_OUT_BASE / DOMAIN
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ta_sig = signature(TrainingArguments.__init__)
ta_params = ta_sig.parameters.keys()

kwargs = {
    "output_dir": str(SAVE_DIR),
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 4,
    "learning_rate": 2e-5,
}

eval_supported = "evaluation_strategy" in ta_params
if eval_supported:
    kwargs["evaluation_strategy"] = "epoch"
if "save_strategy" in ta_params:
    kwargs["save_strategy"] = "epoch"

if "logging_strategy" in ta_params:
    kwargs["logging_strategy"] = "steps"
if "logging_steps" in ta_params:
    kwargs["logging_steps"] = 100
if "save_total_limit" in ta_params:
    kwargs["save_total_limit"] = 3
if "warmup_ratio" in ta_params:
    kwargs["warmup_ratio"] = 0.1
if "fp16" in ta_params:
    kwargs["fp16"] = True

if eval_supported and "load_best_model_at_end" in ta_params:
    kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in ta_params:
        kwargs["metric_for_best_model"] = "f1"
    if "greater_is_better" in ta_params:
        kwargs["greater_is_better"] = True

if "report_to" in ta_params:
    kwargs["report_to"] = ["none"]

hf_log_dir = PROJECT_ROOT / "data" / "logs" / "hf_reranker" / DOMAIN
hf_log_dir.mkdir(parents=True, exist_ok=True)
if "logging_dir" in ta_params:
    kwargs["logging_dir"] = str(hf_log_dir)

logger.info("TrainingArguments kwargs used:")
for k, v in kwargs.items():
    logger.info(f"  {k} = {v}")

training_args = TrainingArguments(**kwargs)

# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logger.info(">>> Starting PAIR-WISE reranker training loop...")
train_start = time.perf_counter()
trainer.train()
train_end = time.perf_counter()
logger.info(f"Training loop finished in {train_end - train_start:.2f} seconds.")

logger.info(">>> Evaluating on test set...")
metrics = trainer.evaluate(test_dataset)
logger.info(f"Test metrics: {metrics}")

best_dir = SAVE_DIR / "best"
best_dir.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(best_dir))
tokenizer.save_pretrained(str(best_dir))
logger.info(f"Saved best reranker model to: {best_dir}")

meta = {
    "model_name": MODEL_NAME,
    "domain": DOMAIN,
    "num_train_examples": len(train_dataset),
    "num_dev_examples": len(dev_dataset),
    "num_test_examples": len(test_dataset),
    "seed": SEED,
    "train_jsonl": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
    "dev_jsonl": str(DEV_PATH.relative_to(PROJECT_ROOT)),
    "test_jsonl": str(TEST_PATH.relative_to(PROJECT_ROOT)),
}
meta_path = best_dir / "reranker_info.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
logger.info(f"Saved reranker metadata to: {meta_path}")

end_time = time.perf_counter()
logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")
logger.info("✅ 11_train_span_reranker (pair-based, L6) completed successfully.")