# src/span_identifier/model.py

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from src.utils.logging_utils import create_logger
from src.span_identifier.preprocess import BILOU_LABELS, LABEL2ID

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
import csv
from datetime import datetime

def log_results_to_csv(span_cfg: dict, metrics: dict):
    """
    Append experiment results to CSV with configuration details.
    """
    results_dir = Path(span_cfg.get("results_dir", "results/span_identification"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = Path(span_cfg.get("results_csv", results_dir / "all_experiments.csv"))
    
    # Build experiment name
    domain = span_cfg.get("domain", "unknown")
    model_name = span_cfg.get("model_name", "bert-base-uncased").split("/")[-1]  # just model name
    level = span_cfg.get("level", "paragraph")
    normalize_punct = span_cfg.get("normalize_punctuation", False)
    punc_str = "punc" if normalize_punct else "no_punc"
    
    exp_name = f"{domain}_{model_name}_{level}_{punc_str}"
    
    # Extract metrics
    row = {
        "experiment_name": exp_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domain": domain,
        "model": model_name,
        "level": level,
        "normalize_punctuation": normalize_punct,
        "eval_f1": metrics.get("eval_f1_seqeval", metrics.get("eval_f1", 0.0)),
        "eval_precision": metrics.get("eval_precision_seqeval", metrics.get("eval_precision", 0.0)),
        "eval_recall": metrics.get("eval_recall_seqeval", metrics.get("eval_recall", 0.0)),
        "eval_loss": metrics.get("eval_loss", 0.0),
        "num_epochs": span_cfg["train"].get("num_epochs", 0),
        "learning_rate": span_cfg["train"].get("learning_rate", 0.0),
        "batch_size": span_cfg["train"].get("batch_size", 0),
    }
    
    # Check if file exists
    file_exists = results_csv.exists()
    
    # Write to CSV
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    return exp_name

def _load_jsonl(path: Path) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _build_hf_datasets(span_cfg: dict, tokenizer) -> DatasetDict:
    train_path = Path(span_cfg["train_file"])
    dev_path = Path(span_cfg["dev_file"])
    test_path = Path(span_cfg["test_file"])

    train_ds = _load_jsonl(train_path)
    dev_ds = _load_jsonl(dev_path)
    test_ds = _load_jsonl(test_path)

    max_seq_length = int(span_cfg.get("max_seq_length", 512))
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def pad_features(ex):
        """
        JSONL already has input_ids, attention_mask, label_ids as integer lists.
        Just pad/truncate to max_seq_length.
        """
        input_ids = ex["input_ids"]
        attention_mask = ex["attention_mask"]
        labels = ex["label_ids"]

        # Truncate if too long
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            labels = labels[:max_seq_length]
        else:
            # Pad if too short
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [LABEL2ID["O"]] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_ds = train_ds.map(pad_features, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(pad_features, remove_columns=dev_ds.column_names)
    test_ds = test_ds.map(pad_features, remove_columns=test_ds.column_names)

    return DatasetDict(train=train_ds, validation=dev_ds, test=test_ds)

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    preds_list = preds.tolist()
    labels_list = labels.tolist()

    pred_tags = [[ID2LABEL[id_] for id_ in seq] for seq in preds_list]
    true_tags = [[ID2LABEL[id_] for id_ in seq] for seq in labels_list]

    return {
        "f1_seqeval": f1_score(true_tags, pred_tags),
        "precision_seqeval": precision_score(true_tags, pred_tags),
        "recall_seqeval": recall_score(true_tags, pred_tags),
    }


def train_model_from_cfg(span_cfg: dict):
    """
    Train a token classification model for BILOU span identification.
    """
    log_dir = Path(span_cfg["log_dir"])
    logger, log_file = create_logger(log_dir, script_name="03_train_span_identifier")
    logger.info("Step 3: Training span identification model")

    model_name = span_cfg["model_name"]
    num_labels = int(span_cfg.get("num_labels", len(BILOU_LABELS)))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    datasets = _build_hf_datasets(span_cfg, tokenizer)

    logger.info(
        f"Loaded datasets: train={len(datasets['train'])}, "
        f"dev={len(datasets['validation'])}, test={len(datasets['test'])}"
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    out_dir = Path(span_cfg["token_dataset_dir"]).parent / "models" / span_cfg["domain"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = span_cfg["train"]

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        # logging
        logging_dir=str(log_dir),
        logging_strategy="steps",  # log every logging_steps steps [web:114][web:115]
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        # simple step-based eval/save (no evaluation_strategy kw)
        eval_steps=int(train_cfg.get("eval_steps", 500)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        # optimization
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        per_device_eval_batch_size=int(train_cfg["batch_size"]),
        num_train_epochs=int(train_cfg["num_epochs"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        seed=int(train_cfg["seed"]),
        report_to=[],  # or ["wandb"] if enabled
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    logger.info(f"Training complete. Best model saved to: {out_dir}")


def evaluate_model_from_cfg(span_cfg: dict):
    """
    Evaluate the trained model on the test split and log seqeval F1/precision/recall.
    """
    log_dir = Path(span_cfg["log_dir"])
    logger, log_file = create_logger(log_dir, script_name="04_eval_span_identifier")
    logger.info("Step 4: Evaluating span identification model on test set")

    model_name = span_cfg["model_name"]
    num_labels = int(span_cfg.get("num_labels", len(BILOU_LABELS)))

    model_dir = Path(span_cfg["token_dataset_dir"]).parent / "models" / span_cfg["domain"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    datasets = _build_hf_datasets(span_cfg, tokenizer)
    test_ds = datasets["test"]

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    logger.info(f"Test examples: {len(test_ds)}")
    metrics = trainer.evaluate(eval_dataset=test_ds)
    logger.info(f"Test metrics (token/BILOU-level): {metrics}")
    
    # ADD THIS: Log to CSV
    exp_name = log_results_to_csv(span_cfg, metrics)
    logger.info(f"Results saved to CSV as: {exp_name}")
