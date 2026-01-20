#!/usr/bin/env python3
# scripts/Span_Identification/15_train_span_identifier.py
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------
# Ensure project root on PYTHONPATH
# ---------------------------------------------------
PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.utils.logging_utils import create_logger
from src.span_identifier.prep.io_jsonl import read_jsonl, write_json
from src.span_identifier.prep.token_labels_bilou import make_label_map, spans_to_bilou_for_tokens, Span


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_domain_from_pipeline_config(pipeline_cfg_path: Path) -> str:
    """
    Reads domain from configs/pipeline_span_id.yaml
    Expected format:
      domain: money-heist
    """
    if not pipeline_cfg_path.exists():
        raise FileNotFoundError(f"Missing pipeline config: {pipeline_cfg_path}")

    cfg = load_yaml(str(pipeline_cfg_path))
    domain = cfg.get("domain")
    if not domain:
        raise ValueError(f"Missing required key 'domain' in {pipeline_cfg_path}")
    return str(domain).strip()


@dataclass
class Item:
    doc_id: str
    text: str
    spans: List[Span]


class SpanIdDataset(Dataset):
    def __init__(self, items: List[Item]):
        self.items = items

    def __len__(self):  # type: ignore
        return len(self.items)

    def __getitem__(self, idx: int):  # type: ignore
        it = self.items[idx]
        return {
            "doc_id": it.doc_id,
            "text": it.text,
            "spans": [{"start": s.start, "end": s.end, "type": s.type} for s in it.spans],
        }


class Collator:
    def __init__(self, tokenizer, label_map: Dict[str, int], max_length: int = 512):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]):
        texts = [b["text"] for b in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        all_labels: List[List[int]] = []
        for i, b in enumerate(batch):
            spans = [Span(**s) for s in b["spans"]]
            offsets = enc["offset_mapping"][i].tolist()
            attn = enc["attention_mask"][i].tolist()
            special = enc["special_tokens_mask"][i].tolist()

            labels = spans_to_bilou_for_tokens(
                offsets=offsets,
                spans=spans,
                label_map=self.label_map,
                ignore_label_id=-100,
            )

            # Ignore padding + special tokens robustly
            labels = [
                (-100 if (attn[j] == 0 or special[j] == 1) else lab)
                for j, lab in enumerate(labels)
            ]
            all_labels.append(labels)

        labels_t = torch.tensor(all_labels, dtype=torch.long)
        enc.pop("offset_mapping")        # not fed into model
        enc.pop("special_tokens_mask")   # not fed into model
        enc["labels"] = labels_t
        return enc


def build_training_args(out_dir: Path, tcfg: Dict[str, Any]) -> TrainingArguments:
    common_kwargs = dict(
        output_dir=str(out_dir),
        learning_rate=float(tcfg.get("lr", 2e-5)),
        per_device_train_batch_size=int(tcfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(tcfg.get("eval_batch_size", 8)),
        num_train_epochs=float(tcfg.get("epochs", 3)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        logging_steps=int(tcfg.get("logging_steps", 50)),
        eval_steps=int(tcfg.get("eval_steps", 200)),
        save_steps=int(tcfg.get("save_steps", 200)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        fp16=bool(tcfg.get("fp16", False)),
        report_to=tcfg.get("report_to", "none"),
        remove_unused_columns=False,
        load_best_model_at_end=bool(tcfg.get("load_best_model_at_end", False)),
        metric_for_best_model=str(tcfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(tcfg.get("greater_is_better", False)),
        seed=int(tcfg.get("seed", 42)),
        data_seed=int(tcfg.get("data_seed", 42)),
    )

    try:
        return TrainingArguments(**common_kwargs, evaluation_strategy="steps")
    except TypeError:
        return TrainingArguments(**common_kwargs, eval_strategy="steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing dataset JSONLs (e.g., data/span_identification/<domain>)",
    )
    ap.add_argument(
        "--suffix",
        default="_paragraph_nopunct",
        help="Dataset suffix used in filenames (e.g., _paragraph_nopunct)",
    )
    ap.add_argument("--cfg", required=True, help="configs/span_identi.yaml")
    ap.add_argument("--out", required=True, help="output dir for checkpoints/logs/models")
    ap.add_argument(
        "--pipeline_config",
        default=str(PROJECT_ROOT / "configs" / "pipeline_span_id.yaml"),
        help="configs/pipeline_span_id.yaml (must contain: domain: <slug>)",
    )
    args = ap.parse_args()

    # ----------------------------
    # Resolve paths + domain (ONLY from pipeline_span_id.yaml)
    # ----------------------------
    data_dir = Path(args.data_dir)
    suffix = args.suffix.strip()
    domain = load_domain_from_pipeline_config(Path(args.pipeline_config))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Logger (same style as script 14)
    # ----------------------------
    LOG_DIR = PROJECT_ROOT / "data" / "logs" / "span_identification" / domain
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger, log_file = create_logger(LOG_DIR, f"15_train_span_identifier_{domain}")
    logger.info(f"Log file: {log_file}")

    logger.info("==== Span Identifier Training ====")
    logger.info(f"DOMAIN: {domain}")
    logger.info(f"PIPELINE_CFG: {args.pipeline_config}")
    logger.info(f"DATA_DIR: {data_dir}")
    logger.info(f"SUFFIX: {suffix}")
    logger.info(f"CFG: {args.cfg}")
    logger.info(f"OUT_DIR: {out_dir}")
    logger.info(f"LOG_DIR: {LOG_DIR}")

    # ----------------------------
    # Load config
    # ----------------------------
    cfg = load_yaml(args.cfg)
    model_name = cfg["model_name"]
    max_length = int(cfg.get("max_length", 512))
    tcfg = cfg.get("train", {})

    logger.info("Config knobs:")
    logger.info(f"  model_name={model_name}")
    logger.info(f"  max_length={max_length}")
    logger.info(f"  train.lr={tcfg.get('lr', 2e-5)}")
    logger.info(f"  train.batch_size={tcfg.get('batch_size', 8)}")
    logger.info(f"  train.eval_batch_size={tcfg.get('eval_batch_size', 8)}")
    logger.info(f"  train.epochs={tcfg.get('epochs', 3)}")
    logger.info(f"  train.fp16={tcfg.get('fp16', False)}")
    logger.info(f"  train.eval_steps={tcfg.get('eval_steps', 200)}")
    logger.info(f"  train.save_steps={tcfg.get('save_steps', 200)}")

    # ----------------------------
    # Load data
    # ----------------------------
    train_path = data_dir / f"train{suffix}.jsonl"
    dev_path = data_dir / f"dev{suffix}.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_path}")

    logger.info(f"Reading train: {train_path}")
    logger.info(f"Reading dev:   {dev_path}")

    train_rows = read_jsonl(str(train_path))
    dev_rows = read_jsonl(str(dev_path))

    def to_items(rows: List[Dict[str, Any]]) -> List[Item]:
        items: List[Item] = []
        for r in rows:
            spans = [
                Span(start=int(s["start"]), end=int(s["end"]), type=s.get("type", "internal"))
                for s in r.get("spans", [])
            ]
            items.append(Item(doc_id=str(r["doc_id"]), text=str(r["text"]), spans=spans))
        return items

    train_items = to_items(train_rows)
    dev_items = to_items(dev_rows)

    train_ds = SpanIdDataset(train_items)
    dev_ds = SpanIdDataset(dev_items)

    logger.info(f"Loaded examples: train={len(train_ds)} dev={len(dev_ds)}")

    # ----------------------------
    # Labels / model
    # ----------------------------
    label_map = make_label_map()
    id2label = {v: k for k, v in label_map.items()}

    logger.info(f"Label map size: {len(label_map)} | labels={list(label_map.keys())}")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_map),
        id2label=id2label,
        label2id=label_map,
    )

    # ----------------------------
    # Training args
    # ----------------------------
    tr_args = build_training_args(out_dir=out_dir, tcfg=tcfg)
    logger.info("TrainingArguments created successfully.")
    logger.info(f"  output_dir={tr_args.output_dir}")

    collator = Collator(tok, label_map=label_map, max_length=max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum()
        total = mask.sum()
        acc = float(correct) / float(total) if total > 0 else 0.0
        return {"token_acc": acc}

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=tok,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    logger.info(f"Saving model + tokenizer to: {out_dir}")
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    meta = {
        "domain": domain,
        "data_dir": str(data_dir),
        "suffix": suffix,
        "train_file": str(train_path),
        "dev_file": str(dev_path),
        "label_map": label_map,
        "id2label": id2label,
        "model_name": model_name,
        "max_length": max_length,
        "train_config": tcfg,
    }
    write_json(str(out_dir / "label_map.json"), meta)

    logger.info("✅ 15_train_span_identifier completed successfully.")
    logger.info(f"Saved to: {out_dir}")
    logger.info(f"Log: {log_file}")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
