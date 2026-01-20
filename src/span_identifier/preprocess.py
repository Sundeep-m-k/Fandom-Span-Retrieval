# src/span_identifier/preprocess.py

import json
from pathlib import Path
import hashlib
from typing import List, Dict, Any
import unicodedata
import re
import pandas as pd
from transformers import AutoTokenizer

from src.utils.logging_utils import create_logger

BILOU_LABELS = ["O", "B-SPAN", "I-SPAN", "L-SPAN", "U-SPAN"]
LABEL2ID = {label: i for i, label in enumerate(BILOU_LABELS)}



def md5_unit_interval(s: str) -> float:
    """Stable float in [0,1) from a string (for reproducible splits)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) / (2**128)
def normalize_punctuation(text: str) -> str:
    """Normalize punctuation variants to canonical forms."""
    if not text:
        return text
    
    # Unicode normalize
    text = unicodedata.normalize('NFKC', text)
    
    # Curly quotes → straight
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Dashes → hyphen
    text = text.replace('—', '-').replace('–', '-')
    
    # Ellipsis
    text = text.replace('…', '...')
    
    # Multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text
# In preprocess.py, in assign_bilou_labels:
def assign_bilou_labels(text, spans, tokenizer, max_seq_length):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=True,
    )
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    labels = [LABEL2ID["O"]] * len(input_ids)  # initialize as IDs, not strings

    for sp in spans:
        start_char = int(sp["start"])
        end_char = int(sp["end"])
        if end_char <= start_char:
            continue

        token_indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:
                continue
            if tok_end <= start_char or tok_start >= end_char:
                continue
            token_indices.append(i)

        if not token_indices:
            continue

        if len(token_indices) == 1:
            labels[token_indices[0]] = LABEL2ID["U-SPAN"]
        else:
            labels[token_indices[0]] = LABEL2ID["B-SPAN"]
            for ti in token_indices[1:-1]:
                labels[ti] = LABEL2ID["I-SPAN"]
            labels[token_indices[-1]] = LABEL2ID["L-SPAN"]

    return input_ids, attention_mask, labels  # return IDs, not tokens





def build_token_dataset_from_cfg(span_cfg: dict):
    """
    Convert paragraph-level master CSV into token-level BIO JSONL splits.
    """
    log_dir = Path(span_cfg["log_dir"])
    logger, log_file = create_logger(log_dir, script_name="02_build_token_dataset")
    logger.info("Step 2: Building token-level BIO dataset from paragraph master")

    paragraph_csv = Path(span_cfg["paragraph_master_csv"])
    if not paragraph_csv.exists():
        logger.error(f"Paragraph master CSV not found: {paragraph_csv}")
        raise FileNotFoundError(f"Paragraph master CSV not found: {paragraph_csv}")

    df = pd.read_csv(paragraph_csv)
    logger.info(f"Loaded paragraph master: {paragraph_csv} (rows={len(df)})")

    model_name = span_cfg["model_name"]
    max_seq_length = int(span_cfg.get("max_seq_length", 512))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    logger.info(f"Loaded tokenizer: {model_name}, max_seq_length={max_seq_length}")

    # ADD THIS: read punctuation normalization flag
    normalize_punct = span_cfg.get("normalize_punctuation", False)
    logger.info(f"Punctuation normalization: {normalize_punct}")

    # Output paths
    out_dir = Path(span_cfg["token_dataset_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(span_cfg["train_file"])
    dev_path = Path(span_cfg["dev_file"])
    test_path = Path(span_cfg["test_file"])

    # Open writers
    train_f = train_path.open("w", encoding="utf-8")
    dev_f = dev_path.open("w", encoding="utf-8")
    test_f = test_path.open("w", encoding="utf-8")

    # Split ratios
    train_ratio = 0.8
    dev_ratio = 0.1

    n_rows = len(df)
    logger.info("Starting paragraph -> token+BILOU conversion")

    num_examples = 0
    num_skipped_empty = 0

    for idx, row in df.iterrows():
        text = row["paragraph_text"]
        if not isinstance(text, str) or not text.strip():
            num_skipped_empty += 1
            continue

        # ADD THIS: apply normalization if enabled
        if normalize_punct:
            text = normalize_punctuation(text)

        # Parse spans_json
        spans_json = row.get("spans_json", "[]")
        try:
            spans = json.loads(spans_json)
        except json.JSONDecodeError:
            logger.warning(f"Row {idx}: invalid spans_json, skipping")
            num_skipped_empty += 1
            continue

        input_ids, attention_mask, labels = assign_bilou_labels(
            text=text,  # now normalized if flag is True
            spans=spans,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        example = {
            "article_id": row.get("article_id"),
            "page_name": row.get("page_name"),
            "section": row.get("section"),
            "paragraph_id": int(row.get("paragraph_id")),
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_ids": labels,
        }

        # Stable split key: article_id + paragraph_id
        key_str = f"{row.get('article_id')}::{row.get('paragraph_id')}"
        r = md5_unit_interval(key_str)

        if r < train_ratio:
            tgt = train_f
        elif r < train_ratio + dev_ratio:
            tgt = dev_f
        else:
            tgt = test_f

        tgt.write(json.dumps(example, ensure_ascii=False) + "\n")
        num_examples += 1

        if (idx + 1) % 500 == 0:
            logger.info(f"Processed {idx+1}/{n_rows} paragraphs -> {num_examples} examples")

    train_f.close()
    dev_f.close()
    test_f.close()

    logger.info(f"Token dataset written:")
    logger.info(f"  train: {train_path}")
    logger.info(f"  dev:   {dev_path}")
    logger.info(f"  test:  {test_path}")
    logger.info(f"Total examples written: {num_examples}")
    logger.info(f"Paragraphs skipped (empty/invalid): {num_skipped_empty}")

