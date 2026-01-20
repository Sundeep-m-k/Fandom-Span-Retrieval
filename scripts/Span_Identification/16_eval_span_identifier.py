#!/usr/bin/env python3
# scripts/Span_Identification/16_eval_span_identifier.py
"""
Evaluate a token-classification span identifier (BILOU tagging) on a JSONL dataset.

CONFIG-DRIVEN VERSION (NO CLI)

- Reads pipeline config from env PIPELINE_CONFIG if set, else defaults to:
    /data/sundeep/Fandom_SI/configs/pipeline_span_id.yaml

Token-level (micro):
  - accuracy
  - TP/FP/FN/TN (LINK vs O)
  - precision / recall / F1
  - specificity
  - ROC-AUC using token scores p(LINK)=1-P(O)

Span-level:
  Macro over docs (within visible truncated window):
    - exact_span_f1_macro (exact char-span match)
    - iou_f1_macro (IoU-based span F1 with greedy matching)
    - span_recall_at_k (1 if any top-K pred span matches any gold at IoU>=thr, else 0; macro avg)
    - pct_gold_empty
    - pct_pred_empty

  Micro over full split (within visible truncated window):
    - exact_span_precision_micro / exact_span_recall_micro / exact_span_f1_micro
    - exact_span_tp_micro / exact_span_fp_micro / exact_span_fn_micro

Also supports:
  - optional per-doc predictions JSONL dump
  - CSV append into results CSV
  - domain-scoped logging
"""

from __future__ import annotations

import sys
import os
import re
from pathlib import Path

# ---------------------------------------------------
# Ensure project root on PYTHONPATH
# ---------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/data/sundeep/Fandom_SI"))
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
from typing import Dict, List, Tuple, Optional, Any

import yaml
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.utils.logging_utils import create_logger
from src.span_identifier.prep.io_jsonl import read_jsonl
from src.span_identifier.prep.token_labels_bilou import make_label_map, spans_to_bilou_for_tokens, Span


# ============================================================
# Template + YAML resolution (safe + multipass + paths.dot support)
# ============================================================

def normalize_template(s: str) -> str:
    # Convert {paths.data_dir} -> {paths[data_dir]} for dict-based formatting
    return re.sub(r"\{paths\.([A-Za-z0-9_]+)\}", r"{paths[\1]}", s)


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def resolve_templates(obj: Any, ctx: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        s = normalize_template(obj)
        return s.format_map(SafeDict(**ctx))
    if isinstance(obj, dict):
        return {k: resolve_templates(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_templates(v, ctx) for v in obj]
    return obj


def load_pipeline_cfg(pipeline_path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}

    if "domain" not in raw or not str(raw["domain"]).strip():
        raise ValueError(f"Missing required key 'domain' in pipeline config: {pipeline_path}")

    project_root = Path(raw.get("project_root", str(PROJECT_ROOT)))
    domain = str(raw["domain"]).strip()
    suffix = str(raw.get("suffix", "")).strip()

    # Base ctx
    ctx: Dict[str, Any] = {
        "project_root": str(project_root),
        "domain": domain,
        "suffix": suffix,
        "paths": {},
    }

    # Resolve raw['paths'] in a multi-pass loop (handles forward refs like {exp_root})
    raw_paths = raw.get("paths", {}) or {}
    resolved_paths = dict(raw_paths)

    for _ in range(10):
        candidate = resolve_templates(resolved_paths, ctx)

        # inject into ctx for both {paths[...]} and {exp_root}-style
        ctx["paths"] = candidate
        for k, v in candidate.items():
            ctx[k] = v

        if candidate == resolved_paths:
            resolved_paths = candidate
            break
        resolved_paths = candidate

    ctx["paths"] = resolved_paths
    for k, v in resolved_paths.items():
        ctx[k] = v

    # Now resolve the full YAML using the final ctx
    resolved_all = resolve_templates(raw, ctx)
    return resolved_all


# ============================================================
# Helper: label map loading
# ============================================================

def safe_load_label_map(ckpt_dir: Path) -> Dict[str, int]:
    lm_path = ckpt_dir / "label_map.json"
    if lm_path.exists():
        obj = json.loads(lm_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "label_map" in obj and isinstance(obj["label_map"], dict):
            return {k: int(v) for k, v in obj["label_map"].items()}
    return make_label_map()


# ============================================================
# Helper: decode predicted token tags back into char spans
# ============================================================

def decode_bilou_spans_from_ids(
    offsets: List[List[int]],
    pred_ids: List[int],
    id2label: Dict[Any, Any],
    ignore_label_id: int = -100,
) -> List[Tuple[int, int]]:

    def lab(i: int) -> str:
        if i == ignore_label_id:
            return "O"
        if i in id2label:
            return str(id2label[i])
        if str(i) in id2label:
            return str(id2label[str(i)])
        return str(i)

    tags = [lab(i) for i in pred_ids]

    spans: List[Tuple[int, int]] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    def flush():
        nonlocal cur_start, cur_end
        if cur_start is not None and cur_end is not None and cur_end > cur_start:
            spans.append((cur_start, cur_end))
        cur_start = None
        cur_end = None

    for (s, e), t in zip(offsets, tags):
        if s == 0 and e == 0:
            continue
        if e <= s:
            continue

        t0 = t.split("-", 1)[0]

        if t0 == "U":
            spans.append((s, e))
            cur_start = None
            cur_end = None
        elif t0 == "B":
            flush()
            cur_start, cur_end = s, e
        elif t0 == "I":
            if cur_start is None:
                cur_start, cur_end = s, e
            else:
                cur_end = e
        elif t0 == "L":
            if cur_start is None:
                spans.append((s, e))
            else:
                cur_end = e
                flush()
        else:
            flush()

    flush()
    return spans


# ============================================================
# Token-level metrics
# ============================================================

def micro_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def specificity(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) else 0.0


def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if y_true.min() == y_true.max():
        return float("nan")

    order = np.argsort(-y_score)
    y_true = y_true[order]

    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        return float("nan")

    tps = 0
    fps = 0
    auc = 0.0
    prev_fpr, prev_tpr = 0.0, 0.0

    for yt in y_true:
        if yt == 1:
            tps += 1
        else:
            fps += 1
        tpr = tps / P
        fpr = fps / N
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr

    return float(auc)


# ============================================================
# Span-level helpers
# ============================================================

def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def exact_span_f1_doc(gold: List[Tuple[int, int]], pred: List[Tuple[int, int]]) -> float:
    gs, ps = set(gold), set(pred)
    tp = len(gs & ps)
    fp = len(ps - gs)
    fn = len(gs - ps)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def iou_f1_doc(gold: List[Tuple[int, int]], pred: List[Tuple[int, int]], thr: float) -> float:
    used_g = set()
    tp = 0

    for pspan in pred:
        best, best_j = -1.0, None
        for j, gspan in enumerate(gold):
            if j in used_g:
                continue
            v = iou(pspan, gspan)
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= thr:
            tp += 1
            used_g.add(best_j)

    fp = len(pred) - tp
    fn = len(gold) - tp

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def span_recall_at_k_doc(
    gold: List[Tuple[int, int]],
    pred_spans: List[Tuple[int, int]],
    pred_scores: List[float],
    k: int,
    thr_iou: float,
) -> float:
    if len(gold) == 0:
        return 0.0
    if not pred_spans:
        return 0.0

    order = np.argsort(-np.array(pred_scores))
    topk_idx = order[: max(1, min(k, len(pred_spans)))]

    for i in topk_idx:
        p = pred_spans[i]
        for g in gold:
            if iou(p, g) >= thr_iou:
                return 1.0
    return 0.0


def compute_max_visible_char(offsets: List[List[int]], attn: List[int]) -> int:
    m = 0
    for (s, e), am in zip(offsets, attn):
        if am == 0:
            continue
        if s == 0 and e == 0:
            continue
        if e > m:
            m = e
    return m


# ============================================================
# CSV saving
# ============================================================

def append_metrics_csv(row: Dict[str, Any], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists()

    if file_exists:
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        fieldnames = sorted(set(existing_header) | set(row.keys()))
        if set(fieldnames) != set(existing_header):
            with out_csv.open("r", encoding="utf-8", newline="") as f:
                dr = csv.DictReader(f)
                old_rows = list(dr)
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                dw = csv.DictWriter(f, fieldnames=fieldnames)
                dw.writeheader()
                for r in old_rows:
                    dw.writerow(r)
                dw.writerow(row)
            return
    else:
        fieldnames = sorted(row.keys())

    with out_csv.open("a", encoding="utf-8", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            dw.writeheader()
        dw.writerow(row)


# ============================================================
# Main (config-driven)
# ============================================================

def main():
    pipeline_path = os.environ.get("PIPELINE_CONFIG")
    if pipeline_path:
        pipe_path = Path(pipeline_path)
    else:
        pipe_path = PROJECT_ROOT / "configs" / "pipeline_span_id.yaml"

    if not pipe_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {pipe_path}")

    cfg = load_pipeline_cfg(pipe_path)

    project_root = Path(cfg.get("project_root", str(PROJECT_ROOT)))
    domain = str(cfg["domain"]).strip()
    suffix = str(cfg.get("suffix", "")).strip()

    paths = cfg.get("paths", {}) or {}
    ev = cfg.get("eval", {}) or {}

    # Required runtime args from resolved config
    data_dir = Path(ev.get("data_dir", paths.get("data_dir", project_root / "data" / "span_identification" / domain)))
    ckpt_dir = Path(ev.get("ckpt", paths.get("ckpt_dir", "")))
    split = str(ev.get("split", "test"))
    max_length = int(ev.get("max_length", 256))
    iou_threshold = float(ev.get("iou_threshold", 0.5))
    span_recall_k = int(ev.get("span_recall_k", 5))

    results_csv = Path(ev.get("results_csv", paths.get("results_csv", project_root / "results" / "span_identifier_metrics.csv")))
    save_preds = ev.get("save_preds", paths.get("preds_test", None))

    if not ckpt_dir or str(ckpt_dir).strip() == "":
        raise ValueError("Missing checkpoint dir in pipeline config. Set eval.ckpt or paths.ckpt_dir.")

    # Ensure dirs
    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    if save_preds:
        Path(save_preds).parent.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(paths.get("logs_dir", project_root / "data" / "logs" / "span_identification" / domain))
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger, log_file = create_logger(logs_dir, f"16_eval_span_identifier_{domain}")
    logger.info(f"Log file: {log_file}")

    logger.info("==== Span Identifier Eval ====")
    logger.info(f"PIPELINE_CFG: {pipe_path}")
    logger.info(f"DOMAIN: {domain}")
    logger.info(f"DATA_DIR: {data_dir}")
    logger.info(f"SPLIT: {split}")
    logger.info(f"SUFFIX: {suffix}")
    logger.info(f"CKPT: {ckpt_dir}")
    logger.info(f"MAX_LENGTH: {max_length}")
    logger.info(f"IOU_THRESHOLD: {iou_threshold}")
    logger.info(f"SPAN_RECALL_K: {span_recall_k}")
    logger.info(f"RESULTS_CSV: {results_csv}")
    if save_preds:
        logger.info(f"SAVE_PREDS: {save_preds}")

    # Load data
    filename = f"{split}{suffix}.jsonl" if suffix else f"{split}.jsonl"
    data_path = Path(data_dir) / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset split file not found: {data_path}")

    logger.info(f"Reading dataset: {data_path}")
    rows = read_jsonl(str(data_path))
    logger.info(f"Docs loaded: {len(rows)}")

    # Load checkpoint model/tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(str(ckpt_dir), local_files_only=True)
    model.to(device)
    model.eval()

    try:
        tok = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True, local_files_only=True)
        logger.info("Tokenizer loaded from checkpoint.")
    except Exception:
        base_name = getattr(model.config, "_name_or_path", None) or "bert-base-uncased"
        tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
        logger.warning(f"Tokenizer not found in checkpoint. Using base tokenizer: {base_name}")

    label_map = safe_load_label_map(Path(ckpt_dir))
    id2label = getattr(model.config, "id2label", {}) or {}

    label2id_cfg = getattr(model.config, "label2id", None)
    if isinstance(label2id_cfg, dict) and "O" in label2id_cfg:
        o_id = int(label2id_cfg["O"])
    else:
        if "O" not in label_map:
            raise ValueError("label_map must contain 'O'.")
        o_id = int(label_map["O"])

    logger.info(f"Using o_id={o_id} | num_labels={model.config.num_labels}")

    pred_f = open(save_preds, "w", encoding="utf-8") if save_preds else None

    # Token-level accumulators
    token_correct = 0
    token_total = 0
    tp = fp = fn = tn = 0
    auc_y_true: List[int] = []
    auc_y_score: List[float] = []

    # Span-level accumulators
    exact_f1_list: List[float] = []
    iou_f1_list: List[float] = []
    srk_list: List[float] = []
    gold_empty = 0
    pred_empty = 0

    exact_tp_micro = 0
    exact_fp_micro = 0
    exact_fn_micro = 0

    thr_iou = float(iou_threshold)
    k = int(span_recall_k)

    logger.info("Starting evaluation loop...")
    with torch.no_grad():
        for idx, r in enumerate(rows, start=1):
            text = r.get("text", "")
            doc_id = r.get("doc_id", "")

            gold_spans = [
                Span(start=int(s["start"]), end=int(s["end"]), type=s.get("type", "internal"))
                for s in r.get("spans", [])
            ]

            enc = tok(
                text,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_offsets_mapping=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            offsets = enc["offset_mapping"][0].tolist()
            attn = enc["attention_mask"][0].tolist()

            gold_labels = spans_to_bilou_for_tokens(
                offsets=offsets,
                spans=gold_spans,
                label_map=label_map,
                ignore_label_id=-100,
            )

            enc_no_offsets = {kk: vv.to(device) for kk, vv in enc.items() if kk != "offset_mapping"}
            out = model(**enc_no_offsets)
            logits = out.logits[0]

            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()

            probs = torch.softmax(logits, dim=-1)
            if not (0 <= o_id < probs.shape[-1]):
                raise ValueError(f"o_id={o_id} out of range for num_labels={probs.shape[-1]}")
            pos_prob = (1.0 - probs[:, o_id]).detach().cpu().numpy().tolist()

            # Token metrics
            for gl, pr, sc, (s, e), am in zip(gold_labels, pred_ids, pos_prob, offsets, attn):
                if am == 0:
                    continue
                if gl == -100:
                    continue
                if s == 0 and e == 0:
                    continue
                if e <= s:
                    continue

                token_total += 1
                token_correct += int(gl == pr)

                gold_is_link = int(gl != o_id)
                pred_is_link = int(pr != o_id)

                if gold_is_link == 1 and pred_is_link == 1:
                    tp += 1
                elif gold_is_link == 0 and pred_is_link == 1:
                    fp += 1
                elif gold_is_link == 1 and pred_is_link == 0:
                    fn += 1
                else:
                    tn += 1

                auc_y_true.append(gold_is_link)
                auc_y_score.append(float(sc))

            # Span metrics
            max_visible_char = compute_max_visible_char(offsets, attn)

            gold_char = [
                (sp.start, sp.end)
                for sp in gold_spans
                if 0 <= sp.start < sp.end <= max_visible_char
            ]

            pred_char = decode_bilou_spans_from_ids(
                offsets=offsets,
                pred_ids=pred_ids,
                id2label=id2label,
                ignore_label_id=-100,
            )
            pred_char = [(a, b) for (a, b) in pred_char if 0 <= a < b <= max_visible_char]

            if len(gold_char) == 0:
                gold_empty += 1
            if len(pred_char) == 0:
                pred_empty += 1

            exact_f1_list.append(exact_span_f1_doc(gold_char, pred_char))
            iou_f1_list.append(iou_f1_doc(gold_char, pred_char, thr=thr_iou))

            gs = set(gold_char)
            ps = set(pred_char)
            exact_tp_micro += len(gs & ps)
            exact_fp_micro += len(ps - gs)
            exact_fn_micro += len(gs - ps)

            tok_pairs = [((int(s), int(e)), float(sc)) for (s, e), sc in zip(offsets, pos_prob)]
            span_scores: List[float] = []
            for (a, b) in pred_char:
                inside = [
                    sc for (ts, te), sc in tok_pairs
                    if not (ts == 0 and te == 0) and te > ts and (ts < b and te > a)
                ]
                span_scores.append(float(np.mean(inside)) if inside else 0.0)

            srk_list.append(span_recall_at_k_doc(gold_char, pred_char, span_scores, k=k, thr_iou=thr_iou))

            if pred_f is not None:
                out_row = {
                    "doc_id": doc_id,
                    "max_visible_char": max_visible_char,
                    "text": text[:max_visible_char],
                    "gold_spans": [{"start": a, "end": b, "type": "internal"} for (a, b) in gold_char],
                    "pred_spans": [{"start": a, "end": b, "type": "internal"} for (a, b) in pred_char],
                }
                pred_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            if idx % 200 == 0:
                logger.info(f"Progress: {idx}/{len(rows)} docs evaluated...")

    if pred_f is not None:
        pred_f.close()
        logger.info(f"Saved predictions to: {save_preds}")

    tok_acc = token_correct / token_total if token_total else 0.0
    prf = micro_prf(tp, fp, fn)
    spec = specificity(tn, fp)

    y_true = np.array(auc_y_true, dtype=int)
    y_score = np.array(auc_y_score, dtype=float)
    auc = roc_auc_np(y_true, y_score)

    exact_macro = float(np.mean(exact_f1_list)) if exact_f1_list else 0.0
    iou_macro = float(np.mean(iou_f1_list)) if iou_f1_list else 0.0
    srk_macro = float(np.mean(srk_list)) if srk_list else 0.0

    pct_gold_empty = gold_empty / len(rows) if rows else 0.0
    pct_pred_empty = pred_empty / len(rows) if rows else 0.0

    exact_p_micro = exact_tp_micro / (exact_tp_micro + exact_fp_micro) if (exact_tp_micro + exact_fp_micro) else 0.0
    exact_r_micro = exact_tp_micro / (exact_tp_micro + exact_fn_micro) if (exact_tp_micro + exact_fn_micro) else 0.0
    exact_f1_micro = (2 * exact_p_micro * exact_r_micro / (exact_p_micro + exact_r_micro)) if (exact_p_micro + exact_r_micro) else 0.0

    summary: Dict[str, Any] = {
        "domain": domain,
        "split": split,
        "suffix": suffix,
        "ckpt": str(ckpt_dir),
        "max_length": max_length,
        "iou_threshold": thr_iou,
        "span_recall_k": k,

        "n_docs": len(rows),
        "n_tokens_eval": token_total,

        "token_accuracy": tok_acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "specificity": spec,
        "roc_auc": auc,

        "exact_span_f1_macro": exact_macro,

        "exact_span_precision_micro": exact_p_micro,
        "exact_span_recall_micro": exact_r_micro,
        "exact_span_f1_micro": exact_f1_micro,
        "exact_span_tp_micro": exact_tp_micro,
        "exact_span_fp_micro": exact_fp_micro,
        "exact_span_fn_micro": exact_fn_micro,

        "iou_f1_macro": iou_macro,
        "span_recall_at_k": srk_macro,

        "pct_gold_empty": pct_gold_empty,
        "pct_pred_empty": pct_pred_empty,
    }

    print("========== Eval ==========")
    print(f"Domain: {domain}")
    print(f"Split: {split}")
    print(f"Docs: {len(rows)}")
    print(f"Tokens eval: {token_total}")
    print("")
    print(f"Token accuracy: {tok_acc:.4f}  (correct={token_correct} total={token_total})")
    print(
        f"Token LINK P/R/F1: {prf['precision']:.4f} {prf['recall']:.4f} {prf['f1']:.4f}  "
        f"(TP={tp} FP={fp} FN={fn} TN={tn})"
    )
    print(f"Token specificity: {spec:.4f}")
    print(f"Token ROC-AUC: {auc:.4f}")
    print("")
    print(f"Span exact F1 (macro): {exact_macro:.4f}")
    print(
        f"Span exact P/R/F1 (micro): {exact_p_micro:.4f} {exact_r_micro:.4f} {exact_f1_micro:.4f} "
        f"(TP={exact_tp_micro} FP={exact_fp_micro} FN={exact_fn_micro})"
    )
    print(f"Span IoU-F1@{thr_iou:.2f} (macro): {iou_macro:.4f}")
    print(f"Span Recall@{k} (macro, IoU@{thr_iou:.2f}): {srk_macro:.4f}")
    print("")
    print(f"pct_gold_empty: {pct_gold_empty:.4f}")
    print(f"pct_pred_empty: {pct_pred_empty:.4f}")

    logger.info("==== Eval Summary ====")
    for kk in [
        "token_accuracy", "precision", "recall", "f1", "specificity", "roc_auc",
        "exact_span_f1_macro",
        "exact_span_precision_micro", "exact_span_recall_micro", "exact_span_f1_micro",
        "iou_f1_macro", "span_recall_at_k",
        "pct_gold_empty", "pct_pred_empty",
    ]:
        logger.info(f"{kk}: {summary[kk]}")

    append_metrics_csv(summary, Path(results_csv))
    logger.info(f"Appended metrics row to CSV: {results_csv}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
