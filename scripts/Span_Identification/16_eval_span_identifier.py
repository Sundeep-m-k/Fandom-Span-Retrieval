#!/usr/bin/env python3
# scripts/Span_Identification/16_eval_span_identifier.py
"""
Evaluate a token-classification span identifier (BILOU tagging) on a JSONL dataset.

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
    - pct_gold_empty (visible gold spans empty)
    - pct_pred_empty (visible predicted spans empty)

  Micro over full split (within visible truncated window):
    - exact_span_precision_micro / exact_span_recall_micro / exact_span_f1_micro
    - exact_span_tp_micro / exact_span_fp_micro / exact_span_fn_micro

Also supports:
  - optional per-doc predictions JSONL dump
  - CSV append into results CSV
  - domain-scoped logging (same create_logger pattern as other scripts)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------
# Ensure project root on PYTHONPATH
# ---------------------------------------------------
PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.utils.logging_utils import create_logger
from src.span_identifier.prep.io_jsonl import read_jsonl
from src.span_identifier.prep.token_labels_bilou import make_label_map, spans_to_bilou_for_tokens, Span


# ============================================================
# Helper: label map loading
# ============================================================

def safe_load_label_map(ckpt_dir: Path) -> Dict[str, int]:
    """
    Load label->id mapping from ckpt_dir/label_map.json if present.
    Training script writes a dict containing "label_map" key.
    """
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
    """
    Convert predicted token label IDs into character-level spans using BILOU decoding.
    Output spans are (start_char, end_char) in document character space.
    """

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

        t0 = t.split("-", 1)[0]  # "B-INTERNAL" -> "B"

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
    """
    Minimal ROC-AUC without sklearn.
    Returns NaN if only one class is present.
    """
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
    """
    Greedy matching: each gold used at most once.
    TP increments when best IoU >= thr.
    """
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
    """
    1 if any of top-K pred spans matches any gold at IoU>=thr, else 0.
    """
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
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing dataset JSONLs (e.g., data/span_identification/<domain>)",
    )
    ap.add_argument(
        "--suffix",
        default="",
        help="Dataset suffix used in filenames (e.g., _paragraph_nopunct). "
             "If provided, reads <split><suffix>.jsonl; else reads <split>.jsonl",
    )
    ap.add_argument("--ckpt", required=True, help="Checkpoint dir (trainer output)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    ap.add_argument("--span_recall_k", type=int, default=5)

    ap.add_argument("--lr", type=float, default=None, help="Optional: record lr into CSV (metadata).")
    ap.add_argument("--epoch", type=int, default=None, help="Optional: record epoch into CSV (metadata).")

    ap.add_argument(
        "--base_tokenizer",
        default=None,
        help="Optional base tokenizer name if checkpoint has no tokenizer files.",
    )
    ap.add_argument(
        "--save_preds",
        default=None,
        help="Optional path to write per-doc predictions JSONL for error analysis.",
    )
    ap.add_argument(
        "--results_csv",
        default=str(PROJECT_ROOT / "results" / "span_identifier_metrics.csv"),
        help="CSV path to append a single metrics row.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    suffix = args.suffix.strip()
    domain = data_dir.name

    # ----------------------------
    # Logger (same technique as other scripts)
    # ----------------------------
    LOG_DIR = PROJECT_ROOT / "data" / "logs" / "span_identification" / domain
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger, log_file = create_logger(LOG_DIR, f"16_eval_span_identifier_{domain}")
    logger.info(f"Log file: {log_file}")

    logger.info("==== Span Identifier Eval ====")
    logger.info(f"DOMAIN: {domain}")
    logger.info(f"DATA_DIR: {data_dir}")
    logger.info(f"SPLIT: {args.split}")
    logger.info(f"SUFFIX: {suffix}")
    logger.info(f"CKPT: {args.ckpt}")
    logger.info(f"MAX_LENGTH: {args.max_length}")
    logger.info(f"IOU_THRESHOLD: {args.iou_threshold}")
    logger.info(f"SPAN_RECALL_K: {args.span_recall_k}")
    logger.info(f"RESULTS_CSV: {args.results_csv}")
    if args.save_preds:
        logger.info(f"SAVE_PREDS: {args.save_preds}")

    # ----------------------------
    # Load data
    # ----------------------------
    filename = f"{args.split}{suffix}.jsonl" if suffix else f"{args.split}.jsonl"
    data_path = data_dir / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset split file not found: {data_path}")

    logger.info(f"Reading dataset: {data_path}")
    rows = read_jsonl(str(data_path))
    logger.info(f"Docs loaded: {len(rows)}")

    # ----------------------------
    # Load checkpoint (model + tokenizer)
    # ----------------------------
    ckpt_dir = Path(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForTokenClassification.from_pretrained(str(ckpt_dir), local_files_only=True)
    model.to(device)
    model.eval()

    try:
        tok = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True, local_files_only=True)
        logger.info("Tokenizer loaded from checkpoint.")
    except Exception:
        base_name = args.base_tokenizer or getattr(model.config, "_name_or_path", None) or "bert-base-uncased"
        tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
        logger.warning(f"Tokenizer not found in checkpoint. Using base tokenizer: {base_name}")

    # label map (prefer ckpt)
    label_map = safe_load_label_map(ckpt_dir)

    # id2label from model config can have int or str keys; decoder handles both
    id2label = getattr(model.config, "id2label", {}) or {}

    # Determine O id robustly:
    # Prefer model.config.label2id (should match model head), else fallback to loaded label_map
    label2id_cfg = getattr(model.config, "label2id", None)
    if isinstance(label2id_cfg, dict) and "O" in label2id_cfg:
        o_id = int(label2id_cfg["O"])
    else:
        if "O" not in label_map:
            raise ValueError("label_map must contain 'O'.")
        o_id = int(label_map["O"])

    logger.info(f"Using o_id={o_id} | num_labels={model.config.num_labels}")

    # Optional predictions dump
    pred_f = open(args.save_preds, "w", encoding="utf-8") if args.save_preds else None

    # ----------------------------
    # Token-level accumulators (micro)
    # ----------------------------
    token_correct = 0
    token_total = 0
    tp = fp = fn = tn = 0
    auc_y_true: List[int] = []
    auc_y_score: List[float] = []

    # ----------------------------
    # Span-level accumulators (macro lists)
    # ----------------------------
    exact_f1_list: List[float] = []
    iou_f1_list: List[float] = []
    srk_list: List[float] = []
    gold_empty = 0
    pred_empty = 0

    # ----------------------------
    # Span-level accumulators (micro counts)
    # ----------------------------
    exact_tp_micro = 0
    exact_fp_micro = 0
    exact_fn_micro = 0

    thr_iou = float(args.iou_threshold)
    k = int(args.span_recall_k)

    # ----------------------------
    # Evaluation loop
    # ----------------------------
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
                max_length=args.max_length,
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
            logits = out.logits[0]  # [T, C]

            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()

            # Token scores for ROC-AUC: p(LINK)=1-P(O)
            probs = torch.softmax(logits, dim=-1)  # [T, C]
            if not (0 <= o_id < probs.shape[-1]):
                raise ValueError(f"o_id={o_id} out of range for num_labels={probs.shape[-1]}")
            pos_prob = (1.0 - probs[:, o_id]).detach().cpu().numpy().tolist()

            # ----------------------------
            # Token metrics (micro)
            # ----------------------------
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

            # ----------------------------
            # Span metrics within visible window
            # ----------------------------
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

            # Macro doc-level
            exact_f1_list.append(exact_span_f1_doc(gold_char, pred_char))
            iou_f1_list.append(iou_f1_doc(gold_char, pred_char, thr=thr_iou))

            # Micro pooled exact counts
            gs = set(gold_char)
            ps = set(pred_char)
            exact_tp_micro += len(gs & ps)
            exact_fp_micro += len(ps - gs)
            exact_fn_micro += len(gs - ps)

            # Score each predicted span by mean token pos_prob inside the span
            tok_pairs = [((int(s), int(e)), float(sc)) for (s, e), sc in zip(offsets, pos_prob)]
            span_scores: List[float] = []
            for (a, b) in pred_char:
                inside = [
                    sc for (ts, te), sc in tok_pairs
                    if not (ts == 0 and te == 0) and te > ts and (ts < b and te > a)
                ]
                span_scores.append(float(np.mean(inside)) if inside else 0.0)

            srk_list.append(span_recall_at_k_doc(gold_char, pred_char, span_scores, k=k, thr_iou=thr_iou))

            # Optional per-doc dump
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
        logger.info(f"Saved predictions to: {args.save_preds}")

    # ----------------------------
    # Final aggregation
    # ----------------------------
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

    # Exact span MICRO metrics
    exact_p_micro = exact_tp_micro / (exact_tp_micro + exact_fp_micro) if (exact_tp_micro + exact_fp_micro) else 0.0
    exact_r_micro = exact_tp_micro / (exact_tp_micro + exact_fn_micro) if (exact_tp_micro + exact_fn_micro) else 0.0
    exact_f1_micro = (2 * exact_p_micro * exact_r_micro / (exact_p_micro + exact_r_micro)) if (exact_p_micro + exact_r_micro) else 0.0

    # ----------------------------
    # Print summary + log summary
    # ----------------------------
    summary: Dict[str, Any] = {
        "domain": domain,
        "split": args.split,
        "suffix": suffix,
        "ckpt": str(ckpt_dir),
        "max_length": args.max_length,
        "iou_threshold": thr_iou,
        "span_recall_k": k,
        "epoch": args.epoch if args.epoch is not None else "",
        "lr": args.lr if args.lr is not None else "",

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
        "iou_f1_macro": iou_macro,
        "span_recall_at_k": srk_macro,

        # NEW: exact span micro
        "exact_span_precision_micro": exact_p_micro,
        "exact_span_recall_micro": exact_r_micro,
        "exact_span_f1_micro": exact_f1_micro,
        "exact_span_tp_micro": exact_tp_micro,
        "exact_span_fp_micro": exact_fp_micro,
        "exact_span_fn_micro": exact_fn_micro,

        "pct_gold_empty": pct_gold_empty,
        "pct_pred_empty": pct_pred_empty,
    }

    print("========== Eval ==========")
    print(f"Domain: {domain}")
    print(f"Split: {args.split}")
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
    print(f"Span exact P/R/F1 (micro): {exact_p_micro:.4f} {exact_r_micro:.4f} {exact_f1_micro:.4f} "
          f"(TP={exact_tp_micro} FP={exact_fp_micro} FN={exact_fn_micro})")
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

    # ----------------------------
    # Append CSV row (always)
    # ----------------------------
    out_csv = Path(args.results_csv)
    append_metrics_csv(summary, out_csv)
    logger.info(f"Appended metrics row to CSV: {out_csv}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
