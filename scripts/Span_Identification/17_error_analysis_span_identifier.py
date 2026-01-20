#!/usr/bin/env python3
"""
Error analysis for Span Identification.

Input: JSONL written by 16_eval_span_identifier.py with keys:
  - doc_id
  - text
  - gold_spans: [{start,end,type}]
  - pred_spans: [{start,end,type}]

Output:
  - counts by error type
  - sample examples per error type with text context
"""

from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def overlap(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    """Return overlap length in characters (0 if disjoint)."""
    s1,e1 = a
    s2,e2 = b
    s = max(s1,s2)
    e = min(e1,e2)
    return max(0, e - s)


def exact(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return a[0] == b[0] and a[1] == b[1]


def extract(text: str, sp: Tuple[int,int]) -> str:
    s,e = sp
    return text[s:e]


def context_snip(text: str, sp: Tuple[int,int], window: int = 60) -> str:
    s,e = sp
    a = max(0, s-window)
    b = min(len(text), e+window)
    return text[a:b].replace("\n", "\\n")


def is_citation_like(t: str) -> bool:
    tt = t.strip()
    if tt in {"↑", "†", "*"}:
        return True
    # patterns like [ 1 ], [1], [ 12 ]
    if tt.startswith("[") and tt.endswith("]"):
        inner = tt[1:-1].strip()
        if inner.isdigit():
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="predictions JSONL from eval script")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    ap.add_argument("--sample_per_type", type=int, default=15)
    ap.add_argument("--context_window", type=int, default=60)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.preds))

    counts = Counter()
    samples = defaultdict(list)

    # We’ll count and sample:
    # TP_exact, FP_spurious, FN_missed, boundary_overlap
    # plus subtypes: citation_fp
    for r in rows:
        text = r["text"]
        gold = [(int(s["start"]), int(s["end"])) for s in r.get("gold_spans", [])]
        pred = [(int(s["start"]), int(s["end"])) for s in r.get("pred_spans", [])]

        gold_set = set(gold)
        pred_set = set(pred)

        # --- Exact matches ---
        for sp in (gold_set & pred_set):
            counts["TP_exact"] += 1

        # --- False negatives (missed gold spans) ---
        for g in (gold_set - pred_set):
            # check if there is any overlapping prediction (then it's boundary error, not a pure miss)
            ov = max((overlap(g, p) for p in pred), default=0)
            if ov > 0:
                counts["Boundary_FN"] += 1
                if len(samples["Boundary_FN"]) < args.sample_per_type:
                    samples["Boundary_FN"].append({
                        "doc_id": r.get("doc_id",""),
                        "gold_span": g, "gold_text": extract(text, g),
                        "note": "Gold overlapped but not exact",
                        "context": context_snip(text, g, args.context_window),
                    })
            else:
                counts["FN_missed"] += 1
                if len(samples["FN_missed"]) < args.sample_per_type:
                    samples["FN_missed"].append({
                        "doc_id": r.get("doc_id",""),
                        "gold_span": g, "gold_text": extract(text, g),
                        "context": context_snip(text, g, args.context_window),
                    })

        # --- False positives (spurious predicted spans) ---
        for p in (pred_set - gold_set):
            ov = max((overlap(p, g) for g in gold), default=0)
            if ov > 0:
                counts["Boundary_FP"] += 1
                if len(samples["Boundary_FP"]) < args.sample_per_type:
                    # show the best-overlap gold
                    best_g = max(gold, key=lambda g: overlap(p, g))
                    samples["Boundary_FP"].append({
                        "doc_id": r.get("doc_id",""),
                        "pred_span": p, "pred_text": extract(text, p),
                        "gold_span_best": best_g, "gold_text_best": extract(text, best_g),
                        "note": "Pred overlapped gold but not exact",
                        "context": context_snip(text, p, args.context_window),
                    })
            else:
                pred_txt = extract(text, p)
                if is_citation_like(pred_txt):
                    counts["FP_citation_like"] += 1
                    key = "FP_citation_like"
                else:
                    counts["FP_spurious"] += 1
                    key = "FP_spurious"

                if len(samples[key]) < args.sample_per_type:
                    samples[key].append({
                        "doc_id": r.get("doc_id",""),
                        "pred_span": p, "pred_text": pred_txt,
                        "context": context_snip(text, p, args.context_window),
                    })

    report = {
        "counts": dict(counts),
        "samples": dict(samples),
        "notes": {
            "TP_exact": "pred span exactly equals a gold span",
            "FN_missed": "gold span has no overlapping predicted span",
            "Boundary_FN": "gold span overlaps a predicted span but boundaries differ",
            "FP_spurious": "pred span has no overlapping gold span",
            "Boundary_FP": "pred span overlaps a gold span but boundaries differ",
            "FP_citation_like": "spurious pred span looks like [1] or ↑ etc.",
        }
    }

    # Print summary
    print("====== Error Analysis Summary ======")
    for k, v in counts.most_common():
        print(f"{k:16s} : {v}")

    print("\n====== Samples (first few) ======")
    for k in ["FP_spurious", "FP_citation_like", "Boundary_FP", "FN_missed", "Boundary_FN"]:
        if k in samples and samples[k]:
            print(f"\n--- {k} ---")
            for ex in samples[k][:min(5, len(samples[k]))]:
                print(json.dumps(ex, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] Wrote report to: {args.out}")


if __name__ == "__main__":
    main()
