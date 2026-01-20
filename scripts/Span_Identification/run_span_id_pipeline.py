#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def must_exist(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")


def run_cmd(cmd: list[str], title: str) -> None:
    print("\n" + "=" * 90)
    print(f"RUN: {title}")
    print("CMD:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def expand_path(
    raw: Any,
    project_root: Path,
    domain: str,
    suffix: str,
) -> Path:
    """
    Expands placeholders in pipeline yaml strings like:
      "{project_root}/scripts/Span_Identification/14_prepare..."
    Supported placeholders:
      - {project_root}
      - {domain}
      - {suffix}
    Also supports relative paths (interpreted relative to project_root).
    """
    if raw is None:
        raise ValueError("Got None where a path string was expected")

    s = str(raw)

    s = s.replace("{project_root}", str(project_root))
    s = s.replace("{domain}", domain)
    s = s.replace("{suffix}", suffix)

    p = Path(s)

    # If still relative, resolve relative to project_root
    if not p.is_absolute():
        p = project_root / p

    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pipeline",
        required=True,
        help="Path to pipeline yaml (e.g., configs/pipeline_span_id.yaml)",
    )
    args = ap.parse_args()

    pipe_path = Path(args.pipeline)
    must_exist(pipe_path, "Pipeline config")

    cfg = load_yaml(pipe_path)

    project_root = Path(cfg.get("project_root", "/data/sundeep/Fandom_SI"))

    if "domain" not in cfg or not str(cfg["domain"]).strip():
        raise ValueError(f"Missing required key 'domain' in pipeline config: {pipe_path}")
    domain = str(cfg["domain"]).strip()

    suffix = str(cfg.get("suffix", "")).strip()

    # Derived data dir (output of script 14)
    data_dir = project_root / "data" / "span_identification" / domain

    # -----------------------------
    # Step 14: Prepare dataset
    # -----------------------------
    prep = cfg["prep"]
    prep_script = expand_path(prep["script"], project_root, domain, suffix)
    prep_cfg = expand_path(prep["config"], project_root, domain, suffix)

    must_exist(prep_script, "Prep script (14)")
    must_exist(prep_cfg, "Prep config yaml")

    prep_cmd = [
        sys.executable,
        str(prep_script),
        "--pipeline_config",
        str(pipe_path),
        "--config",
        str(prep_cfg),
        "--out_suffix",
        suffix,
        "--para_sep",
        str(prep.get("para_sep", "\n\n")),
    ]
    if bool(prep.get("remove_punct", False)):
        prep_cmd.append("--remove_punct")
    if bool(prep.get("require_spans", False)):
        prep_cmd.append("--require_spans")

    run_cmd(prep_cmd, "STEP 14: Prepare span-id dataset")

    # Verify outputs exist
    must_exist(data_dir, "Data dir (span_identification/<domain>)")
    train_file = data_dir / f"train{suffix}.jsonl"
    dev_file = data_dir / f"dev{suffix}.jsonl"
    test_file = data_dir / f"test{suffix}.jsonl"
    must_exist(train_file, "Train JSONL")
    must_exist(dev_file, "Dev JSONL")
    must_exist(test_file, "Test JSONL")

    # -----------------------------
    # Step 15: Train model
    # -----------------------------
    train = cfg["train"]
    train_script = expand_path(train["script"], project_root, domain, suffix)
    train_cfg = expand_path(train["cfg"], project_root, domain, suffix)
    out_dir = expand_path(train["out_dir"], project_root, domain, suffix)

    must_exist(train_script, "Train script (15)")
    must_exist(train_cfg, "Train config yaml (span_identi.yaml)")

    train_cmd = [
        sys.executable,
        str(train_script),
        "--pipeline_config",
        str(pipe_path),
        "--data_dir",
        str(data_dir),
        "--suffix",
        suffix,
        "--cfg",
        str(train_cfg),
        "--out",
        str(out_dir),
    ]

    run_cmd(train_cmd, "STEP 15: Train span identifier")

    # Verify model artifacts exist
    must_exist(out_dir, "Training output dir")
    must_exist(out_dir / "label_map.json", "label_map.json (training metadata)")

    # -----------------------------
    # Step 16: Evaluate model
    # -----------------------------
    ev = cfg["eval"]
    eval_script = expand_path(ev["script"], project_root, domain, suffix)
    must_exist(eval_script, "Eval script (16)")

    split = str(ev.get("split", "test"))
    max_length = int(ev.get("max_length", 256))
    iou_thr = float(ev.get("iou_threshold", 0.5))
    srk = int(ev.get("span_recall_k", 5))

    results_csv_raw = ev.get("results_csv", project_root / "results" / "span_identifier_metrics.csv")
    results_csv = expand_path(results_csv_raw, project_root, domain, suffix)

    save_preds_raw = ev.get("save_preds", None)
    save_preds: Optional[Path] = None
    if save_preds_raw:
        save_preds = expand_path(save_preds_raw, project_root, domain, suffix)

    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--data_dir",
        str(data_dir),
        "--suffix",
        suffix,
        "--split",
        split,
        "--ckpt",
        str(out_dir),
        "--max_length",
        str(max_length),
        "--iou_threshold",
        str(iou_thr),
        "--span_recall_k",
        str(srk),
        "--results_csv",
        str(results_csv),
    ]
    if save_preds:
        eval_cmd += ["--save_preds", str(save_preds)]

    run_cmd(eval_cmd, "STEP 16: Evaluate span identifier")

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"- Domain: {domain}")
    print(f"- Data: {data_dir}")
    print(f"- Model: {out_dir}")
    print(f"- Metrics CSV: {results_csv}")
    if save_preds:
        print(f"- Predictions: {save_preds}")


if __name__ == "__main__":
    main()