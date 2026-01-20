#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# --------------------------------------------------------------------------------------
# YAML + template resolution
# --------------------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def must_exist(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], title: str, env: Optional[dict[str, str]] = None) -> None:
    print("\n" + "=" * 90)
    print(f"RUN: {title}")
    print("CMD:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True, env=env)


def normalize_template(s: str) -> str:
    """
    Convert dot-style dict access into bracket-style for str.format_map.

    Example:
      "{paths.data_dir}" -> "{paths[data_dir]}"

    This allows YAML templates using dot notation even though `paths` is a dict.
    """
    return re.sub(r"\{paths\.([A-Za-z0-9_]+)\}", r"{paths[\1]}", s)


class SafeDict(dict):
    """
    Safe formatter mapping: leaves unknown placeholders untouched.
    Example: "{exp_root}/{run_name}".format_map(SafeDict(...))
    If exp_root missing -> "{exp_root}/{run_name}" stays as-is for next pass.
    """
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def resolve_templates(obj: Any, ctx: Dict[str, Any]) -> Any:
    """
    Recursively resolve templates using SafeDict + format_map,
    so forward references don't crash (e.g., ckpt_dir uses {exp_root}).
    Also supports dot-style {paths.data_dir} by normalizing it first.
    """
    if isinstance(obj, str):
        s = normalize_template(obj)
        return s.format_map(SafeDict(**ctx))
    if isinstance(obj, dict):
        return {k: resolve_templates(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_templates(v, ctx) for v in obj]
    return obj


def make_abs(p: Path, project_root: Path) -> Path:
    return p if p.is_absolute() else (project_root / p)


def expand_path(raw: Any, project_root: Path, ctx: Dict[str, Any]) -> Path:
    """
    Expand a yaml string path that may contain templates and may be relative.
    Uses SafeDict so missing keys don't crash, and supports {paths.xxx}.
    """
    if raw is None:
        raise ValueError("Got None where a path string was expected")

    s = normalize_template(str(raw)).format_map(SafeDict(**ctx))
    p = Path(s)
    return make_abs(p, project_root)


def build_context(cfg: Dict[str, Any], project_root: Path, domain: str, suffix: str) -> Dict[str, Any]:
    """
    Build a formatting context and resolve cfg['paths'] even if it contains
    forward references like ckpt_dir: "{exp_root}/{run_name}".

    Strategy:
      - Start ctx with base keys
      - Iteratively resolve paths up to 10 times or until stable
      - Inject resolved path keys both into ctx['paths'] and top-level ctx
        so templates can use {exp_root} as well as {paths[exp_root]} / {paths.exp_root}
    """
    ctx: Dict[str, Any] = {
        "project_root": str(project_root),
        "domain": domain,
        "suffix": suffix,
        "paths": {},
    }

    raw_paths = cfg.get("paths", {}) or {}
    resolved_paths = dict(raw_paths)

    for _ in range(10):
        candidate = resolve_templates(resolved_paths, ctx)

        # inject into ctx (both nested and top-level)
        ctx["paths"] = candidate
        for k, v in candidate.items():
            ctx[k] = v

        if candidate == resolved_paths:
            resolved_paths = candidate
            break
        resolved_paths = candidate

    # final commit
    ctx["paths"] = resolved_paths
    for k, v in resolved_paths.items():
        ctx[k] = v

    return ctx


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

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

    # Build template context
    ctx = build_context(cfg, project_root, domain, suffix)

    # Prefer YAML-defined data_dir if provided, else default
    if cfg.get("paths", {}).get("data_dir"):
        data_dir = expand_path(cfg["paths"]["data_dir"], project_root, ctx)
    else:
        data_dir = project_root / "data" / "span_identification" / domain

    # -----------------------------
    # Ensure directories exist (auto-create)
    # -----------------------------
    ensure_dirs_list = cfg.get("ensure_dirs")
    if ensure_dirs_list:
        for d in ensure_dirs_list:
            ensure_dir(expand_path(d, project_root, ctx))
    else:
        ensure_dir(data_dir)

        logs_dir = cfg.get("paths", {}).get("logs_dir")
        if logs_dir:
            ensure_dir(expand_path(logs_dir, project_root, ctx))
        else:
            ensure_dir(project_root / "data" / "logs" / "span_identification" / domain)

        results_dir = cfg.get("paths", {}).get("results_dir")
        if results_dir:
            ensure_dir(expand_path(results_dir, project_root, ctx))
        else:
            ensure_dir(project_root / "results")

        ckpt_dir = cfg.get("paths", {}).get("ckpt_dir")
        if ckpt_dir:
            ensure_dir(expand_path(ckpt_dir, project_root, ctx))

    # -----------------------------
    # Step 14: Prepare dataset
    # -----------------------------
    prep = cfg["prep"]
    prep_script = expand_path(prep["script"], project_root, ctx)
    prep_cfg = expand_path(prep["config"], project_root, ctx)

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
    train_script = expand_path(train["script"], project_root, ctx)

    # Accept multiple YAML key names for the training config
    train_cfg_raw = (
        train.get("cfg")
        or train.get("config")
        or train.get("config_yaml")
        or train.get("train_cfg")
    )
    if not train_cfg_raw:
        raise KeyError(
            "Missing training config path in pipeline YAML under train: "
            "expected one of ['cfg', 'config', 'config_yaml', 'train_cfg']"
        )
    train_cfg = expand_path(train_cfg_raw, project_root, ctx)

    # out_dir: prefer YAML, else prefer paths.ckpt_dir, else default under experiments
    if "out_dir" in train:
        out_dir = expand_path(train["out_dir"], project_root, ctx)
    elif cfg.get("paths", {}).get("ckpt_dir"):
        out_dir = expand_path(cfg["paths"]["ckpt_dir"], project_root, ctx)
    else:
        out_dir = project_root / "experiments" / "span_id" / domain / f"bert_bilou{suffix}"

    ensure_dir(out_dir)

    must_exist(train_script, "Train script (15)")
    must_exist(train_cfg, "Train config yaml")

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


    if "out_dir" in train:
        out_dir = expand_path(train["out_dir"], project_root, ctx)
    elif cfg.get("paths", {}).get("ckpt_dir"):
        out_dir = expand_path(cfg["paths"]["ckpt_dir"], project_root, ctx)
    else:
        out_dir = project_root / "experiments" / "span_id" / domain / f"bert_bilou{suffix}"

    ensure_dir(out_dir)

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

    must_exist(out_dir, "Training output dir")
    must_exist(out_dir / "label_map.json", "label_map.json (training metadata)")

    # -----------------------------
    # Step 16: Evaluate model (NO CLI ARGS)
    # -----------------------------
    ev = cfg["eval"]
    eval_script = expand_path(ev["script"], project_root, ctx)
    must_exist(eval_script, "Eval script (16)")

    env = os.environ.copy()
    env["PIPELINE_CONFIG"] = str(pipe_path)
    env["PROJECT_ROOT"] = str(project_root)

    eval_cmd = [
        sys.executable,
        str(eval_script),
    ]

    run_cmd(eval_cmd, "STEP 16: Evaluate span identifier (CONFIG-DRIVEN, NO CLI)", env=env)

    # Derive post-run reporting paths
    results_csv_raw = ev.get("results_csv") or cfg.get("paths", {}).get("results_csv")
    if results_csv_raw:
        results_csv = expand_path(results_csv_raw, project_root, ctx)
    else:
        results_csv = project_root / "results" / "span_identifier_metrics.csv"

    save_preds_raw = ev.get("save_preds") or cfg.get("paths", {}).get("preds_test")
    save_preds: Optional[Path] = None
    if save_preds_raw:
        save_preds = expand_path(save_preds_raw, project_root, ctx)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"- Domain: {domain}")
    print(f"- Data: {data_dir}")
    print(f"- Model: {out_dir}")
    print(f"- Metrics CSV: {results_csv}")
    if save_preds:
        print(f"- Predictions: {save_preds}")


if __name__ == "__main__":
    main()
