#!/usr/bin/env python3
"""
run_pipeline.py

Simple pipeline runner for the Fandom_SI project.

Reads configs/pipeline.yaml and lets you:

- Run the full pipeline 00 → 13
- Run from a specific step to the end
    python scripts/run_pipeline.py --from-step 04_compute_span_embeddings

- Run up to a specific step
    python scripts/run_pipeline.py --to-step 08_train_article_reranker

- Run a custom subset of steps (in pipeline order)
    python scripts/run_pipeline.py --only 03_build_spans_from_sections,04_compute_span_embeddings,05_build_faiss_index

- Just list steps
    python scripts/run_pipeline.py --list

Flags:
    --continue-on-error   → don’t stop the whole pipeline if one step fails
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path("/data/sundeep/Fandom_SI")
PIPELINE_YAML = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_pipeline():
    if not PIPELINE_YAML.exists():
        print(f"[ERROR] pipeline config not found: {PIPELINE_YAML}", file=sys.stderr)
        sys.exit(1)

    with open(PIPELINE_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    steps = cfg.get("steps", [])
    if not steps:
        print("[ERROR] No steps defined in pipeline.yaml", file=sys.stderr)
        sys.exit(1)

    # Keep original order, but also build index by id
    step_by_id = {s["id"]: s for s in steps}
    return steps, step_by_id, cfg


def list_steps(steps):
    print("Available pipeline steps (in order):")
    for s in steps:
        deps = s.get("depends_on") or []
        deps_str = ", ".join(deps) if deps else "-"
        print(f"  {s['id']:>26}  |  {s['name']}  |  depends_on: {deps_str}")


def parse_args():
    p = argparse.ArgumentParser(description="Fandom_SI pipeline runner")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--from-step", type=str, help="Start from this step id (inclusive)")
    g.add_argument("--to-step", type=str, help="Run up to this step id (inclusive)")
    g.add_argument(
        "--only",
        type=str,
        help="Comma-separated list of step ids to run (respecting pipeline order)",
    )
    p.add_argument("--list", action="store_true", help="List available steps and exit")

    # 🔥 FIXED — use store_true for Python 3.10 compatibility
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set, continues even if a step fails",
    )

    return p.parse_args()


def resolve_steps_to_run(steps, step_by_id, args):
    # No specific selection → full pipeline
    if not args.from_step and not args.to_step and not args.only:
        return steps

    # --only: pick given IDs in pipeline order
    if args.only:
        requested = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [rid for rid in requested if rid not in step_by_id]
        if unknown:
            print(f"[ERROR] Unknown step ids in --only: {unknown}", file=sys.stderr)
            sys.exit(1)

        id_set = set(requested)
        ordered = [s for s in steps if s["id"] in id_set]
        return ordered

    # range mode: from / to
    step_ids = [s["id"] for s in steps]

    start_idx = 0
    end_idx = len(steps) - 1

    if args.from_step:
        if args.from_step not in step_by_id:
            print(f"[ERROR] Unknown --from-step id: {args.from_step}", file=sys.stderr)
            sys.exit(1)
        start_idx = step_ids.index(args.from_step)

    if args.to_step:
        if args.to_step not in step_by_id:
            print(f"[ERROR] Unknown --to-step id: {args.to_step}", file=sys.stderr)
            sys.exit(1)
        end_idx = step_ids.index(args.to_step)

    if start_idx > end_idx:
        print("[ERROR] from-step occurs after to-step in pipeline order.", file=sys.stderr)
        sys.exit(1)

    return steps[start_idx : end_idx + 1]


def run_step(step):
    script_rel = step["script"]
    script_path = PROJECT_ROOT / script_rel

    if not script_path.exists():
        print(f"[ERROR] Script not found for step {step['id']}: {script_path}", file=sys.stderr)
        return False

    print("\n" + "=" * 80)
    print(f"[RUN] {step['id']} - {step['name']}")
    print(f"      script: {script_path}")
    print("=" * 80)

    # Use current Python interpreter
    cmd = [sys.executable, str(script_path)]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"[OK] Step {step['id']} completed with exit code {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Step {step['id']} failed with exit code {e.returncode}", file=sys.stderr)
        return False


def main():
    steps, step_by_id, cfg = load_pipeline()
    args = parse_args()

    if args.list:
        list_steps(steps)
        return

    steps_to_run = resolve_steps_to_run(steps, step_by_id, args)

    print("\nPipeline: ", cfg.get("pipeline_name", "unnamed"))
    print("Project root:", cfg.get("project_root", PROJECT_ROOT))
    print("Steps to run (in order):")
    for s in steps_to_run:
        print(f"  - {s['id']}  ({s['name']})")

    for step in steps_to_run:
        ok = run_step(step)
        if not ok and not args.continue_on_error:
            print("\n[STOP] Aborting pipeline because a step failed.")
            sys.exit(1)

    print("\n[DONE] Pipeline run completed.")


if __name__ == "__main__":
    main()