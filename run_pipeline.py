#!/usr/bin/env python
"""Minimal pipeline runner (official guideline aligned).

Flow:
  1) Retrieval candidates (TF-IDF over class name+keywords)
  2) Optional LLM silver labeling (<= 1000 calls, prompts/outputs saved to files)
  3) Train student classifier on silver labels
  4) Inference on test set -> submission with exactly 2 or 3 labels per doc
  5) Verify submission format and row count

This runner intentionally avoids extra stages to keep the code simple and stable.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def run_stage(name: str, script: str, project_root: Path, args: list[str] | None = None) -> float:
    """Run a pipeline stage and return elapsed time."""
    start = time.time()
    logging.info("=" * 60)
    logging.info(name)
    logging.info("=" * 60)
    
    src_dir = project_root / "src"
    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)

    result = subprocess.run(
        cmd,
        cwd=src_dir,
        env={**os.environ, "PYTHONPATH": str(src_dir)},
    )
    if result.returncode != 0:
        logging.error("Stage %s failed with return code %d", name, result.returncode)
        sys.exit(result.returncode)
    
    elapsed = time.time() - start
    logging.info("%s completed in %.1fs", name, elapsed)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ML pipeline")
    parser.add_argument("--student-id", type=str, help="Your student ID for submission filename")
    parser.add_argument("--force", action="store_true", help="Force re-run LLM labeling even if silver files exist")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM labeling stage (uses existing silver_simple.jsonl)")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM refinement (<=1000 calls, prompts/responses saved)")
    parser.add_argument(
        "--use-gnn",
        action="store_true",
        help="Use the GNN classifier for training + inference (gnn_classifier.py/gnn_inference.py) instead of the baseline student model.",
    )
    args = parser.parse_args()

    # Set student ID from args or environment
    if args.student_id:
        os.environ["STUDENT_ID"] = args.student_id
    elif "STUDENT_ID" not in os.environ:
        logging.warning("STUDENT_ID not set. Using 'STUDENT_ID' as placeholder.")
        logging.warning("Run with: python run_pipeline.py --student-id 2021320045")
        os.environ["STUDENT_ID"] = "2021320045"

    student_id = os.environ["STUDENT_ID"]
    logging.info("Student ID: %s", student_id)

    project_root = Path(__file__).parent.resolve()
    src_dir = project_root / "src"

    # Check for OpenAI API key (needed only if enabling LLM refinement)
    key_file = project_root / "artifacts" / "llm_calls" / "openai.key"
    if args.use_llm and not os.getenv("OPENAI_API_KEY") and not key_file.exists():
        logging.warning("No OpenAI API key found. LLM refinement will fail unless you set OPENAI_API_KEY or create %s", key_file)

    pipeline_start = time.time()

    stages: list[tuple[str, str, list[str] | None]] = [
        ("[1/5] Candidate Retrieval", "retrieval.py", None),
        ("[2/5] Graph Build", "graph_build.py", None),
    ]

    silver_args: list[str] = []
    if args.force:
        silver_args.append("--force")
    if args.skip_llm:
        silver_args.append("--no-llm")
    if args.use_llm:
        silver_args.append("--use-llm")

    stages.append(("[3/5] Silver Labeling", "silver_labeling.py", silver_args or None))

    if bool(args.use_gnn):
        # Run from src/; use project-root relative paths for saved model directory.
        stages.append(("[4/5] Training (GNN)", "gnn_classifier.py", ["--save-dir", "..\\student_gnn"]))
        stages.append(
            (
                "[5/5] Inference (GNN)",
                "gnn_inference.py",
                [
                    "--model-dir",
                    "..\\student_gnn",
                    "--student-id",
                    student_id,
                ],
            )
        )
    else:
        stages.append(("[4/5] Training", "training.py", None))
        infer_args: list[str] = []
        if args.use_llm:
            infer_args.append("--use-llm")

        stages.append(("[5/5] Inference", "inference.py", infer_args or None))

    timings = []
    for name, script, stage_args in stages:
        elapsed = run_stage(name, script, project_root, args=stage_args)
        timings.append((name, elapsed))

    # Verify is cheap; run it at the end regardless
    _ = run_stage("[final] Verify Submission", "verify.py", project_root, args=None)

    total_time = time.time() - pipeline_start

    # Summary
    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 60)
    for name, elapsed in timings:
        logging.info("  %s: %.1fs", name, elapsed)
    logging.info("-" * 60)
    logging.info("  Total: %.1fs (%.1f minutes)", total_time, total_time / 60)

    submission_file = project_root / "output" / f"{student_id}_final.csv"
    if submission_file.exists():
        logging.info("Submission saved to: %s", submission_file)
        # Count lines for verification
        with open(submission_file, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f) - 1  # exclude header
        logging.info("Submission contains %d predictions", lines)
    else:
        logging.error("Submission file not found: %s", submission_file)


if __name__ == "__main__":
    main()
