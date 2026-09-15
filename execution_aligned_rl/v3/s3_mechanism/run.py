"""S3 orchestrator. Protocol is already frozen. Preflight must pass before workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from execution_aligned_rl.v3.s3_mechanism.protocol import DEFAULT_PATHS, PROTOCOL_ID, experiment_dir
from execution_aligned_rl.v3.serialization import dump_json


def cpu_env(repo: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["JAX_ENABLE_X64"] = "0"
    env.pop("XLA_FLAGS", None)
    return env


def run_mod(repo: Path, module: str, args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *args], check=True, cwd=str(repo), env=cpu_env(repo))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    parser.add_argument("--stage", choices=["preflight", "workers", "analyze", "all"], default="preflight")
    args = parser.parse_args()
    repo = Path(args.repo)
    exp = experiment_dir(repo)
    exp.mkdir(parents=True, exist_ok=True)
    common = [
        "--repo",
        str(repo),
        "--dataset-dir",
        DEFAULT_PATHS["dataset_dir"],
        "--official-source",
        DEFAULT_PATHS["official_source"],
        "--checkpoint-dir",
        DEFAULT_PATHS["checkpoint_dir"],
    ]
    if args.stage in ("preflight", "all"):
        print("PREFLIGHT", flush=True)
        run_mod(repo, "execution_aligned_rl.v3.s3_mechanism.preflight", [])
        print("PREFLIGHT_DONE", flush=True)
        if args.stage == "preflight":
            return
    if args.stage in ("workers", "all"):
        marker = exp / "preflight" / "preflight_pass.json"
        if not marker.exists():
            dump_json(exp / "decision.json", {"status": "EA35_HOLD_ENGINEERING", "reason": "preflight_missing", "s4_unlocked": False, "protocol_id": PROTOCOL_ID})
            raise SystemExit(2)
        print("WORKER_A", flush=True)
        run_mod(repo, "execution_aligned_rl.v3.s3_mechanism.worker", ["--worker-id", "A", *common])
        print("WORKER_B", flush=True)
        run_mod(repo, "execution_aligned_rl.v3.s3_mechanism.worker", ["--worker-id", "B", *common])
        print("WORKERS_DONE", flush=True)
        if args.stage == "workers":
            return
    if args.stage in ("analyze", "all"):
        print("ANALYZE", flush=True)
        run_mod(repo, "execution_aligned_rl.v3.s3_mechanism.analyze", [])


if __name__ == "__main__":
    main()
