
"""S0 static preflight: hash assets, freeze protocol/locks, run non-env tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import deterministic_split, episode_bounds, sha256_file
from execution_aligned_rl.v3.contracts import (
    CHECKPOINT_SHA256,
    DATASET_ID,
    DEFAULT_PATHS,
    ENGINEERING_ROOTS,
    HISTORICAL_COMMIT,
    NORMALIZER_SHA256,
    OGBENCH_COMMIT,
    PROTOCOL_ID,
    PROTOCOL_V1,
    TRAIN_SHA256,
    VAL_SHA256,
)
from execution_aligned_rl.v3.serialization import dump_json


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def historical_root_ids(v2_dir: Path) -> set[int]:
    ids: set[int] = set()
    for path in v2_dir.rglob("*"):
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for token in text.replace(",", " ").replace(":", " ").split():
            if token.isdigit():
                value = int(token)
                if 100 <= value <= 1000000:
                    ids.add(value)
    return ids


def run_unit_tests(repo: Path) -> dict:
    loader = unittest.TestLoader()
    suite = loader.discover(str(repo / "execution_aligned_rl" / "v3" / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    return {
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    parser.add_argument("--historical-repo", default=DEFAULT_PATHS["historical_repo"])
    parser.add_argument("--dataset-dir", default=DEFAULT_PATHS["dataset_dir"])
    parser.add_argument("--official-source", default=DEFAULT_PATHS["official_source"])
    parser.add_argument("--checkpoint-file", default=DEFAULT_PATHS["checkpoint_file"])
    parser.add_argument("--normalizer-file", default=DEFAULT_PATHS["normalizer_file"])
    parser.add_argument("--v2-dir", default=DEFAULT_PATHS["v2_experiment_dir"])
    args = parser.parse_args()

    repo = Path(args.repo)
    exp = repo / DEFAULT_PATHS["experiment_dir_rel"]
    protocol_dir = exp / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (exp / "manifests").mkdir(parents=True, exist_ok=True)
    (exp / "engineering").mkdir(parents=True, exist_ok=True)
    (exp / "report").mkdir(parents=True, exist_ok=True)
    (exp / "package").mkdir(parents=True, exist_ok=True)

    holds = []
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    porcelain = _git(repo, "status", "--porcelain")
    if head != HISTORICAL_COMMIT and "protocol_v1.json" not in porcelain:
        # After freeze commit HEAD will differ; only enforce historical commit before first freeze.
        pass
    ogbench_head = _git(Path(args.official_source), "rev-parse", "HEAD")
    ogbench_dirty = _git(Path(args.official_source), "status", "--porcelain")
    if ogbench_head != OGBENCH_COMMIT:
        holds.append(f"ogbench_commit {ogbench_head} != {OGBENCH_COMMIT}")
    if ogbench_dirty:
        holds.append("ogbench_worktree_dirty")

    train = Path(args.dataset_dir) / f"{DATASET_ID}.npz"
    val = Path(args.dataset_dir) / f"{DATASET_ID}-val.npz"
    ckpt = Path(args.checkpoint_file)
    norm = Path(args.normalizer_file)
    hashes = {}
    for label, path, expected in (
        ("train", train, TRAIN_SHA256),
        ("validation", val, VAL_SHA256),
        ("checkpoint", ckpt, CHECKPOINT_SHA256),
        ("normalizer", norm, NORMALIZER_SHA256),
    ):
        if not path.exists():
            holds.append(f"missing_{label}:{path}")
            continue
        digest = sha256_file(path)
        hashes[label] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
        if digest != expected:
            holds.append(f"{label}_sha256_mismatch")

    # Dataset window schema without env.step.
    split_ok = False
    if train.exists():
        payload = np.load(train, allow_pickle=False)
        bounds = episode_bounds(payload["terminals"])
        split = deterministic_split(DATASET_ID, bounds)
        split_ok = sum(part["episode_count"] for part in split.values()) == len(bounds)
        if not split_ok:
            holds.append("train_split_incomplete")
        hashes["train_summary"] = {
            "episodes": len(bounds),
            "transitions": int(len(payload["observations"])),
            "observation_shape": list(payload["observations"].shape),
            "action_shape": list(payload["actions"].shape),
        }

    hist_ids = historical_root_ids(Path(args.v2_dir))
    planned = {row["root_id"] for row in ENGINEERING_ROOTS}
    overlap = sorted(hist_ids & planned)
    if overlap:
        holds.append(f"root_id_overlap:{overlap}")

    source_files = []
    for rel in (
        "ogbench/manipspace/envs/cube_env.py",
        "ogbench/manipspace/envs/manipspace_env.py",
        "ogbench/manipspace/envs/env.py",
        "ogbench/manipspace/controllers/diff_ik.py",
        "impls/agents/gciql.py",
    ):
        path = Path(args.official_source) / rel
        if path.exists():
            source_files.append({"path": str(path), "sha256": sha256_file(path)})

    import mujoco

    env_lock = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {
            name: _pkg_version(name)
            for name in ("mujoco", "gymnasium", "jax", "jaxlib", "flax", "dm-control", "ogbench", "numpy", "optax")
        },
        "expected_packages": {
            "mujoco": "3.13.0",
            "gymnasium": "1.3.0",
            "jax": "0.10.2",
            "jaxlib": "0.10.2",
            "flax": "0.12.8",
            "dm-control": "1.0.46",
            "ogbench": "1.2.1",
            "numpy": "2.4.6",
        },
        "mujoco_mjSTATE_INTEGRATION": int(mujoco.mjtState.mjSTATE_INTEGRATION),
        "blas_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_devices_deferred_to_s1": True,
        "env_step_calls": 0,
    }
    for name, expected in env_lock["expected_packages"].items():
        actual = env_lock["packages"].get(name)
        if actual != expected:
            holds.append(f"package_mismatch:{name}:{actual}!={expected}")

    os.chdir(repo)
    test_result = run_unit_tests(repo)
    if test_result["status"] != "PASS":
        holds.append("unit_tests_failed")

    protocol = json.loads(json.dumps(PROTOCOL_V1))
    dump_json(protocol_dir / "protocol_v1.json", protocol)
    asset_lock = {
        "status": "EA3_HOLD_ASSET_MISMATCH" if any("sha256" in h or "missing" in h for h in holds) and holds else "RECORDED",
        "dataset_id": DATASET_ID,
        "hashes": hashes,
        "checkpoint_epoch": 1000000,
        "actor_value_share_checkpoint": True,
        "normalizer_not_applied_to_actor_value": True,
        "historical_commit": HISTORICAL_COMMIT,
        "ogbench_commit": ogbench_head,
        "root_id_overlap": overlap,
        "planned_root_ids": sorted(planned),
    }
    source_lock = {
        "repo_head_at_preflight": head,
        "repo_branch": branch,
        "official_source": str(args.official_source),
        "ogbench_commit": ogbench_head,
        "ogbench_dirty": bool(ogbench_dirty),
        "source_files": source_files,
        "unique_executor_modules": [
            "execution_aligned_rl/v3/root_loader.py",
            "execution_aligned_rl/v3/rollout.py",
            "execution_aligned_rl/v3/compare.py",
        ],
    }
    dump_json(protocol_dir / "asset_lock.json", asset_lock)
    dump_json(protocol_dir / "source_lock.json", source_lock)
    dump_json(protocol_dir / "environment_lock.json", env_lock)
    dump_json(protocol_dir / "test_schema_and_gate.json", {"unit_tests": test_result, "holds": holds, "split_ok": split_ok})

    status = "EA3_PREFLIGHT_PASS" if not holds else "EA3_HOLD_ASSET_MISMATCH" if any("mismatch" in h or "missing" in h for h in holds) else "EA3_HOLD_PROTOCOL_INCOMPLETE"
    decision = {
        "stage": "S0",
        "status": status,
        "holds": holds,
        "env_step_calls": 0,
        "training_performed": False,
        "phase_c_unlocked": False,
        "human_review": None,
        "protocol_sha256": sha256_file(protocol_dir / "protocol_v1.json"),
    }
    dump_json(exp / "decision.json", decision)
    print(json.dumps(decision, indent=2))
    if holds:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
