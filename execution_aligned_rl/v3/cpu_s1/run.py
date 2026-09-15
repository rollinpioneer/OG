
"""Orchestrate R0 + CPU S1. Collector exits before workers. No JAX import here."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.cpu_adoption.inference_check import run_inference_check
from execution_aligned_rl.v3.cpu_s1.protocol import (
    BUDGET,
    DEFAULT_PATHS,
    PROTOCOL,
    PROTOCOL_ID,
    REGRESSION_D3_INDEX,
    STEPS,
)
from execution_aligned_rl.v3.hashing import stable_int
from execution_aligned_rl.v3.serialization import dump_json, load_json


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


def freeze_root_specs(exp: Path) -> dict:
    train = np.load(Path(DEFAULT_PATHS["dataset_dir"]) / "cube-double-play-v0.npz", allow_pickle=False)
    n_train = int(len(train["observations"]))
    roots = []
    for i in range(15):
        old_id = 690000 + i
        roots.append(
            {
                "set": "regression",
                "root_id": 710000 + i,
                "reset_seed": 690000 + i,
                "task_id": 1 + i // 3,
                "planned_decision_step": STEPS[i % 3],
                "d3_target_index": int(REGRESSION_D3_INDEX[old_id]),
                "d3_index_source": "frozen_gpu_s1_manifest",
            }
        )
    for i in range(15):
        root_id = 711000 + i
        formula = f"ea3-cpu-s1-v1:d3-endpoint:{root_id}"
        roots.append(
            {
                "set": "holdout",
                "root_id": root_id,
                "reset_seed": 691000 + i,
                "task_id": 1 + i // 3,
                "planned_decision_step": STEPS[i % 3],
                "d3_target_index": int(stable_int(formula) % n_train),
                "d3_index_source": formula,
            }
        )
    payload = {"n_train": n_train, "roots": roots, "protocol_id": PROTOCOL_ID}
    dump_json(exp / "protocol" / "root_specs.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    args = parser.parse_args()
    repo = Path(args.repo)
    exp = repo / DEFAULT_PATHS["experiment_rel"]
    for part in ("protocol", "identities", "manifests", "engineering", "report", "package"):
        (exp / part).mkdir(parents=True, exist_ok=True)
    dump_json(exp / "protocol" / "cpu_s1_protocol.json", PROTOCOL)
    dump_json(
        exp / "protocol" / "runtime_identity_contract_v2.json",
        {
            "schema": "runtime_identity_v2",
            "excluded_from_hash": ["CPU scaling MHz", "CPU MHz", "BogoMIPS", "pid", "timestamps", "wall time", "utilization", "memory load", "role"],
            "included": [
                "canonical_env",
                "cpu_stable architecture/vendor/model/family/model/stepping",
                "python/packages",
                "checkpoint/data sha256",
                "ogbench commit",
                "jax backend/devices/x64/matmul",
                "no_xla_flags",
                "comparator_contract_v2_sha256",
            ],
            "lscpu_raw": "diagnostics_only",
        },
    )
    freeze_root_specs(exp)

    print("R0_IDENTITY", flush=True)
    shas = []
    for role in ("r0_collector", "r0_worker_A", "r0_worker_B", "r0_finalizer"):
        out = exp / "identities" / f"{role}.json"
        run_mod(repo, "execution_aligned_rl.v3.cpu_s1.runtime_identity", ["--role", role, "--out", str(out)])
        shas.append(load_json(out)["runtime_identity_sha256"])
    if len(set(shas)) != 1:
        decision = {"stage": "R0", "status": "EA33_HOLD_RUNTIME_IDENTITY", "shas": shas, "s2_unlocked": False, "protocol_id": PROTOCOL_ID}
        dump_json(exp / "decision.json", decision)
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)
    print(json.dumps({"r0": "PASS", "sha256": shas[0]}), flush=True)

    print("PRE_S1_INFERENCE", flush=True)
    pre = run_inference_check(repo, exp, "pre_s1", (0, 1))
    dump_json(exp / "pre_s1_inference_check.json", pre)
    if pre["status"] != "PASS":
        decision = {"stage": "PRE_S1", "status": "EA33_HOLD_RUNTIME_IDENTITY", "check": pre, "s2_unlocked": False, "protocol_id": PROTOCOL_ID}
        dump_json(exp / "decision.json", decision)
        raise SystemExit(2)

    common = [
        "--repo",
        str(repo),
        "--dataset-dir",
        DEFAULT_PATHS["dataset_dir"],
        "--official-source",
        DEFAULT_PATHS["official_source"],
        "--checkpoint-dir",
        DEFAULT_PATHS["checkpoint_dir"],
        "--root-store",
        DEFAULT_PATHS["root_store"],
    ]
    print("ACQUIRE", flush=True)
    run_mod(repo, "execution_aligned_rl.v3.cpu_s1.process", ["acquire", *common])
    print("WORKER_A", flush=True)
    run_mod(repo, "execution_aligned_rl.v3.cpu_s1.process", ["worker", "--worker-id", "A", *common])
    print("WORKER_B", flush=True)
    run_mod(repo, "execution_aligned_rl.v3.cpu_s1.process", ["worker", "--worker-id", "B", *common])

    print("POST_S1_INFERENCE", flush=True)
    post = run_inference_check(repo, exp, "post_s1", (2, 3))
    dump_json(exp / "post_s1_inference_check.json", post)

    print("FINALIZE", flush=True)
    run_mod(repo, "execution_aligned_rl.v3.cpu_s1.finalize", ["--repo", str(repo), "--root-store", DEFAULT_PATHS["root_store"]])
    decision = load_json(exp / "decision.json")
    if post["status"] != "PASS" and decision.get("status") == "EA33_CPU_S1_QUALIFIED":
        decision["status"] = "EA33_HOLD_RUNTIME_IDENTITY"
        decision["post_s1_inference"] = post["status"]
        dump_json(exp / "decision.json", decision)
        raise SystemExit(3)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
