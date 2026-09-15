
"""Re-run 2 fresh CPU workers on the frozen Q0 corpus and compare hashes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from execution_aligned_rl.v3.cpu_adoption.protocol import DEFAULT_PATHS
from execution_aligned_rl.v3.serialization import dump_json, load_json


def pairwise_max_abs(arrays: list[np.ndarray]) -> float:
    worst = 0.0
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            worst = max(worst, float(np.max(np.abs(arrays[i] - arrays[j]))))
    return worst


def run_cpu_worker(repo: Path, corpus_dir: Path, out_path: Path, process_id: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["JAX_ENABLE_X64"] = "0"
    env.pop("XLA_FLAGS", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "execution_aligned_rl.v3.runtime_qual.worker",
        "--runtime",
        "CPU_SINGLE_THREAD",
        "--process-id",
        str(process_id),
        "--corpus-dir",
        str(corpus_dir),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)
    return load_json(out_path)


def summarize_pair(payloads: list[dict], q0_reference: dict | None) -> dict:
    actor_a = [np.asarray(r["action"], dtype=np.float64) for r in payloads[0]["actor"]]
    actor_b = [np.asarray(r["action"], dtype=np.float64) for r in payloads[1]["actor"]]
    value_a = [np.asarray(r["value"], dtype=np.float64) for r in payloads[0]["value"]]
    value_b = [np.asarray(r["value"], dtype=np.float64) for r in payloads[1]["value"]]
    actor_hash_a = [r["action_sha256"] for r in payloads[0]["actor"]]
    actor_hash_b = [r["action_sha256"] for r in payloads[1]["actor"]]
    value_hash_a = [r["value_sha256"] for r in payloads[0]["value"]]
    value_hash_b = [r["value_sha256"] for r in payloads[1]["value"]]
    actor_pair = pairwise_max_abs([np.stack(actor_a), np.stack(actor_b)]) if False else max(float(np.max(np.abs(a - b))) for a, b in zip(actor_a, actor_b))
    value_pair = max(float(np.max(np.abs(a - b))) for a, b in zip(value_a, value_b))
    actor_sha_same = actor_hash_a == actor_hash_b
    value_sha_same = value_hash_a == value_hash_b
    q0_actor_same = None
    q0_value_same = None
    if q0_reference is not None:
        q0_actor = [r["action_sha256"] for r in q0_reference["actor"]]
        q0_value = [r["value_sha256"] for r in q0_reference["value"]]
        q0_actor_same = q0_actor == actor_hash_a == actor_hash_b
        q0_value_same = q0_value == value_hash_a == value_hash_b
    ok = actor_pair == 0.0 and value_pair == 0.0 and actor_sha_same and value_sha_same and (q0_actor_same is not False) and (q0_value_same is not False)
    return {
        "n_processes": len(payloads),
        "n_actor": len(actor_a),
        "n_value": len(value_a),
        "actor_pairwise_max_abs": actor_pair,
        "value_pairwise_max_abs": value_pair,
        "actor_sha256_identical": actor_sha_same,
        "value_sha256_identical": value_sha_same,
        "matches_q0_cpu_actor_sha256": q0_actor_same,
        "matches_q0_cpu_value_sha256": q0_value_same,
        "status": "PASS" if ok else "FAIL",
        "backends": [p["env"]["jax_default_backend"] for p in payloads],
        "jax_enable_x64": [p["env"]["jax_enable_x64"] for p in payloads],
    }


def run_inference_check(repo: Path, exp: Path, tag: str, process_ids: tuple[int, int]) -> dict:
    q0 = Path(DEFAULT_PATHS["q0_repo"]) / DEFAULT_PATHS["q0_experiment_rel"]
    corpus_dir = q0 / "corpus"
    q0_ref = load_json(q0 / "processes" / "CPU_SINGLE_THREAD" / "process_00.json")
    payloads = []
    for pid in process_ids:
        out = exp / "inference_checks" / tag / f"process_{pid:02d}.json"
        payloads.append(run_cpu_worker(repo, corpus_dir, out, pid))
    summary = summarize_pair(payloads, q0_ref)
    summary["tag"] = tag
    summary["corpus_dir"] = str(corpus_dir)
    dump_json(exp / f"{tag}_inference_check.json", summary)
    return summary
