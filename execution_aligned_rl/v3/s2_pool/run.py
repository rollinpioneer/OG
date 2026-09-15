
"""S2 orchestrator. Protocol is already frozen. Collector exits before workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from execution_aligned_rl.v3.cpu_adoption.inference_check import run_inference_check
from execution_aligned_rl.v3.s2_pool.protocol import DEFAULT_PATHS, EXPECTED_IDENTITY, PROTOCOL_ID
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


def run_mod(repo, module, args):
    subprocess.run([sys.executable, "-m", module, *args], check=True, cwd=str(repo), env=cpu_env(repo))


def write_root_specs(exp: Path) -> None:
    plan = load_json(exp / "protocol" / "formal_root_plan.json")
    roots = []
    for r in plan["roots"]:
        roots.append(
            {
                "set": "formal",
                "root_id": r["root_id"],
                "reset_seed": r["reset_seed"],
                "task_id": r["task_id"],
                "planned_decision_step": r["planned_decision_step"],
                "d3_target_index": r["qualification_target_index"],
                "is_deep": r["is_deep"],
            }
        )
    dump_json(exp / "protocol" / "root_specs.json", {"protocol_id": PROTOCOL_ID, "roots": roots})


def main() -> None:
    repo = Path(DEFAULT_PATHS["repo"])
    exp = repo / DEFAULT_PATHS["experiment_rel"]
    for part in ("identities", "manifests", "engineering", "report", "package"):
        (exp / part).mkdir(parents=True, exist_ok=True)
    write_root_specs(exp)

    print("R0_IDENTITY", flush=True)
    shas = []
    for role in ("r0_collector", "r0_worker_A", "r0_worker_B", "r0_finalizer"):
        out = exp / "identities" / f"{role}.json"
        run_mod(repo, "execution_aligned_rl.v3.cpu_s1.runtime_identity", ["--role", role, "--out", str(out)])
        shas.append(load_json(out)["runtime_identity_sha256"])
    if len(set(shas)) != 1 or shas[0] != EXPECTED_IDENTITY:
        dump_json(exp / "decision.json", {"status": "EA34_HOLD_RUNTIME_IDENTITY", "shas": shas, "expected": EXPECTED_IDENTITY, "s3_unlocked": False})
        raise SystemExit(2)
    dump_json(exp / "runtime_identity_manifest.json", {"sha256": shas[0], "n": 4, "match_expected": True})

    print("PRE_S2_INFERENCE", flush=True)
    pre = run_inference_check(repo, exp, "pre_s2", (0, 1))
    dump_json(exp / "pre_s2_inference_check.json", pre)
    if pre["status"] != "PASS":
        dump_json(exp / "decision.json", {"status": "EA34_HOLD_RUNTIME_IDENTITY", "phase": "pre", "s3_unlocked": False})
        raise SystemExit(2)

    common = ["--repo", str(repo), "--dataset-dir", DEFAULT_PATHS["dataset_dir"], "--official-source", DEFAULT_PATHS["official_source"], "--checkpoint-dir", DEFAULT_PATHS["checkpoint_dir"], "--root-store", DEFAULT_PATHS["root_store"]]
    print("ACQUIRE", flush=True)
    run_mod(repo, "execution_aligned_rl.v3.s2_pool.process", ["acquire", *common])
    print("WORKER_A", flush=True)
    run_mod(repo, "execution_aligned_rl.v3.s2_pool.process", ["worker", "--worker-id", "A", *common])
    print("WORKER_B", flush=True)
    run_mod(repo, "execution_aligned_rl.v3.s2_pool.process", ["worker", "--worker-id", "B", *common])
    print("POST_S2_INFERENCE", flush=True)
    post = run_inference_check(repo, exp, "post_s2", (2, 3))
    dump_json(exp / "post_s2_inference_check.json", post)
    print("FINALIZE", flush=True)
    run_mod(repo, "execution_aligned_rl.v3.s2_pool.finalize", ["--repo", str(repo)])
    if post["status"] != "PASS":
        d = load_json(exp / "decision.json")
        if d.get("status") == "EA34_ROOT_AND_CANDIDATE_POOL_LOCKED":
            d["status"] = "EA34_HOLD_RUNTIME_IDENTITY"
            dump_json(exp / "decision.json", d)
            raise SystemExit(3)


if __name__ == "__main__":
    main()
