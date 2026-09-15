
"""Q2 driver: lock CPU runtime, pre/post inference checks, 250-episode official eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.cpu_adoption.inference_check import run_inference_check
from execution_aligned_rl.v3.cpu_adoption.lock import apply_canonical_cpu_env, collect_lock, lock_sha256, validate_lock
from execution_aligned_rl.v3.cpu_adoption.protocol import (
    BASELINE_COMMIT,
    DEFAULT_PATHS,
    GATES,
    PROTOCOL,
    PROTOCOL_ID,
    Q0_BRANCH,
    S1_COMMIT,
)
from execution_aligned_rl.v3.serialization import dump_json, load_json


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()


def write_zip(exp: Path) -> str:
    zip_path = exp / "package" / "ea_v3_q2_cpu_adoption_lightweight.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    names = [
        "protocol/cpu_adoption_protocol.json",
        "cpu_runtime_lock.json",
        "pre_eval_inference_check.json",
        "cpu_backbone_evaluation.json",
        "cpu_backbone_episode_results.jsonl",
        "post_eval_inference_check.json",
        "CPU_RUNTIME_ADOPTION_REPORT.md",
        "decision.json",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in names:
            path = exp / rel
            if path.exists():
                zf.write(path, arcname=rel)
    digest = sha256_file(zip_path)
    (exp / "package" / "ea_v3_q2_cpu_adoption_lightweight.zip.sha256").write_text(digest + "  ea_v3_q2_cpu_adoption_lightweight.zip\n", encoding="utf-8")
    return digest


def decide(pre, post, evaluation, holds) -> str:
    if holds:
        return "EA32_HOLD_ASSET_MISMATCH"
    if pre["status"] != "PASS" or post["status"] != "PASS":
        return "EA32_HOLD_CPU_RUNTIME_DRIFT"
    if not evaluation.get("capability_pass"):
        return "EA32_HOLD_CPU_BACKBONE_CAPABILITY"
    if evaluation.get("n_episodes") != GATES["n_episodes_total"]:
        return "EA32_HOLD_ENGINEERING"
    return "EA32_CPU_CAPABILITY_PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo)
    exp = repo / DEFAULT_PATHS["experiment_rel"]
    (exp / "protocol").mkdir(parents=True, exist_ok=True)
    (exp / "package").mkdir(exist_ok=True)

    holds = []
    q0_head = _git(Path(DEFAULT_PATHS["q0_repo"]), "rev-parse", "HEAD")
    if q0_head != BASELINE_COMMIT:
        holds.append(f"q0_head_mismatch:{q0_head}")
    s1_head = _git(Path(DEFAULT_PATHS["s1_repo"]), "rev-parse", "HEAD")
    if s1_head != S1_COMMIT:
        holds.append(f"s1_head_mismatch:{s1_head}")

    dump_json(exp / "protocol" / "cpu_adoption_protocol.json", PROTOCOL)

    apply_canonical_cpu_env()
    lock = collect_lock(include_jax=True)
    lock["cpu_runtime_environment_sha256"] = lock_sha256(lock)
    lock["q0_branch"] = Q0_BRANCH
    lock["q0_commit"] = q0_head
    lock["s1_commit"] = s1_head
    lock_holds = validate_lock(lock)
    holds.extend(lock_holds)
    dump_json(exp / "cpu_runtime_lock.json", {**lock, "validation_holds": holds})
    if any("mismatch" in h or h.startswith("package_") or h == "python_version" or h.startswith("env_") or h.startswith("backend") for h in holds):
        decision = {"stage": "Q2", "status": "EA32_HOLD_ASSET_MISMATCH", "holds": holds, "protocol_id": PROTOCOL_ID}
        dump_json(exp / "decision.json", decision)
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)

    print("PRE_EVAL_CHECK", flush=True)
    pre = run_inference_check(repo, exp, "pre_eval", (0, 1))
    dump_json(exp / "pre_eval_inference_check.json", pre)
    if pre["status"] != "PASS":
        decision = {"stage": "Q2", "status": "EA32_HOLD_CPU_RUNTIME_DRIFT", "phase": "pre_eval", "check": pre, "holds": holds, "protocol_id": PROTOCOL_ID, "s1_retry_run": False}
        dump_json(exp / "decision.json", decision)
        (exp / "CPU_RUNTIME_ADOPTION_REPORT.md").write_text("# CPU Runtime Adoption\n\nStatus: `EA32_HOLD_CPU_RUNTIME_DRIFT` (pre-eval)\n", encoding="utf-8")
        print(json.dumps(decision, indent=2))
        raise SystemExit(3)

    if not args.skip_eval:
        print("CPU_BACKBONE_EVAL", flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["JAX_ENABLE_X64"] = "0"
        env.pop("XLA_FLAGS", None)
        subprocess.run(
            [sys.executable, "-m", "execution_aligned_rl.v3.cpu_adoption.evaluate_cpu", "--out-dir", str(exp), "--numpy-seed", "0"],
            check=True,
            cwd=str(repo),
            env=env,
        )
    evaluation = load_json(exp / "cpu_backbone_evaluation.json")

    print("POST_EVAL_CHECK", flush=True)
    post = run_inference_check(repo, exp, "post_eval", (2, 3))
    dump_json(exp / "post_eval_inference_check.json", post)

    status = decide(pre, post, evaluation, holds)
    decision = {
        "stage": "Q2",
        "status": status,
        "holds": holds,
        "pre_eval": pre["status"],
        "post_eval": post["status"],
        "overall_success": evaluation.get("overall_success"),
        "gpu_reference_overall_success": 0.36,
        "nonzero_success_tasks": evaluation.get("nonzero_success_tasks"),
        "finite_action_rate": evaluation.get("finite_action_rate"),
        "bounds_action_rate": evaluation.get("bounds_action_rate"),
        "cpu_runtime_environment_sha256": lock["cpu_runtime_environment_sha256"],
        "env_step_calls": "official_evaluator_250_episodes",
        "s1_retry_run": False,
        "s2_unlocked": False,
        "phase_c_unlocked": False,
        "training_performed": False,
        "human_review": None,
        "protocol_id": PROTOCOL_ID,
    }
    dump_json(exp / "decision.json", decision)
    report = "\n".join(
        [
            "# EA-V3 CPU Runtime Adoption Report",
            "",
            f"Status: `{status}`",
            "",
            f"- Canonical CPU runtime SHA256: `{lock['cpu_runtime_environment_sha256']}`",
            f"- Pre-eval inference check: `{pre['status']}` actor pairwise `{pre['actor_pairwise_max_abs']}`, value pairwise `{pre['value_pairwise_max_abs']}`, matches Q0 `{pre.get('matches_q0_cpu_actor_sha256')}`",
            f"- CPU backbone overall success: `{evaluation.get('overall_success')}` (gate >= 0.26; GPU reference 0.36 engineering-only)",
            f"- Nonzero-success tasks: `{evaluation.get('nonzero_success_tasks')}` / 5 (gate >= 4)",
            f"- Action finite/bounds: `{evaluation.get('finite_action_rate')}` / `{evaluation.get('bounds_action_rate')}`",
            f"- Post-eval inference check: `{post['status']}` actor pairwise `{post['actor_pairwise_max_abs']}`, value pairwise `{post['value_pairwise_max_abs']}`",
            "",
            "No new XLA flags. No S1 retry. No S2. Checkpoint and thresholds unchanged.",
            "",
            json.dumps({"decision": decision, "evaluation_tasks": evaluation.get("tasks")}, indent=2, default=str),
            "",
        ]
    )
    (exp / "CPU_RUNTIME_ADOPTION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    digest = write_zip(exp)
    print(json.dumps({"status": status, "zip_sha256": digest, "overall_success": evaluation.get("overall_success")}, indent=2))
    if status != "EA32_CPU_CAPABILITY_PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
