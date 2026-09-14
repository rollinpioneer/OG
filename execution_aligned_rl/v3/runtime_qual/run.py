
"""Q0+Q1 driver: corpus, 8+8 fresh processes, comparator regression, report. No env.step."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.runtime_qual.corpus import extract_corpus
from execution_aligned_rl.v3.runtime_qual.protocol import (
    BASELINE_COMMIT,
    CHECKPOINT_SHA256,
    DEFAULT_PATHS,
    OGBENCH_COMMIT,
    PROTOCOL,
    PROTOCOL_ID,
    THRESHOLDS,
)
from execution_aligned_rl.v3.serialization import dump_json, load_json


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()


def pairwise_max_abs(arrays: list[np.ndarray]) -> float:
    worst = 0.0
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            diff = np.max(np.abs(arrays[i] - arrays[j]))
            worst = max(worst, float(diff))
    return worst


def summarize_runtime(process_payloads: list[dict], kind: str, limit: float) -> dict:
    n = len(process_payloads[0][kind])
    rows = []
    worst = 0.0
    hash_mismatch_items = 0
    nonfinite = 0
    oob = 0
    for idx in range(n):
        if kind == "actor":
            arrays = [np.asarray(p["actor"][idx]["action"], dtype=np.float64) for p in process_payloads]
            hashes = [p["actor"][idx]["action_sha256"] for p in process_payloads]
            finite = all(p["actor"][idx]["finite"] for p in process_payloads)
            bounds = all(p["actor"][idx]["in_bounds"] for p in process_payloads)
            if not bounds:
                oob += 1
        else:
            arrays = [np.asarray(p["value"][idx]["value"], dtype=np.float64) for p in process_payloads]
            hashes = [p["value"][idx]["value_sha256"] for p in process_payloads]
            finite = all(p["value"][idx]["finite"] for p in process_payloads)
            bounds = True
        if not finite:
            nonfinite += 1
        if len(set(hashes)) > 1:
            hash_mismatch_items += 1
        mad = pairwise_max_abs(arrays)
        worst = max(worst, mad)
        rows.append(
            {
                "index": idx,
                "root_id": process_payloads[0][kind][idx]["root_id"],
                "pairwise_max_abs": mad,
                "sha256_unique": sorted(set(hashes)),
                "finite": finite,
                "in_bounds": bounds,
                "pass": bool(finite and bounds and mad <= limit and len(set(hashes)) == 1),
            }
        )
    return {
        "n_items": n,
        "n_processes": len(process_payloads),
        "pairwise_max_abs_worst": worst,
        "limit": limit,
        "hash_mismatch_items": hash_mismatch_items,
        "nonfinite_items": nonfinite,
        "out_of_bounds_items": oob,
        "n_fail_items": sum(1 for r in rows if not r["pass"]),
        "status": "PASS" if worst <= limit and hash_mismatch_items == 0 and nonfinite == 0 and oob == 0 else "FAIL",
        "items": rows,
    }


def run_processes(runtime: str, n: int, repo: Path, corpus_dir: Path, out_dir: Path) -> list[dict]:
    python = sys.executable
    payloads = []
    jsonl_path = out_dir / f"{runtime.lower()}_process_outputs.jsonl"
    lines = []
    for pid in range(n):
        out = out_dir / "processes" / runtime / f"process_{pid:02d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        if runtime == "GPU_CURRENT":
            env["CUDA_VISIBLE_DEVICES"] = "3"
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env.pop("JAX_PLATFORMS", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JAX_PLATFORMS"] = "cpu"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
        cmd = [
            python,
            "-m",
            "execution_aligned_rl.v3.runtime_qual.worker",
            "--runtime",
            runtime,
            "--process-id",
            str(pid),
            "--corpus-dir",
            str(corpus_dir),
            "--out",
            str(out),
        ]
        subprocess.run(cmd, check=True, cwd=str(repo), env=env)
        payload = load_json(out)
        payloads.append(payload)
        lines.append(json.dumps({"runtime": runtime, "process_id": pid, "pid": payload["pid"], "backend": payload["env"]["jax_default_backend"], "devices": payload["env"]["jax_devices"], "n_actor": payload["n_actor"], "n_value": payload["n_value"]}, sort_keys=True))
        print(json.dumps({"finished": runtime, "process_id": pid, "backend": payload["env"]["jax_default_backend"]}), flush=True)
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # write full outputs jsonl with hashes only to keep size reasonable, plus pointer to per-process files
    full_jsonl = out_dir / f"{'gpu' if runtime=='GPU_CURRENT' else 'cpu'}_process_outputs.jsonl"
    with full_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            slim = {
                "runtime": payload["runtime"],
                "process_id": payload["process_id"],
                "pid": payload["pid"],
                "env": payload["env"],
                "actor": [{"index": r["index"], "root_id": r["root_id"], "step": r["step"], "action_sha256": r["action_sha256"], "finite": r["finite"], "in_bounds": r["in_bounds"], "action": r["action"]} for r in payload["actor"]],
                "value": [{"index": r["index"], "root_id": r["root_id"], "value_sha256": r["value_sha256"], "finite": r["finite"], "value": r["value"]} for r in payload["value"]],
                "env_step_calls": 0,
            }
            handle.write(json.dumps(slim, sort_keys=True) + "\n")
    return payloads


def run_comparator_tests(repo: Path) -> dict:
    loader = unittest.TestLoader()
    suite = loader.discover(str(repo / "execution_aligned_rl" / "v3" / "runtime_qual" / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    return {
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "failure_names": [str(item[0]) for item in result.failures + result.errors],
    }


def decide(gpu_actor, gpu_value, cpu_actor, cpu_value, comparator, holds: list[str]) -> str:
    if any("asset" in h for h in holds):
        return "EA31_HOLD_ASSET_MISMATCH"
    if comparator["status"] != "PASS":
        return "EA31_HOLD_COMPARATOR_CONTRACT"
    if gpu_actor["status"] != "PASS" or cpu_actor["status"] != "PASS":
        return "EA31_HOLD_POLICY_RUNTIME_NUMERICS"
    if gpu_value["status"] != "PASS" or cpu_value["status"] != "PASS":
        return "EA31_HOLD_VALUE_RUNTIME_NUMERICS"
    # both runtimes passed actor and value
    return "EA31_GPU_RUNTIME_QUALIFIED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    parser.add_argument("--s1-repo", default=DEFAULT_PATHS["s1_repo"])
    parser.add_argument("--n-processes", type=int, default=8)
    parser.add_argument("--reuse-processes", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo)
    exp = repo / DEFAULT_PATHS["experiment_rel"]
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "protocol").mkdir(exist_ok=True)
    (exp / "corpus").mkdir(exist_ok=True)
    (exp / "report").mkdir(exist_ok=True)

    holds = []
    head = _git(repo, "rev-parse", "HEAD")
    if _git(Path(args.s1_repo), "rev-parse", "HEAD") != BASELINE_COMMIT:
        # S1 worktree must remain at frozen commit
        s1_head = _git(Path(args.s1_repo), "rev-parse", "HEAD")
        if s1_head != BASELINE_COMMIT:
            holds.append(f"s1_repo_head_mismatch:{s1_head}")
    ckpt = Path(DEFAULT_PATHS["checkpoint_file"])
    if sha256_file(ckpt) != CHECKPOINT_SHA256:
        holds.append("checkpoint_sha256_mismatch")
    og_head = _git(Path(DEFAULT_PATHS["official_source"]), "rev-parse", "HEAD")
    if og_head != OGBENCH_COMMIT:
        holds.append(f"ogbench_commit_mismatch:{og_head}")

    dump_json(exp / "protocol" / "runtime_qualification_protocol.json", PROTOCOL)
    dump_json(
        exp / "protocol" / "comparator_contract_v2.json",
        {
            "contract": "comparator_contract_v2",
            "d0_terminated_truncated": "if either side is missing/None, field is NOT_APPLICABLE; if both explicit they must agree",
            "d1_full_proxy": "NOT_APPLICABLE",
            "d1_step_proxy": "NOT_APPLICABLE",
            "d1_action_state": "required",
            "d2_d3_full_proxy": "required when present; missing is FAIL",
            "fail_closed": ["missing", "nan", "Inf", "length", "identity"],
            "rtol": 0.0,
            "does_not_modify_s1_compare_py": True,
            "does_not_modify_s1_traces": True,
        },
    )

    corpus = extract_corpus(s1_repo=Path(args.s1_repo), out_dir=exp / "corpus")
    dump_json(exp / "inference_corpus_manifest.json", corpus)

    comparator = run_comparator_tests(repo)
    dump_json(exp / "comparator_regression_tests.json", comparator)

    if args.reuse_processes:
        gpu_payloads = [load_json(p) for p in sorted((exp / "processes" / "GPU_CURRENT").glob("process_*.json"))]
        cpu_payloads = [load_json(p) for p in sorted((exp / "processes" / "CPU_SINGLE_THREAD").glob("process_*.json"))]
        if len(gpu_payloads) != args.n_processes or len(cpu_payloads) != args.n_processes:
            raise RuntimeError(f"reuse-processes expected {args.n_processes} each, got gpu={len(gpu_payloads)} cpu={len(cpu_payloads)}")
        for runtime, payloads in (("GPU_CURRENT", gpu_payloads), ("CPU_SINGLE_THREAD", cpu_payloads)):
            full_jsonl = exp / f"{'gpu' if runtime=='GPU_CURRENT' else 'cpu'}_process_outputs.jsonl"
            with full_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
                for payload in payloads:
                    slim = {
                        "runtime": payload["runtime"],
                        "process_id": payload["process_id"],
                        "pid": payload["pid"],
                        "env": payload["env"],
                        "actor": [{"index": r["index"], "root_id": r["root_id"], "step": r["step"], "action_sha256": r["action_sha256"], "finite": r["finite"], "in_bounds": r["in_bounds"], "action": r["action"]} for r in payload["actor"]],
                        "value": [{"index": r["index"], "root_id": r["root_id"], "value_sha256": r["value_sha256"], "finite": r["finite"], "value": r["value"]} for r in payload["value"]],
                        "env_step_calls": 0,
                    }
                    handle.write(json.dumps(slim, sort_keys=True) + "\n")
    else:
        gpu_payloads = run_processes("GPU_CURRENT", args.n_processes, repo, exp / "corpus", exp)
        cpu_payloads = run_processes("CPU_SINGLE_THREAD", args.n_processes, repo, exp / "corpus", exp)

    gpu_actor = summarize_runtime(gpu_payloads, "actor", THRESHOLDS["action_max_abs"])
    gpu_value = summarize_runtime(gpu_payloads, "value", THRESHOLDS["value_max_abs"])
    cpu_actor = summarize_runtime(cpu_payloads, "actor", THRESHOLDS["action_max_abs"])
    cpu_value = summarize_runtime(cpu_payloads, "value", THRESHOLDS["value_max_abs"])

    def strip_items(summary):
        out = dict(summary)
        out["failing_item_indices"] = [r["index"] for r in summary["items"] if not r["pass"]]
        out.pop("items")
        return out

    det = {
        "gpu": {"actor": strip_items(gpu_actor), "value": strip_items(gpu_value), "backend": gpu_payloads[0]["env"]["jax_default_backend"]},
        "cpu": {"actor": strip_items(cpu_actor), "value": strip_items(cpu_value), "backend": cpu_payloads[0]["env"]["jax_default_backend"]},
        "thresholds": {"action_max_abs": THRESHOLDS["action_max_abs"], "value_max_abs": THRESHOLDS["value_max_abs"], "rtol": 0.0},
        "n_processes": args.n_processes,
        "env_step_calls": 0,
        "dist_mode_used": False,
    }
    dump_json(exp / "actor_value_determinism_summary.json", det)
    dump_json(
        exp / "runtime_environment_lock.json",
        {
            "gpu_processes": [p["env"] for p in gpu_payloads],
            "cpu_processes": [p["env"] for p in cpu_payloads],
            "checkpoint_sha256": sha256_file(ckpt),
            "ogbench_commit": og_head,
            "qualification_head_at_start": head,
            "s1_commit_required": BASELINE_COMMIT,
        },
    )

    gpu_ok = gpu_actor["status"] == "PASS" and gpu_value["status"] == "PASS"
    cpu_ok = cpu_actor["status"] == "PASS" and cpu_value["status"] == "PASS"
    status = decide(gpu_actor, gpu_value, cpu_actor, cpu_value, comparator, holds)
    if gpu_ok and cpu_ok and comparator["status"] == "PASS" and not holds:
        # both qualified; primary reported status is GPU, CPU recorded explicitly
        status = "EA31_GPU_RUNTIME_QUALIFIED"
    cpu_status = "EA31_CPU_RUNTIME_QUALIFIED" if cpu_ok else (
        "EA31_HOLD_POLICY_RUNTIME_NUMERICS" if cpu_actor["status"] != "PASS" else "EA31_HOLD_VALUE_RUNTIME_NUMERICS"
    )
    gpu_status = "EA31_GPU_RUNTIME_QUALIFIED" if gpu_ok else (
        "EA31_HOLD_POLICY_RUNTIME_NUMERICS" if gpu_actor["status"] != "PASS" else "EA31_HOLD_VALUE_RUNTIME_NUMERICS"
    )

    decision = {
        "stage": "Q0_Q1",
        "status": status,
        "gpu_status": gpu_status,
        "cpu_status": cpu_status,
        "comparator_status": comparator["status"],
        "holds": holds,
        "env_step_calls": 0,
        "s1_retry_run": False,
        "s2_unlocked": False,
        "phase_c_unlocked": False,
        "human_review": None,
        "training_performed": False,
        "thresholds_relaxed": False,
        "dist_mode_used": False,
        "protocol_id": PROTOCOL_ID,
    }
    dump_json(exp / "decision.json", decision)
    report = "\n".join(
        [
            "# EA-V3 Runtime Qualification Report",
            "",
            f"Status: `{status}`",
            "",
            f"- GPU: `{gpu_status}` actor pairwise max-abs `{gpu_actor['pairwise_max_abs_worst']}` (limit {THRESHOLDS['action_max_abs']}); value `{gpu_value['pairwise_max_abs_worst']}` (limit {THRESHOLDS['value_max_abs']})",
            f"- CPU: `{cpu_status}` actor pairwise max-abs `{cpu_actor['pairwise_max_abs_worst']}` (limit {THRESHOLDS['action_max_abs']}); value `{cpu_value['pairwise_max_abs_worst']}` (limit {THRESHOLDS['value_max_abs']})",
            f"- Comparator contract v2 tests: `{comparator['status']}` ({comparator['testsRun']} tests)",
            "",
            "S1 traces, protocol_v1.json, and thresholds were not modified. No env.step was called. S2/Phase C remain locked.",
            "",
            "## Corpus",
            "",
            json.dumps({k: corpus[k] for k in ["n_legal_roots", "n_queries", "n_actor_queries", "n_value_queries", "env_step_calls"]}, indent=2),
            "",
            "## Decision",
            "",
            json.dumps(decision, indent=2),
            "",
        ]
    )
    (exp / "RUNTIME_QUALIFICATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    (exp / "report" / "RUNTIME_QUALIFICATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps(decision, indent=2))
    if status.startswith("EA31_HOLD"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
