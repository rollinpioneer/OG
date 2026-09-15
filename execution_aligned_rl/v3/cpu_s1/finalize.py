
"""CPU S1 finalize: comparator v2, negatives, coverage, packaging. No JAX required except identity."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.cpu_s1.protocol import BUDGET, DEFAULT_PATHS, PROTOCOL_ID
from execution_aligned_rl.v3.cpu_s1.runtime_identity import apply_cpu_env, write_identity
from execution_aligned_rl.v3.root_bundle import load_trace, read_root_bundle, root_dir
from execution_aligned_rl.v3.runtime_qual.compare_v2 import compare_traces_v2
from execution_aligned_rl.v3.serialization import dump_json, load_json


def experiment_dir(repo: Path) -> Path:
    return repo / DEFAULT_PATHS["experiment_rel"]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _open_loop_from_original(original: dict, n):
    src = original["steps"] if n is None else original["steps"][:n]
    steps = [dict(s) for s in src]
    out = dict(original)
    out["steps"] = steps
    return out


def _coverage(roots: list[dict], *, require_mid: bool) -> dict:
    legal = [r for r in roots if r.get("legal")]
    tasks = {int(r["task_id"]) for r in legal}
    steps = {int(r["planned_decision_step"]) for r in legal}
    ok = len(legal) >= 10 and tasks == {1, 2, 3, 4, 5}
    if require_mid:
        ok = ok and (125 in steps) and (250 in steps)
    return {
        "legal": len(legal),
        "planned": len(roots),
        "tasks": sorted(tasks),
        "decision_steps_present": sorted(steps),
        "ok": ok,
        "ineligible": [r["root_id"] for r in roots if not r.get("legal")],
    }


def negative_tests(exp: Path, sample_bundle: dict, expected_identity_sha: str) -> dict:
    original = dict(sample_bundle["original_live_probe"])
    original["goal_observation"] = sample_bundle["goal_observation"]
    ident = dict(original.get("identity") or {})
    ident.update({"root_id": sample_bundle["manifest"]["root_id"], "task_id": sample_bundle["manifest"]["task_id"], "protocol_id": PROTOCOL_ID})
    original["identity"] = ident
    cases = []

    tmp = exp / "engineering" / "negative" / "byteflip"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(sample_bundle["directory"], tmp)
    target = tmp / "goal_observation.npy"
    data = bytearray(target.read_bytes())
    data[-1] = (data[-1] + 1) % 256
    target.write_bytes(bytes(data))
    try:
        read_root_bundle(tmp, verify=True)
        hash_detected = False
        hash_error = "no_exception"
    except Exception as exc:
        hash_detected = True
        hash_error = type(exc).__name__ + ": " + str(exc)
    cases.append({"name": "byte_flip_root_file", "expected": "hash_failure", "detected": hash_detected, "detail": hash_error})

    other = dict(original)
    other["goal_observation"] = np.asarray(original["goal_observation"]).copy()
    other["goal_observation"][19] += 0.25
    cmp_goal = compare_traces_v2(original, other, phase="D3")
    cases.append({"name": "replace_37d_goal_keep_task", "expected": "GOAL_ENCODING", "detected": "GOAL_ENCODING" in cmp_goal["reasons"], "status": cmp_goal["status"], "reasons": cmp_goal["reasons"]})

    other = dict(original)
    other["elapsed_steps_start"] = int(original.get("elapsed_steps_start") or 0) + 1
    cmp_elapsed = compare_traces_v2(original, other, phase="D3")
    cases.append({"name": "elapsed_plus_one", "expected": "ELAPSED_START", "detected": "ELAPSED_START" in cmp_elapsed["reasons"], "status": cmp_elapsed["status"]})

    other = dict(original)
    other["steps"] = list(original["steps"])[:-1]
    cmp_len = compare_traces_v2(original, other, phase="D3")
    cases.append({"name": "drop_last_step", "expected": "LENGTH_MISMATCH", "detected": "LENGTH_MISMATCH" in cmp_len["reasons"], "status": cmp_len["status"]})

    nan_trace = dict(original)
    nan_steps = [dict(s) for s in original["steps"]]
    if nan_steps:
        obs = np.asarray(nan_steps[-1]["observation"]).copy()
        obs[0] = np.nan
        nan_steps[-1]["observation"] = obs
    nan_trace["steps"] = nan_steps
    cmp_nan = compare_traces_v2(original, nan_trace, phase="D3")
    cases.append({"name": "nan_observation", "expected": "NONFINITE", "detected": cmp_nan["status"] == "FAIL", "status": cmp_nan["status"]})

    biased = dict(original)
    bsteps = [dict(s) for s in original["steps"]]
    if bsteps:
        bsteps[0]["action"] = np.asarray(bsteps[0]["action"]) + 1e-3
    biased["steps"] = bsteps
    cmp_act = compare_traces_v2(original, biased, phase="D3")
    cases.append({"name": "action_bias", "expected": "action threshold", "detected": cmp_act["status"] == "FAIL", "status": cmp_act["status"]})

    other = dict(original)
    other["continued_after_terminal"] = True
    cmp_term = compare_traces_v2(original, other, phase="D3")
    cases.append({"name": "continue_after_terminal", "expected": "TERMINATION_PROTOCOL", "detected": "TERMINATION_PROTOCOL" in cmp_term["reasons"], "status": cmp_term["status"]})

    fake = dict(original)
    fake["identity"] = dict(original.get("identity") or {})
    fake["identity"]["root_id"] = int(sample_bundle["manifest"]["root_id"]) + 999
    cmp_wrong = compare_traces_v2(original, fake, phase="D3")
    cases.append({"name": "wrong_root_identity", "expected": "IDENTITY_root_id", "detected": any("IDENTITY" in r for r in cmp_wrong["reasons"]), "status": cmp_wrong["status"]})

    ident_path = exp / "identities" / "collector.json"
    ident = load_json(ident_path)
    mutated = dict(ident)
    mutated["jax_enable_x64"] = True
    from execution_aligned_rl.v3.cpu_s1.runtime_identity import identity_sha256

    mutated_sha = identity_sha256(mutated)
    mismatch = mutated_sha != expected_identity_sha
    cases.append({"name": "runtime_identity_mismatch", "expected": "identity_sha256_change", "detected": mismatch, "left": expected_identity_sha, "right": mutated_sha})

    passed = all(case["detected"] for case in cases)
    return {"status": "PASS" if passed else "FAIL", "cases": cases, "n_cases": len(cases), "passed_cases": sum(1 for c in cases if c["detected"])}


def finalize_main(args) -> None:
    apply_cpu_env()
    repo = Path(args.repo)
    exp = experiment_dir(repo)
    ident = write_identity(exp / "identities" / "finalizer.json", "finalizer")
    store = Path(args.root_store)
    all_rows = load_json(exp / "manifests" / "all_roots.json")["roots"]
    legal_ids = [r["root_id"] for r in all_rows if r.get("legal")]
    rows = []
    all_pass = True
    for row in all_rows:
        if not row.get("legal"):
            continue
        root_id = row["root_id"]
        bundle = read_root_bundle(root_dir(store, PROTOCOL_ID, root_id), verify=True)
        original = dict(bundle["original_live_probe"])
        original["goal_observation"] = bundle["goal_observation"]
        original.setdefault("identity", {})
        original["identity"].update({"root_id": root_id, "task_id": row["task_id"], "protocol_id": PROTOCOL_ID})
        d0a = load_json(exp / "engineering" / "workers" / "A" / str(root_id) / "primary" / "d0.json")
        d0b = load_json(exp / "engineering" / "workers" / "B" / str(root_id) / "primary" / "d0.json")
        d1a = load_trace(exp / "engineering" / "workers" / "A" / str(root_id) / "primary" / "d1.npz")
        d1b = load_trace(exp / "engineering" / "workers" / "B" / str(root_id) / "primary" / "d1.npz")
        d2a = load_trace(exp / "engineering" / "workers" / "A" / str(root_id) / "primary" / "d2.npz")
        d2b = load_trace(exp / "engineering" / "workers" / "B" / str(root_id) / "primary" / "d2.npz")
        d3a = load_trace(exp / "engineering" / "workers" / "A" / str(root_id) / "primary" / "d3.npz")
        d3b = load_trace(exp / "engineering" / "workers" / "B" / str(root_id) / "primary" / "d3.npz")
        for tr in (d1a, d1b, d2a, d2b, d3a, d3b):
            tr["goal_observation"] = bundle["goal_observation"]
            tr.setdefault("identity", {})
            tr["identity"].update({"root_id": root_id, "task_id": row["task_id"], "protocol_id": PROTOCOL_ID})
        cmp = {
            "D0_A": d0a,
            "D0_B": d0b,
            "D1_A_vs_original": compare_traces_v2(_open_loop_from_original(original, 1), d1a, phase="D1"),
            "D1_B_vs_original": compare_traces_v2(_open_loop_from_original(original, 1), d1b, phase="D1"),
            "D2_A_vs_original": compare_traces_v2(_open_loop_from_original(original, None), d2a, phase="D2"),
            "D2_B_vs_original": compare_traces_v2(_open_loop_from_original(original, None), d2b, phase="D2"),
            "D3_A_vs_original_live": compare_traces_v2(original, d3a, phase="D3"),
            "D3_B_vs_original_live": compare_traces_v2(original, d3b, phase="D3"),
            "D3_A_vs_B": compare_traces_v2(d3a, d3b, phase="D3"),
        }
        root_pass = all(cmp[k]["status"] == "PASS" for k in cmp)
        all_pass = all_pass and root_pass
        rec = {"root_id": root_id, "set": row["set"], "task_id": row["task_id"], "planned_decision_step": row["planned_decision_step"], "pass": root_pass, "comparisons": cmp}
        rows.append(rec)
        dump_json(exp / "engineering" / f"compare_{root_id}.json", rec)
        print(json.dumps({"compared": root_id, "pass": root_pass}), flush=True)
    with (exp / "engineering" / "engineering_comparisons.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for rec in rows:
            handle.write(json.dumps(json.loads(json.dumps(rec, default=_json_default)), sort_keys=True) + "\n")

    sample_id = legal_ids[0]
    sample = read_root_bundle(root_dir(store, PROTOCOL_ID, sample_id), verify=True)
    negatives = negative_tests(exp, sample, ident["runtime_identity_sha256"])
    dump_json(exp / "engineering" / "negative_test_results.json", negatives)

    acq = load_json(exp / "engineering" / "acquisition" / "counters.json")
    wa = load_json(exp / "engineering" / "workers" / "A" / "counters.json")
    wb = load_json(exp / "engineering" / "workers" / "B" / "counters.json")
    external = int(acq["external_control_steps"] + wa["external_control_steps"] + wb["external_control_steps"])
    reg = _coverage([r for r in all_rows if r["set"] == "regression"], require_mid=False)
    hol = _coverage([r for r in all_rows if r["set"] == "holdout"], require_mid=True)
    coverage_ok = reg["ok"] and hol["ok"]
    aba_ok = True
    aba = {}
    for set_name in ("regression", "holdout"):
        path = exp / "engineering" / "workers" / "A" / f"aba_{set_name}.json"
        if not path.exists():
            aba_ok = False
            aba[set_name] = {"status": "MISSING"}
            continue
        payload = load_json(path)
        aba[set_name] = payload
        if payload["A1_vs_A2_d3"]["status"] != "PASS":
            aba_ok = False
    identities = {}
    shas = []
    for name in ("r0_collector", "r0_worker_A", "r0_worker_B", "r0_finalizer", "collector", "worker_A", "worker_B", "finalizer"):
        path = exp / "identities" / f"{name}.json"
        if path.exists():
            identities[name] = load_json(path)["runtime_identity_sha256"]
            shas.append(identities[name])
    identity_ok = len(set(shas)) == 1 and len(shas) >= 4
    budget_ok = external <= BUDGET

    status = "EA33_CPU_S1_QUALIFIED"
    if not identity_ok:
        status = "EA33_HOLD_RUNTIME_IDENTITY"
    elif not coverage_ok:
        status = "EA33_HOLD_COVERAGE"
    elif not all_pass:
        status = "EA33_HOLD_TRACE_MISMATCH"
    elif negatives["status"] != "PASS":
        status = "EA33_HOLD_NEGATIVE_TESTS"
    elif not budget_ok:
        status = "EA33_HOLD_BUDGET"
    elif not aba_ok:
        status = "EA33_HOLD_ENGINEERING"

    estimate = {
        "external_control_steps": external,
        "budget": BUDGET,
        "reset_internal_steps": int(acq.get("reset_internal_steps", 0) + wa.get("reset_internal_steps", 0) + wb.get("reset_internal_steps", 0)),
        "regression_legal": reg["legal"],
        "holdout_legal": hol["legal"],
        "training_performed": False,
        "runtime": "CPU_SINGLE_THREAD",
        "result_source": "ENV_EVALUATED",
        "training_eligible": False,
    }
    dump_json(exp / "report" / "resource_estimate.json", estimate)
    inventory = []
    for path in sorted(Path(store).rglob("*")):
        if path.is_file():
            inventory.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump_json(exp / "package" / "full_root_storage_inventory.json", {"n_files": len(inventory), "files": inventory})
    report = [
        "# EA-V3 CPU S1 Qualification Report",
        "",
        f"Status: `{status}`",
        "",
        json.dumps({"regression": reg, "holdout": hol, "external_control_steps": external, "negatives": negatives["status"], "aba_ok": aba_ok, "identity_ok": identity_ok, "identities": identities}, indent=2),
        "",
        "## Per-root",
        "",
    ]
    for rec in rows:
        report.append(f"- {rec['set']} {rec['root_id']} task {rec['task_id']} step {rec['planned_decision_step']}: {'PASS' if rec['pass'] else 'FAIL'}")
    (exp / "report" / "CPU_S1_QUALIFICATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    dump_json(exp / "cpu_runtime_identity.json", load_json(exp / "identities" / "finalizer.json"))
    diag = exp / "identities" / "finalizer_diagnostics.json"
    if diag.exists():
        shutil.copyfile(diag, exp / "cpu_runtime_diagnostics.json")
    decision = {
        "stage": "R0_S1_CPU",
        "status": status,
        "s2_unlocked": False,
        "phase_c_unlocked": False,
        "training_performed": False,
        "human_review": None,
        "runtime": "CPU_SINGLE_THREAD",
        "gpu_used_for_roots": False,
        "protocol_id": PROTOCOL_ID,
        "external_control_steps": external,
        "regression_coverage": reg,
        "holdout_coverage": hol,
        "negative_tests": negatives["status"],
        "runtime_identity_sha256": ident["runtime_identity_sha256"],
    }
    dump_json(exp / "decision.json", decision)
    zip_path = exp / "package" / "ea_v3_cpu_s1_lightweight.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in [
            "protocol/cpu_s1_protocol.json",
            "protocol/runtime_identity_contract_v2.json",
            "cpu_runtime_identity.json",
            "cpu_runtime_diagnostics.json",
            "pre_s1_inference_check.json",
            "post_s1_inference_check.json",
            "manifests/regression_root_manifest.json",
            "manifests/holdout_root_manifest.json",
            "engineering/engineering_comparisons.jsonl",
            "engineering/negative_test_results.json",
            "report/CPU_S1_QUALIFICATION_REPORT.md",
            "report/resource_estimate.json",
            "decision.json",
            "package/full_root_storage_inventory.json",
        ]:
            path = exp / rel
            if path.exists():
                zf.write(path, arcname=rel)
    digest = sha256_file(zip_path)
    (exp / "package" / "ea_v3_cpu_s1_lightweight.zip.sha256").write_text(digest + "  ea_v3_cpu_s1_lightweight.zip\n", encoding="utf-8")
    print(json.dumps({"status": status, "external": external, "zip": digest}, indent=2))
    if status != "EA33_CPU_S1_QUALIFIED":
        raise SystemExit(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    parser.add_argument("--root-store", default=DEFAULT_PATHS["root_store"])
    args = parser.parse_args()
    finalize_main(args)


if __name__ == "__main__":
    main()
