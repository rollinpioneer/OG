
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.cpu_s1.finalize import negative_tests
from execution_aligned_rl.v3.cpu_s1.runtime_identity import apply_cpu_env, write_identity
from execution_aligned_rl.v3.root_bundle import load_trace, read_root_bundle, root_dir
from execution_aligned_rl.v3.runtime_qual.compare_v2 import compare_traces_v2
from execution_aligned_rl.v3.s2_pool.candidates import retrieve
from execution_aligned_rl.v3.s2_pool.protocol import BUDGET, DEFAULT_PATHS, EXPECTED_IDENTITY, PROTOCOL_ID
from execution_aligned_rl.v3.serialization import dump_json, load_json


def _jd(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    return str(v)


def _slice(original, n):
    src = original["steps"] if n is None else original["steps"][:n]
    out = dict(original)
    out["steps"] = [dict(s) for s in src]
    return out


def main() -> None:
    apply_cpu_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    args = parser.parse_args()
    repo = Path(args.repo)
    exp = repo / DEFAULT_PATHS["experiment_rel"]
    ident = write_identity(exp / "identities" / "finalizer.json", "finalizer")
    store = Path(DEFAULT_PATHS["root_store"])
    all_rows = load_json(exp / "manifests" / "all_roots.json")["roots"]
    deep_plan = load_json(exp / "protocol" / "deep_root_plan.json")
    deep_ids = set(deep_plan["deep_root_ids"])
    rows = []
    all_pass = True
    for rec in all_rows:
        if not rec.get("legal"):
            continue
        rid = rec["root_id"]
        bundle = read_root_bundle(root_dir(store, PROTOCOL_ID, rid), verify=True)
        original = dict(bundle["original_live_probe"])
        original["goal_observation"] = bundle["goal_observation"]
        original.setdefault("identity", {}).update({"root_id": rid, "task_id": rec["task_id"], "protocol_id": PROTOCOL_ID})
        d0a = load_json(exp / "engineering" / "workers" / "A" / str(rid) / "primary" / "d0.json")
        d0b = load_json(exp / "engineering" / "workers" / "B" / str(rid) / "primary" / "d0.json")
        traces = {}
        for w, name in (("A", "d1"), ("B", "d1"), ("A", "d2"), ("B", "d2"), ("A", "d3"), ("B", "d3")):
            tr = load_trace(exp / "engineering" / "workers" / w / str(rid) / "primary" / f"{name}.npz")
            tr["goal_observation"] = bundle["goal_observation"]
            tr.setdefault("identity", {}).update({"root_id": rid, "task_id": rec["task_id"], "protocol_id": PROTOCOL_ID})
            traces[f"{name}_{w}"] = tr
        cmp = {
            "D0_A": d0a, "D0_B": d0b,
            "D1_A": compare_traces_v2(_slice(original, 1), traces["d1_A"], phase="D1"),
            "D1_B": compare_traces_v2(_slice(original, 1), traces["d1_B"], phase="D1"),
            "D2_A": compare_traces_v2(_slice(original, None), traces["d2_A"], phase="D2"),
            "D2_B": compare_traces_v2(_slice(original, None), traces["d2_B"], phase="D2"),
            "D3_A": compare_traces_v2(original, traces["d3_A"], phase="D3"),
            "D3_B": compare_traces_v2(original, traces["d3_B"], phase="D3"),
            "D3_AB": compare_traces_v2(traces["d3_A"], traces["d3_B"], phase="D3"),
        }
        ok = all(cmp[k]["status"] == "PASS" for k in cmp)
        all_pass = all_pass and ok
        rows.append({"root_id": rid, "task_id": rec["task_id"], "deep": rid in deep_ids, "pass": ok, "planned_decision_step": rec["planned_decision_step"]})
        dump_json(exp / "engineering" / f"compare_{rid}.json", {"root_id": rid, "pass": ok, "comparisons": cmp})
        print({"compared": rid, "pass": ok}, flush=True)
    with (exp / "root_qualification.jsonl").open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, sort_keys=True) + "\n")
    with (exp / "engineering" / "engineering_comparisons.jsonl").open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, sort_keys=True) + "\n")

    # A-pool-A
    apool_ok = True
    apool_rows = []
    for rid in range(720000, 720005):
        if rid not in [r["root_id"] for r in all_rows if r.get("legal")]:
            continue
        before = load_trace(exp / "engineering" / "workers" / "A" / str(rid) / "primary" / "d3.npz")
        after = load_trace(exp / "engineering" / "workers" / "A" / str(rid) / "apool_after" / "d3.npz")
        before.setdefault("identity", {}).update({"root_id": rid, "protocol_id": PROTOCOL_ID})
        after.setdefault("identity", {}).update({"root_id": rid, "protocol_id": PROTOCOL_ID})
        cmp = compare_traces_v2(before, after, phase="D3")
        apool_ok = apool_ok and cmp["status"] == "PASS"
        apool_rows.append({"root_id": rid, "status": cmp["status"]})
    dump_json(exp / "engineering" / "apoola.json", {"ok": apool_ok, "rows": apool_rows})

    legal = [r for r in all_rows if r.get("legal")]
    by_task = {t: [r for r in legal if r["task_id"] == t] for t in range(1, 6)}
    legal_deep = [r for r in legal if r["root_id"] in deep_ids]
    deep_by_task = {t: [r for r in legal_deep if r["task_id"] == t] for t in range(1, 6)}
    coverage = {
        "legal": len(legal),
        "planned": 80,
        "per_task": {str(t): len(by_task[t]) for t in range(1, 6)},
        "legal_deep": len(legal_deep),
        "deep_planned": 40,
        "deep_per_task": {str(t): len(deep_by_task[t]) for t in range(1, 6)},
        "ineligible": [r["root_id"] for r in all_rows if not r.get("legal")],
    }
    cov_ok = (
        coverage["legal"] >= 60
        and all(v >= 8 for v in coverage["per_task"].values())
        and coverage["legal_deep"] >= 30
        and all(v >= 5 for v in coverage["deep_per_task"].values())
    )
    sample = read_root_bundle(root_dir(store, PROTOCOL_ID, legal[0]["root_id"]), verify=True)
    negatives = negative_tests(exp, sample, ident["runtime_identity_sha256"])
    dump_json(exp / "engineering" / "negative_test_results.json", negatives)
    acq = load_json(exp / "engineering" / "acquisition" / "counters.json")
    wa = load_json(exp / "engineering" / "workers" / "A" / "counters.json")
    wb = load_json(exp / "engineering" / "workers" / "B" / "counters.json")
    external = int(acq["external_control_steps"] + wa["external_control_steps"] + wb["external_control_steps"])
    budget_ok = external <= BUDGET
    identity_ok = ident["runtime_identity_sha256"] == EXPECTED_IDENTITY

    cand = retrieve(exp)
    dump_json(exp / "root_pool_manifest.json", {"protocol_id": PROTOCOL_ID, "roots": all_rows, "coverage": coverage})

    status = "EA34_ROOT_AND_CANDIDATE_POOL_LOCKED"
    if not identity_ok:
        status = "EA34_HOLD_RUNTIME_IDENTITY"
    elif not all_pass:
        status = "EA34_HOLD_ROOT_RECONSTRUCTION"
    elif not cov_ok:
        status = "EA34_HOLD_COVERAGE"
    elif cand.get("status") != "PASS":
        status = "EA34_HOLD_CANDIDATE_DATA"
    elif negatives["status"] != "PASS":
        status = "EA34_HOLD_NEGATIVE_TESTS"
    elif not budget_ok:
        status = "EA34_HOLD_BUDGET"
    elif not apool_ok:
        status = "EA34_HOLD_ENGINEERING"

    dump_json(exp / "report" / "resource_estimate.json", {"external_control_steps": external, "budget": BUDGET})
    inv = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(store.rglob("*")) if p.is_file()]
    dump_json(exp / "full_root_storage_inventory.json", {"n_files": len(inv), "files": inv})
    dump_json(exp / "package" / "full_root_storage_inventory.json", {"n_files": len(inv), "files": inv})
    decision = {
        "stage": "S2",
        "status": status,
        "s3_unlocked": False,
        "training_performed": False,
        "human_review": None,
        "protocol_id": PROTOCOL_ID,
        "external_control_steps": external,
        "coverage": coverage,
        "negatives": negatives["status"],
        "candidates": cand.get("status"),
        "apoola": apool_ok,
        "runtime_identity_sha256": ident["runtime_identity_sha256"],
    }
    dump_json(exp / "decision.json", decision)
    (exp / "S2_POOL_LOCK_REPORT.md").write_text(
        "# S2 Pool Lock Report\n\nStatus: `" + status + "`\n\n" + json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    (exp / "report" / "S2_POOL_LOCK_REPORT.md").write_text((exp / "S2_POOL_LOCK_REPORT.md").read_text(encoding="utf-8"), encoding="utf-8")
    zpath = exp / "package" / "ea_v3_s2_lightweight.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in [
            "protocol/s2_protocol.json",
            "protocol/formal_root_plan.json",
            "protocol/deep_root_plan.json",
            "protocol/candidate_retrieval_contract.json",
            "protocol/runtime_identity_contract_v2.json",
            "root_pool_manifest.json",
            "root_qualification.jsonl",
            "candidate_manifest.json",
            "candidates.npz",
            "candidate_retrieval_audit.json",
            "candidate_index_manifest.json",
            "normalizer_manifest.json",
            "pre_s2_inference_check.json",
            "post_s2_inference_check.json",
            "runtime_identity_manifest.json",
            "S2_POOL_LOCK_REPORT.md",
            "decision.json",
            "full_root_storage_inventory.json",
            "engineering/negative_test_results.json",
        ]:
            path = exp / rel
            if path.exists():
                zf.write(path, arcname=rel)
    (exp / "package" / "ea_v3_s2_lightweight.zip.sha256").write_text(sha256_file(zpath) + "  ea_v3_s2_lightweight.zip\n", encoding="utf-8")
    print(json.dumps({"status": status, "external": external, "legal": coverage["legal"]}, indent=2))
    if status != "EA34_ROOT_AND_CANDIDATE_POOL_LOCKED":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
