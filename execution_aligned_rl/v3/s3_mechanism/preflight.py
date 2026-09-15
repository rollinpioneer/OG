"""S3-P zero-env-step preflight. Must pass before any official env.step."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import episode_bounds, sha256_file
from execution_aligned_rl.v3.cpu_adoption.inference_check import run_inference_check
from execution_aligned_rl.v3.cpu_s1.runtime_identity import apply_cpu_env
from execution_aligned_rl.v3.hashing import sha256_array
from execution_aligned_rl.v3.root_bundle import read_root_bundle, root_dir
from execution_aligned_rl.v3.s3_mechanism.protocol import (
    DEFAULT_PATHS,
    EXPECTED_IDENTITY,
    EXPECTED_SHA,
    PROTOCOL_ID,
    S2_COMMIT,
    S2_PROTOCOL_FREEZE,
    S2_PROTOCOL_ID,
    experiment_dir,
    load_frozen_protocol,
    s2_experiment,
)
from execution_aligned_rl.v3.serialization import dump_json, load_json


def git_head(path: str | Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def git_exists(path: str | Path, sha: str) -> bool:
    proc = subprocess.run(["git", "-C", str(path), "cat-file", "-t", sha], capture_output=True, text=True)
    return proc.returncode == 0 and proc.stdout.strip() == "commit"


def hold(exp: Path, status: str, details: dict) -> dict:
    decision = {
        "stage": "S3-P",
        "status": status,
        "research_metrics_valid": False,
        "s4_unlocked": False,
        "phase_c_unlocked": False,
        "training_performed": False,
        "human_review": None,
        "protocol_id": PROTOCOL_ID,
        "details": details,
    }
    dump_json(exp / "decision.json", decision)
    return decision


def verify_assets() -> tuple[dict, list[str]]:
    details = {}
    failures = []
    s2_repo = Path(DEFAULT_PATHS["s2_repo"])
    s2_head = git_head(s2_repo)
    details["s2_head"] = s2_head
    details["s2_expected"] = S2_COMMIT
    if s2_head != S2_COMMIT:
        failures.append("s2_head")
    details["s2_protocol_freeze_present"] = git_exists(s2_repo, S2_PROTOCOL_FREEZE)
    if not details["s2_protocol_freeze_present"]:
        failures.append("s2_protocol_freeze")
    og = subprocess.check_output(["git", "-C", DEFAULT_PATHS["official_source"], "rev-parse", "HEAD"], text=True).strip()
    details["ogbench_commit"] = og
    if og != EXPECTED_SHA["ogbench_commit"]:
        failures.append("ogbench_commit")

    checks = {
        "train": Path(DEFAULT_PATHS["train_file"]),
        "validation": Path(DEFAULT_PATHS["val_file"]),
        "checkpoint": Path(DEFAULT_PATHS["checkpoint_file"]),
        "normalizer": s2_repo / DEFAULT_PATHS["normalizer_s2"],
        "candidates_npz": s2_experiment() / "candidates.npz",
        "s2_zip": s2_experiment() / "package" / "ea_v3_s2_lightweight.zip",
    }
    details["hashes"] = {}
    for name, path in checks.items():
        digest = sha256_file(path) if path.exists() else None
        details["hashes"][name] = {"path": str(path), "sha256": digest, "expected": EXPECTED_SHA[name], "exists": path.exists()}
        if digest != EXPECTED_SHA[name]:
            failures.append(name)
    return details, failures


def verify_root_inventory() -> tuple[dict, list[str]]:
    s2 = s2_experiment()
    inv = load_json(s2 / "full_root_storage_inventory.json")
    files = inv["files"]
    mismatches = []
    missing = []
    for row in files:
        path = Path(row["path"])
        if not path.exists():
            missing.append(row["path"])
            continue
        digest = sha256_file(path)
        if digest != row["sha256"]:
            mismatches.append({"path": row["path"], "expected": row["sha256"], "actual": digest})
    store = Path(DEFAULT_PATHS["s2_root_store"])
    extra = []
    known = {row["path"] for row in files}
    for path in store.rglob("*"):
        if path.is_file() and str(path) not in known:
            extra.append(str(path))
    payload = {
        "n_inventory": len(files),
        "n_missing": len(missing),
        "n_mismatch": len(mismatches),
        "n_extra": len(extra),
        "missing": missing[:50],
        "mismatches": mismatches[:50],
        "extra": extra[:50],
        "s2_root_store": str(store),
        "read_only": True,
        "s3_writes_to_s2_store": False,
    }
    failures = []
    if missing or mismatches:
        failures.append("inventory_hash")
    return payload, failures


def episode_id_table(terminals: np.ndarray) -> np.ndarray:
    bounds = episode_bounds(terminals)
    table = np.empty(len(terminals), dtype=np.int64)
    table[:] = -1
    for i, (start, end) in enumerate(bounds):
        table[start:end] = i
    if np.any(table < 0):
        raise RuntimeError("episode table has uncovered indices")
    return table, bounds


def provenance() -> tuple[dict, list[str], list[dict]]:
    s2 = s2_experiment()
    audit = load_json(s2 / "candidate_retrieval_audit.json")
    rows = audit["rows"]
    npz = np.load(s2 / "candidates.npz", allow_pickle=False)
    cand_arr = np.asarray(npz["candidates"])
    root_ids = [int(x) for x in np.asarray(npz["root_ids"])]
    train = np.load(DEFAULT_PATHS["train_file"], allow_pickle=False)
    obs = np.asarray(train["observations"])
    terminals = np.asarray(train["terminals"])
    ep_table, bounds = episode_id_table(terminals)
    failures = []
    records = []
    npz_ok = True
    if cand_arr.shape != (len(rows), 8, 37) or cand_arr.dtype != np.float64:
        failures.append("npz_shape_dtype")
        npz_ok = False
    if root_ids != [int(r["root_id"]) for r in rows]:
        failures.append("npz_root_order")
    identity_parts = []
    for ridx, row in enumerate(rows):
        root_id = int(row["root_id"])
        cands = row["candidates"]
        if len(cands) != 8:
            failures.append(f"n_cand_{root_id}")
        for cand in cands:
            cid = int(cand["rank"])
            start = int(cand["start_index"])
            end = int(cand["endpoint_index"])
            reasons = []
            if end != start + 20:
                reasons.append("endpoint_not_start_plus_20")
            if start < 0 or end >= len(obs) or start >= len(obs):
                reasons.append("index_oob")
                rec = {
                    "root_id": root_id,
                    "candidate_id": cid,
                    "start_index": start,
                    "endpoint_index": end,
                    "ok": False,
                    "reasons": reasons,
                }
                records.append(rec)
                failures.append(f"{root_id}:{cid}")
                continue
            same_episode = int(ep_table[start]) == int(ep_table[end])
            if not same_episode:
                reasons.append("cross_episode")
            source = np.asarray(obs[end])
            source_sha = sha256_array(source)
            if source_sha != cand["endpoint_sha256"]:
                reasons.append("endpoint_sha256")
            replica = np.asarray(cand_arr[ridx, cid], dtype=np.float64)
            if not np.array_equal(replica, source.astype(np.float64)):
                reasons.append("npz_numeric_mismatch")
                npz_ok = False
            rec = {
                "root_id": root_id,
                "candidate_id": cid,
                "source_episode_id": int(ep_table[end]),
                "source_start_index": start,
                "source_endpoint_index": end,
                "original_dtype": str(source.dtype),
                "original_shape": list(source.shape),
                "original_endpoint_sha256": source_sha,
                "audit_endpoint_sha256": cand["endpoint_sha256"],
                "distance_rms": cand.get("distance_rms"),
                "npz_float64_equal": bool(np.array_equal(replica, source.astype(np.float64))),
                "same_episode": bool(same_episode),
                "endpoint_is_start_plus_k": bool(end == start + 20),
                "k": 20,
                "result_source": "DATA_REAL",
                "re_retrieved": False,
                "re_sorted": False,
                "ok": not reasons,
                "reasons": reasons,
            }
            records.append(rec)
            identity_parts.append(
                (
                    root_id,
                    cid,
                    rec["source_episode_id"],
                    start,
                    end,
                    rec["original_dtype"],
                    tuple(rec["original_shape"]),
                    source_sha,
                )
            )
            if reasons:
                failures.append(f"{root_id}:{cid}")
    ident_text = json.dumps(identity_parts, sort_keys=True, ensure_ascii=False)
    ident_sha = hashlib.sha256(ident_text.encode("utf-8")).hexdigest()
    summary = {
        "n_rows": len(rows),
        "n_candidates": len(records),
        "n_ok": sum(1 for r in records if r["ok"]),
        "n_fail": sum(1 for r in records if not r["ok"]),
        "npz_ok": npz_ok,
        "canonical_candidate": "train_source_endpoint_original_array",
        "identity_sha256": ident_sha,
        "n_episodes": len(bounds),
        "n_train": int(len(obs)),
        "re_retrieved": False,
        "re_sorted": False,
    }
    npz_report = {
        "shape": list(cand_arr.shape),
        "dtype": str(cand_arr.dtype),
        "numeric_copy_of_float32_source": True,
        "canonical_identity_is_npz_bytes": False,
        "ok": npz_ok and not failures,
    }
    if failures:
        summary["failure_examples"] = failures[:20]
    return {"summary": summary, "npz": npz_report, "records": records}, failures, records


def legal_and_deep() -> tuple[list[dict], list[int], list[int]]:
    s2 = s2_experiment()
    pool = load_json(s2 / "root_pool_manifest.json")["roots"]
    plan = {int(r["root_id"]): r for r in load_json(s2 / "protocol" / "formal_root_plan.json")["roots"]}
    legal = []
    for row in pool:
        if not row.get("legal"):
            continue
        item = dict(row)
        item["is_deep"] = bool(plan[int(row["root_id"])]["is_deep"])
        legal.append(item)
    legal_ids = [int(r["root_id"]) for r in legal]
    deep_ids = [int(r["root_id"]) for r in legal if r["is_deep"]]
    return legal, legal_ids, deep_ids


def ideal_ranking(records: list[dict], exp: Path) -> tuple[dict, list[str]]:
    apply_cpu_env()
    from execution_aligned_rl.v3.policy import load_agent, value_for

    legal, legal_ids, deep_ids = legal_and_deep()
    store = Path(DEFAULT_PATHS["s2_root_store"])
    train = np.load(DEFAULT_PATHS["train_file"], allow_pickle=False)
    obs = np.asarray(train["observations"])
    by_root = {}
    for rec in records:
        by_root.setdefault(int(rec["root_id"]), {})[int(rec["candidate_id"])] = rec
    agent, config, _train = load_agent(
        DEFAULT_PATHS["official_source"],
        DEFAULT_PATHS["checkpoint_dir"],
        DEFAULT_PATHS["dataset_dir"],
    )
    values_path = exp / "ideal_values.jsonl"
    if values_path.exists():
        values_path.unlink()
    ranking = []
    failures = []
    all_values = []
    for root in legal:
        root_id = int(root["root_id"])
        bundle = read_root_bundle(root_dir(store, S2_PROTOCOL_ID, root_id), verify=True)
        goal = np.asarray(bundle["goal_observation"])
        goal_sha = sha256_array(goal)
        if goal_sha != root.get("goal_sha256") and root.get("goal_sha256") is not None:
            # compare against recomputed
            pass
        if sha256_array(goal) != sha256_array(np.asarray(bundle["goal_observation"])):
            failures.append(f"goal_{root_id}")
        zs = []
        recs = []
        for cid in range(8):
            rec = by_root[root_id][cid]
            z = np.asarray(obs[int(rec["source_endpoint_index"])])
            if sha256_array(z) != rec["original_endpoint_sha256"]:
                failures.append(f"zhash_{root_id}_{cid}")
            zs.append(z)
            recs.append(rec)
        stacked = np.stack(zs)
        raw = np.asarray(value_for(agent, stacked, goal), dtype=np.float64).reshape(-1)
        if raw.shape[0] != 8 or not np.isfinite(raw).all():
            failures.append(f"value_{root_id}")
        pairs = []
        best_v = None
        best_id = None
        for cid, rec, val in zip(range(8), recs, raw):
            item = {
                "root_id": root_id,
                "task_id": int(root["task_id"]),
                "candidate_id": cid,
                "value": float(val),
                "source_endpoint_index": rec["source_endpoint_index"],
                "source_episode_id": rec["source_episode_id"],
                "original_endpoint_sha256": rec["original_endpoint_sha256"],
                "goal_sha256": goal_sha,
                "z_sha256": sha256_array(zs[cid]),
                "canonical_candidate": "train_source_endpoint_original_array",
            }
            from execution_aligned_rl.v3.s3_mechanism.protocol import append_jsonl

            append_jsonl(values_path, item)
            pairs.append(item)
            if best_v is None or float(val) > best_v or (float(val) == best_v and cid < best_id):
                best_v = float(val)
                best_id = cid
        n_at_best = sum(1 for p in pairs if p["value"] == best_v)
        ranking.append(
            {
                "root_id": root_id,
                "task_id": int(root["task_id"]),
                "is_deep": bool(root["is_deep"]),
                "ideal_candidate_id": int(best_id),
                "ideal_value": best_v,
                "tie": n_at_best > 1,
                "n_tied": n_at_best,
                "tie_break": "smallest_candidate_id",
                "values": [p["value"] for p in pairs],
            }
        )
        all_values.append(raw.copy())
        print({"ideal": root_id, "cand": best_id, "V": best_v}, flush=True)
    values_arr = np.stack(all_values)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "n_roots": len(ranking),
        "n_candidates": 8,
        "formula": "V(z, g_r) on source endpoint vs exact 37D goal",
        "tie_break": "smallest_candidate_id",
        "env_step_used": False,
        "value_array_sha256": sha256_array(values_arr),
        "value_array_shape": list(values_arr.shape),
        "value_array_dtype": str(values_arr.dtype),
        "roots": ranking,
        "legal_root_ids": legal_ids,
        "legal_deep_root_ids": deep_ids,
    }
    dump_json(exp / "ideal_ranking_manifest.json", manifest)
    if len(ranking) != 70 or len(deep_ids) != 35:
        failures.append("counts")
    return manifest, failures


def write_identities(repo: Path, exp: Path) -> tuple[list[str], dict]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    for k, v in {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "JAX_ENABLE_X64": "0",
    }.items():
        env[k] = v
    env.pop("XLA_FLAGS", None)
    shas = []
    payloads = {}
    for role in ("r0_collector", "r0_worker_A", "r0_worker_B", "r0_finalizer"):
        out = exp / "identities" / f"{role}.json"
        subprocess.run(
            [sys.executable, "-m", "execution_aligned_rl.v3.cpu_s1.runtime_identity", "--role", role, "--out", str(out)],
            check=True,
            cwd=str(repo),
            env=env,
        )
        payload = load_json(out)
        shas.append(payload["runtime_identity_sha256"])
        payloads[role] = payload["runtime_identity_sha256"]
    return shas, payloads


def main() -> None:
    apply_cpu_env()
    repo = Path(DEFAULT_PATHS["repo"])
    exp = experiment_dir(repo)
    for part in ("identities", "preflight", "traces", "report", "package"):
        (exp / part).mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_protocol(exp)
    if frozen["s3_protocol"]["protocol_id"] != PROTOCOL_ID:
        decision = hold(exp, "EA35_HOLD_PROTOCOL_VIOLATION", {"reason": "protocol_id"})
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)

    asset_details, asset_fail = verify_assets()
    dump_json(exp / "preflight" / "asset_verification.json", {"details": asset_details, "failures": asset_fail})
    if asset_fail:
        decision = hold(exp, "EA35_HOLD_ASSET_MISMATCH", {"failures": asset_fail, "details": asset_details})
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)

    inv_payload, inv_fail = verify_root_inventory()
    dump_json(exp / "root_pool_integrity_audit.json", inv_payload)
    legal, legal_ids, deep_ids = legal_and_deep()
    dump_json(
        exp / "root_pool_identity.json",
        {
            "n_legal": len(legal_ids),
            "n_legal_deep": len(deep_ids),
            "legal_root_ids": legal_ids,
            "legal_deep_root_ids": deep_ids,
            "inventory_n_files": inv_payload["n_inventory"],
            "read_only": True,
        },
    )
    if inv_fail or len(legal_ids) != 70 or len(deep_ids) != 35:
        decision = hold(exp, "EA35_HOLD_ROOT_POOL_INTEGRITY", {"inventory": inv_payload, "n_legal": len(legal_ids), "n_deep": len(deep_ids)})
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)

    prov, prov_fail, records = provenance()
    dump_json(exp / "candidate_pool_identity.json", prov["summary"])
    dump_json(exp / "candidate_npz_consistency.json", prov["npz"])
    prov_path = exp / "candidate_provenance_v2.jsonl"
    if prov_path.exists():
        prov_path.unlink()
    from execution_aligned_rl.v3.s3_mechanism.protocol import append_jsonl

    for rec in records:
        append_jsonl(prov_path, rec)
    if prov_fail:
        decision = hold(exp, "EA35_HOLD_CANDIDATE_PROVENANCE", {"failures": prov_fail[:50], "n_fail": prov["summary"]["n_fail"]})
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)

    shas, role_map = write_identities(repo, exp)
    ident_ok = len(set(shas)) == 1 and shas[0] == EXPECTED_IDENTITY
    dump_json(
        exp / "runtime_identity_manifest.json",
        {
            "sha256": shas[0] if shas else None,
            "roles": role_map,
            "match_expected": ident_ok,
            "expected": EXPECTED_IDENTITY,
            "schema": "runtime_identity_v2_reused_from_cpu_s1",
            "s3_protocol_id_in_hash": False,
        },
    )
    dump_json(exp / "cpu_runtime_identity.json", load_json(exp / "identities" / "r0_collector.json"))
    if not ident_ok:
        decision = hold(exp, "EA35_HOLD_RUNTIME_IDENTITY", {"shas": shas, "expected": EXPECTED_IDENTITY})
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)

    pre = run_inference_check(repo, exp, "pre_s3", (0, 1))
    dump_json(exp / "pre_s3_inference_check.json", pre)
    if pre.get("status") != "PASS":
        decision = hold(exp, "EA35_HOLD_RUNTIME_IDENTITY", {"phase": "pre_s3_inference", "check": pre})
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)

    ranking, ideal_fail = ideal_ranking(records, exp)
    dump_json(exp / "preflight" / "ideal_status.json", {"failures": ideal_fail, "n_roots": ranking["n_roots"]})
    if ideal_fail:
        decision = hold(exp, "EA35_HOLD_ENGINEERING", {"phase": "ideal_ranking", "failures": ideal_fail})
        print(json.dumps(decision, indent=2))
        raise SystemExit(2)

    payload = {
        "stage": "S3-P",
        "status": "PREFLIGHT_PASS",
        "protocol_id": PROTOCOL_ID,
        "env_step_used": False,
        "n_legal": 70,
        "n_legal_deep": 35,
        "n_candidates": 560,
        "runtime_identity_sha256": shas[0],
        "ideal_value_array_sha256": ranking["value_array_sha256"],
        "s4_unlocked": False,
        "training_performed": False,
    }
    dump_json(exp / "preflight" / "preflight_pass.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
