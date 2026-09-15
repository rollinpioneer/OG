"""S3 analysis, engineering gates, report, and lightweight package."""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.cpu_adoption.inference_check import run_inference_check
from execution_aligned_rl.v3.hashing import sha256_array
from execution_aligned_rl.v3.root_bundle import load_trace, read_root_bundle, root_dir
from execution_aligned_rl.v3.runtime_qual.compare_v2 import compare_traces_v2
from execution_aligned_rl.v3.s3_mechanism.preflight import legal_and_deep, verify_root_inventory
from execution_aligned_rl.v3.s3_mechanism.protocol import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    BUDGET,
    DEFAULT_PATHS,
    EXPECTED_IDENTITY,
    PROTOCOL_ID,
    S2_PROTOCOL_ID,
    SENTINELS,
    VALUE_TOLERANCE,
    experiment_dir,
    read_jsonl,
    s2_experiment,
)
from execution_aligned_rl.v3.serialization import dump_json, load_json


def load_goal(root_id: int):
    store = Path(DEFAULT_PATHS["s2_root_store"])
    bundle = read_root_bundle(root_dir(store, S2_PROTOCOL_ID, root_id), verify=True)
    return np.asarray(bundle["goal_observation"], dtype=np.float64)


def attach_goal(trace: dict, goal: np.ndarray) -> dict:
    out = dict(trace)
    out["goal_observation"] = np.asarray(goal, dtype=np.float64)
    return out


def cmp_status(path_a: Path, path_b: Path, goal, phase: str = "D3") -> dict:
    if not path_a.exists() or not path_b.exists():
        return {"status": "FAIL", "reasons": ["MISSING_TRACE"], "path_a": str(path_a), "path_b": str(path_b)}
    a = attach_goal(load_trace(path_a), goal)
    b = attach_goal(load_trace(path_b), goal)
    result = compare_traces_v2(a, b, phase=phase)
    return {
        "status": result["status"],
        "reasons": result.get("reasons"),
        "first_divergent_step": result.get("first_divergent_step"),
        "n_steps_a": result.get("n_steps_a"),
        "n_steps_b": result.get("n_steps_b"),
        "full_proxy": result.get("full_proxy"),
    }


def index_rows(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        out[key] = row
    return out


def first_success_step(row: dict):
    flags = row.get("success") or []
    for i, flag in enumerate(flags):
        if flag:
            return i + 1
    return None


def analyze(repo: Path) -> dict:
    exp = experiment_dir(repo)
    legal, legal_ids, deep_ids = legal_and_deep()
    legal_map = {int(r["root_id"]): r for r in legal}
    ranking = load_json(exp / "ideal_ranking_manifest.json")
    ideal_of = {int(r["root_id"]): int(r["ideal_candidate_id"]) for r in ranking["roots"]}
    traces = exp / "traces"

    short_a = read_jsonl(exp / "short_branch_results_A.jsonl")
    short_b = read_jsonl(exp / "short_branch_results_B.jsonl")
    deep_a = read_jsonl(exp / "deep_branch_results_A.jsonl")
    deep_b = read_jsonl(exp / "deep_branch_results_B.jsonl")
    direct_a = read_jsonl(exp / "direct_results_A.jsonl")
    direct_b = read_jsonl(exp / "direct_results_B.jsonl")
    sa = index_rows(short_a, ("root_id", "candidate_id"))
    sb = index_rows(short_b, ("root_id", "candidate_id"))
    da = index_rows(deep_a, ("root_id", "candidate_id"))
    db = index_rows(deep_b, ("root_id", "candidate_id"))
    dia = index_rows(direct_a, ("root_id",))
    dib = index_rows(direct_b, ("root_id",))

    engineering_failures = []
    holds = []

    inv_payload, inv_fail = verify_root_inventory()
    dump_json(exp / "post_s3_root_pool_integrity_audit.json", inv_payload)
    if inv_fail:
        engineering_failures.append("root_inventory_changed")
        holds.append("EA35_HOLD_ROOT_POOL_INTEGRITY")

    ident_a = load_json(exp / "identities" / "worker_A.json")["runtime_identity_sha256"]
    ident_b = load_json(exp / "identities" / "worker_B.json")["runtime_identity_sha256"]
    if ident_a != EXPECTED_IDENTITY or ident_b != EXPECTED_IDENTITY:
        engineering_failures.append("worker_identity")
        holds.append("EA35_HOLD_RUNTIME_IDENTITY")

    repro = []
    prefix_cons = []
    goal_cache = {}

    def goal_of(root_id: int):
        if root_id not in goal_cache:
            goal_cache[root_id] = load_goal(root_id)
        return goal_cache[root_id]

    for root_id in legal_ids:
        goal = goal_of(root_id)
        for cid in range(8):
            key = (root_id, cid)
            if key not in sa or key not in sb:
                engineering_failures.append(f"missing_short_{root_id}_{cid}")
                holds.append("EA35_HOLD_PROTOCOL_VIOLATION")
                continue
            cmp = cmp_status(traces / "A" / "short" / f"{root_id}_{cid}.npz", traces / "B" / "short" / f"{root_id}_{cid}.npz", goal)
            repro.append({"kind": "short", "root_id": root_id, "candidate_id": cid, **cmp})
            if cmp["status"] != "PASS":
                engineering_failures.append(f"short_repro_{root_id}_{cid}")
                holds.append("EA35_HOLD_BRANCH_REPRODUCIBILITY")
        if root_id in deep_ids:
            for cid in range(8):
                key = (root_id, cid)
                if key not in da or key not in db:
                    engineering_failures.append(f"missing_deep_{root_id}_{cid}")
                    holds.append("EA35_HOLD_PROTOCOL_VIOLATION")
                    continue
                cmp = cmp_status(traces / "A" / "deep" / f"{root_id}_{cid}.npz", traces / "B" / "deep" / f"{root_id}_{cid}.npz", goal)
                repro.append({"kind": "deep", "root_id": root_id, "candidate_id": cid, **cmp})
                if cmp["status"] != "PASS":
                    engineering_failures.append(f"deep_repro_{root_id}_{cid}")
                    holds.append("EA35_HOLD_BRANCH_REPRODUCIBILITY")
                pcmp = cmp_status(
                    traces / "A" / "short" / f"{root_id}_{cid}.npz",
                    traces / "A" / "deep_prefix" / f"{root_id}_{cid}.npz",
                    goal,
                )
                pcmp_b = cmp_status(
                    traces / "B" / "short" / f"{root_id}_{cid}.npz",
                    traces / "B" / "deep_prefix" / f"{root_id}_{cid}.npz",
                    goal,
                )
                prefix_cons.append({"worker": "A", "root_id": root_id, "candidate_id": cid, **pcmp})
                prefix_cons.append({"worker": "B", "root_id": root_id, "candidate_id": cid, **pcmp_b})
                if pcmp["status"] != "PASS" or pcmp_b["status"] != "PASS":
                    engineering_failures.append(f"prefix_{root_id}_{cid}")
                    holds.append("EA35_HOLD_BRANCH_REPRODUCIBILITY")
        if (root_id,) not in dia or (root_id,) not in dib:
            engineering_failures.append(f"missing_direct_{root_id}")
            holds.append("EA35_HOLD_PROTOCOL_VIOLATION")
        else:
            cmp = cmp_status(traces / "A" / "direct" / f"{root_id}.npz", traces / "B" / "direct" / f"{root_id}.npz", goal)
            repro.append({"kind": "direct", "root_id": root_id, **cmp})
            if cmp["status"] != "PASS":
                engineering_failures.append(f"direct_repro_{root_id}")
                holds.append("EA35_HOLD_BRANCH_REPRODUCIBILITY")

    sentinel_report = []
    for rid in SENTINELS:
        goal = goal_of(int(rid))
        for worker in ("A", "B"):
            pre = traces / worker / "sentinel_pre" / f"{rid}.npz"
            post = traces / worker / "sentinel_post" / f"{rid}.npz"
            cmp = cmp_status(pre, post, goal)
            sentinel_report.append({"worker": worker, "root_id": rid, "pair": "pre_post", **cmp})
            if cmp["status"] != "PASS":
                engineering_failures.append(f"sentinel_{worker}_{rid}")
                holds.append("EA35_HOLD_BRANCH_REPRODUCIBILITY")
        cmp_ab = cmp_status(traces / "A" / "sentinel_pre" / f"{rid}.npz", traces / "B" / "sentinel_pre" / f"{rid}.npz", goal)
        sentinel_report.append({"worker": "A_vs_B", "root_id": rid, "pair": "pre", **cmp_ab})
        if cmp_ab["status"] != "PASS":
            engineering_failures.append(f"sentinel_ab_{rid}")
            holds.append("EA35_HOLD_BRANCH_REPRODUCIBILITY")

    # field / remaining horizon checks
    for row in short_a + short_b + deep_a + deep_b + direct_a + direct_b:
        if row.get("result_source") != "ENV_EVALUATED" or row.get("training_eligible") is not False:
            engineering_failures.append("result_flags")
            holds.append("EA35_HOLD_PROTOCOL_VIOLATION")
        if row.get("continued_after_terminal"):
            engineering_failures.append("continued_after_terminal")
            holds.append("EA35_HOLD_PROTOCOL_VIOLATION")
        if row.get("elapsed_start_matches_decision") is False:
            engineering_failures.append("remaining_horizon")
            holds.append("EA35_HOLD_PROTOCOL_VIOLATION")

    if len(short_a) != 560 or len(short_b) != 560 or len(deep_a) != 280 or len(deep_b) != 280 or len(direct_a) != 70 or len(direct_b) != 70:
        engineering_failures.append("counts")
        holds.append("EA35_HOLD_PROTOCOL_VIOLATION")

    counters_a = load_json(exp / "engineering" / "workers" / "A" / "counters.json")
    counters_b = load_json(exp / "engineering" / "workers" / "B" / "counters.json")
    external = int(counters_a["external_control_steps"]) + int(counters_b["external_control_steps"])
    if external > BUDGET:
        engineering_failures.append("budget")
        holds.append("EA35_HOLD_BUDGET")

    post = run_inference_check(repo, exp, "post_s3", (2, 3))
    dump_json(exp / "post_s3_inference_check.json", post)
    if post.get("status") != "PASS":
        engineering_failures.append("post_inference")
        holds.append("EA35_HOLD_RUNTIME_IDENTITY")

    dump_json(
        exp / "branch_reproducibility.json",
        {
            "n": len(repro),
            "n_pass": sum(1 for r in repro if r["status"] == "PASS"),
            "n_fail": sum(1 for r in repro if r["status"] != "PASS"),
            "rows": [r for r in repro if r["status"] != "PASS"] + [r for r in repro if r["status"] == "PASS"][:5],
            "all": repro,
        },
    )
    dump_json(
        exp / "standalone_deep_prefix_consistency.json",
        {
            "n": len(prefix_cons),
            "n_pass": sum(1 for r in prefix_cons if r["status"] == "PASS"),
            "n_fail": sum(1 for r in prefix_cons if r["status"] != "PASS"),
            "all": prefix_cons,
        },
    )
    dump_json(exp / "sentinel_results.json", {"rows": sentinel_report, "n_pass": sum(1 for r in sentinel_report if r["status"] == "PASS")})
    dump_json(
        exp / "execution_cost_manifest.json",
        {
            "worker_A": counters_a,
            "worker_B": counters_b,
            "external_control_steps_total": external,
            "budget": BUDGET,
        },
    )

    priority = load_json(exp / "protocol" / "status_priority.json")["engineering"]
    unique_holds = [h for h in priority if h in set(holds)]
    for h in holds:
        if h not in unique_holds:
            unique_holds.append(h)
    research_valid = not unique_holds

    # Science uses worker A as primary after A==B.
    table = []
    wrong = []
    distinguishable = []
    ideal_miss = []
    paired = []
    spearman_rows = []
    for root_id in legal_ids:
        rec = legal_map[root_id]
        proxies = [float(sa[(root_id, cid)]["proxy"]) for cid in range(8)]
        ideal_id = ideal_of[root_id]
        max_proxy = max(proxies)
        # proxy selector: max proxy, smallest id tie, without looking at full success
        proxy_id = min(i for i, val in enumerate(proxies) if val == max_proxy)
        gap = max_proxy - proxies[ideal_id]
        wrong_flag = bool(gap > VALUE_TOLERANCE)
        wrong.append(wrong_flag)
        row = {
            "root_id": root_id,
            "task_id": int(rec["task_id"]),
            "is_deep": root_id in deep_ids,
            "ideal_candidate_id": ideal_id,
            "proxy_candidate_id": int(proxy_id),
            "proxies": proxies,
            "proxy_gap_vs_ideal": gap,
            "wrong_selection": wrong_flag,
            "direct_success": bool(dia[(root_id,)]["ever_success"]),
            "direct_n_steps": dia[(root_id,)]["n_steps"],
            "direct_completion_steps": first_success_step(dia[(root_id,)]),
        }
        if root_id in deep_ids:
            succ = [bool(da[(root_id, cid)]["ever_success"]) for cid in range(8)]
            n_success = sum(succ)
            dist = not (all(succ) or not any(succ))
            # distinguishable: not all the same
            dist = len(set(succ)) > 1
            distinguishable.append(dist)
            miss = (n_success > 0) and (not succ[ideal_id])
            ideal_miss.append(miss)
            gain = int(succ[proxy_id]) - int(succ[ideal_id])
            paired.append(gain)
            if len(set(proxies)) > 1 and len(set(int(x) for x in succ)) > 1:
                rho, pval = spearmanr(proxies, [int(x) for x in succ])
            else:
                rho, pval = (float("nan"), float("nan"))
            spearman_rows.append(float(rho) if rho == rho else None)
            full_id = None
            # oracle full: max success, smallest id among successes if any, else smallest id
            if any(succ):
                full_id = min(i for i, flag in enumerate(succ) if flag)
            else:
                full_id = 0
            # wait: ORACLE_FULL maximizes success; ties -> smallest candidate id. If multiple success=True, smallest id among successes is NOT maximizing uniquely... all True are equal so smallest id among max. Yes min i with succ[i]==max(succ).
            max_s = max(int(x) for x in succ)
            full_id = min(i for i, flag in enumerate(succ) if int(flag) == max_s)
            row.update(
                {
                    "successes": succ,
                    "n_success_candidates": n_success,
                    "distinguishable": dist,
                    "ideal_full_miss": miss,
                    "proxy_paired_gain": gain,
                    "oracle_full_candidate_id": int(full_id),
                    "ideal_full_success": bool(succ[ideal_id]),
                    "proxy_full_success": bool(succ[proxy_id]),
                    "oracle_full_success": bool(succ[full_id]),
                    "uniform_expectation": float(np.mean(succ)),
                    "spearman_proxy_vs_success": None if rho != rho else float(rho),
                    "completion_steps": [first_success_step(da[(root_id, cid)]) for cid in range(8)],
                }
            )
        table.append(row)

    from execution_aligned_rl.v3.s3_mechanism.protocol import append_jsonl

    table_path = exp / "mechanism_root_table.jsonl"
    if table_path.exists():
        table_path.unlink()
    for row in table:
        append_jsonl(table_path, row)

    n_wrong = int(sum(wrong))
    n_dist = int(sum(distinguishable))
    n_miss = int(sum(ideal_miss))
    net_gain = int(sum(paired))
    wrong_rate = n_wrong / 70.0

    # stratified bootstrap by task, unit=root
    by_task = defaultdict(list)
    for row in table:
        by_task[int(row["task_id"])].append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def metrics_of(sample: list[dict]) -> dict:
        w = [r["wrong_selection"] for r in sample]
        d = [r["distinguishable"] for r in sample if r["is_deep"]]
        m = [r["ideal_full_miss"] for r in sample if r["is_deep"]]
        g = [r["proxy_paired_gain"] for r in sample if r["is_deep"]]
        return {
            "wrong_selection_rate": float(np.mean(w)) if w else float("nan"),
            "distinguishable": int(sum(d)),
            "ideal_miss": int(sum(m)),
            "paired_net_gain": int(sum(g)),
        }

    boots = []
    for _ in range(BOOTSTRAP_REPS):
        sample = []
        for task_id in sorted(by_task):
            rows = by_task[task_id]
            idx = rng.integers(0, len(rows), size=len(rows))
            sample.extend(rows[i] for i in idx)
        boots.append(metrics_of(sample))

    def ci(key: str):
        arr = np.asarray([b[key] for b in boots], dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "lo": float(np.quantile(arr, 0.025)),
            "hi": float(np.quantile(arr, 0.975)),
        }

    bootstrap = {
        "repetitions": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "unit": "root",
        "stratified_by_task": True,
        "wrong_selection_rate": ci("wrong_selection_rate"),
        "distinguishable": ci("distinguishable"),
        "ideal_miss": ci("ideal_miss"),
        "paired_net_gain": ci("paired_net_gain"),
    }
    dump_json(exp / "bootstrap_intervals.json", bootstrap)

    task_summary = {}
    for task_id, rows in sorted(by_task.items()):
        deep_rows = [r for r in rows if r["is_deep"]]
        task_summary[str(task_id)] = {
            "n_roots": len(rows),
            "n_deep": len(deep_rows),
            "wrong_selection": int(sum(r["wrong_selection"] for r in rows)),
            "distinguishable": int(sum(r["distinguishable"] for r in deep_rows)),
            "ideal_miss": int(sum(r["ideal_full_miss"] for r in deep_rows)),
            "paired_net_gain": int(sum(r["proxy_paired_gain"] for r in deep_rows)),
            "ideal_full_success": float(np.mean([r["ideal_full_success"] for r in deep_rows])) if deep_rows else None,
            "proxy_full_success": float(np.mean([r["proxy_full_success"] for r in deep_rows])) if deep_rows else None,
            "direct_success": float(np.mean([r["direct_success"] for r in rows])),
            "uniform_expectation": float(np.mean([r["uniform_expectation"] for r in deep_rows])) if deep_rows else None,
        }
    dump_json(exp / "task_stratified_summary.json", task_summary)

    gates = {
        "wrong_selection_min_count": 7,
        "distinguishable_min": 9,
        "ideal_miss_min": 5,
        "paired_net_gain_min": 0,
        "wrong_selection_count": n_wrong,
        "wrong_selection_rate": wrong_rate,
        "distinguishable": n_dist,
        "ideal_miss": n_miss,
        "paired_net_gain": net_gain,
        "wrong_ok": n_wrong >= 7,
        "dist_ok": n_dist >= 9,
        "miss_ok": n_miss >= 5,
        "gain_ok": net_gain > 0,
    }
    if research_valid:
        if gates["wrong_ok"] and gates["dist_ok"] and gates["miss_ok"] and gates["gain_ok"]:
            status = "EA35_MECHANISM_GAP_SUPPORTED"
        elif gates["dist_ok"] and gates["miss_ok"] and not gates["gain_ok"]:
            status = "EA35_GAP_EXISTS_PROXY_MISALIGNED"
        elif gates["wrong_ok"] and not (gates["dist_ok"] and gates["miss_ok"]):
            status = "EA35_GAP_PROXY_ONLY"
        else:
            status = "EA35_NO_USEFUL_GAP"
    else:
        status = unique_holds[0]

    deep_table = [r for r in table if r["is_deep"]]
    summary = {
        "protocol_id": PROTOCOL_ID,
        "research_metrics_valid": research_valid,
        "n_legal_roots": 70,
        "n_legal_deep_roots": 35,
        "wrong_selection_count": n_wrong,
        "wrong_selection_rate": wrong_rate,
        "full_distinguishable": n_dist,
        "ideal_full_miss": n_miss,
        "proxy_paired_net_success_gain": net_gain,
        "ideal_full_success_rate": float(np.mean([r["ideal_full_success"] for r in deep_table])),
        "proxy_full_success_rate": float(np.mean([r["proxy_full_success"] for r in deep_table])),
        "oracle_full_success_rate": float(np.mean([r["oracle_full_success"] for r in deep_table])),
        "direct_success_rate": float(np.mean([r["direct_success"] for r in table])),
        "uniform_expectation": float(np.mean([r["uniform_expectation"] for r in deep_table])),
        "mean_spearman_proxy_vs_success": float(np.nanmean([r.get("spearman_proxy_vs_success") for r in deep_table if r.get("spearman_proxy_vs_success") is not None] or [np.nan])),
        "ideal_value_ties": int(sum(1 for r in ranking["roots"] if r.get("tie"))),
        "proxy_ties": int(sum(1 for r in table if r["proxies"].count(max(r["proxies"])) > 1)),
        "gates": gates,
        "engineering_failures": engineering_failures[:100],
        "engineering_holds": unique_holds,
    }
    dump_json(exp / "mechanism_summary.json", summary)

    decision = {
        "stage": "S3",
        "status": status,
        "research_metrics_valid": research_valid,
        "s4_unlocked": False,
        "phase_c_unlocked": False,
        "training_performed": False,
        "human_review": None,
        "protocol_id": PROTOCOL_ID,
        "runtime_identity_sha256": EXPECTED_IDENTITY,
        "external_control_steps": external,
        "gates": gates,
        "engineering_holds": unique_holds,
    }
    dump_json(exp / "decision.json", decision)

    report = []
    report.append("# S3 Mechanism Report")
    report.append("")
    report.append(f"Status: `{status}`")
    report.append("")
    report.append(f"- research_metrics_valid: `{research_valid}`")
    report.append(f"- s4_unlocked: `false`")
    report.append(f"- training_performed: `false`")
    report.append(f"- runtime identity: `{EXPECTED_IDENTITY}`")
    report.append(f"- external control steps: `{external}` / `{BUDGET}`")
    report.append("")
    report.append("## Engineering")
    report.append("")
    report.append(json.dumps({"holds": unique_holds, "n_failures": len(engineering_failures)}, indent=2))
    report.append("")
    report.append("## Science")
    report.append("")
    report.append(json.dumps({k: summary[k] for k in summary if k not in ("engineering_failures",)}, indent=2))
    report.append("")
    report.append("## Bootstrap 95% CI")
    report.append("")
    report.append(json.dumps(bootstrap, indent=2))
    report.append("")
    (exp / "S3_MECHANISM_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (exp / "report" / "S3_MECHANISM_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    inv = []
    for path in sorted(exp.rglob("*")):
        if path.is_file():
            inv.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump_json(exp / "full_root_storage_inventory.json", {"n_files": len(inv), "note": "S3 experiment files; S2 root store untouched", "files": inv})
    dump_json(exp / "package" / "full_root_storage_inventory.json", {"n_files": len(inv), "files": inv})

    zpath = exp / "package" / "ea_v3_s3_lightweight.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in [
            "protocol/s3_protocol.json",
            "protocol/execution_matrix.json",
            "protocol/analysis_plan.json",
            "protocol/status_priority.json",
            "protocol/s2_deviation_register.json",
            "candidate_provenance_v2.jsonl",
            "candidate_pool_identity.json",
            "candidate_npz_consistency.json",
            "root_pool_integrity_audit.json",
            "root_pool_identity.json",
            "ideal_values.jsonl",
            "ideal_ranking_manifest.json",
            "pre_s3_inference_check.json",
            "short_branch_results_A.jsonl",
            "short_branch_results_B.jsonl",
            "deep_branch_results_A.jsonl",
            "deep_branch_results_B.jsonl",
            "direct_results_A.jsonl",
            "direct_results_B.jsonl",
            "sentinel_results.json",
            "branch_reproducibility.json",
            "standalone_deep_prefix_consistency.json",
            "execution_cost_manifest.json",
            "post_s3_inference_check.json",
            "mechanism_root_table.jsonl",
            "mechanism_summary.json",
            "task_stratified_summary.json",
            "bootstrap_intervals.json",
            "S3_MECHANISM_REPORT.md",
            "decision.json",
            "runtime_identity_manifest.json",
        ]:
            path = exp / rel
            if path.exists():
                zf.write(path, arcname=rel)
    (exp / "package" / "ea_v3_s3_lightweight.zip.sha256").write_text(sha256_file(zpath) + "  ea_v3_s3_lightweight.zip\n", encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    return decision


def main() -> None:
    from execution_aligned_rl.v3.cpu_s1.runtime_identity import apply_cpu_env

    apply_cpu_env()
    analyze(Path(DEFAULT_PATHS["repo"]))


if __name__ == "__main__":
    main()
