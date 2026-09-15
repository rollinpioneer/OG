"""Orchestrate S3 postmortem. CPU env before JAX. Zero env.step."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.cpu_adoption.inference_check import run_inference_check
from execution_aligned_rl.v3.cpu_s1.runtime_identity import apply_cpu_env, write_identity
from execution_aligned_rl.v3.hashing import sha256_array
from execution_aligned_rl.v3.s3_postmortem.analyze import (
    QueryBudget,
    cube_obs_success_audit,
    fixed_deep_bootstrap,
    integrity,
    reproduce,
    run_core,
    write_reports,
)
from execution_aligned_rl.v3.s3_postmortem.fixtures import run_fixtures
from execution_aligned_rl.v3.s3_postmortem.protocol import (
    ACTOR_QUERY_BUDGET,
    DEFAULT_REPO,
    EXPECTED_IDENTITY,
    PROTOCOL_ID,
    S2_PROTOCOL_ID,
    S2_STORE,
    S3_EXP,
    experiment_dir,
)
from execution_aligned_rl.v3.serialization import dump_json, load_json


def hold(exp: Path, status: str, details: dict) -> dict:
    decision = {
        "stage": "S3_POSTMORTEM",
        "status": status,
        "s4_unlocked": False,
        "method_training_authorized": False,
        "new_environment_evaluation_authorized": False,
        "training_performed": False,
        "new_environment_steps": 0,
        "human_review": None,
        "modifies_s3_decision": False,
        "s3_status_preserved": "EA35_GAP_EXISTS_PROXY_MISALIGNED",
        "protocol_id": PROTOCOL_ID,
        "details": details,
        "analysis_kind": "exploratory_postmortem",
    }
    dump_json(exp / "decision.json", decision)
    return decision


def load_goals(legal):
    goals = {}
    for rid in legal:
        path = S2_STORE / S2_PROTOCOL_ID / f"{int(rid):06d}" / "goal_observation.npy"
        goals[int(rid)] = np.load(path)
    return goals


def offline_audit(repro, goals, agent, budget: QueryBudget) -> dict:
    from execution_aligned_rl.v3.policy import action_for, prng_key

    succ = cube_obs_success_audit(repro, goals)
    train = np.load("/home/__compress_data/xushijie/ea_v2_cube_data/cube-double-play-v0.npz", allow_pickle=False)
    obs = np.asarray(train["observations"])
    acts = np.asarray(train["actions"])
    terminals = np.asarray(train["terminals"])
    rng = np.random.default_rng(350301)
    n = min(ACTOR_QUERY_BUDGET, len(obs) - 1)
    idx = rng.choice(np.arange(0, len(obs) - 1), size=n, replace=False)
    diffs = []
    for i in idx:
        if bool(terminals[i]):
            continue
        o = obs[int(i)]
        g = obs[int(i) + 1]
        key = prng_key(f"ea35x:support:{int(i)}")
        a = np.asarray(action_for(agent, o, g, key), dtype=np.float64)
        budget.actor += 1
        if budget.actor > ACTOR_QUERY_BUDGET:
            break
        diffs.append(float(np.max(np.abs(a - np.asarray(acts[int(i)], dtype=np.float64)))))
    return {
        "public_obs_goal_success_reconstruction": succ,
        "recorded_window_is_not_target_policy_rollout": True,
        "s3_env_labels_training_eligible": False,
        "true_terminal_vs_truncation_vs_budget": {
            "cube_success_does_not_set_terminated": True,
            "s3_deep_tail_stop_on_success": True,
            "s3_prefix_does_not_stop_on_success": True,
            "time_limit_truncated_at_500": True,
        },
        "actor_support_diagnostic": {
            "n": len(diffs),
            "max_abs_vs_recorded_action_mean": float(np.mean(diffs)) if diffs else None,
            "max_abs_vs_recorded_action_p90": float(np.quantile(diffs, 0.9)) if diffs else None,
            "note": "distance to behavior action at the same index, not proof of coverage",
        },
        "future_fqe_supervision": {
            "requires": "target actor bootstrap on train (o,a,o') plus remaining budget R and reconstructible success",
            "cannot_use": "S3 ENV_EVALUATED labels as train y",
            "cannot_treat_recorded_later_actions_as_pi_g": True,
        },
    }


def main() -> None:
    apply_cpu_env()
    started = time.time()
    repo = DEFAULT_REPO
    exp = experiment_dir(repo)
    for part in ("protocol", "manifests", "checks", "analysis", "report", "package"):
        (exp / part).mkdir(parents=True, exist_ok=True)

    fixtures = run_fixtures()
    dump_json(exp / "checks" / "analysis_unit_tests.json", fixtures)
    if fixtures["status"] != "PASS":
        hold(exp, "EA35X_HOLD_ANALYSIS_INTEGRITY", {"phase": "fixtures", "fixtures": fixtures})
        raise SystemExit(2)

    integ, data = integrity(exp)
    if integ["status"] != "PASS":
        status = "EA35X_HOLD_REQUIRED_INPUT_MISSING" if "missing_trace" in integ["failures"] else "EA35X_HOLD_ANALYSIS_INTEGRITY"
        if "runtime_identity" in integ["failures"]:
            status = "EA35X_HOLD_RUNTIME_IDENTITY"
        hold(exp, status, {"integrity": integ})
        raise SystemExit(2)

    ident = write_identity(exp / "manifests" / "runtime_identity.json", "postmortem")
    dump_json(exp / "manifests" / "runtime_identity_manifest.json", {"sha256": ident["runtime_identity_sha256"], "expected": EXPECTED_IDENTITY, "match": ident["runtime_identity_sha256"] == EXPECTED_IDENTITY})
    if ident["runtime_identity_sha256"] != EXPECTED_IDENTITY:
        hold(exp, "EA35X_HOLD_RUNTIME_IDENTITY", {"sha": ident["runtime_identity_sha256"]})
        raise SystemExit(2)

    pre = run_inference_check(repo, exp, "pre_postmortem", (0, 1))
    dump_json(exp / "checks" / "pre_inference_check.json", pre)
    if pre.get("status") != "PASS":
        hold(exp, "EA35X_HOLD_RUNTIME_IDENTITY", {"phase": "pre_inference", "check": pre})
        raise SystemExit(2)

    repro = reproduce(data)
    dump_json(exp / "checks" / "s3_reproduction_check.json", {"got": repro["got"], "expected": repro["expected"], "mismatches": repro["mismatches"], "match": repro["match"]})
    if not repro["match"]:
        hold(exp, "EA35X_HOLD_ANALYSIS_INTEGRITY", {"phase": "reproduction", "mismatches": repro["mismatches"]})
        raise SystemExit(2)

    dump_json(
        exp / "analysis" / "same_denominator_comparison.json",
        {
            "DIRECT_on_all_70": repro["got"]["direct_success_70"],
            "DIRECT_on_same_35_deep": repro["got"]["direct_success_35"],
            "IDEAL_on_same_35_deep": repro["got"]["ideal_success_35"],
            "ORACLE_PROXY_on_same_35_deep": repro["got"]["proxy_success_35"],
            "ORACLE_FULL_on_same_35_deep": repro["got"]["full_success_35"],
            "do_not_compare_direct70_to_deep35": True,
        },
    )
    dump_json(exp / "analysis" / "paired_rescue_harm.json", repro["rescue_harm"])
    boot = fixed_deep_bootstrap(repro)
    hist = load_json(S3_EXP / "bootstrap_intervals.json")
    dump_json(exp / "analysis" / "fixed_deep_bootstrap.json", {"historical_bootstrap": hist, "fixed_deep_supplement": boot})

    goals = load_goals(repro["legal"])
    from execution_aligned_rl.v3.policy import load_agent, value_for

    agent, config, train = load_agent(
        "/home/__compress_data/xushijie/og_runs/ea_v2_v1/src/ogbench",
        "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0",
        "/home/__compress_data/xushijie/ea_v2_cube_data",
    )

    def vfn(obs, goal):
        return value_for(agent, obs, goal)

    qbudget = QueryBudget()
    core = run_core(exp, data, repro, goals, vfn, qbudget)
    dump_json(exp / "analysis" / "horizon_selector_summary.json", core["horizon_summary"])
    dump_json(exp / "analysis" / "known_outcome_baseline.json", {"by_h": core["known_summary"], "note": "uses revealed success/term/trunc only; not a deployable method"})
    dump_json(exp / "analysis" / "unresolved_prefix_analysis.json", {"by_h": core["unresolved_summary"], "shared_pairs": core["shared_unresolved"]})
    dump_json(exp / "analysis" / "epsilon_switch_diagnostic.json", core["eps"])
    dump_json(exp / "analysis" / "proxy_decomposition_summary.json", core["decomp_summary"])
    dump_json(exp / "analysis" / "tail_return_residual_summary.json", core["residual_summary"])

    dump_json(
        exp / "analysis" / "value_target_contract_audit.json",
        {
            "value_loss": "IQL expectile regression of V toward min(target_Q); not explicit FQE of the frozen actor",
            "actor_loss": "AWR or DDPG+BC on a separately trained actor; S3 samples with temperature=0",
            "gc_negative": "training relabel uses 0 if s==g else -1 when gc_negative=True",
            "cube_success": "all cubes within 0.04 of mocap xyz; task mode requires all cubes",
            "value_uses_remaining_budget": False,
            "terminated_truncated_in_proxy": "history compute_proxy zeros tail iff terminated or truncated; success-without-done does not zero tail",
            "s3_prefix_stop_on_success": False,
            "s3_tail_direct_stop_on_success": True,
            "n_success_without_done_short_or_decomp": core["n_success_without_done"],
            "public_observation_markov_proven": False,
            "TARGET_CONTRACT_DISCREPANCY": [
                "success can be true while terminated and truncated are false",
                "historical J uses term/trunc mask, not success mask",
            ],
            "unproven": ["whether expectile V mismatch is the unique cause of proxy misalignment"],
        },
    )

    offline = offline_audit(repro, goals, agent, qbudget)
    dump_json(exp / "analysis" / "offline_supervision_audit.json", offline)
    (exp / "report" / "OFFLINE_SUPERVISION_FEASIBILITY.md").write_text("# placeholder\n", encoding="utf-8")

    # task/budget stratified
    by_task = {}
    for rid in repro["deep"]:
        t = str(repro["table"][rid]["task_id"])
        by_task.setdefault(t, {"n": 0, "ideal": 0, "proxy": 0})
        by_task[t]["n"] += 1
        by_task[t]["ideal"] += int(repro["ideal_ok"][repro["deep"].index(rid)])
        by_task[t]["proxy"] += int(repro["proxy_ok"][repro["deep"].index(rid)])
    dump_json(exp / "analysis" / "task_budget_stratified_summary.json", {"by_task_deep": by_task, "remaining_horizon_note": "decision remaining budget is 500-decision_elapsed; stored on jsonl"})

    post = run_inference_check(repo, exp, "post_postmortem", (2, 3))
    dump_json(exp / "checks" / "post_inference_check.json", post)
    if post.get("status") != "PASS":
        hold(exp, "EA35X_HOLD_RUNTIME_IDENTITY", {"phase": "post_inference"})
        raise SystemExit(2)

    # evidence
    h5 = core["horizon_summary"]["5"]["J_h"]["success"]
    h80 = core["horizon_summary"]["80"]["J_h"]["success"]
    known80 = core["horizon_summary"]["80"]["KNOWN_OUTCOME_ONLY"]["success"]
    unres5 = core["unresolved_summary"]["5"]
    unres80 = core["unresolved_summary"]["80"]
    evidence = {
        "target_policy_mismatch": {
            "value": "possible_not_identified_as_unique_cause",
            "numbers": {"value_is_expectile_not_FQE": True, "success_without_done": core["n_success_without_done"]},
            "source": "ogbench impls/agents/gciql.py value_loss; S3 compute_proxy",
            "unidentified": ["whether replacing V with FQE would reverse the -5 net"],
        },
        "longer_horizon_descriptive_signal": {
            "value": h80 > h5,
            "numbers": {str(h): core["horizon_summary"][str(h)]["J_h"]["success"] for h in (1, 5, 10, 20, 40, 80)},
            "source": "deep traces J_h",
            "unidentified": ["decision-time predictability"],
        },
        "signal_beyond_revealed_outcomes": {
            "value": None if unres80["coverage"] != "OK" else (unres80["J_h"]["success"] > unres80["KNOWN_OUTCOME_ONLY"]["success"]),
            "numbers": {"unresolved_h5": unres5, "unresolved_h80_n": unres80["n_roots"], "known_h80_success": known80, "J_h80_success": h80},
            "source": "ALL_UNRESOLVED_h and KNOWN_OUTCOME_ONLY_h",
            "unidentified": ["offline learnability of any residual signal"],
        },
        "epsilon_switch_signal": {
            "value": core["eps"]["epsilon_switch"]["success"] > core["eps"]["always_IDEAL"]["success"],
            "numbers": core["eps"],
            "source": "fixed epsilon 0.5495205402374268",
            "unidentified": ["whether a learned rejector would generalize"],
        },
        "offline_supervision_feasibility": {
            "value": bool(offline["public_obs_goal_success_reconstruction"]["agreement_rate"] == 1.0) if offline["public_obs_goal_success_reconstruction"]["agreement_rate"] is not None else False,
            "numbers": offline["public_obs_goal_success_reconstruction"],
            "source": "cube_env._compute_successes 0.04 xyz; public obs cube xyz scaled by 10",
            "unidentified": ["sufficient target-actor coverage for FQE"],
        },
    }

    status = "EA35X_POSTMORTEM_COMPLETE"
    decision = {
        "stage": "S3_POSTMORTEM",
        "status": status,
        "s4_unlocked": False,
        "method_training_authorized": False,
        "new_environment_evaluation_authorized": False,
        "training_performed": False,
        "new_environment_steps": 0,
        "human_review": None,
        "modifies_s3_decision": False,
        "s3_status_preserved": "EA35_GAP_EXISTS_PROXY_MISALIGNED",
        "protocol_id": PROTOCOL_ID,
        "analysis_kind": "exploratory_postmortem",
        "research_metrics_valid_s3": True,
        "evidence": evidence,
        "runtime_identity_sha256": EXPECTED_IDENTITY,
        "unique_extra_value_queries": qbudget.value,
        "extra_support_actor_queries": qbudget.actor,
    }
    dump_json(exp / "decision.json", decision)

    usage = {
        "new_environment_steps": 0,
        "training_updates": 0,
        "gpu": 0,
        "unique_extra_value_queries": qbudget.value,
        "extra_support_actor_queries": qbudget.actor,
        "wall_seconds": time.time() - started,
        "value_query_budget": 10000,
        "actor_query_budget": 512,
    }
    write_reports(exp, decision, repro, core["horizon_summary"], core["eps"], core["residual_summary"], core["decomp_summary"], offline, usage)

    inv = []
    for path in sorted(exp.rglob("*")):
        if path.is_file():
            inv.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump_json(exp / "manifests" / "derived_artifact_inventory.json", {"n_files": len(inv), "files": inv})

    zpath = exp / "package" / "ea35x_postmortem_lightweight.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in exp.rglob("*"):
            if not p.is_file():
                continue
            if "traces" in p.parts or p.suffix == ".npy":
                continue
            rel = p.relative_to(exp).as_posix()
            if rel.startswith("package/"):
                continue
            zf.write(p, arcname=rel)
    (exp / "package" / "ea35x_postmortem_lightweight.zip.sha256").write_text(sha256_file(zpath) + "  ea35x_postmortem_lightweight.zip\n", encoding="utf-8")
    print(json.dumps({"status": status, "value_queries": qbudget.value, "actor_queries": qbudget.actor}, indent=2), flush=True)


if __name__ == "__main__":
    main()
