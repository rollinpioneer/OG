
"""S1 engineering pilot: acquire 15 roots, two file-only workers, comparisons, negative tests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import jax
import numpy as np
import ogbench

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.compare import compare_states, compare_traces, physical_diagnostics
from execution_aligned_rl.v3.contracts import (
    DEFAULT_PATHS,
    ENGINEERING_ROOTS,
    PROTOCOL_ID,
    PROTOCOL_V1,
)
from execution_aligned_rl.v3.env_state import (
    capture_integration,
    capture_observation_stage,
    capture_python_state,
    elapsed_steps,
    exact_goal,
    measure_state,
    perturb_state,
    physical_task_targets,
    public_observation,
    restore_direct,
)
from execution_aligned_rl.v3.hashing import sha256_array, stable_int, stable_uint32
from execution_aligned_rl.v3.policy import FrozenPolicy, action_for, load_agent, prng_key, value_for
from execution_aligned_rl.v3.root_bundle import read_root_bundle, root_dir, save_trace_npz, write_root_bundle
from execution_aligned_rl.v3.root_loader import load_root
from execution_aligned_rl.v3.rollout import rollout_segment
from execution_aligned_rl.v3.serialization import dump_json, load_json


def experiment_dir(repo: Path) -> Path:
    return repo / DEFAULT_PATHS["experiment_dir_rel"]


def make_env(dataset_dir: str):
    return ogbench.make_env_and_datasets("cube-double-play-v0", dataset_dir=dataset_dir, env_only=True)


def allocate_env(env, seed: int, task_id: int, counters: dict):
    np.random.seed(seed)
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed, options={"task_id": int(task_id)})
    counters["reset_internal_steps"] = int(counters.get("reset_internal_steps", 0)) + 2
    counters["resets"] = int(counters.get("resets", 0)) + 1
    return np.asarray(observation, dtype=np.float64).copy(), info


def prefix_key(root_id: int, step: int):
    return prng_key(f"ea3-mr-v1:prefix:{root_id}:{step}")


def probe_key(root_id: int, step: int):
    return prng_key(f"ea3-mr-v1:probe:{root_id}:{step}")


def d3_index(root_id: int, n_train: int) -> int:
    return stable_int(f"ea3-mr-v1:d3-endpoint:{root_id}") % int(n_train)


def acquire(args) -> None:
    repo = Path(args.repo)
    exp = experiment_dir(repo)
    store = Path(args.root_store)
    store.mkdir(parents=True, exist_ok=True)
    out = exp / "engineering" / "acquisition"
    out.mkdir(parents=True, exist_ok=True)
    agent, config, train = load_agent(args.official_source, args.checkpoint_dir, args.dataset_dir)
    policy = FrozenPolicy(agent)
    env = make_env(args.dataset_dir)
    counters = {"external_control_steps": 0, "reset_internal_steps": 0, "resets": 0}
    train_obs = np.asarray(train["observations"])
    records = []
    started = time.time()
    for spec in ENGINEERING_ROOTS:
        root_id = int(spec["root_id"])
        task_id = int(spec["task_id"])
        planned = int(spec["planned_decision_step"])
        observation, info = allocate_env(env, spec["reset_seed"], task_id, counters)
        goal = np.asarray(info["goal"], dtype=np.float64).copy()
        env_goal = exact_goal(env)
        if float(np.max(np.abs(goal - env_goal))) > 0.0:
            raise RuntimeError(f"reset info goal != _cur_goal_ob for root {root_id}")
        z_index = d3_index(root_id, len(train_obs))
        z = np.asarray(train_obs[z_index], dtype=np.float64).copy()
        bootstrap = {
            "initial_observation": observation.copy(),
            "goal_observation": goal.copy(),
            "bootstrap_integration": capture_integration(env),
            "bootstrap_python_state": capture_python_state(env),
            "bootstrap_observation_stage": capture_observation_stage(env),
            "task_state": physical_task_targets(env),
        }
        prefix_actions = []
        prefix_steps = []
        ineligible = False
        ineligible_reason = None
        for step in range(planned):
            action = action_for(agent, observation, goal, prefix_key(root_id, step))
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            counters["external_control_steps"] += 1
            measured = measure_state(
                env,
                observation=next_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=step_info,
                goal=goal,
            )
            prefix_actions.append(np.asarray(action, dtype=np.float64).copy())
            prefix_steps.append(measured)
            observation = np.asarray(next_obs, dtype=np.float64).copy()
            if terminated or truncated:
                ineligible = True
                ineligible_reason = "INELIGIBLE_PREFIX_TERMINAL"
                break
        prefix_arr = np.zeros((0, 5), dtype=np.float64) if not prefix_actions else np.stack(prefix_actions)
        decision_obs = observation.copy()
        bundle = {
            **bootstrap,
            "prefix_actions": prefix_arr,
            "decision_observation": decision_obs,
            "decision_integration": capture_integration(env),
            "decision_python_state": capture_python_state(env),
            "decision_observation_stage": capture_observation_stage(env),
            "d3_target_observation": z,
            "prefix_trace": {
                "identity": {"root_id": root_id, "task_id": task_id, "protocol_id": PROTOCOL_ID},
                "steps": prefix_steps,
                "status": "OK" if prefix_steps or planned == 0 else "EMPTY",
                "full_proxy": None,
            },
            "original_live_probe": None,
            "manifest": {
                "protocol_id": PROTOCOL_ID,
                "root_id": root_id,
                "task_id": task_id,
                "planned_decision_step": planned,
                "actual_prefix_length": int(len(prefix_actions)),
                "reset_seed": spec["reset_seed"],
                "legal": (not ineligible),
                "ineligible_reason": ineligible_reason,
                "result_source": "ENV_EVALUATED",
                "training_eligible": False,
                "d3_target_index": int(z_index),
                "decision_elapsed_steps": elapsed_steps(env),
                "decision_success": bool(getattr(env.unwrapped, "_success", False)),
                "goal_sha256": sha256_array(goal),
                "d3_target_sha256": sha256_array(z),
            },
        }
        if not ineligible:
            probe = rollout_segment(
                env,
                policy,
                decision_obs,
                goal,
                steps=5,
                key_schedule=[probe_key(root_id, i) for i in range(5)],
                frozen_value=agent,
                exact_goal=goal,
                mode="closed_loop",
                identity={"root_id": root_id, "task_id": task_id, "protocol_id": PROTOCOL_ID, "role": "original_live_probe"},
                step_counter=counters,
                probe_goal=z,
            )
            bundle["original_live_probe"] = probe
            bundle["manifest"]["probe_steps"] = len(probe["steps"])
            bundle["manifest"]["probe_full_proxy"] = probe.get("full_proxy")
        dest = root_dir(store, PROTOCOL_ID, root_id)
        if dest.exists():
            shutil.rmtree(dest)
        hashes = write_root_bundle(dest, bundle)
        record = {
            **bundle["manifest"],
            "store_path": str(dest),
            "file_hashes": hashes,
        }
        dump_json(out / f"{root_id}.json", record)
        records.append(record)
        print(json.dumps({"acquired": root_id, "legal": record["legal"], "prefix": record["actual_prefix_length"]}), flush=True)
    dump_json(out / "counters.json", {**counters, "wall_seconds": time.time() - started})
    dump_json(exp / "manifests" / "engineering_root_manifest.json", {"roots": records, "protocol_id": PROTOCOL_ID})


def _load_legal_roots(exp: Path, store: Path):
    manifest = load_json(exp / "manifests" / "engineering_root_manifest.json")
    legal = []
    for row in manifest["roots"]:
        bundle = read_root_bundle(root_dir(store, PROTOCOL_ID, row["root_id"]), verify=True)
        if bundle["manifest"]["legal"]:
            legal.append(bundle)
    return legal, manifest


def worker(args) -> None:
    repo = Path(args.repo)
    exp = experiment_dir(repo)
    store = Path(args.root_store)
    worker_id = args.worker_id
    dest = exp / "engineering" / "workers" / worker_id
    dest.mkdir(parents=True, exist_ok=True)
    agent, config, train = load_agent(args.official_source, args.checkpoint_dir, args.dataset_dir)
    policy = FrozenPolicy(agent)
    env = make_env(args.dataset_dir)
    counters = {"external_control_steps": 0, "reset_internal_steps": 0, "resets": 0, "worker_id": worker_id}
    legal, _ = _load_legal_roots(exp, store)
    started = time.time()
    results = []
    aba_traces = {}

    def run_root(bundle, tag: str):
        spec = bundle["manifest"]
        root_id = spec["root_id"]
        allocate_env(env, 0, spec["task_id"], counters)
        loaded = load_root(bundle, mode="BOOTSTRAP_PREFIX_REPLAY", strict=True, env=env, step_counter=counters)
        goal = loaded.exact_goal
        z = loaded.context["d3_target"]
        captured = measure_state(env, observation=loaded.observation, goal=goal)
        perturb_state(env)
        restore_obs = restore_direct(
            env,
            bundle["decision_integration"],
            bundle["decision_python_state"],
            bundle["decision_observation_stage"],
        )
        d0 = compare_states(
            {
                "observation": bundle["decision_observation"],
                "integration": bundle["decision_integration"],
                "qpos": bundle["decision_python_state"].get("qpos"),
                "qvel": bundle["decision_python_state"].get("qvel"),
                "act": bundle["decision_python_state"].get("act"),
                "ctrl": bundle["decision_python_state"].get("ctrl"),
                "warmstart": bundle["decision_python_state"].get("warmstart"),
                "goal_observation": goal,
                "elapsed_steps": spec.get("decision_elapsed_steps"),
                "task_id": spec["task_id"],
                "success": spec.get("decision_success"),
                "terminated": False,
                "truncated": False,
            },
            measure_state(env, observation=restore_obs, goal=goal),
            has_actions=False,
        )
        original = bundle["original_live_probe"]
        orig_actions = [step["action"] for step in original["steps"]]
        keys = [probe_key(root_id, i) for i in range(5)]
        identity = {"root_id": root_id, "task_id": spec["task_id"], "protocol_id": PROTOCOL_ID, "worker": worker_id, "tag": tag}

        def back_to_decision():
            obs = restore_direct(
                env,
                bundle["decision_integration"],
                bundle["decision_python_state"],
                bundle["decision_observation_stage"],
            )
            return np.asarray(obs, dtype=np.float64).copy()

        obs = back_to_decision()
        d1 = rollout_segment(
            env, policy, obs, goal, steps=1, key_schedule=keys[:1], frozen_value=agent, exact_goal=goal,
            mode="open_loop", actions=orig_actions[:1], identity={**identity, "test": "D1"}, step_counter=counters, probe_goal=z,
        )
        obs = back_to_decision()
        d2 = rollout_segment(
            env, policy, obs, goal, steps=len(orig_actions), key_schedule=keys, frozen_value=agent, exact_goal=goal,
            mode="open_loop", actions=orig_actions, identity={**identity, "test": "D2"}, step_counter=counters, probe_goal=z,
        )
        obs = back_to_decision()
        d3 = rollout_segment(
            env, policy, obs, goal, steps=5, key_schedule=keys, frozen_value=agent, exact_goal=goal,
            mode="closed_loop", identity={**identity, "test": "D3"}, step_counter=counters, probe_goal=z,
        )
        root_out = dest / str(root_id) / tag
        root_out.mkdir(parents=True, exist_ok=True)
        save_trace_npz(root_out / "d1.npz", d1)
        save_trace_npz(root_out / "d2.npz", d2)
        save_trace_npz(root_out / "d3.npz", d3)
        dump_json(root_out / "d0.json", d0)
        dump_json(root_out / "loader_proof.json", loaded.proof)
        return {"root_id": root_id, "d0": d0, "d1": d1, "d2": d2, "d3": d3}

    for bundle in legal:
        rec = run_root(bundle, "primary")
        results.append({"root_id": rec["root_id"], "d0_status": rec["d0"]["status"]})
        print(json.dumps({"worker": worker_id, "root": rec["root_id"], "d0": rec["d0"]["status"]}), flush=True)

    if len(legal) >= 2:
        a, b = legal[0], legal[1]
        rec_a1 = run_root(a, "aba_A1")
        rec_b = run_root(b, "aba_B")
        rec_a2 = run_root(a, "aba_A2")
        aba = {
            "A_id": a["manifest"]["root_id"],
            "B_id": b["manifest"]["root_id"],
            "A1_vs_A2_d3": compare_traces(rec_a1["d3"], rec_a2["d3"]),
            "A1_vs_B_d3_should_fail_if_goals_differ": compare_traces(rec_a1["d3"], rec_b["d3"]),
        }
        dump_json(dest / "aba.json", aba)
    dump_json(dest / "results.json", {"results": results, "counters": counters, "wall_seconds": time.time() - started})
    dump_json(dest / "counters.json", counters)


def _trace_from_worker(exp: Path, worker_id: str, root_id: int, name: str):
    from execution_aligned_rl.v3.root_bundle import load_trace
    return load_trace(exp / "engineering" / "workers" / worker_id / str(root_id) / "primary" / f"{name}.npz")


def negative_tests(exp: Path, store: Path, sample_bundle: dict) -> dict:
    cases = []
    original = dict(sample_bundle["original_live_probe"])
    original["goal_observation"] = sample_bundle["goal_observation"]
    ident = dict(original.get("identity") or {})
    ident["root_id"] = sample_bundle["manifest"]["root_id"]
    ident["task_id"] = sample_bundle["manifest"]["task_id"]
    ident["protocol_id"] = PROTOCOL_ID
    original["identity"] = ident
    worker_a = _trace_from_worker(exp, "A", sample_bundle["manifest"]["root_id"], "d3")

    # 1. byte flip hash
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

    # 2. replace 37D goal
    mutated = json.loads(json.dumps({k: v for k, v in original.items() if k != "steps" and k != "goal_observation"}, default=str))
    # reconstruct proper trace objects
    other = dict(original)
    other["goal_observation"] = np.asarray(original["goal_observation"]).copy()
    other["goal_observation"][19] += 0.25
    cmp_goal = compare_traces(original, other)
    cases.append({"name": "replace_37d_goal_keep_task", "expected": "GOAL_ENCODING", "detected": "GOAL_ENCODING" in cmp_goal["reasons"], "status": cmp_goal["status"], "reasons": cmp_goal["reasons"]})

    # 3. elapsed + 1
    other = dict(original)
    other["elapsed_steps_start"] = int(original.get("elapsed_steps_start") or 0) + 1
    cmp_elapsed = compare_traces(original, other)
    cases.append({"name": "elapsed_plus_one", "expected": "ELAPSED_START", "detected": "ELAPSED_START" in cmp_elapsed["reasons"], "status": cmp_elapsed["status"], "reasons": cmp_elapsed["reasons"]})

    # 4. drop last step / NaN
    other = dict(original)
    other["steps"] = list(original["steps"])[:-1]
    cmp_len = compare_traces(original, other)
    nan_trace = dict(original)
    nan_steps = [dict(s) for s in original["steps"]]
    if nan_steps:
        obs = np.asarray(nan_steps[-1]["observation"]).copy()
        obs[0] = np.nan
        nan_steps[-1]["observation"] = obs
    nan_trace["steps"] = nan_steps
    cmp_nan = compare_traces(original, nan_trace)
    cases.append({"name": "drop_last_step", "expected": "LENGTH_MISMATCH", "detected": "LENGTH_MISMATCH" in cmp_len["reasons"], "status": cmp_len["status"], "reasons": cmp_len["reasons"]})
    cases.append({"name": "nan_observation", "expected": "NONFINITE", "detected": cmp_nan["status"] == "FAIL", "status": cmp_nan["status"], "first_divergent_step": cmp_nan["first_divergent_step"]})

    # 5. action / state bias
    biased = dict(original)
    bsteps = [dict(s) for s in original["steps"]]
    if bsteps:
        bsteps[0]["action"] = np.asarray(bsteps[0]["action"]) + 1e-3
    biased["steps"] = bsteps
    cmp_act = compare_traces(original, biased)
    cases.append({"name": "action_bias", "expected": "action threshold", "detected": cmp_act["status"] == "FAIL", "status": cmp_act["status"], "first_divergent_step": cmp_act["first_divergent_step"]})

    # 6. continued after terminal
    other = dict(original)
    other["continued_after_terminal"] = True
    cmp_term = compare_traces(original, other)
    cases.append({"name": "continue_after_terminal", "expected": "TERMINATION_PROTOCOL", "detected": "TERMINATION_PROTOCOL" in cmp_term["reasons"], "status": cmp_term["status"]})

    # 7. wrong root / skip restore: compare original live vs a zero action synthetic
    fake = dict(original)
    fake["identity"] = dict(original.get("identity") or {})
    fake["identity"]["root_id"] = int(sample_bundle["manifest"]["root_id"]) + 999
    cmp_wrong = compare_traces(original, fake)
    cases.append({"name": "wrong_root_identity", "expected": "IDENTITY_root_id", "detected": any("IDENTITY" in r for r in cmp_wrong["reasons"]), "status": cmp_wrong["status"], "reasons": cmp_wrong["reasons"]})

    # original vs worker should be compared in finalize, not here
    passed = all(case["detected"] for case in cases)
    return {"status": "PASS" if passed else "FAIL", "cases": cases, "n_cases": len(cases), "passed_cases": sum(1 for c in cases if c["detected"])}


def finalize(args) -> None:
    repo = Path(args.repo)
    exp = experiment_dir(repo)
    store = Path(args.root_store)
    legal, manifest = _load_legal_roots(exp, store)
    rows = []
    all_pass = True
    for bundle in legal:
        root_id = bundle["manifest"]["root_id"]
        original = bundle["original_live_probe"]
        d0a = load_json(exp / "engineering" / "workers" / "A" / str(root_id) / "primary" / "d0.json")
        d0b = load_json(exp / "engineering" / "workers" / "B" / str(root_id) / "primary" / "d0.json")
        d1a = _trace_from_worker(exp, "A", root_id, "d1")
        d1b = _trace_from_worker(exp, "B", root_id, "d1")
        d2a = _trace_from_worker(exp, "A", root_id, "d2")
        d2b = _trace_from_worker(exp, "B", root_id, "d2")
        d3a = _trace_from_worker(exp, "A", root_id, "d3")
        d3b = _trace_from_worker(exp, "B", root_id, "d3")
        cmp = {
            "D0_A": d0a,
            "D0_B": d0b,
            "D1_A_vs_original_prefix_action": compare_traces(_open_loop_from_original(original, 1), d1a),
            "D1_B_vs_original": compare_traces(_open_loop_from_original(original, 1), d1b),
            "D2_A_vs_original": compare_traces(_open_loop_from_original(original, None), d2a),
            "D2_B_vs_original": compare_traces(_open_loop_from_original(original, None), d2b),
            "D3_A_vs_original_live": compare_traces(original, d3a),
            "D3_B_vs_original_live": compare_traces(original, d3b),
            "D3_A_vs_B": compare_traces(d3a, d3b),
            "physical_A": physical_diagnostics(original["steps"][-1]["observation"], d3a["steps"][-1]["observation"]) if original["steps"] and d3a["steps"] else {"status": "FAIL", "reason": "EMPTY"},
        }
        root_pass = all(
            (cmp[k]["status"] == "PASS")
            for k in (
                "D0_A", "D0_B",
                "D1_A_vs_original_prefix_action", "D1_B_vs_original",
                "D2_A_vs_original", "D2_B_vs_original",
                "D3_A_vs_original_live", "D3_B_vs_original_live", "D3_A_vs_B",
            )
        )
        all_pass = all_pass and root_pass
        row = {"root_id": root_id, "task_id": bundle["manifest"]["task_id"], "planned_decision_step": bundle["manifest"]["planned_decision_step"], "pass": root_pass, "comparisons": cmp}
        rows.append(row)
        dump_json(exp / "engineering" / f"compare_{root_id}.json", row)
    comparisons_path = exp / "engineering" / "engineering_comparisons.jsonl"
    with comparisons_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            slim = json.loads(json.dumps(row, default=_json_default))
            handle.write(json.dumps(slim, sort_keys=True) + "\n")
    sample = legal[0] if legal else None
    negatives = negative_tests(exp, store, sample) if sample else {"status": "FAIL", "cases": [], "reason": "no_legal_root"}
    dump_json(exp / "engineering" / "negative_test_results.json", negatives)

    acq = load_json(exp / "engineering" / "acquisition" / "counters.json")
    wa = load_json(exp / "engineering" / "workers" / "A" / "counters.json")
    wb = load_json(exp / "engineering" / "workers" / "B" / "counters.json")
    external = int(acq["external_control_steps"] + wa["external_control_steps"] + wb["external_control_steps"])
    coverage = _coverage(manifest["roots"])
    aba = None
    aba_path = exp / "engineering" / "workers" / "A" / "aba.json"
    if aba_path.exists():
        aba = load_json(aba_path)
    budget_ok = external <= int(PROTOCOL_V1["engineering"]["external_control_step_budget"])
    engineering_pass = (
        all_pass
        and negatives["status"] == "PASS"
        and coverage["ok"]
        and budget_ok
        and aba is not None
        and aba["A1_vs_A2_d3"]["status"] == "PASS"
    )
    status = "EA3_ENGINEERING_PILOT_PASS" if engineering_pass else "EA3_HOLD_ENGINEERING_PILOT"
    if not coverage["ok"]:
        status = "EA3_HOLD_COVERAGE"
    if not budget_ok:
        status = "EA3_HOLD_BUDGET"
    if negatives["status"] != "PASS":
        status = "EA3_HOLD_NEGATIVE_TESTS"
    if not all_pass:
        status = "EA3_HOLD_TRACE_MISMATCH"

    estimate = {
        "external_control_steps": external,
        "reset_internal_steps": int(acq["reset_internal_steps"] + wa["reset_internal_steps"] + wb["reset_internal_steps"]),
        "budget": PROTOCOL_V1["engineering"]["external_control_step_budget"],
        "legal_roots": coverage["legal"],
        "planned_roots": 15,
        "wall_seconds_acquisition": acq.get("wall_seconds"),
        "gpu_used": True,
        "training_performed": False,
        "result_source": "ENV_EVALUATED",
        "training_eligible": False,
    }
    dump_json(exp / "report" / "resource_estimate.json", estimate)
    inventory = []
    for path in sorted(Path(store).rglob("*")):
        if path.is_file():
            inventory.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump_json(exp / "package" / "full_root_storage_inventory.json", {"files": inventory, "n_files": len(inventory)})

    report = _render_report(status, coverage, estimate, rows, negatives, aba)
    (exp / "report" / "ENGINEERING_PILOT_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    protocol_sha = sha256_file(exp / "protocol" / "protocol_v1.json")
    decision = {
        "stage": "S1",
        "status": status,
        "engineering_pass": engineering_pass,
        "coverage": coverage,
        "external_control_steps": external,
        "negative_tests": negatives["status"],
        "phase_c_unlocked": False,
        "s2_unlocked": False,
        "human_review": None,
        "protocol_sha256": protocol_sha,
        "training_performed": False,
        "result_source": "ENV_EVALUATED",
        "training_eligible": False,
    }
    dump_json(exp / "decision.json", decision)
    _write_zip(exp)
    print(json.dumps({"status": status, "external_control_steps": external, "legal": coverage["legal"]}, indent=2))
    if not engineering_pass:
        raise SystemExit(3)


def _open_loop_from_original(original: dict, n):
    src = original["steps"] if n is None else original["steps"][:n]
    steps = [dict(s) for s in src]
    out = dict(original)
    out["steps"] = steps
    if n is not None:
        out["full_proxy"] = None
        for step in steps:
            step["step_proxy"] = None
    return out


def _coverage(roots: list[dict]) -> dict:
    legal = [r for r in roots if r.get("legal")]
    tasks = {int(r["task_id"]) for r in legal}
    steps = {int(r["planned_decision_step"]) for r in legal}
    ok = len(legal) >= 10 and tasks == {1, 2, 3, 4, 5} and (125 in steps) and (250 in steps)
    return {
        "legal": len(legal),
        "planned": len(roots),
        "tasks": sorted(tasks),
        "decision_steps_present": sorted(steps),
        "ok": ok,
        "ineligible": [r["root_id"] for r in roots if not r.get("legal")],
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _render_report(status, coverage, estimate, rows, negatives, aba) -> str:
    failed = [row["root_id"] for row in rows if not row["pass"]]
    lines = [
        "# EA-V3 Engineering Pilot Report",
        "",
        f"Status: `{status}`",
        "",
        "This report covers S0+S1 only. Formal root pool (S2) was not created. No model was trained. Phase C remains locked.",
        "",
        "## Coverage",
        "",
        json.dumps(coverage, indent=2),
        "",
        "## Resource",
        "",
        json.dumps(estimate, indent=2),
        "",
        "## Per-root pass",
        "",
    ]
    for row in rows:
        lines.append(f"- root {row['root_id']} task {row['task_id']} step {row['planned_decision_step']}: {'PASS' if row['pass'] else 'FAIL'}")
    lines.extend(["", "## Failed roots", "", json.dumps(failed), "", "## Negative tests", "", json.dumps(negatives, indent=2, default=_json_default), "", "## A-B-A", "", json.dumps(aba, indent=2, default=_json_default), ""])
    return "\n".join(lines) + "\n"


def _write_zip(exp: Path) -> None:
    package = exp / "package"
    zip_path = package / "ea_v3_s1_lightweight.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in [
            "protocol/protocol_v1.json",
            "protocol/asset_lock.json",
            "protocol/source_lock.json",
            "protocol/environment_lock.json",
            "protocol/test_schema_and_gate.json",
            "manifests/engineering_root_manifest.json",
            "engineering/engineering_comparisons.jsonl",
            "engineering/negative_test_results.json",
            "report/ENGINEERING_PILOT_REPORT.md",
            "report/resource_estimate.json",
            "decision.json",
            "package/full_root_storage_inventory.json",
        ]:
            path = exp / rel
            if path.exists():
                zf.write(path, arcname=rel)
        # include one example root if present
        workers = exp / "engineering" / "workers" / "A"
        if workers.exists():
            for path in workers.rglob("*.json"):
                zf.write(path, arcname=str(path.relative_to(exp)))
    digest = sha256_file(zip_path)
    (package / "ea_v3_s1_lightweight.zip.sha256").write_text(digest + "  ea_v3_s1_lightweight.zip\n", encoding="utf-8")


def run_s1(args) -> None:
    python = sys.executable
    common = [
        python,
        "-m",
        "execution_aligned_rl.v3.engineering_pilot",
        "--repo",
        args.repo,
        "--dataset-dir",
        args.dataset_dir,
        "--official-source",
        args.official_source,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--root-store",
        args.root_store,
    ]
    subprocess.run(common + ["acquire"], check=True, cwd=args.repo)
    subprocess.run(common + ["worker", "--worker-id", "A"], check=True, cwd=args.repo)
    subprocess.run(common + ["worker", "--worker-id", "B"], check=True, cwd=args.repo)
    subprocess.run(common + ["finalize"], check=True, cwd=args.repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["acquire", "worker", "finalize", "run-s1"])
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    parser.add_argument("--dataset-dir", default=DEFAULT_PATHS["dataset_dir"])
    parser.add_argument("--official-source", default=DEFAULT_PATHS["official_source"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_PATHS["checkpoint_dir"])
    parser.add_argument("--root-store", default=DEFAULT_PATHS["root_store"])
    parser.add_argument("--worker-id", default="A")
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        run_s1(args)


if __name__ == "__main__":
    main()
