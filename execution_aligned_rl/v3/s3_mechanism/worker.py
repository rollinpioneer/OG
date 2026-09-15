"""S3 env workers. CPU env is applied before JAX import."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from execution_aligned_rl.v3.cpu_s1.runtime_identity import apply_cpu_env, write_identity
from execution_aligned_rl.v3.hashing import sha256_array
from execution_aligned_rl.v3.s3_mechanism.protocol import (
    BUDGET,
    DEFAULT_PATHS,
    EXPECTED_IDENTITY,
    HORIZON,
    M,
    PROTOCOL_ID,
    S2_PROTOCOL_ID,
    SENTINELS,
    append_jsonl,
    candidate_key_label,
    direct_key_label,
    experiment_dir,
    sentinel_key_label,
    tail_key_label,
)
from execution_aligned_rl.v3.serialization import dump_json, load_json


def _boot() -> None:
    apply_cpu_env()


def make_env(dataset_dir: str):
    import ogbench

    return ogbench.make_env_and_datasets("cube-double-play-v0", dataset_dir=dataset_dir, env_only=True)


def allocate_env(env, seed: int, task_id: int, counters: dict):
    import numpy as np

    np.random.seed(seed)
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed, options={"task_id": int(task_id)})
    counters["reset_internal_steps"] = int(counters.get("reset_internal_steps", 0)) + 2
    counters["resets"] = int(counters.get("resets", 0)) + 1
    return np.asarray(observation, dtype=np.float64).copy(), info


def compact_trace(trace: dict) -> dict:
    keep = (
        "observation",
        "action",
        "reward",
        "success",
        "terminated",
        "truncated",
        "integration",
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "warmstart",
        "time",
        "elapsed_steps",
        "task_id",
        "goal_observation",
        "step_proxy",
    )
    out = dict(trace)
    out["steps"] = [{k: step.get(k) for k in keep} for step in (trace.get("steps") or [])]
    return out


def summarize_trace(trace: dict) -> dict:
    steps = trace.get("steps") or []
    rewards = [float(s["reward"]) for s in steps if s.get("reward") is not None]
    successes = [bool(s.get("success")) for s in steps]
    terminated = bool(steps[-1]["terminated"]) if steps else False
    truncated = bool(steps[-1]["truncated"]) if steps else False
    reason = "empty"
    if steps:
        if successes[-1] or any(successes):
            reason = "success"
        if terminated:
            reason = "terminated"
        elif truncated:
            reason = "truncated"
        elif not any(successes):
            reason = "horizon"
        if any(successes) and not terminated and not truncated:
            reason = "success"
    return {
        "n_steps": len(steps),
        "terminated": terminated,
        "truncated": truncated,
        "ever_success": any(successes),
        "final_success": bool(successes[-1]) if successes else False,
        "undiscounted_env_return": float(sum(rewards)) if rewards else 0.0,
        "elapsed_start": trace.get("elapsed_steps_start"),
        "elapsed_end": None if not steps else steps[-1].get("elapsed_steps"),
        "continued_after_terminal": bool(trace.get("continued_after_terminal")),
        "full_proxy": trace.get("full_proxy"),
        "termination_reason": reason,
        "action_sha256": [sha256_array(s["action"]) for s in steps if s.get("action") is not None],
        "observation_sha256": [sha256_array(s["observation"]) for s in steps if s.get("observation") is not None],
        "reward": rewards,
        "success": successes,
        "elapsed_steps": [s.get("elapsed_steps") for s in steps],
        "result_source": "ENV_EVALUATED",
        "training_eligible": False,
    }


def remaining_of(env) -> tuple[int, int]:
    from execution_aligned_rl.v3.env_state import elapsed_steps

    elapsed = elapsed_steps(env)
    elapsed_i = 0 if elapsed is None else int(elapsed)
    return max(0, HORIZON - elapsed_i), elapsed_i


def key_fn(label: str):
    from execution_aligned_rl.v3.policy import prng_key

    return prng_key(label)


def rebuild(env, bundle, counters):
    from execution_aligned_rl.v3.root_loader import load_root

    spec = bundle["manifest"]
    allocate_env(env, 0, spec["task_id"], counters)
    return load_root(bundle, mode="BOOTSTRAP_PREFIX_REPLAY", strict=True, env=env, step_counter=counters)


def save_branch(path: Path, trace: dict) -> None:
    from execution_aligned_rl.v3.root_bundle import save_trace_npz

    save_trace_npz(path, compact_trace(trace))


def worker_main(args) -> None:
    _boot()
    import numpy as np
    from execution_aligned_rl.v3.env_state import elapsed_steps
    from execution_aligned_rl.v3.policy import FrozenPolicy, load_agent
    from execution_aligned_rl.v3.root_bundle import read_root_bundle, root_dir
    from execution_aligned_rl.v3.rollout import compute_proxy, rollout_segment

    repo = Path(args.repo)
    exp = experiment_dir(repo)
    worker_id = str(args.worker_id)
    ident = write_identity(exp / "identities" / f"worker_{worker_id}.json", f"worker_{worker_id}")
    if ident["runtime_identity_sha256"] != EXPECTED_IDENTITY:
        dump_json(
            exp / f"worker_{worker_id}_identity_fail.json",
            {"status": "EA35_HOLD_RUNTIME_IDENTITY", "sha": ident["runtime_identity_sha256"]},
        )
        raise SystemExit(2)

    s2 = Path(DEFAULT_PATHS["s2_repo"]) / DEFAULT_PATHS["s2_experiment_rel"]
    pool = load_json(s2 / "root_pool_manifest.json")["roots"]
    plan = {int(r["root_id"]): r for r in load_json(s2 / "protocol" / "formal_root_plan.json")["roots"]}
    audit = {int(r["root_id"]): r for r in load_json(s2 / "candidate_retrieval_audit.json")["rows"]}
    legal = [r for r in pool if r.get("legal")]
    legal.sort(key=lambda r: int(r["root_id"]))
    if worker_id == "B":
        legal = list(reversed(legal))
    cand_ids = list(range(8)) if worker_id == "A" else list(range(7, -1, -1))
    sentinels = list(SENTINELS) if worker_id == "A" else list(reversed(SENTINELS))

    train = np.load(DEFAULT_PATHS["train_file"], allow_pickle=False)
    obs = np.asarray(train["observations"])
    store = Path(DEFAULT_PATHS["s2_root_store"])
    dest = exp / "engineering" / "workers" / worker_id
    dest.mkdir(parents=True, exist_ok=True)
    traces = exp / "traces" / worker_id
    traces.mkdir(parents=True, exist_ok=True)

    short_path = exp / f"short_branch_results_{worker_id}.jsonl"
    deep_path = exp / f"deep_branch_results_{worker_id}.jsonl"
    direct_path = exp / f"direct_results_{worker_id}.jsonl"
    for path in (short_path, deep_path, direct_path):
        if path.exists():
            path.unlink()

    agent, config, _train = load_agent(args.official_source, args.checkpoint_dir, args.dataset_dir)
    policy = FrozenPolicy(agent)
    env = make_env(args.dataset_dir)
    counters = {
        "external_control_steps": 0,
        "reset_internal_steps": 0,
        "resets": 0,
        "worker_id": worker_id,
        "runtime_identity_sha256": ident["runtime_identity_sha256"],
    }
    started = time.time()
    cache = {}

    def bundle_of(root_id: int):
        if root_id not in cache:
            cache[root_id] = read_root_bundle(root_dir(store, S2_PROTOCOL_ID, root_id), verify=True)
        return cache[root_id]

    def check_budget():
        if int(counters["external_control_steps"]) > BUDGET:
            dump_json(dest / "budget_exceeded.json", {"counters": counters})
            raise RuntimeError("EA35_HOLD_BUDGET")

    def source_z(root_id: int, cand_id: int):
        meta = audit[int(root_id)]["candidates"][int(cand_id)]
        idx = int(meta["endpoint_index"])
        z = np.asarray(obs[idx])
        return z, meta

    def run_short(root, cand_id: int):
        root_id = int(root["root_id"])
        bundle = bundle_of(root_id)
        loaded = rebuild(env, bundle, counters)
        rem, elapsed = remaining_of(env)
        n = min(M, rem)
        z, meta = source_z(root_id, cand_id)
        goal = loaded.exact_goal
        keys = [key_fn(candidate_key_label(root_id, cand_id, i)) for i in range(n)]
        identity = {
            "root_id": root_id,
            "task_id": int(root["task_id"]),
            "protocol_id": PROTOCOL_ID,
            "candidate_id": cand_id,
            "branch": "short",
            "goal_sha256": sha256_array(goal),
            "probe_goal_sha256": sha256_array(z),
        }
        trace = compact_trace(
            rollout_segment(
                env,
                policy,
                loaded.observation,
                goal,
                steps=n,
                key_schedule=keys,
                frozen_value=agent,
                exact_goal=goal,
                probe_goal=z,
                identity=identity,
                step_counter=counters,
            )
        )
        save_branch(traces / "short" / f"{root_id}_{cand_id}.npz", trace)
        summary = summarize_trace(trace)
        row = {
            **summary,
            "worker_id": worker_id,
            "root_id": root_id,
            "task_id": int(root["task_id"]),
            "candidate_id": cand_id,
            "is_deep_root": bool(plan[root_id]["is_deep"]),
            "branch": "short",
            "source_start_index": int(meta["start_index"]),
            "source_endpoint_index": int(meta["endpoint_index"]),
            "original_endpoint_sha256": sha256_array(z),
            "audit_endpoint_sha256": meta["endpoint_sha256"],
            "goal_sha256": sha256_array(goal),
            "remaining_horizon_at_start": rem,
            "decision_elapsed_steps": root.get("decision_elapsed_steps"),
            "elapsed_start_matches_decision": elapsed == int(root.get("decision_elapsed_steps", elapsed)),
            "proxy": trace.get("full_proxy"),
            "key_labels": [candidate_key_label(root_id, cand_id, i) for i in range(n)],
            "runtime_identity_sha256": ident["runtime_identity_sha256"],
        }
        append_jsonl(short_path, row)
        check_budget()
        print({"worker": worker_id, "short": root_id, "cand": cand_id, "n": summary["n_steps"], "proxy": row["proxy"]}, flush=True)
        return trace, row

    def run_deep(root, cand_id: int):
        root_id = int(root["root_id"])
        bundle = bundle_of(root_id)
        loaded = rebuild(env, bundle, counters)
        rem, elapsed = remaining_of(env)
        n = min(M, rem)
        z, meta = source_z(root_id, cand_id)
        goal = loaded.exact_goal
        keys = [key_fn(candidate_key_label(root_id, cand_id, i)) for i in range(n)]
        identity = {
            "root_id": root_id,
            "task_id": int(root["task_id"]),
            "protocol_id": PROTOCOL_ID,
            "candidate_id": cand_id,
            "branch": "deep_prefix",
            "goal_sha256": sha256_array(goal),
            "probe_goal_sha256": sha256_array(z),
        }
        prefix = compact_trace(
            rollout_segment(
                env,
                policy,
                loaded.observation,
                goal,
                steps=n,
                key_schedule=keys,
                frozen_value=agent,
                exact_goal=goal,
                probe_goal=z,
                identity=identity,
                step_counter=counters,
            )
        )
        save_branch(traces / "deep_prefix" / f"{root_id}_{cand_id}.npz", prefix)
        tail = {
            "identity": {**identity, "branch": "deep_tail", "probe_goal_sha256": sha256_array(goal)},
            "status": "EMPTY",
            "steps": [],
            "full_proxy": None,
            "elapsed_steps_start": elapsed_steps(env),
            "continued_after_terminal": False,
            "goal_observation": goal,
            "goal_sha256": sha256_array(goal),
            "probe_goal_sha256": sha256_array(goal),
        }
        prefix_term = False
        if prefix["steps"]:
            last = prefix["steps"][-1]
            prefix_term = bool(last.get("terminated") or last.get("truncated"))
        if prefix["steps"] and not prefix_term and not prefix.get("continued_after_terminal"):
            last = prefix["steps"][-1]
            rem2 = max(0, HORIZON - int(last.get("elapsed_steps") or elapsed))
            tkeys = [key_fn(tail_key_label(root_id, cand_id, i)) for i in range(rem2)]
            tail = compact_trace(
                rollout_segment(
                    env,
                    policy,
                    last["observation"],
                    goal,
                    steps=rem2,
                    key_schedule=tkeys,
                    frozen_value=agent,
                    exact_goal=goal,
                    probe_goal=goal,
                    identity=tail["identity"],
                    step_counter=counters,
                    stop_on_success=True,
                )
            )
        combined_steps = list(prefix.get("steps") or []) + list(tail.get("steps") or [])
        combined = {
            "identity": {**identity, "branch": "deep"},
            "status": "OK" if combined_steps else "EMPTY",
            "steps": combined_steps,
            "elapsed_steps_start": prefix.get("elapsed_steps_start"),
            "continued_after_terminal": bool(prefix.get("continued_after_terminal") or tail.get("continued_after_terminal")),
            "goal_observation": goal,
            "goal_sha256": sha256_array(goal),
            "probe_goal_sha256": sha256_array(z),
            "full_proxy": None,
            "result_source": "ENV_EVALUATED",
            "training_eligible": False,
        }
        if combined_steps:
            combined["full_proxy"] = float(compute_proxy(combined, agent, goal))
        save_branch(traces / "deep" / f"{root_id}_{cand_id}.npz", combined)
        summary = summarize_trace(combined)
        prefix_summary = summarize_trace(prefix)
        row = {
            **summary,
            "worker_id": worker_id,
            "root_id": root_id,
            "task_id": int(root["task_id"]),
            "candidate_id": cand_id,
            "is_deep_root": True,
            "branch": "deep",
            "source_start_index": int(meta["start_index"]),
            "source_endpoint_index": int(meta["endpoint_index"]),
            "original_endpoint_sha256": sha256_array(z),
            "goal_sha256": sha256_array(goal),
            "remaining_horizon_at_start": rem,
            "decision_elapsed_steps": root.get("decision_elapsed_steps"),
            "proxy_5": prefix.get("full_proxy"),
            "prefix_n_steps": prefix_summary["n_steps"],
            "tail_n_steps": len(tail.get("steps") or []),
            "key_labels_prefix": [candidate_key_label(root_id, cand_id, i) for i in range(n)],
            "runtime_identity_sha256": ident["runtime_identity_sha256"],
        }
        append_jsonl(deep_path, row)
        check_budget()
        print({"worker": worker_id, "deep": root_id, "cand": cand_id, "n": summary["n_steps"], "success": summary["ever_success"]}, flush=True)
        return combined, row

    def run_direct(root):
        root_id = int(root["root_id"])
        bundle = bundle_of(root_id)
        loaded = rebuild(env, bundle, counters)
        rem, elapsed = remaining_of(env)
        goal = loaded.exact_goal
        keys = [key_fn(direct_key_label(root_id, i)) for i in range(rem)]
        identity = {
            "root_id": root_id,
            "task_id": int(root["task_id"]),
            "protocol_id": PROTOCOL_ID,
            "branch": "direct",
            "goal_sha256": sha256_array(goal),
            "probe_goal_sha256": sha256_array(goal),
        }
        trace = compact_trace(
            rollout_segment(
                env,
                policy,
                loaded.observation,
                goal,
                steps=rem,
                key_schedule=keys,
                frozen_value=agent,
                exact_goal=goal,
                probe_goal=goal,
                identity=identity,
                step_counter=counters,
                stop_on_success=True,
            )
        )
        save_branch(traces / "direct" / f"{root_id}.npz", trace)
        summary = summarize_trace(trace)
        row = {
            **summary,
            "worker_id": worker_id,
            "root_id": root_id,
            "task_id": int(root["task_id"]),
            "branch": "direct",
            "goal_sha256": sha256_array(goal),
            "remaining_horizon_at_start": rem,
            "decision_elapsed_steps": root.get("decision_elapsed_steps"),
            "proxy": trace.get("full_proxy"),
            "runtime_identity_sha256": ident["runtime_identity_sha256"],
        }
        append_jsonl(direct_path, row)
        check_budget()
        print({"worker": worker_id, "direct": root_id, "n": summary["n_steps"], "success": summary["ever_success"]}, flush=True)
        return trace, row

    def run_sentinel(root_id: int, phase: str):
        bundle = bundle_of(root_id)
        spec = bundle["manifest"]
        loaded = rebuild(env, bundle, counters)
        rem, elapsed = remaining_of(env)
        n = min(M, rem)
        goal = loaded.exact_goal
        z = loaded.context["d3_target"]
        keys = [key_fn(sentinel_key_label(root_id, i)) for i in range(n)]
        identity = {
            "root_id": int(root_id),
            "task_id": int(spec["task_id"]),
            "protocol_id": PROTOCOL_ID,
            "branch": "sentinel",
            "phase": phase,
            "goal_sha256": sha256_array(goal),
            "probe_goal_sha256": sha256_array(z),
        }
        trace = compact_trace(
            rollout_segment(
                env,
                policy,
                loaded.observation,
                goal,
                steps=n,
                key_schedule=keys,
                frozen_value=agent,
                exact_goal=goal,
                probe_goal=z,
                identity=identity,
                step_counter=counters,
            )
        )
        save_branch(traces / f"sentinel_{phase}" / f"{root_id}.npz", trace)
        check_budget()
        print({"worker": worker_id, "sentinel": root_id, "phase": phase, "n": len(trace.get("steps") or [])}, flush=True)
        return summarize_trace(trace) | {"root_id": int(root_id), "phase": phase, "task_id": int(spec["task_id"])}

    sentinel_rows = []
    for rid in sentinels:
        sentinel_rows.append(run_sentinel(rid, "pre"))
    for root in legal:
        for cid in cand_ids:
            run_short(root, cid)
        if bool(plan[int(root["root_id"])]["is_deep"]):
            for cid in cand_ids:
                run_deep(root, cid)
        run_direct(root)
    for rid in sentinels:
        sentinel_rows.append(run_sentinel(rid, "post"))

    dump_json(dest / "sentinel_rows.json", sentinel_rows)
    dump_json(
        dest / "counters.json",
        {**counters, "wall_seconds": time.time() - started, "n_legal": len(legal), "candidate_order": cand_ids},
    )
    print(json.dumps({"worker": worker_id, "done": True, "external": counters["external_control_steps"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    parser.add_argument("--dataset-dir", default=DEFAULT_PATHS["dataset_dir"])
    parser.add_argument("--official-source", default=DEFAULT_PATHS["official_source"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_PATHS["checkpoint_dir"])
    parser.add_argument("--worker-id", required=True, choices=["A", "B"])
    args = parser.parse_args()
    worker_main(args)


if __name__ == "__main__":
    main()
