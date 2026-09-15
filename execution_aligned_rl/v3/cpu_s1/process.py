
"""CPU S1 acquire/worker/finalize. apply_cpu_env() runs before JAX import."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from execution_aligned_rl.v3.cpu_s1.protocol import DEFAULT_PATHS, PROTOCOL_ID
from execution_aligned_rl.v3.cpu_s1.runtime_identity import apply_cpu_env, identity_sha256, write_identity
from execution_aligned_rl.v3.serialization import dump_json, load_json


def _boot():
    apply_cpu_env()


def specs(exp: Path) -> list[dict]:
    return load_json(exp / "protocol" / "root_specs.json")["roots"]


def experiment_dir(repo: Path) -> Path:
    return repo / DEFAULT_PATHS["experiment_rel"]


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


def prefix_key(root_id: int, step: int):
    from execution_aligned_rl.v3.policy import prng_key

    return prng_key(f"ea3-cpu-s1-v1:prefix:{root_id}:{step}")


def probe_key(root_id: int, step: int):
    from execution_aligned_rl.v3.policy import prng_key

    return prng_key(f"ea3-cpu-s1-v1:probe:{root_id}:{step}")


def acquire_main(args) -> None:
    _boot()
    import numpy as np
    from execution_aligned_rl.v3.env_state import (
        capture_integration,
        capture_observation_stage,
        capture_python_state,
        elapsed_steps,
        exact_goal,
        measure_state,
        physical_task_targets,
    )
    from execution_aligned_rl.v3.hashing import sha256_array
    from execution_aligned_rl.v3.policy import FrozenPolicy, action_for, load_agent
    from execution_aligned_rl.v3.root_bundle import root_dir, write_root_bundle
    from execution_aligned_rl.v3.rollout import rollout_segment

    repo = Path(args.repo)
    exp = experiment_dir(repo)
    ident = write_identity(exp / "identities" / "collector.json", "collector")
    store = Path(args.root_store)
    store.mkdir(parents=True, exist_ok=True)
    out = exp / "engineering" / "acquisition"
    out.mkdir(parents=True, exist_ok=True)
    (exp / "manifests").mkdir(parents=True, exist_ok=True)
    (exp / "identities").mkdir(parents=True, exist_ok=True)
    agent, config, train = load_agent(args.official_source, args.checkpoint_dir, args.dataset_dir)
    policy = FrozenPolicy(agent)
    env = make_env(args.dataset_dir)
    counters = {"external_control_steps": 0, "reset_internal_steps": 0, "resets": 0, "runtime_identity_sha256": ident["runtime_identity_sha256"]}
    train_obs = np.asarray(train["observations"])
    records = []
    started = time.time()
    for spec in specs(exp):
        root_id = int(spec["root_id"])
        task_id = int(spec["task_id"])
        planned = int(spec["planned_decision_step"])
        observation, info = allocate_env(env, spec["reset_seed"], task_id, counters)
        goal = np.asarray(info["goal"], dtype=np.float64).copy()
        if float(np.max(np.abs(goal - exact_goal(env)))) > 0.0:
            raise RuntimeError(f"goal mismatch root {root_id}")
        z_index = int(spec["d3_target_index"])
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
            prefix_actions.append(np.asarray(action, dtype=np.float64).copy())
            prefix_steps.append(
                measure_state(env, observation=next_obs, action=action, reward=reward, terminated=terminated, truncated=truncated, info=step_info, goal=goal)
            )
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
            "prefix_trace": {"identity": {"root_id": root_id, "task_id": task_id, "protocol_id": PROTOCOL_ID}, "steps": prefix_steps, "status": "OK" if prefix_steps or planned == 0 else "EMPTY", "full_proxy": None},
            "original_live_probe": None,
            "manifest": {
                "protocol_id": PROTOCOL_ID,
                "root_id": root_id,
                "set": spec["set"],
                "task_id": task_id,
                "planned_decision_step": planned,
                "actual_prefix_length": int(len(prefix_actions)),
                "reset_seed": spec["reset_seed"],
                "legal": (not ineligible),
                "ineligible_reason": ineligible_reason,
                "result_source": "ENV_EVALUATED",
                "training_eligible": False,
                "d3_target_index": z_index,
                "decision_elapsed_steps": elapsed_steps(env),
                "decision_success": bool(getattr(env.unwrapped, "_success", False)),
                "goal_sha256": sha256_array(goal),
                "d3_target_sha256": sha256_array(z),
                "runtime": "CPU_SINGLE_THREAD",
            },
        }
        if not ineligible:
            probe = rollout_segment(
                env, policy, decision_obs, goal, steps=5,
                key_schedule=[probe_key(root_id, i) for i in range(5)],
                frozen_value=agent, exact_goal=goal, mode="closed_loop",
                identity={"root_id": root_id, "task_id": task_id, "protocol_id": PROTOCOL_ID, "role": "original_live_probe"},
                step_counter=counters, probe_goal=z,
            )
            bundle["original_live_probe"] = probe
            bundle["manifest"]["probe_steps"] = len(probe["steps"])
            bundle["manifest"]["probe_full_proxy"] = probe.get("full_proxy")
        dest = root_dir(store, PROTOCOL_ID, root_id)
        if dest.exists():
            shutil.rmtree(dest)
        hashes = write_root_bundle(dest, bundle)
        record = {**bundle["manifest"], "store_path": str(dest), "file_hashes": hashes}
        dump_json(out / f"{root_id}.json", record)
        records.append(record)
        print(json.dumps({"acquired": root_id, "set": spec["set"], "legal": record["legal"], "prefix": record["actual_prefix_length"]}), flush=True)
    dump_json(out / "counters.json", {**counters, "wall_seconds": time.time() - started})
    by_set = {"regression": [r for r in records if r["set"] == "regression"], "holdout": [r for r in records if r["set"] == "holdout"]}
    dump_json(exp / "manifests" / "regression_root_manifest.json", {"set": "regression", "protocol_id": PROTOCOL_ID, "roots": by_set["regression"]})
    dump_json(exp / "manifests" / "holdout_root_manifest.json", {"set": "holdout", "protocol_id": PROTOCOL_ID, "roots": by_set["holdout"]})
    dump_json(exp / "manifests" / "all_roots.json", {"roots": records})


def _legal_bundles(exp: Path, store: Path, set_name: str | None = None):
    from execution_aligned_rl.v3.root_bundle import read_root_bundle, root_dir

    rows = load_json(exp / "manifests" / "all_roots.json")["roots"]
    out = []
    for row in rows:
        if set_name and row["set"] != set_name:
            continue
        bundle = read_root_bundle(root_dir(store, PROTOCOL_ID, row["root_id"]), verify=True)
        if bundle["manifest"]["legal"]:
            out.append(bundle)
    return out


def worker_main(args) -> None:
    _boot()
    import numpy as np
    from execution_aligned_rl.v3.env_state import measure_state, perturb_state, restore_direct
    from execution_aligned_rl.v3.policy import FrozenPolicy, load_agent
    from execution_aligned_rl.v3.root_bundle import save_trace_npz
    from execution_aligned_rl.v3.root_loader import load_root
    from execution_aligned_rl.v3.rollout import rollout_segment
    from execution_aligned_rl.v3.runtime_qual.compare_v2 import compare_states_v2, compare_traces_v2

    repo = Path(args.repo)
    exp = experiment_dir(repo)
    worker_id = args.worker_id
    ident = write_identity(exp / "identities" / f"worker_{worker_id}.json", f"worker_{worker_id}")
    store = Path(args.root_store)
    dest = exp / "engineering" / "workers" / worker_id
    dest.mkdir(parents=True, exist_ok=True)
    agent, config, train = load_agent(args.official_source, args.checkpoint_dir, args.dataset_dir)
    policy = FrozenPolicy(agent)
    env = make_env(args.dataset_dir)
    counters = {"external_control_steps": 0, "reset_internal_steps": 0, "resets": 0, "worker_id": worker_id, "runtime_identity_sha256": ident["runtime_identity_sha256"]}
    started = time.time()

    def run_root(bundle, tag: str):
        spec = bundle["manifest"]
        root_id = spec["root_id"]
        allocate_env(env, 0, spec["task_id"], counters)
        loaded = load_root(bundle, mode="BOOTSTRAP_PREFIX_REPLAY", strict=True, env=env, step_counter=counters)
        goal = loaded.exact_goal
        z = loaded.context["d3_target"]
        perturb_state(env)
        restore_obs = restore_direct(env, bundle["decision_integration"], bundle["decision_python_state"], bundle["decision_observation_stage"])
        left = {
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
        }
        right = measure_state(env, observation=restore_obs, goal=goal)
        for k in ("terminated", "truncated"):
            left.pop(k, None)
            right[k] = None
        d0 = compare_states_v2(left, right, phase="D0")
        original = bundle["original_live_probe"]
        orig_actions = [step["action"] for step in original["steps"]]
        keys = [probe_key(root_id, i) for i in range(5)]
        identity = {"root_id": root_id, "task_id": spec["task_id"], "protocol_id": PROTOCOL_ID, "worker": worker_id, "tag": tag}

        def back_to_decision():
            return np.asarray(restore_direct(env, bundle["decision_integration"], bundle["decision_python_state"], bundle["decision_observation_stage"]), dtype=np.float64).copy()

        obs = back_to_decision()
        d1 = rollout_segment(env, policy, obs, goal, steps=1, key_schedule=keys[:1], frozen_value=agent, exact_goal=goal, mode="open_loop", actions=orig_actions[:1], identity={**identity, "test": "D1"}, step_counter=counters, probe_goal=z)
        obs = back_to_decision()
        d2 = rollout_segment(env, policy, obs, goal, steps=len(orig_actions), key_schedule=keys, frozen_value=agent, exact_goal=goal, mode="open_loop", actions=orig_actions, identity={**identity, "test": "D2"}, step_counter=counters, probe_goal=z)
        obs = back_to_decision()
        d3 = rollout_segment(env, policy, obs, goal, steps=5, key_schedule=keys, frozen_value=agent, exact_goal=goal, mode="closed_loop", identity={**identity, "test": "D3"}, step_counter=counters, probe_goal=z)
        root_out = dest / str(root_id) / tag
        root_out.mkdir(parents=True, exist_ok=True)
        save_trace_npz(root_out / "d1.npz", d1)
        save_trace_npz(root_out / "d2.npz", d2)
        save_trace_npz(root_out / "d3.npz", d3)
        dump_json(root_out / "d0.json", d0)
        dump_json(root_out / "loader_proof.json", loaded.proof)
        return {"root_id": root_id, "d0": d0, "d1": d1, "d2": d2, "d3": d3}

    legal = _legal_bundles(exp, store)
    results = []
    for bundle in legal:
        rec = run_root(bundle, "primary")
        results.append({"root_id": rec["root_id"], "d0_status": rec["d0"]["status"]})
        print(json.dumps({"worker": worker_id, "root": rec["root_id"], "d0": rec["d0"]["status"]}), flush=True)

    for set_name in ("regression", "holdout"):
        group = [b for b in legal if b["manifest"].get("set") == set_name]
        if len(group) < 2:
            continue
        a, b = group[0], group[1]
        rec_a1 = run_root(a, f"aba_{set_name}_A1")
        rec_b = run_root(b, f"aba_{set_name}_B")
        rec_a2 = run_root(a, f"aba_{set_name}_A2")
        aba = {
            "set": set_name,
            "A_id": a["manifest"]["root_id"],
            "B_id": b["manifest"]["root_id"],
            "A1_vs_A2_d3": compare_traces_v2(rec_a1["d3"], rec_a2["d3"], phase="D3"),
            "A1_vs_B_d3": compare_traces_v2(rec_a1["d3"], rec_b["d3"], phase="D3"),
        }
        dump_json(dest / f"aba_{set_name}.json", aba)
    dump_json(dest / "results.json", {"results": results, "counters": counters, "wall_seconds": time.time() - started})
    dump_json(dest / "counters.json", counters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["acquire", "worker"])
    parser.add_argument("--repo", default=DEFAULT_PATHS["repo"])
    parser.add_argument("--dataset-dir", default=DEFAULT_PATHS["dataset_dir"])
    parser.add_argument("--official-source", default=DEFAULT_PATHS["official_source"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_PATHS["checkpoint_dir"])
    parser.add_argument("--root-store", default=DEFAULT_PATHS["root_store"])
    parser.add_argument("--worker-id", default="A")
    args = parser.parse_args()
    if args.command == "acquire":
        acquire_main(args)
    else:
        worker_main(args)


if __name__ == "__main__":
    main()
