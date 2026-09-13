"""Bounded phase-0 mechanism diagnostic; environment forks are evaluation-only."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import numpy as np
import ogbench
from scipy.stats import spearmanr

from execution_aligned_rl.agents.train_candidate_prior import CandidatePrior
from execution_aligned_rl.data.audit_assets import capture_snapshot, restore_snapshot, sha256_file


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def array_hash(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def load_agent(official_source: str, checkpoint_dir: str, epoch: int, train, config_overrides: dict):
    impls = str(Path(official_source) / "impls")
    if impls not in sys.path:
        sys.path.insert(0, impls)
    from agents.gciql import GCIQLAgent, get_config
    from utils.datasets import Dataset, GCDataset
    from utils.flax_utils import restore_agent

    config = get_config()
    config.encoder = None
    config.frame_stack = None
    for key, value in config_overrides.items():
        config[key] = value
    gc_dataset = GCDataset(Dataset.create(**train), config)
    batch = gc_dataset.sample(1)
    agent = GCIQLAgent.create(0, batch["observations"], batch["actions"], config)
    return restore_agent(agent, checkpoint_dir, epoch), config


def action_for(agent, observation, goal, key):
    return np.asarray(agent.sample_actions(observation, goals=goal, seed=key, temperature=0.0))


def value_for(agent, states, goal):
    goals = np.broadcast_to(np.asarray(goal), np.asarray(states).shape)
    return np.asarray(agent.network.select("value")(np.asarray(states), goals))


def sample_candidates(prior_payload, state, count: int, seed: int):
    config = prior_payload["config"]
    norm = prior_payload["normalization"]
    model = CandidatePrior(tuple(config["hidden_dims"]), len(norm["obs_mean"]))
    x = (np.asarray(state) - norm["obs_mean"]) / norm["obs_std"]
    mean, log_std = model.apply({"params": prior_payload["params"]}, x[None])
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((count, x.shape[-1]))
    candidates = (np.asarray(mean)[0] + np.exp(np.asarray(log_std)) * noise) * norm["obs_std"] + norm["obs_mean"]
    return np.clip(candidates, norm["target_min"], norm["target_max"])


def run_steps(env, agent, goal, start_key, steps: int, gamma: float):
    observation = env.unwrapped.get_ob()
    rewards = []
    terminated = truncated = False
    success = 0.0
    for offset in range(steps):
        key = jax.random.fold_in(start_key, offset)
        action = action_for(agent, observation, goal, key)
        observation, env_reward, terminated, truncated, info = env.step(action)
        rewards.append(float(env_reward) - 1.0)
        success = max(success, float(info.get("success", 0.0)))
        if terminated or truncated:
            break
    discounted = sum((gamma**idx) * reward for idx, reward in enumerate(rewards))
    return {
        "observation": np.asarray(observation),
        "discounted_training_reward": float(discounted),
        "environment_reward_sum": float(sum(r + 1.0 for r in rewards)),
        "steps": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": success,
    }


def rank_corr(x, y):
    if len(set(np.round(x, 10))) < 2 or len(set(np.round(y, 10))) < 2:
        return None
    value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="antmaze-large-stitch-v0")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, default=1000000)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--max-branch", type=int, default=40)
    parser.add_argument("--deep-roots", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env, train, _ = ogbench.make_env_and_datasets(args.dataset, dataset_dir=args.dataset_dir)
    agent, config = load_agent(args.official_source, args.checkpoint_dir, args.checkpoint_epoch, train, {"alpha": 0.3})
    with Path(args.candidate_checkpoint).open("rb") as handle:
        prior = pickle.load(handle)

    rng = np.random.default_rng(20260913)
    sample_count = min(8192, len(train["observations"]))
    state_ids = rng.choice(len(train["observations"]), sample_count, replace=False)
    goal_ids = rng.choice(len(train["observations"]), sample_count, replace=False)
    offline_values = value_for(agent, np.asarray(train["observations"])[state_ids], np.asarray(train["observations"])[goal_ids])
    value_std = float(np.std(offline_values))
    tolerance = 0.05 * value_std

    task_count = len(env.unwrapped.task_infos)
    max_steps = int(env.spec.max_episode_steps)
    roots = list(range(args.roots))
    deep_set = set(sorted(roots, key=lambda root: stable_int(f"deep:{root}"))[: args.deep_roots])
    raw_rows = []
    root_rows = []
    started = time.time()

    for root_id in roots:
        task_id = root_id % task_count + 1
        fraction = (0.0, 0.25, 0.5)[stable_int(f"fraction:{root_id}") % 3]
        decision_step = int(max_steps * fraction)
        np.random.seed(root_id)
        env.action_space.seed(root_id)
        observation, info = env.reset(seed=root_id, options={"task_id": task_id})
        final_goal = np.asarray(info["goal"])
        prefix_done = False
        for step in range(decision_step):
            action = action_for(agent, observation, final_goal, jax.random.PRNGKey(stable_int(f"prefix:{root_id}:{step}") & 0xFFFFFFFF))
            observation, _, terminated, truncated, step_info = env.step(action)
            if terminated or truncated:
                prefix_done = True
                break
        if prefix_done:
            root_rows.append({"root_id": root_id, "task_id": task_id, "legal": False, "reason": "prefix_terminated"})
            continue

        snapshot = capture_snapshot(env)
        snapshot_id = array_hash(snapshot["qpos"], snapshot["qvel"], [snapshot["time"]])
        candidates = sample_candidates(prior, observation, args.candidates, stable_int(f"candidates:{root_id}"))
        candidate_set_hash = array_hash(candidates)
        ideal_values = value_for(agent, candidates, final_goal)
        candidate_rows = []
        for candidate_id, candidate in enumerate(candidates):
            band_results = []
            for band in range(2):
                restore_snapshot(env, snapshot)
                key = jax.random.PRNGKey(stable_int(f"branch:{root_id}:{candidate_id}:{band}") & 0xFFFFFFFF)
                first = run_steps(env, agent, candidate, key, args.m, float(config.discount))
                tail_m = (
                    0.0
                    if first["terminated"] or first["truncated"]
                    else float(value_for(agent, first["observation"][None], final_goal)[0])
                )
                proxy_m = first["discounted_training_reward"] + (float(config.discount) ** first["steps"]) * tail_m
                restore_snapshot(env, snapshot)
                full_k = run_steps(env, agent, candidate, key, args.k, float(config.discount))
                tail_k = (
                    0.0
                    if full_k["terminated"] or full_k["truncated"]
                    else float(value_for(agent, full_k["observation"][None], final_goal)[0])
                )
                proxy_k = full_k["discounted_training_reward"] + (float(config.discount) ** full_k["steps"]) * tail_k
                restore_snapshot(env, snapshot)
                full_40 = run_steps(env, agent, candidate, key, args.max_branch, float(config.discount))
                band_results.append((first, proxy_m, full_k, proxy_k, full_40))

            reproducible = bool(
                np.array_equal(band_results[0][0]["observation"], band_results[1][0]["observation"])
                and band_results[0][1] == band_results[1][1]
            )
            deep_return = deep_success = None
            if root_id in deep_set:
                restore_snapshot(env, snapshot)
                key = jax.random.PRNGKey(stable_int(f"deep:{root_id}:{candidate_id}") & 0xFFFFFFFF)
                first = run_steps(env, agent, candidate, key, args.m, float(config.discount))
                remaining = max_steps - decision_step - first["steps"]
                if first["terminated"] or first["truncated"] or remaining <= 0:
                    deep_return = first["environment_reward_sum"]
                    deep_success = first["success"]
                else:
                    continuation = run_steps(
                        env, agent, final_goal, jax.random.fold_in(key, 999), remaining, float(config.discount)
                    )
                    deep_return = first["environment_reward_sum"] + continuation["environment_reward_sum"]
                    deep_success = max(first["success"], continuation["success"])

            row = {
                "experiment_id": "ea_v2_v1_antmaze_seed0",
                "dataset_id": args.dataset,
                "task_id": task_id,
                "root_id": root_id,
                "training_seed": 0,
                "snapshot_hash": snapshot_id,
                "candidate_set_hash": candidate_set_hash,
                "candidate_id": candidate_id,
                "k": args.k,
                "m": args.m,
                "ideal_value": float(ideal_values[candidate_id]),
                "env_proxy_m": float(band_results[0][1]),
                "env_proxy_k": float(band_results[0][3]),
                "env_40_success": float(band_results[0][4]["success"]),
                "full_env_return": deep_return,
                "full_env_success": deep_success,
                "result_source": "ENV_EVALUATED",
                "training_eligible": False,
                "reproducible_band": reproducible,
                "evaluated_steps": args.max_branch,
            }
            raw_rows.append(row)
            candidate_rows.append(row)

        ideal_top = max(candidate_rows, key=lambda row: row["ideal_value"])
        proxy_top = max(candidate_rows, key=lambda row: row["env_proxy_m"])
        proxy_regret = proxy_top["env_proxy_m"] - ideal_top["env_proxy_m"]
        root_summary = {
            "root_id": root_id,
            "task_id": task_id,
            "legal": True,
            "decision_step": decision_step,
            "ideal_proxy_spearman": rank_corr(
                [row["ideal_value"] for row in candidate_rows], [row["env_proxy_m"] for row in candidate_rows]
            ),
            "ideal_top_candidate": ideal_top["candidate_id"],
            "proxy_top_candidate": proxy_top["candidate_id"],
            "proxy_regret": float(proxy_regret),
            "proxy_wrong_beyond_tolerance": bool(proxy_regret > tolerance),
            "proxy_indistinguishable": len({round(row["env_proxy_m"], 8) for row in candidate_rows}) == 1,
            "all_reproducible": all(row["reproducible_band"] for row in candidate_rows),
        }
        if root_id in deep_set:
            full_top = max(candidate_rows, key=lambda row: (row["full_env_success"], row["full_env_return"]))
            root_summary.update(
                {
                    "full_top_candidate": full_top["candidate_id"],
                    "ideal_top_full_success": ideal_top["full_env_success"],
                    "best_full_success": full_top["full_env_success"],
                    "full_gap_present": bool(full_top["full_env_success"] > ideal_top["full_env_success"]),
                    "full_indistinguishable": len({row["full_env_success"] for row in candidate_rows}) == 1,
                }
            )
        root_rows.append(root_summary)
        print(json.dumps({"root": root_id, "legal": True, "elapsed": time.time() - started}), flush=True)

    raw_path = out / "phase0_candidates.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    roots_path = out / "phase0_roots.csv"
    all_keys = sorted({key for row in root_rows for key in row})
    with roots_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(root_rows)

    legal = [row for row in root_rows if row["legal"]]
    deep = [row for row in legal if "full_gap_present" in row]
    wrong_rate = float(np.mean([row["proxy_wrong_beyond_tolerance"] for row in legal])) if legal else 0.0
    full_gap_rate = float(np.mean([row["full_gap_present"] for row in deep])) if deep else 0.0
    correlations = [row["ideal_proxy_spearman"] for row in legal if row["ideal_proxy_spearman"] is not None]
    if len(legal) < 24:
        status = "HOLD_INSUFFICIENT_LEGAL_ROOTS"
    elif wrong_rate >= 0.1 and full_gap_rate > 0:
        status = "GAP_PRESENT_PROVISIONAL"
    elif wrong_rate >= 0.1:
        status = "GAP_PROXY_ONLY"
    else:
        status = "NO_USEFUL_GAP_ON_THIS_TASK"
    summary = {
        "status": status,
        "legal_roots": len(legal),
        "planned_roots": args.roots,
        "deep_roots_evaluated": len(deep),
        "value_std_data_real": value_std,
        "value_tolerance": tolerance,
        "proxy_wrong_selection_rate": wrong_rate,
        "full_gap_rate_on_preregistered_deep_roots": full_gap_rate,
        "mean_ideal_proxy_spearman": float(np.mean(correlations)) if correlations else None,
        "proxy_indistinguishable_rate": float(np.mean([row["proxy_indistinguishable"] for row in legal])) if legal else None,
        "full_indistinguishable_rate": float(np.mean([row["full_indistinguishable"] for row in deep])) if deep else None,
        "all_environment_branches_training_eligible": False,
        "raw_sha256": sha256_file(raw_path),
        "roots_sha256": sha256_file(roots_path),
        "elapsed_seconds": time.time() - started,
    }
    (out / "phase0_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
