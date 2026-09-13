"""Freeze and audit phase-B value, controller, and candidate-prior artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import jax
import numpy as np
import ogbench

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.evaluation.evaluate_gap import action_for, load_agent, value_for


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-code-commit", required=True)
    parser.add_argument("--epoch", type=int, default=1_000_000)
    args = parser.parse_args()

    env, train, val = ogbench.make_env_and_datasets("antmaze-large-stitch-v0", dataset_dir=args.dataset_dir)
    agent, config = load_agent(args.official_source, args.checkpoint_dir, args.epoch, train, {"alpha": 0.3})
    rng = np.random.default_rng(20260913)
    ids = rng.choice(len(val["observations"]), size=min(4096, len(val["observations"])), replace=False)
    observations = np.asarray(val["observations"])[ids]
    adjacent = np.asarray(val["next_observations"])[ids]
    self_actions = np.stack(
        [action_for(agent, state, state, jax.random.PRNGKey(idx)) for idx, state in enumerate(observations[:256])]
    )
    adjacent_actions = np.stack(
        [action_for(agent, state, goal, jax.random.PRNGKey(idx + 256)) for idx, (state, goal) in enumerate(zip(observations[:256], adjacent[:256]))]
    )
    values = value_for(agent, observations, adjacent)
    evaluation = json.loads((Path(args.checkpoint_dir) / "evaluation.json").read_text(encoding="utf-8"))
    candidate_audit = json.loads((Path(args.candidate_dir) / "candidate_audit.json").read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint_dir) / f"params_{args.epoch}.pkl"
    candidate_checkpoint = Path(args.candidate_dir) / "candidate_prior.pkl"
    official_commit = subprocess.check_output(
        ["git", "-C", args.official_source, "rev-parse", "HEAD"], text=True
    ).strip()
    action_ok = bool(
        np.isfinite(self_actions).all()
        and np.isfinite(adjacent_actions).all()
        and np.all(self_actions >= env.action_space.low - 1e-6)
        and np.all(self_actions <= env.action_space.high + 1e-6)
        and np.all(adjacent_actions >= env.action_space.low - 1e-6)
        and np.all(adjacent_actions <= env.action_space.high + 1e-6)
    )
    status = "PASS"
    if not action_ok or candidate_audit["status"] != "PASS" or evaluation["overall_success"] <= 0:
        status = "HOLD_BACKBONE_OR_CANDIDATES"
    manifest = {
        "status": status,
        "dataset_id": "antmaze-large-stitch-v0",
        "training_code_commit": args.training_code_commit,
        "official_ogbench_commit": official_commit,
        "implementation": "official OGBench GCIQL modules via local no-wandb runner",
        "upstream_runner_difference": "external wandb call removed; stable local paths; fixed-final checkpoint",
        "seed": 0,
        "updates": args.epoch,
        "checkpoint_rule": "fixed_final_checkpoint",
        "value_checkpoint_sha256": sha256_file(checkpoint),
        "controller_checkpoint_sha256": sha256_file(checkpoint),
        "shared_gciql_checkpoint": True,
        "candidate_prior_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "candidate_prior_config": candidate_audit,
        "controller": {
            "goal_conditioned": True,
            "explicit_remaining_horizon_input": False,
            "k_protocol": 20,
            "action_interface_finite_and_bounded": action_ok,
            "self_goal_action_abs_mean": float(np.abs(self_actions).mean()),
            "adjacent_goal_action_abs_mean": float(np.abs(adjacent_actions).mean()),
        },
        "value": {
            "finite_rate": float(np.isfinite(values).mean()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        },
        "official_environment_evaluation": evaluation,
    }
    contract = {
        "dataset_id": "antmaze-large-stitch-v0",
        "discount": float(config.discount),
        "environment_reward": {"failure": 0.0, "success": 1.0},
        "gciql_training_reward": {"non_goal": -1.0, "goal": 0.0},
        "planning_conversion": "r_training = r_environment - 1",
        "value_semantics": "frozen conservative ranking critic under official GCIQL relabeling; not a success probability",
        "goal_success_mask": "1 - relabeled_goal_success",
        "dataset_terminal_semantics": "episode/file boundary only; never substituted for goal-success mask",
        "time_limit_semantics": "official env.spec.max_episode_steps and TimeLimit elapsed_steps",
        "absorbing_tail": "tail value set to zero after success/termination",
    }
    write_json(Path(args.output_dir) / "backbone_manifest.json", manifest)
    write_json(Path(args.output_dir) / "reward_value_contract.json", contract)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
