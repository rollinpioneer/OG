"""Thin, offline-logging runner around the pinned official OGBench GCIQL agent."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
import ogbench


def scalarize(mapping):
    return {key: float(np.asarray(value)) for key, value in mapping.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--dataset", default="antmaze-large-stitch-v0")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=1_000_000)
    parser.add_argument("--log-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-tasks", type=int)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()

    impls = str(Path(args.official_source) / "impls")
    sys.path.insert(0, impls)
    from agents.gciql import GCIQLAgent, get_config
    from utils.datasets import Dataset, GCDataset
    from utils.evaluation import evaluate
    from utils.flax_utils import save_agent

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = get_config()
    config.encoder = None
    config.frame_stack = None
    config.alpha = args.alpha
    env, train_data, val_data = ogbench.make_env_and_datasets(args.dataset, dataset_dir=args.dataset_dir)
    train = GCDataset(Dataset.create(**train_data), config)
    val = GCDataset(Dataset.create(**val_data), config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    example = train.sample(1)
    agent = GCIQLAgent.create(args.seed, example["observations"], example["actions"], config)

    run_config = {
        "dataset_id": args.dataset,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "log_interval": args.log_interval,
        "eval_episodes": args.eval_episodes,
        "eval_tasks": args.eval_tasks,
        "agent": config.to_dict(),
        "upstream_runner_difference": "wandb removed; stable local output path; fixed-final checkpoint",
        "result_source": "DATA_REAL",
    }
    (output / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_csv = output / "train.csv"
    started = last_time = time.time()
    fieldnames = None
    handle = train_csv.open("w", newline="", encoding="utf-8")
    writer = None
    for step in range(1, args.train_steps + 1):
        batch = train.sample(config.batch_size)
        agent, update_info = agent.update(batch)
        if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
            val_batch = val.sample(config.batch_size)
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            row = {"step": step, "wall_seconds": time.time() - started, "seconds_per_update": (time.time() - last_time) / (1 if step == 1 else args.log_interval)}
            row.update({f"training/{key}": value for key, value in scalarize(update_info).items()})
            row.update({f"validation/{key}": value for key, value in scalarize(val_info).items()})
            if writer is None:
                fieldnames = list(row)
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            last_time = time.time()
            print(json.dumps({"step": step, "wall_seconds": row["wall_seconds"]}), flush=True)
    handle.close()
    save_agent(agent, str(output), args.train_steps)

    task_infos = env.unwrapped.task_infos
    task_count = args.eval_tasks or len(task_infos)
    evaluation = {}
    successes = []
    for task_id in range(1, task_count + 1):
        info, _, _ = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=args.eval_episodes,
            num_video_episodes=0,
            video_frame_skip=3,
            eval_temperature=0.0,
            eval_gaussian=None,
        )
        task_result = scalarize(info)
        evaluation[f"task{task_id}"] = task_result
        successes.append(task_result["success"])
    evaluation["overall_success"] = float(np.mean(successes))
    evaluation["wall_seconds"] = time.time() - started
    evaluation["result_source"] = "ENV_EVALUATED"
    evaluation["training_eligible"] = False
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main()

