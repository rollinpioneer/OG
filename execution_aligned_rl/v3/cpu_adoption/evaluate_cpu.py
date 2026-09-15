
"""Official Cube evaluator on the canonical CPU runtime. 5 tasks x 50 episodes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from execution_aligned_rl.v3.cpu_adoption.lock import apply_canonical_cpu_env, collect_lock, lock_sha256, validate_lock
from execution_aligned_rl.v3.cpu_adoption.protocol import DEFAULT_PATHS, GATES
from execution_aligned_rl.v3.serialization import dump_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--numpy-seed", type=int, default=0)
    args = parser.parse_args()
    apply_canonical_cpu_env()

    import numpy as np
    import ogbench

    from execution_aligned_rl.v3.policy import load_agent

    impls = str(Path(DEFAULT_PATHS["official_source"]) / "impls")
    if impls not in sys.path:
        sys.path.insert(0, impls)
    from utils.evaluation import evaluate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "cpu_backbone_episode_results.jsonl"
    lock = collect_lock(include_jax=True)
    lock["cpu_runtime_environment_sha256"] = lock_sha256(lock)
    holds = validate_lock(lock)
    dump_json(out_dir / "cpu_runtime_lock_eval_process.json", {**lock, "validation_holds": holds})
    if holds:
        dump_json(out_dir / "cpu_backbone_evaluation.json", {"status": "EA32_HOLD_ASSET_MISMATCH", "holds": holds})
        raise SystemExit(2)

    np.random.seed(args.numpy_seed)
    env = ogbench.make_env_and_datasets("cube-double-play-v0", dataset_dir=DEFAULT_PATHS["dataset_dir"], env_only=True)
    agent, config, _train = load_agent(
        DEFAULT_PATHS["official_source"],
        DEFAULT_PATHS["checkpoint_dir"],
        DEFAULT_PATHS["dataset_dir"],
    )
    started = time.time()
    task_rows = {}
    n_finite = 0
    n_bounds = 0
    n_actions = 0
    successes = []
    lines = []
    for task_id in range(1, 6):
        stats, trajs, _renders = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=GATES["episodes_per_task"],
            num_video_episodes=0,
            eval_temperature=0.0,
            eval_gaussian=None,
        )
        ep_success = []
        for ep_i, traj in enumerate(trajs):
            actions = np.asarray(traj["action"], dtype=np.float64)
            if actions.ndim == 1:
                actions = actions[None]
            finite = bool(np.isfinite(actions).all())
            bounds = bool(np.all(actions >= -1.0) and np.all(actions <= 1.0))
            info_last = traj["info"][-1] if traj.get("info") else {}
            success = float(info_last.get("success", 0.0))
            terminated = bool(traj["done"][-1]) if traj.get("done") else False
            n_act = int(len(actions))
            n_actions += n_act
            n_finite += n_act if finite else 0
            n_bounds += n_act if bounds else int(np.sum((actions >= -1.0) & (actions <= 1.0) & np.isfinite(actions)))
            if not finite:
                n_finite += 0
            else:
                # if finite but we already added n_act; for mixed-bounds count per-action
                pass
            # per-action counts
            finite_count = int(np.isfinite(actions).all(axis=1).sum()) if actions.ndim == 2 else int(np.isfinite(actions).all())
            bounds_count = int(np.all((actions >= -1.0) & (actions <= 1.0), axis=1).sum()) if actions.ndim == 2 else int(np.all((actions >= -1.0) & (actions <= 1.0)))
            row = {
                "task_id": task_id,
                "episode": ep_i,
                "success": success,
                "n_actions": n_act,
                "finite": finite,
                "bounds": bounds,
                "finite_action_count": finite_count,
                "bounds_action_count": bounds_count,
                "reward_sum": float(np.sum(traj["reward"])) if traj.get("reward") is not None else None,
                "result_source": "ENV_EVALUATED",
                "training_eligible": False,
                "runtime": "CPU_SINGLE_THREAD",
            }
            lines.append(json.dumps(row, sort_keys=True))
            ep_success.append(success)
            successes.append(success)
        task_rows[f"task{task_id}"] = {
            "success": float(np.mean(ep_success)),
            "n_episodes": len(ep_success),
            "n_success_episodes": int(np.sum(np.asarray(ep_success) > 0)),
            "official_stats_success": float(stats.get("success", np.mean(ep_success))),
            "result_source": "ENV_EVALUATED",
        }
        print(json.dumps({"task_id": task_id, "success": task_rows[f"task{task_id}"]["success"]}), flush=True)

    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # recount finite/bounds per action from jsonl for the gate
    finite_actions = 0
    bound_actions = 0
    total_actions = 0
    for line in lines:
        rec = json.loads(line)
        total_actions += rec["n_actions"]
        finite_actions += rec["finite_action_count"]
        bound_actions += rec["bounds_action_count"]
    overall = float(np.mean(successes))
    nonzero_tasks = sum(1 for k, v in task_rows.items() if v["success"] > 0)
    finite_rate = finite_actions / total_actions if total_actions else 0.0
    bounds_rate = bound_actions / total_actions if total_actions else 0.0
    capability = (
        overall >= GATES["overall_success_min"]
        and nonzero_tasks >= GATES["min_tasks_with_nonzero_success"]
        and finite_rate >= GATES["action_finite_rate"]
        and bounds_rate >= GATES["action_bounds_rate"]
    )
    evaluation = {
        "overall_success": overall,
        "gpu_reference_overall_success": GATES["gpu_reference_overall_success"],
        "gpu_reference_role": "engineering_qualification_only",
        "n_episodes": len(successes),
        "episodes_per_task": GATES["episodes_per_task"],
        "nonzero_success_tasks": nonzero_tasks,
        "finite_action_rate": finite_rate,
        "bounds_action_rate": bounds_rate,
        "n_actions": total_actions,
        "tasks": task_rows,
        "numpy_seed_before_eval": args.numpy_seed,
        "temperature": 0.0,
        "eval_gaussian": None,
        "result_source": "ENV_EVALUATED",
        "training_eligible": False,
        "training_performed": False,
        "runtime": "CPU_SINGLE_THREAD",
        "wall_seconds": time.time() - started,
        "capability_pass": capability,
        "cpu_runtime_environment_sha256": lock["cpu_runtime_environment_sha256"],
    }
    dump_json(out_dir / "cpu_backbone_evaluation.json", evaluation)
    print(json.dumps({"overall_success": overall, "capability_pass": capability, "finite_action_rate": finite_rate, "bounds_action_rate": bounds_rate}, indent=2))


if __name__ == "__main__":
    main()
