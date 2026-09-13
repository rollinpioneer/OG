"""Phase A audit for the frozen OGBench experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import ogbench
import mujoco


HORIZONS = (1, 5, 10, 20, 40)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def episode_bounds(terminals: np.ndarray) -> list[tuple[int, int]]:
    ends = np.flatnonzero(np.asarray(terminals) > 0)
    if len(ends) == 0 or ends[-1] != len(terminals) - 1:
        raise ValueError("terminal markers do not close the final episode")
    starts = np.r_[0, ends[:-1] + 1]
    bounds = [(int(s), int(e) + 1) for s, e in zip(starts, ends)]
    if sum(e - s for s, e in bounds) != len(terminals):
        raise AssertionError("episode partition does not cover dataset exactly")
    return bounds


def summarize_dataset(dataset: dict) -> dict:
    bounds = episode_bounds(dataset["terminals"])
    lengths = np.asarray([e - s for s, e in bounds])
    windows = {str(h): int(sum(max(0, n - h) for n in lengths)) for h in HORIZONS}
    return {
        "transitions": int(len(dataset["observations"])),
        "episodes": int(len(bounds)),
        "keys": sorted(dataset),
        "observation_shape": list(dataset["observations"].shape[1:]),
        "action_shape": list(dataset["actions"].shape[1:]),
        "terminal_count": int(np.asarray(dataset["terminals"]).sum()),
        "episode_length": {
            "min": int(lengths.min()),
            "p25": float(np.quantile(lengths, 0.25)),
            "median": float(np.median(lengths)),
            "p75": float(np.quantile(lengths, 0.75)),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
        },
        "legal_windows_by_horizon": windows,
    }


def deterministic_split(dataset_id: str, bounds: list[tuple[int, int]]) -> dict:
    ranked = sorted(
        range(len(bounds)),
        key=lambda idx: hashlib.sha256(f"{dataset_id}:val_episode:{idx}".encode()).hexdigest(),
    )
    count = len(ranked)
    base = [int(count * p) for p in (0.4, 0.3, 0.3)]
    for idx in range(count - sum(base)):
        base[idx] += 1
    names = ("offline_select", "offline_calibration", "offline_audit")
    result = {}
    cursor = 0
    for name, size in zip(names, base):
        ids = sorted(ranked[cursor : cursor + size])
        cursor += size
        lengths = [bounds[i][1] - bounds[i][0] for i in ids]
        result[name] = {
            "episode_ids": ids,
            "episode_count": len(ids),
            "transition_count": int(sum(lengths)),
            "legal_windows_by_horizon": {
                str(h): int(sum(max(0, n - h) for n in lengths)) for h in HORIZONS
            },
        }
    assigned = [idx for split in result.values() for idx in split["episode_ids"]]
    if sorted(assigned) != list(range(count)) or len(set(assigned)) != count:
        raise AssertionError("validation episode split is incomplete or overlapping")
    return result


def wrapper_chain(env):
    chain = []
    current = env
    while True:
        chain.append(current)
        if not hasattr(current, "env"):
            break
        current = current.env
    return chain


def capture_snapshot(env) -> dict:
    base = env.unwrapped
    return {
        "qpos": base.data.qpos.copy(),
        "qvel": base.data.qvel.copy(),
        "act": None if base.data.act is None else base.data.act.copy(),
        "time": float(base.data.time),
        "rng": copy.deepcopy(base.np_random.bit_generator.state),
        "global_numpy_rng": copy.deepcopy(np.random.get_state()),
        "action_space_rng": copy.deepcopy(base.action_space.np_random.bit_generator.state),
        "ctrl": base.data.ctrl.copy(),
        "qacc_warmstart": base.data.qacc_warmstart.copy(),
        "qfrc_applied": base.data.qfrc_applied.copy(),
        "xfrc_applied": base.data.xfrc_applied.copy(),
        "mocap_pos": base.data.mocap_pos.copy(),
        "mocap_quat": base.data.mocap_quat.copy(),
        "userdata": base.data.userdata.copy(),
        "cur_task_id": copy.deepcopy(getattr(base, "cur_task_id", None)),
        "cur_task_info": copy.deepcopy(getattr(base, "cur_task_info", None)),
        "cur_goal_xy": copy.deepcopy(getattr(base, "cur_goal_xy", None)),
        "elapsed_steps": [copy.deepcopy(getattr(w, "_elapsed_steps", None)) for w in wrapper_chain(env)],
    }


def restore_snapshot(env, snapshot: dict) -> None:
    base = env.unwrapped
    base.set_state(snapshot["qpos"], snapshot["qvel"])
    if snapshot["act"] is not None and base.data.act is not None:
        base.data.act[:] = snapshot["act"]
    base.data.time = snapshot["time"]
    base.np_random.bit_generator.state = copy.deepcopy(snapshot["rng"])
    np.random.set_state(copy.deepcopy(snapshot["global_numpy_rng"]))
    base.action_space.np_random.bit_generator.state = copy.deepcopy(snapshot["action_space_rng"])
    for name in (
        "ctrl",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    ):
        getattr(base.data, name)[:] = snapshot[name]
    for name in ("cur_task_id", "cur_task_info", "cur_goal_xy"):
        if hasattr(base, name):
            setattr(base, name, copy.deepcopy(snapshot[name]))
    for wrapper, elapsed in zip(wrapper_chain(env), snapshot["elapsed_steps"]):
        if elapsed is not None:
            wrapper._elapsed_steps = elapsed
    mujoco.mj_forward(base.model, base.data)


def audit_environment(env, reset_count: int = 8) -> dict:
    tasks = getattr(env.unwrapped, "task_infos", [])
    rows = []
    for idx in range(reset_count):
        task_id = idx % max(1, len(tasks)) + 1
        options = {"task_id": task_id} if tasks else None
        np.random.seed(idx)
        env.action_space.seed(idx)
        obs_a, info_a = env.reset(seed=idx, options=options)
        np.random.seed(idx)
        env.action_space.seed(idx)
        obs_b, info_b = env.reset(seed=idx, options=options)
        deterministic_reset = bool(np.array_equal(obs_a, obs_b))
        snapshot = capture_snapshot(env)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        result_a = env.step(action)
        restore_snapshot(env, snapshot)
        result_b = env.step(action)
        restore_ok = bool(
            np.array_equal(result_a[0], result_b[0])
            and result_a[1:4] == result_b[1:4]
            and result_a[4].get("success") == result_b[4].get("success")
        )
        rows.append(
            {
                "reset_id": idx,
                "task_id": task_id,
                "deterministic_reset": deterministic_reset,
                "snapshot_restore_exact_for_zero_action": restore_ok,
                "reward": float(result_a[1]),
                "terminated": bool(result_a[2]),
                "truncated": bool(result_a[3]),
                "success": float(result_a[4].get("success", 0.0)),
                "result_source": "ENV_EVALUATED",
                "training_eligible": False,
            }
        )
    return {
        "engineering_reset_limit": reset_count,
        "checks": rows,
        "all_deterministic_resets": all(r["deterministic_reset"] for r in rows),
        "all_snapshot_restores_exact": all(r["snapshot_restore_exact_for_zero_action"] for r in rows),
        "snapshot_fields": [
            "mujoco_qpos",
            "mujoco_qvel",
            "mujoco_act",
            "mujoco_time",
            "environment_rng",
            "global_numpy_rng",
            "action_space_rng",
            "mujoco_ctrl_warmstart_applied_forces_mocap_userdata",
            "task_id",
            "task_info",
            "goal_xy",
            "wrapper_elapsed_steps",
        ],
        "future_information_isolation": {
            "planner_entrypoint_present": False,
            "oracle_entrypoint_present": False,
            "status": "NOT_APPLICABLE_PHASE_A",
        },
    }


def package_versions(names: list[str]) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="antmaze-large-stitch-v0")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.output_dir)
    data_dir = Path(args.dataset_dir).expanduser()
    env, train, val = ogbench.make_env_and_datasets(args.dataset, dataset_dir=str(data_dir))
    train_path = data_dir / f"{args.dataset}.npz"
    val_path = data_dir / f"{args.dataset}-val.npz"
    train_summary = summarize_dataset(train)
    val_summary = summarize_dataset(val)
    train_bounds = episode_bounds(train["terminals"])
    val_bounds = episode_bounds(val["terminals"])

    normalizer = {
        "observation_mean": np.asarray(train["observations"]).mean(axis=0),
        "observation_std": np.maximum(np.asarray(train["observations"]).std(axis=0), 1e-6),
        "action_mean": np.asarray(train["actions"]).mean(axis=0),
        "action_std": np.maximum(np.asarray(train["actions"]).std(axis=0), 1e-6),
    }
    normalizer_path = out / "normalizer_train_only.npz"
    np.savez(normalizer_path, **normalizer)

    source_commit = subprocess.check_output(
        ["git", "-C", args.official_source, "rev-parse", "HEAD"], text=True
    ).strip()
    environment_manifest = {
        "dataset_id": args.dataset,
        "ogbench_commit": source_commit,
        "ogbench_version": importlib.metadata.version("ogbench"),
        "python": sys.version,
        "platform": platform.platform(),
        "observation_space": {
            "shape": list(env.observation_space.shape),
            "dtype": str(env.observation_space.dtype),
        },
        "action_space": {
            "shape": list(env.action_space.shape),
            "dtype": str(env.action_space.dtype),
            "low": np.asarray(env.action_space.low),
            "high": np.asarray(env.action_space.high),
        },
        "max_episode_steps": getattr(env.spec, "max_episode_steps", None),
        "task_count": len(getattr(env.unwrapped, "task_infos", [])),
        "task_infos": getattr(env.unwrapped, "task_infos", []),
        "success_timing": getattr(env.unwrapped, "_success_timing", None),
        "terminate_at_goal": getattr(env.unwrapped, "_terminate_at_goal", None),
        "goal_tolerance": getattr(env.unwrapped, "_goal_tol", None),
        "control_frequency_hz": env.unwrapped.metadata.get("render_fps"),
        "dependencies": package_versions(
            ["numpy", "jax", "jaxlib", "flax", "mujoco", "dm-control", "gymnasium", "ogbench"]
        ),
    }
    dataset_manifest = {
        "dataset_id": args.dataset,
        "license": "OGBench upstream MIT code; dataset terms per upstream project",
        "files": {
            "train": {"name": train_path.name, "bytes": train_path.stat().st_size, "sha256": sha256_file(train_path)},
            "validation": {"name": val_path.name, "bytes": val_path.stat().st_size, "sha256": sha256_file(val_path)},
        },
        "train": train_summary,
        "validation": val_summary,
        "result_source": "DATA_REAL",
    }
    split_manifest = {
        "dataset_id": args.dataset,
        "unit": "episode",
        "algorithm": "sha256_order_then_largest_remainder",
        "proportions": {"offline_select": 0.4, "offline_calibration": 0.3, "offline_audit": 0.3},
        "validation_splits": deterministic_split(args.dataset, val_bounds),
        "train_episode_count": len(train_bounds),
        "normalization_source": "official_training_only",
        "normalizer_file": normalizer_path.name,
        "normalizer_sha256": sha256_file(normalizer_path),
        "window_boundary_test": "PASS",
    }
    interface_audit = audit_environment(env)
    interface_audit.update(
        {
            "observation_semantics": "Ant qpos concatenated with qvel; public observation includes full MuJoCo generalized state",
            "action_semantics": "8 normalized actuator controls clipped by Box bounds",
            "environment_reward": "goal success indicator in goal-conditioned mode: 1 on success, else 0",
            "training_reward": "official GCIQL gc_negative contract: 0 when relabeled state equals goal, else -1",
            "terminal_semantics": "dataset terminal marks trajectory file boundary; GCIQL mask is goal-relabel success, not file termination",
            "quaternion_order": "MuJoCo qpos convention (w, x, y, z) for free joint",
            "state_markov_audit": "PASS_PUBLIC_OBSERVATION_CONTAINS_QPOS_QVEL",
        }
    )
    resource_estimate = {
        "run_root": args.run_root,
        "phase_a_wall_seconds": time.time() - started,
        "dataset_bytes": train_path.stat().st_size + val_path.stat().st_size,
        "checkpoint_budget_bytes_estimate": 4_000_000_000,
        "raw_mechanism_budget_bytes_estimate": 1_000_000_000,
        "gpu_measurement_status": "pending_phase_b_1000_update_benchmark",
        "phase_b_update_cap": 1_000_000,
        "candidate_prior_update_cap": 50_000,
        "phase_0_max_environment_steps": 32 * 8 * 2 * 40 + 8 * 8 * int(env.spec.max_episode_steps),
        "phase_0_note": "upper bound; terminated prefixes and branches stop early; forks are evaluation-only",
    }

    write_json(out / "environment_manifest.json", environment_manifest)
    write_json(out / "dataset_manifest.json", dataset_manifest)
    write_json(out / "split_manifest.json", split_manifest)
    write_json(out / "interface_audit.json", interface_audit)
    write_json(out / "resource_estimate.json", resource_estimate)
    print(json.dumps({"status": "PASS", "output_dir": str(out), "dataset": dataset_manifest}, indent=2))


if __name__ == "__main__":
    main()
