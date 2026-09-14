"""Unique root loader. BOOTSTRAP_PREFIX_REPLAY is the primary path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from execution_aligned_rl.v3.compare import compare_states
from execution_aligned_rl.v3.contracts import PROTOCOL_ID
from execution_aligned_rl.v3.env_state import (
    elapsed_steps,
    exact_goal,
    measure_state,
    public_observation,
    restore_direct,
)
from execution_aligned_rl.v3.hashing import sha256_array
from execution_aligned_rl.v3.root_bundle import read_root_bundle


@dataclass
class LoadedRoot:
    env: Any
    observation: np.ndarray
    exact_goal: np.ndarray
    context: dict
    proof: dict


def _count_step(counter: dict | None) -> None:
    if counter is not None:
        counter["external_control_steps"] = int(counter.get("external_control_steps", 0)) + 1


def load_root(bundle, mode: str = "BOOTSTRAP_PREFIX_REPLAY", strict: bool = True, env=None, step_counter: dict | None = None) -> LoadedRoot:
    if isinstance(bundle, (str, bytes)):
        raise TypeError("bundle must be a loaded dict from read_root_bundle")
    if env is None:
        raise ValueError("load_root requires an allocated env; reset may be used only to allocate")
    mode = str(mode)
    goal = np.asarray(bundle["goal_observation"], dtype=np.float64).copy()
    proof = {
        "mode": mode,
        "protocol_id": PROTOCOL_ID,
        "root_id": bundle["manifest"]["root_id"],
        "goal_sha256": sha256_array(goal),
        "file_hash_verified": True,
    }

    if mode == "DIRECT_DECISION_RESTORE":
        obs = restore_direct(
            env,
            bundle["decision_integration"],
            bundle["decision_python_state"],
            bundle["decision_observation_stage"],
        )
        measured = measure_state(env, observation=obs, goal=goal)
        cmp = compare_states(
            {
                "observation": bundle["decision_observation"],
                "integration": bundle["decision_integration"],
                "qpos": bundle["decision_python_state"].get("qpos"),
                "qvel": bundle["decision_python_state"].get("qvel"),
                "act": bundle["decision_python_state"].get("act"),
                "ctrl": bundle["decision_python_state"].get("ctrl"),
                "warmstart": bundle["decision_python_state"].get("warmstart"),
                "goal_observation": goal,
                "elapsed_steps": (bundle["manifest"] or {}).get("decision_elapsed_steps"),
                "task_id": bundle["manifest"]["task_id"],
                "success": bundle["manifest"].get("decision_success"),
                "terminated": False,
                "truncated": False,
            },
            measured,
            has_actions=False,
        )
        proof["decision_compare"] = cmp
        if strict and cmp["status"] != "PASS":
            raise RuntimeError(f"DIRECT_DECISION_RESTORE failed: {cmp['failures']}")
        context = {
            "bundle": bundle,
            "mode": mode,
            "d3_target": np.asarray(bundle["d3_target_observation"], dtype=np.float64).copy(),
            "prefix_actions": np.asarray(bundle["prefix_actions"]),
        }
        return LoadedRoot(env=env, observation=np.asarray(obs, dtype=np.float64).copy(), exact_goal=goal, context=context, proof=proof)

    if mode != "BOOTSTRAP_PREFIX_REPLAY":
        raise ValueError(f"unknown load mode {mode}")

    obs0 = restore_direct(
        env,
        bundle["bootstrap_integration"],
        bundle["bootstrap_python_state"],
        bundle["bootstrap_observation_stage"],
    )
    bootstrap_cmp = compare_states(
        {
            "observation": bundle["initial_observation"],
            "integration": bundle["bootstrap_integration"],
            "goal_observation": goal,
            "qpos": bundle["bootstrap_python_state"].get("qpos"),
            "qvel": bundle["bootstrap_python_state"].get("qvel"),
            "act": bundle["bootstrap_python_state"].get("act"),
            "ctrl": bundle["bootstrap_python_state"].get("ctrl"),
            "warmstart": bundle["bootstrap_python_state"].get("warmstart"),
            "elapsed_steps": 0,
            "task_id": bundle["manifest"]["task_id"],
            "success": False,
            "terminated": False,
            "truncated": False,
        },
        measure_state(env, observation=obs0, goal=goal, terminated=False, truncated=False, info={"success": False}),
        has_actions=False,
    )
    proof["bootstrap_compare"] = bootstrap_cmp
    restored_goal = exact_goal(env)
    goal_diff = float(np.max(np.abs(np.asarray(restored_goal) - goal))) if restored_goal.size == goal.size else float("inf")
    proof["restored_goal_max_abs"] = goal_diff
    if strict and (bootstrap_cmp["status"] != "PASS" or goal_diff > 0.0):
        raise RuntimeError(f"bootstrap restore failed: {bootstrap_cmp['failures']} goal_diff={goal_diff}")

    prefix = np.asarray(bundle["prefix_actions"])
    if prefix.size == 0:
        prefix = prefix.reshape(0, 5)
    obs = np.asarray(obs0, dtype=np.float64).copy()
    terminated = truncated = False
    last_info = {}
    for action in prefix:
        if terminated or truncated:
            raise RuntimeError("prefix continued after terminal; refusing")
        next_obs, reward, terminated, truncated, info = env.step(np.asarray(action))
        _count_step(step_counter)
        obs = np.asarray(next_obs, dtype=np.float64).copy()
        last_info = info
        if terminated or truncated:
            break
    decision_cmp = compare_states(
        {
            "observation": bundle["decision_observation"],
            "integration": bundle["decision_integration"],
            "goal_observation": goal,
            "qpos": bundle["decision_python_state"].get("qpos"),
            "qvel": bundle["decision_python_state"].get("qvel"),
            "act": bundle["decision_python_state"].get("act"),
            "ctrl": bundle["decision_python_state"].get("ctrl"),
            "warmstart": bundle["decision_python_state"].get("warmstart"),
            "elapsed_steps": bundle["manifest"].get("decision_elapsed_steps"),
            "task_id": bundle["manifest"]["task_id"],
            "success": bundle["manifest"].get("decision_success"),
            "terminated": False,
            "truncated": False,
        },
        measure_state(env, observation=obs, goal=goal, terminated=terminated, truncated=truncated, info=last_info),
        has_actions=False,
    )
    proof["decision_compare"] = decision_cmp
    proof["prefix_replay_terminated"] = bool(terminated or truncated)
    if strict and (terminated or truncated):
        raise RuntimeError("prefix replay terminated before decision")
    if strict and decision_cmp["status"] != "PASS":
        raise RuntimeError(f"prefix replay did not recover decision: {decision_cmp['failures']}")

    context = {
        "bundle": bundle,
        "mode": mode,
        "d3_target": np.asarray(bundle["d3_target_observation"], dtype=np.float64).copy(),
        "prefix_actions": prefix,
        "elapsed_steps": elapsed_steps(env),
    }
    return LoadedRoot(env=env, observation=obs, exact_goal=goal, context=context, proof=proof)