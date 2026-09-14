"""Unique rollout_segment and compute_proxy. Subsequent observations come only from env.step."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from execution_aligned_rl.v3.contracts import PROTOCOL_V1
from execution_aligned_rl.v3.env_state import elapsed_steps, measure_state
from execution_aligned_rl.v3.hashing import sha256_array
from execution_aligned_rl.v3.policy import value_for


def compute_proxy(trace: dict, frozen_value, exact_goal, reward_contract: dict | None = None) -> float:
    contract = reward_contract or PROTOCOL_V1["reward_value_contract"]
    steps = list(trace.get("steps") or [])
    if not steps:
        raise ValueError("compute_proxy on empty trace must not be reported as 0; caller should use NOT_APPLICABLE")
    gamma = float(contract["discount"])
    total = 0.0
    terminated = truncated = False
    last_obs = None
    for idx, step in enumerate(steps):
        reward = float(step["reward"])
        r_train = reward - 1.0
        total += (gamma ** idx) * r_train
        terminated = bool(step["terminated"])
        truncated = bool(step["truncated"])
        last_obs = step["observation"]
        if terminated or truncated:
            if idx != len(steps) - 1:
                trace["continued_after_terminal"] = True
            break
    if terminated or truncated:
        tail = 0.0
    else:
        tail = float(value_for(frozen_value, np.asarray(last_obs), exact_goal)[0])
    proxy = total + (gamma ** len(steps)) * tail
    return float(proxy)


def rollout_segment(
    env,
    policy,
    observation,
    goal,
    steps: int,
    key_schedule: Sequence,
    *,
    frozen_value=None,
    exact_goal=None,
    mode: str = "closed_loop",
    actions: Sequence | None = None,
    identity: dict | None = None,
    step_counter: dict | None = None,
    probe_goal=None,
) -> dict:
    if steps < 0:
        raise ValueError("steps must be >= 0")
    obs = np.asarray(observation, dtype=np.float64).copy()
    goal = np.asarray(goal, dtype=np.float64).copy()
    exact = goal if exact_goal is None else np.asarray(exact_goal, dtype=np.float64).copy()
    policy_goal = goal if probe_goal is None else np.asarray(probe_goal, dtype=np.float64).copy()
    recorded = []
    continued_after_terminal = False
    terminated = truncated = False
    start_elapsed = elapsed_steps(env)

    if mode == "open_loop":
        if actions is None:
            raise ValueError("open_loop rollout requires actions")
        plan = [np.asarray(a, dtype=np.float64) for a in actions]
        n = min(steps, len(plan))
    else:
        n = steps
        plan = None

    for idx in range(n):
        if terminated or truncated:
            continued_after_terminal = True
            break
        if mode == "open_loop":
            action = np.asarray(plan[idx], dtype=np.float64).copy()
        else:
            key = key_schedule[idx]
            action = np.asarray(policy(obs, policy_goal, key), dtype=np.float64).copy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        if step_counter is not None:
            step_counter["external_control_steps"] = int(step_counter.get("external_control_steps", 0)) + 1
        next_obs = np.asarray(next_obs, dtype=np.float64).copy()
        measured = measure_state(
            env,
            observation=next_obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            goal=exact,
        )
        measured["step_proxy"] = None
        recorded.append(measured)
        obs = next_obs
        if terminated or truncated:
            break

    trace = {
        "identity": identity or {},
        "status": "EMPTY" if not recorded else "OK",
        "steps": recorded,
        "elapsed_steps_start": start_elapsed,
        "continued_after_terminal": continued_after_terminal,
        "key_schedule": [int(np.asarray(k).reshape(-1)[0]) if np.asarray(k).size == 1 else None for k in (key_schedule or [])],
        "goal_observation": exact,
        "goal_sha256": sha256_array(exact),
        "probe_goal_sha256": sha256_array(policy_goal),
        "mode": mode,
        "result_source": "ENV_EVALUATED",
        "training_eligible": False,
        "full_proxy": None,
    }
    if recorded and frozen_value is not None:
        proxy = compute_proxy(trace, frozen_value, exact)
        trace["full_proxy"] = proxy
        # attach the same full proxy to the last step for pairwise comparison convenience
        recorded[-1]["step_proxy"] = proxy
        for earlier in recorded[:-1]:
            earlier["step_proxy"] = proxy
    return trace