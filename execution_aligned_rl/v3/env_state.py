"""Capture/restore of Cube integration, Python, wrapper, RNG, and observation-stage fields."""

from __future__ import annotations

import copy
import random
from typing import Any

import mujoco
import numpy as np

from execution_aligned_rl.v3.serialization import decode_nested, decode_rng_state, encode_nested, encode_rng_state

INTEGRATION_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

PYTHON_ATTRS = (
    "cur_task_id",
    "cur_task_info",
    "_cur_goal_ob",
    "_success",
    "_reset_next_step",
    "_prev_qpos",
    "_prev_qvel",
    "_prev_ob_info",
    "_render_goal",
    "_target_block",
    "_mode",
    "_success_timing",
    "_terminate_at_goal",
    "_use_oracle_rep",
    "_reward_task_id",
    "_ob_type",
)

OBS_STAGE_FIELDS = (
    "cfrc_ext",
    "site_xpos",
    "site_xmat",
    "xpos",
    "xmat",
    "xquat",
    "mocap_pos",
    "mocap_quat",
    "sensordata",
)


def wrapper_chain(env) -> list:
    chain = []
    current = env
    while True:
        chain.append(current)
        if not hasattr(current, "env"):
            break
        current = current.env
    return chain


def base_env(env):
    return env.unwrapped


def integration_size(env) -> int:
    model = base_env(env).model
    return int(mujoco.mj_stateSize(model, INTEGRATION_SPEC))


def capture_integration(env) -> np.ndarray:
    model = base_env(env).model
    data = base_env(env).data
    size = mujoco.mj_stateSize(model, INTEGRATION_SPEC)
    out = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(model, data, out, INTEGRATION_SPEC)
    return out.copy()


def restore_integration(env, integration: np.ndarray) -> None:
    model = base_env(env).model
    data = base_env(env).data
    mujoco.mj_setState(model, data, np.asarray(integration, dtype=np.float64), INTEGRATION_SPEC)


def capture_observation_stage(env) -> dict[str, np.ndarray]:
    data = base_env(env).data
    out = {}
    for name in OBS_STAGE_FIELDS:
        value = getattr(data, name, None)
        if value is None:
            continue
        out[name] = np.array(value, copy=True)
    return out


def restore_observation_stage(env, payload: dict[str, np.ndarray]) -> None:
    data = base_env(env).data
    for name, array in payload.items():
        target = getattr(data, name, None)
        if target is None:
            continue
        np.copyto(target, np.asarray(array))


def capture_python_state(env) -> dict[str, Any]:
    base = base_env(env)
    payload: dict[str, Any] = {}
    for name in PYTHON_ATTRS:
        if hasattr(base, name):
            payload[name] = copy.deepcopy(getattr(base, name))
        else:
            payload[name] = {"__missing__": True}
    payload["elapsed_steps_by_wrapper"] = [copy.deepcopy(getattr(wrapper, "_elapsed_steps", None)) for wrapper in wrapper_chain(env)]
    payload["env_np_random"] = encode_rng_state(copy.deepcopy(base.np_random.bit_generator.state))
    payload["action_space_np_random"] = encode_rng_state(copy.deepcopy(env.action_space.np_random.bit_generator.state))
    payload["global_numpy_rng"] = encode_rng_state(copy.deepcopy(np.random.get_state()))
    payload["python_random_state"] = encode_nested(random.getstate())
    payload["qpos"] = base.data.qpos.copy()
    payload["qvel"] = base.data.qvel.copy()
    payload["act"] = None if base.data.act is None else base.data.act.copy()
    payload["ctrl"] = base.data.ctrl.copy()
    payload["warmstart"] = base.data.qacc_warmstart.copy()
    payload["time"] = float(base.data.time)
    payload["n_steps"] = int(getattr(base, "_n_steps", 1))
    if hasattr(base, "_cur_goal_ob") and base._cur_goal_ob is not None:
        payload["_cur_goal_ob"] = np.asarray(base._cur_goal_ob).copy()
    if hasattr(base, "cur_task_info") and base.cur_task_info is not None:
        payload["cur_task_info"] = encode_nested(base.cur_task_info)
    return payload


def restore_python_state(env, payload: dict[str, Any]) -> None:
    base = base_env(env)
    for name in PYTHON_ATTRS:
        if name not in payload:
            continue
        value = payload[name]
        if isinstance(value, dict) and value.get("__missing__"):
            continue
        if name == "cur_task_info":
            setattr(base, name, decode_nested(value))
            continue
        if name == "_cur_goal_ob" and value is not None:
            setattr(base, name, np.asarray(value).copy())
            continue
        if hasattr(base, name):
            setattr(base, name, copy.deepcopy(value) if not isinstance(value, np.ndarray) else np.asarray(value).copy())
    for wrapper, elapsed in zip(wrapper_chain(env), payload.get("elapsed_steps_by_wrapper") or []):
        if elapsed is not None and hasattr(wrapper, "_elapsed_steps"):
            wrapper._elapsed_steps = copy.deepcopy(elapsed)
    if "env_np_random" in payload:
        base.np_random.bit_generator.state = decode_rng_state(payload["env_np_random"])
    if "action_space_np_random" in payload:
        env.action_space.np_random.bit_generator.state = decode_rng_state(payload["action_space_np_random"])
    if "global_numpy_rng" in payload:
        np.random.set_state(decode_rng_state(payload["global_numpy_rng"]))
    if "python_random_state" in payload:
        st = decode_nested(payload["python_random_state"])
        if isinstance(st, list):
            st = tuple(tuple(x) if isinstance(x, list) else x for x in st)
        random.setstate(st)


def public_observation(env) -> np.ndarray:
    return np.asarray(base_env(env).compute_observation(), dtype=np.float64).copy()


def exact_goal(env) -> np.ndarray:
    goal = getattr(base_env(env), "_cur_goal_ob", None)
    if goal is None:
        raise RuntimeError("environment has no _cur_goal_ob")
    return np.asarray(goal, dtype=np.float64).copy()


def physical_task_targets(env) -> dict[str, Any]:
    base = base_env(env)
    targets = []
    n = int(getattr(base, "_num_cubes", 0))
    for i in range(n):
        mocap_id = int(base._cube_target_mocap_ids[i])
        joint = base.data.joint(f"object_joint_{i}")
        targets.append(
            {
                "cube_index": i,
                "mocap_id": mocap_id,
                "target_pos": base.data.mocap_pos[mocap_id].copy(),
                "target_quat": base.data.mocap_quat[mocap_id].copy(),
                "object_pos": np.asarray(joint.qpos[:3]).copy(),
                "object_quat": np.asarray(joint.qpos[3:]).copy(),
            }
        )
    return {
        "task_id": getattr(base, "cur_task_id", None),
        "task_info": encode_nested(getattr(base, "cur_task_info", None)),
        "cubes": targets,
        "success": bool(getattr(base, "_success", False)),
    }


def elapsed_steps(env):
    for wrapper in wrapper_chain(env):
        if hasattr(wrapper, "_elapsed_steps") and getattr(wrapper, "_elapsed_steps") is not None:
            return int(wrapper._elapsed_steps)
    return None


def measure_state(env, *, observation=None, action=None, reward=None, terminated=None, truncated=None, info=None, goal=None, proxy=None) -> dict:
    base = base_env(env)
    obs = np.asarray(observation if observation is not None else public_observation(env), dtype=np.float64).copy()
    info = info or {}
    return {
        "observation": obs,
        "action": None if action is None else np.asarray(action, dtype=np.float64).copy(),
        "reward": None if reward is None else float(reward),
        "success": info.get("success", getattr(base, "_success", None)),
        "terminated": None if terminated is None else bool(terminated),
        "truncated": None if truncated is None else bool(truncated),
        "integration": capture_integration(env),
        "qpos": base.data.qpos.copy(),
        "qvel": base.data.qvel.copy(),
        "act": None if base.data.act is None else base.data.act.copy(),
        "ctrl": base.data.ctrl.copy(),
        "warmstart": base.data.qacc_warmstart.copy(),
        "time": float(base.data.time),
        "elapsed_steps": elapsed_steps(env),
        "task_id": getattr(base, "cur_task_id", None),
        "goal_observation": None if goal is None else np.asarray(goal, dtype=np.float64).copy(),
        "proxy": proxy,
        "python_state": capture_python_state(env),
        "observation_stage": capture_observation_stage(env),
    }


def restore_direct(env, integration, python_state, observation_stage) -> np.ndarray:
    restore_integration(env, integration)
    restore_python_state(env, python_state)
    restore_observation_stage(env, observation_stage)
    return public_observation(env)


def perturb_state(env) -> None:
    base = base_env(env)
    if base.data.qpos.size:
        base.data.qpos[0] = float(base.data.qpos[0]) + 0.17
    if base.data.qvel.size:
        base.data.qvel[0] = float(base.data.qvel[0]) + 0.09
    mujoco.mj_forward(base.model, base.data)