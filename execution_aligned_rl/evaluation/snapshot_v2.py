"""Shared MuJoCo snapshot_v2 capture/restore interface for evaluation branches."""

from __future__ import annotations

import copy
import numpy as np
import mujoco

from execution_aligned_rl.data.audit_assets import capture_snapshot, wrapper_chain

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def capture_snapshot_v2(env) -> dict:
    base = env.unwrapped
    size = mujoco.mj_stateSize(base.model, SPEC)
    integration = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(base.model, base.data, integration, SPEC)
    return {"integration_state": integration.copy(), "python_state": capture_snapshot(env)}


def restore_snapshot_v2(env, snapshot: dict) -> None:
    base = env.unwrapped
    mujoco.mj_setState(base.model, base.data, snapshot["integration_state"], SPEC)
    py = snapshot["python_state"]
    base.np_random.bit_generator.state = copy.deepcopy(py["rng"])
    np.random.set_state(copy.deepcopy(py["global_numpy_rng"]))
    base.action_space.np_random.bit_generator.state = copy.deepcopy(py["action_space_rng"])
    for name in ("cur_task_id", "cur_task_info", "cur_goal_xy"):
        if hasattr(base, name):
            setattr(base, name, copy.deepcopy(py[name]))
    for wrapper, elapsed in zip(wrapper_chain(env), py["elapsed_steps"]):
        if elapsed is not None:
            wrapper._elapsed_steps = copy.deepcopy(elapsed)
    mujoco.mj_forward(base.model, base.data)
