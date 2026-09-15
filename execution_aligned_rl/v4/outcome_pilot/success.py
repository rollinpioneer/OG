
" Official Cube success from public observation/goal. No env."
from __future__ import annotations
import numpy as np

XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float64)
XYZ_SCALER = 10.0
TOL = 0.04
CUBE0 = slice(19, 22)
CUBE1 = slice(28, 31)

def cube_xyz_from_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float64)
    p0 = obs[..., 19:22] / XYZ_SCALER + XYZ_CENTER
    p1 = obs[..., 28:31] / XYZ_SCALER + XYZ_CENTER
    return np.stack([p0, p1], axis=-2)

def success_from_obs_goal(obs, goal) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    po = cube_xyz_from_obs(obs)
    pg = cube_xyz_from_obs(goal)
    dist = np.linalg.norm(po - pg, axis=-1)
    return np.all(dist <= TOL, axis=-1)
