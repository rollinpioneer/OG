"""Feature packing for finite-budget critics. No S3 labels."""
from __future__ import annotations

import numpy as np

from execution_aligned_rl.v4.outcome_pilot.const import NORM_FILE


def load_norm():
    z = np.load(NORM_FILE)
    mean = np.asarray(z["observation_mean"], dtype=np.float32)
    std = np.asarray(z["observation_std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    amean = np.asarray(z["action_mean"], dtype=np.float32)
    astd = np.asarray(z["action_std"], dtype=np.float32)
    astd = np.where(astd == 0, 1.0, astd)
    return mean, std, amean, astd


def nrm_obs(x, mean, std):
    return (np.asarray(x, dtype=np.float32) - mean) / std


def nrm_act(a, amean, astd):
    return (np.asarray(a, dtype=np.float32) - amean) / astd


def pack_features(o, a, z, g, j, R, method, mean, std, amean, astd):
    o_n = nrm_obs(o, mean, std)
    a_n = nrm_act(a, amean, astd)
    g_n = nrm_obs(g, mean, std)
    j_n = (np.asarray(j, dtype=np.float32).reshape(-1, 1) / 5.0)
    r_n = (np.asarray(R, dtype=np.float32).reshape(-1, 1) / 500.0)
    if method == "B2":
        return np.concatenate([o_n, a_n, g_n, r_n], axis=-1)
    z_n = nrm_obs(z, mean, std)
    if method == "B3":
        return np.concatenate([o_n, a_n, z_n, g_n, j_n], axis=-1)
    return np.concatenate([o_n, a_n, z_n, g_n, j_n, r_n], axis=-1)


def feature_dim(method: str) -> int:
    if method == "B2":
        return 37 + 5 + 37 + 1
    if method == "B3":
        return 37 + 5 + 37 + 37 + 1
    return 37 + 5 + 37 + 37 + 1 + 1
