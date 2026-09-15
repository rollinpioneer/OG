
"Episode-safe offline dataset helpers. Never reads S3 labels."
from __future__ import annotations
from pathlib import Path
import numpy as np
from execution_aligned_rl.data.audit_assets import episode_bounds, sha256_file
from execution_aligned_rl.v4.outcome_pilot.const import K, TRAIN_FILE, VAL_FILE
from execution_aligned_rl.v4.outcome_pilot.success import success_from_obs_goal

def load_split(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    obs = np.asarray(z['observations'])
    act = np.asarray(z['actions'])
    term = np.asarray(z['terminals'])
    bounds = episode_bounds(term)
    n = len(obs)
    ep_id = np.empty(n, dtype=np.int32)
    ep_end = np.empty(n, dtype=np.int32)
    for i,(s,e) in enumerate(bounds):
        ep_id[s:e] = i
        ep_end[s:e] = e-1
    is_last = np.zeros(n, dtype=bool)
    is_last[np.array([e-1 for s,e in bounds])] = True
    # last index of each episode is a transition? terminals at last row of episode
    return {
        'obs': obs, 'act': act, 'term': term, 'bounds': bounds,
        'ep_id': ep_id, 'ep_end': ep_end, 'n': n, 'n_ep': len(bounds),
        'is_last': is_last,
    }

def legal_k_starts(terminals: np.ndarray, k: int = K) -> np.ndarray:
    bounds = episode_bounds(terminals)
    starts = []
    for s,e in bounds:
        n = e-s
        for i in range(max(0, n-k)):
            starts.append(s+i)
    return np.asarray(starts, dtype=np.int64)

def classify_ends(ds: dict) -> dict:
    obs, last = ds['obs'], ds['is_last']
    # for last index, success vs a dummy? episode end success needs a goal.
    # Data-collection success isn't stored; we classify later per sampled g.
    n_last = int(last.sum())
    return {'n_transitions': ds['n'], 'n_episodes': ds['n_ep'], 'n_last': n_last}

def remaining_in_episode(ds, idx: np.ndarray) -> np.ndarray:
    return ds['ep_end'][idx] - idx
