"""Frozen GCIQL actor/value loading. Inputs are raw observations; temperature is 0."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np

from execution_aligned_rl.v3.contracts import DEFAULT_PATHS
from execution_aligned_rl.v3.hashing import stable_uint32


def load_train_arrays(dataset_dir: str, dataset_id: str = "cube-double-play-v0"):
    import ogbench

    path = Path(dataset_dir) / f"{dataset_id}.npz"
    payload = np.load(path, allow_pickle=False)
    train = {key: payload[key] for key in payload.files}
    return train


def load_agent(
    official_source: str | None = None,
    checkpoint_dir: str | None = None,
    dataset_dir: str | None = None,
    epoch: int = 1_000_000,
):
    official_source = official_source or DEFAULT_PATHS["official_source"]
    checkpoint_dir = checkpoint_dir or DEFAULT_PATHS["checkpoint_dir"]
    dataset_dir = dataset_dir or DEFAULT_PATHS["dataset_dir"]
    impls = str(Path(official_source) / "impls")
    if impls not in sys.path:
        sys.path.insert(0, impls)
    from agents.gciql import GCIQLAgent, get_config
    from utils.datasets import Dataset, GCDataset
    from utils.flax_utils import restore_agent

    train = load_train_arrays(dataset_dir)
    config = get_config()
    config.encoder = None
    config.frame_stack = None
    config.alpha = 1.0
    config.actor_p_randomgoal = 0.0
    config.actor_p_trajgoal = 1.0
    config.actor_p_curgoal = 0.0
    gc_dataset = GCDataset(Dataset.create(**train), config)
    batch = gc_dataset.sample(1)
    agent = GCIQLAgent.create(0, batch["observations"], batch["actions"], config)
    agent = restore_agent(agent, checkpoint_dir, epoch)
    return agent, config, train


def prng_key(label: str):
    return jax.random.PRNGKey(stable_uint32(label) & 0xFFFFFFFF)


def action_for(agent, observation, goal, key):
    obs = np.asarray(observation)
    gol = np.asarray(goal)
    return np.asarray(agent.sample_actions(obs, goals=gol, seed=key, temperature=0.0), dtype=np.float64).copy()


def value_for(agent, states, goal):
    states = np.asarray(states)
    if states.ndim == 1:
        states = states[None]
    goals = np.broadcast_to(np.asarray(goal), states.shape)
    return np.asarray(agent.network.select("value")(states, goals), dtype=np.float64)


class FrozenPolicy:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, observation, goal, key):
        return action_for(self.agent, observation, goal, key)