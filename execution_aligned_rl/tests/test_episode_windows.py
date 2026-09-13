import numpy as np

from execution_aligned_rl.data.audit_assets import deterministic_split, episode_bounds


def test_episode_bounds_and_split_do_not_overlap():
    terminals = np.array([0, 0, 1, 0, 1, 1], dtype=np.float32)
    bounds = episode_bounds(terminals)
    assert bounds == [(0, 3), (3, 5), (5, 6)]
    split = deterministic_split("test", bounds)
    ids = [idx for value in split.values() for idx in value["episode_ids"]]
    assert sorted(ids) == [0, 1, 2]
    assert len(ids) == len(set(ids))

