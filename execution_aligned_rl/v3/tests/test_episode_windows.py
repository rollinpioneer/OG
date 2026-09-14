
import unittest

import numpy as np

from execution_aligned_rl.data.audit_assets import deterministic_split, episode_bounds


class EpisodeWindowTests(unittest.TestCase):
    def test_windows_do_not_cross_episode_boundaries(self):
        terminals = np.array([0, 0, 1, 0, 0, 1, 1], dtype=np.float32)
        bounds = episode_bounds(terminals)
        self.assertEqual(bounds, [(0, 3), (3, 6), (6, 7)])
        self.assertEqual(sum(end - start for start, end in bounds), len(terminals))
        horizon = 2
        legal = []
        for start, end in bounds:
            legal.extend(range(start, end - horizon))
        self.assertNotIn(2, legal)
        self.assertNotIn(5, legal)

    def test_split_is_episode_then_window(self):
        terminals = np.zeros(20, dtype=np.float32)
        terminals[[4, 9, 14, 19]] = 1
        bounds = episode_bounds(terminals)
        self.assertEqual(len(bounds), 4)
        split = deterministic_split("cube-double-play-v0", bounds)
        ids = [idx for part in split.values() for idx in part["episode_ids"]]
        self.assertEqual(sorted(ids), [0, 1, 2, 3])
        self.assertEqual(len(ids), len(set(ids)))
