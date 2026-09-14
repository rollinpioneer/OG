
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from execution_aligned_rl.v3.root_bundle import write_root_bundle
from execution_aligned_rl.v3.serialization import encode_nested


class TaskStateJsonTests(unittest.TestCase):
    def test_encode_nested_task_state_is_jsonable(self):
        payload = {
            "task_id": 1,
            "cubes": [{"target_pos": np.zeros(3), "object_quat": np.ones(4)}],
        }
        encoded = encode_nested(payload)
        json.dumps(encoded)

    def test_write_root_bundle_accepts_ndarray_task_state(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw) / "root"
            bundle = {
                "initial_observation": np.zeros(37),
                "goal_observation": np.zeros(37),
                "bootstrap_integration": np.zeros(8),
                "prefix_actions": np.zeros((0, 5)),
                "decision_observation": np.zeros(37),
                "decision_integration": np.zeros(8),
                "d3_target_observation": np.zeros(37),
                "bootstrap_python_state": {"qpos": np.zeros(4)},
                "decision_python_state": {"qpos": np.zeros(4)},
                "bootstrap_observation_stage": {"cfrc_ext": np.zeros((2, 6))},
                "decision_observation_stage": {"cfrc_ext": np.zeros((2, 6))},
                "task_state": {"cubes": [{"target_pos": np.array([0.1, 0.2, 0.3])}]},
                "manifest": {"root_id": 690000, "task_id": 1},
            }
            write_root_bundle(directory, bundle)
            text = (directory / "task_state.json").read_text(encoding="utf-8")
            json.loads(text)
