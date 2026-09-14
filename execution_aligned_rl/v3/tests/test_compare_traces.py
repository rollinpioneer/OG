
import unittest

import numpy as np

from execution_aligned_rl.v3.compare import compare_traces, compare_states


def _step(obs=None, act=None, **kwargs):
    obs = np.zeros(37, dtype=np.float64) if obs is None else np.asarray(obs, dtype=np.float64)
    act = np.zeros(5, dtype=np.float64) if act is None else np.asarray(act, dtype=np.float64)
    row = {
        "observation": obs.copy(),
        "action": act.copy(),
        "integration": np.zeros(8, dtype=np.float64),
        "qpos": np.zeros(4),
        "qvel": np.zeros(4),
        "act": np.zeros(2),
        "ctrl": np.zeros(2),
        "warmstart": np.zeros(4),
        "reward": 0.0,
        "success": 0.0,
        "terminated": False,
        "truncated": False,
        "elapsed_steps": 1,
        "step_proxy": 0.1,
    }
    row.update(kwargs)
    return row


def _trace(steps, **kwargs):
    payload = {
        "identity": {"root_id": 690000, "task_id": 1, "protocol_id": "ea_v3_materialized_roots_v1"},
        "steps": steps,
        "full_proxy": 0.1,
        "elapsed_steps_start": 0,
        "goal_observation": np.zeros(37),
        "continued_after_terminal": False,
        "status": "OK",
    }
    payload.update(kwargs)
    return payload


class CompareTests(unittest.TestCase):
    def test_identical_traces_pass_and_first_divergent_is_none(self):
        steps = [_step(elapsed_steps=i + 1) for i in range(3)]
        result = compare_traces(_trace(steps), _trace([_step(elapsed_steps=i + 1) for i in range(3)]))
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["first_divergent_step"])

    def test_empty_trace_fails(self):
        result = compare_traces(_trace([]), _trace([_step()]))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("EMPTY_STEPS", result["reasons"])
        self.assertIsNotNone(result["first_divergent_step"])

    def test_length_mismatch_fails(self):
        result = compare_traces(_trace([_step(), _step()]), _trace([_step()]))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("LENGTH_MISMATCH", result["reasons"])

    def test_nan_fails(self):
        bad = _step()
        bad["observation"][3] = np.nan
        result = compare_traces(_trace([_step()]), _trace([bad]))
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNotNone(result["first_divergent_step"])

    def test_action_bias_fails(self):
        biased = _step()
        biased["action"] = biased["action"] + 1e-3
        result = compare_traces(_trace([_step()]), _trace([biased]))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["first_divergent_step"], 0)

    def test_goal_identity_fails(self):
        other = _trace([_step()])
        other["goal_observation"] = np.ones(37)
        result = compare_traces(_trace([_step()]), other)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("GOAL_ENCODING", result["reasons"])

    def test_elapsed_mismatch_fails(self):
        other = _trace([_step()], elapsed_steps_start=1)
        result = compare_traces(_trace([_step()]), other)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("ELAPSED_START", result["reasons"])

    def test_post_terminal_step_fails(self):
        other = _trace([_step()], continued_after_terminal=True)
        result = compare_traces(_trace([_step()]), other)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("TERMINATION_PROTOCOL", result["reasons"])

    def test_d0_proxy_not_applicable(self):
        state = {
            "observation": np.zeros(37),
            "integration": np.zeros(4),
            "qpos": np.zeros(4),
            "qvel": np.zeros(4),
            "act": np.zeros(2),
            "ctrl": np.zeros(2),
            "warmstart": np.zeros(4),
            "goal_observation": np.zeros(37),
            "elapsed_steps": 0,
            "success": False,
            "terminated": False,
            "truncated": False,
            "task_id": 1,
        }
        result = compare_states(state, state, has_actions=False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["fields"]["proxy"]["status"], "NOT_APPLICABLE")
        self.assertEqual(result["fields"]["action"]["status"], "NOT_APPLICABLE")

    def test_empty_act_arrays_are_equal(self):
        state = {
            "observation": np.zeros(37),
            "integration": np.zeros(4),
            "qpos": np.zeros(4),
            "qvel": np.zeros(4),
            "act": np.zeros(0),
            "ctrl": np.zeros(2),
            "warmstart": np.zeros(4),
            "goal_observation": np.zeros(37),
            "elapsed_steps": 0,
            "success": False,
            "terminated": False,
            "truncated": False,
            "task_id": 1,
        }
        result = compare_states(state, state, has_actions=False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["fields"]["act"]["status"], "PASS")
