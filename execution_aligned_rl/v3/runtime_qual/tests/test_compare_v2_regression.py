
"""Regression tests built from the frozen S1 D0/D1 failure modes. No env.step."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from execution_aligned_rl.v3.compare import compare_states as compare_states_v1
from execution_aligned_rl.v3.compare import compare_traces as compare_traces_v1
from execution_aligned_rl.v3.runtime_qual.compare_v2 import compare_states_v2, compare_traces_v2
from execution_aligned_rl.v3.runtime_qual.protocol import DEFAULT_PATHS, THRESHOLDS


def _state(**kwargs):
    base = {
        "observation": np.zeros(37),
        "integration": np.zeros(8),
        "qpos": np.zeros(4),
        "qvel": np.zeros(4),
        "act": np.zeros(0),
        "ctrl": np.zeros(2),
        "warmstart": np.zeros(4),
        "goal_observation": np.zeros(37),
        "elapsed_steps": 0,
        "success": False,
        "task_id": 1,
    }
    base.update(kwargs)
    return base


def _step(**kwargs):
    row = {
        "observation": np.zeros(37),
        "action": np.zeros(5),
        "integration": np.zeros(8),
        "qpos": np.zeros(4),
        "qvel": np.zeros(4),
        "act": np.zeros(0),
        "ctrl": np.zeros(2),
        "warmstart": np.zeros(4),
        "reward": 0.0,
        "success": 0.0,
        "terminated": False,
        "truncated": False,
        "elapsed_steps": 1,
        "step_proxy": None,
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


class S1D0RegressionTests(unittest.TestCase):
    def test_s1_d0_false_vs_none_failed_v1_and_is_na_in_v2(self):
        left = _state(terminated=False, truncated=False)
        right = _state(terminated=None, truncated=None)
        v1 = compare_states_v1(left, right, has_actions=False)
        self.assertEqual(v1["status"], "FAIL")
        self.assertIn("terminated", v1["failures"])
        self.assertIn("truncated", v1["failures"])
        v2 = compare_states_v2(left, right, phase="D0")
        self.assertEqual(v2["status"], "PASS")
        self.assertEqual(v2["fields"]["terminated"]["status"], "NOT_APPLICABLE")
        self.assertEqual(v2["fields"]["truncated"]["status"], "NOT_APPLICABLE")
        self.assertEqual(v2["fields"]["observation"]["status"], "PASS")
        self.assertEqual(v2["fields"]["action"]["status"], "NOT_APPLICABLE")

    def test_s1_recorded_d0_json_is_only_terminal_missing(self):
        path = Path(DEFAULT_PATHS["s1_repo"]) / DEFAULT_PATHS["s1_experiment_rel"] / "engineering" / "workers" / "A" / "690000" / "primary" / "d0.json"
        if not path.exists():
            self.skipTest("S1 d0.json not mounted")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["failures"], ["terminated", "truncated"])
        self.assertEqual(payload["fields"]["observation"]["status"], "PASS")
        self.assertEqual(payload["fields"]["goal_observation"]["status"], "PASS")
        self.assertEqual(payload["fields"]["integration"]["status"], "PASS")


class S1D1RegressionTests(unittest.TestCase):
    def test_s1_d1_full_proxy_bookkeeping_failed_v1_and_is_na_in_v2(self):
        steps = [_step()]
        a = _trace(steps, full_proxy=0.25)
        b = _trace([_step()], full_proxy=0.01)
        v1 = compare_traces_v1(a, b)
        self.assertEqual(v1["status"], "FAIL")
        self.assertIn("FULL_PROXY", v1["reasons"])
        self.assertEqual(v1["steps"][0]["failures"], [])
        v2 = compare_traces_v2(a, b, phase="D1")
        self.assertEqual(v2["status"], "PASS")
        self.assertEqual(v2["full_proxy"]["status"], "NOT_APPLICABLE")
        self.assertEqual(v2["steps"][0]["proxy"]["status"], "NOT_APPLICABLE")
        self.assertEqual(v2["steps"][0]["action"]["status"], "PASS")

    def test_d1_still_fails_on_action_bias(self):
        biased = _step()
        biased["action"] = biased["action"] + 1e-3
        v2 = compare_traces_v2(_trace([_step()], full_proxy=0.2), _trace([biased], full_proxy=0.2), phase="D1")
        self.assertEqual(v2["status"], "FAIL")
        self.assertIn("action", v2["steps"][0]["failures"])

    def test_s1_recorded_d1_json_is_full_proxy_only(self):
        path = Path(DEFAULT_PATHS["s1_repo"]) / DEFAULT_PATHS["s1_experiment_rel"] / "engineering" / "compare_690000.json"
        if not path.exists():
            self.skipTest("S1 compare json not mounted")
        payload = json.loads(path.read_text(encoding="utf-8"))
        d1 = payload["comparisons"]["D1_A_vs_original_prefix_action"]
        self.assertEqual(d1["status"], "FAIL")
        self.assertEqual(d1["reasons"], ["FULL_PROXY"])
        self.assertEqual(d1["steps"][0]["failures"], [])


class FailClosedTests(unittest.TestCase):
    def test_empty_nan_length_identity_still_fail(self):
        self.assertEqual(compare_traces_v2(_trace([]), _trace([_step()]), phase="D3")["status"], "FAIL")
        nan_step = _step()
        nan_step["observation"][0] = np.nan
        self.assertEqual(compare_traces_v2(_trace([_step()]), _trace([nan_step]), phase="D3")["status"], "FAIL")
        self.assertEqual(compare_traces_v2(_trace([_step(), _step()]), _trace([_step()]), phase="D3")["status"], "FAIL")
        other = _trace([_step()])
        other["identity"]["root_id"] = 699999
        result = compare_traces_v2(_trace([_step()]), other, phase="D3")
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(r.startswith("IDENTITY") for r in result["reasons"]))

    def test_d2_still_compares_full_proxy(self):
        a = _trace([_step()], full_proxy=0.2)
        b = _trace([_step()], full_proxy=0.2 + 1e-3)
        result = compare_traces_v2(a, b, phase="D2")
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("FULL_PROXY", result["reasons"])

    def test_thresholds_unchanged(self):
        self.assertEqual(THRESHOLDS["action_max_abs"], 1e-7)
        self.assertEqual(THRESHOLDS["value_max_abs"], 1e-5)
        self.assertEqual(THRESHOLDS["rtol"], 0.0)
