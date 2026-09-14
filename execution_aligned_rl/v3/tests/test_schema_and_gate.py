
import unittest

from execution_aligned_rl.v3.contracts import PROTOCOL_V1, ENGINEERING_ROOTS, THRESHOLDS


REQUIRED_PROTOCOL_KEYS = (
    "protocol_id",
    "schema_version",
    "status_machine",
    "scope",
    "assets",
    "restore",
    "unique_executor",
    "thresholds",
    "engineering",
    "budget",
    "reward_value_contract",
)


class SchemaTests(unittest.TestCase):
    def test_protocol_complete(self):
        for key in REQUIRED_PROTOCOL_KEYS:
            self.assertIn(key, PROTOCOL_V1)
        self.assertFalse(PROTOCOL_V1["status_machine"]["finalizer_may_rewrite_protocol"])
        self.assertFalse(PROTOCOL_V1["status_machine"]["phase_c_unlocked"])
        self.assertEqual(PROTOCOL_V1["restore"]["primary_mode"], "BOOTSTRAP_PREFIX_REPLAY")
        self.assertEqual(float(THRESHOLDS["rtol"]), 0.0)
        self.assertEqual(len(ENGINEERING_ROOTS), 15)
        ids = [row["root_id"] for row in ENGINEERING_ROOTS]
        self.assertEqual(ids, list(range(690000, 690015)))
        self.assertEqual(PROTOCOL_V1["engineering"]["external_control_step_budget"], 30000)
        self.assertFalse(PROTOCOL_V1["scope"]["training_eligible"])

    def test_hold_defaults(self):
        self.assertEqual(PROTOCOL_V1["status_machine"]["default_on_missing_field"], "HOLD")
        self.assertIn("S2_formal_root_pool", PROTOCOL_V1["scope"]["forbidden"])
