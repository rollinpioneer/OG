"""EA-V3 S3 postmortem constants."""

from __future__ import annotations

from pathlib import Path

PROTOCOL_ID = "ea_v3_s3_postmortem_v1"
EXPECTED_IDENTITY = "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8"
S3_COMMIT = "3882a7d7e8028f3416af17e6827a2e7c13c3fc0f"
GAMMA = 0.99
EPSILON = 0.5495205402374268
M = 5
HORIZONS = (1, 5, 10, 20, 40, 80)
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 350301
J5_ATOL = 1e-5
DEEP_TASK_COUNTS = {1: 5, 2: 6, 3: 8, 4: 8, 5: 8}
VALUE_QUERY_BUDGET = 10000
ACTOR_QUERY_BUDGET = 512

S3_EXP = Path("/home/__compress_data/xushijie/OG_ea_v3_s3_mechanism/experiments/execution_aligned/ea_v3_s3_mechanism_v1")
S3_TRACES = S3_EXP / "traces"
S2_EXP = Path("/home/__compress_data/xushijie/OG_ea_v3_s2_pool/experiments/execution_aligned/ea_v3_s2_root_candidate_pool_v1")
S2_STORE = Path("/home/__compress_data/xushijie/ea_v3_root_store/ea_v3_s2_root_candidate_pool_v1")
S2_PROTOCOL_ID = "ea_v3_s2_root_candidate_pool_v1"
DEFAULT_REPO = Path("/home/__compress_data/xushijie/OG_ea_v3_s3_postmortem")
EXP_REL = Path("experiments/execution_aligned/ea_v3_s3_postmortem_v1")


def experiment_dir(repo: Path | None = None) -> Path:
    return Path(repo or DEFAULT_REPO) / EXP_REL
