"""EA-V3 S3 mechanism protocol helpers. Frozen JSON files are the source of truth."""

from __future__ import annotations

import json
from pathlib import Path

from execution_aligned_rl.v3.serialization import load_json

PROTOCOL_ID = "ea_v3_s3_mechanism_v1"
EXPECTED_IDENTITY = "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8"
S2_COMMIT = "eba81e0ac662ecd93a29814d1b4c7e611465fa2b"
S2_PROTOCOL_FREEZE = "3d9a4efd5d01baf9508ce6042e59f403ab339fdf"
S2_PROTOCOL_ID = "ea_v3_s2_root_candidate_pool_v1"
BUDGET = 800000
M = 5
K = 20
N_CANDIDATES = 8
HORIZON = 500
GAMMA = 0.99
VALUE_TOLERANCE = 0.5495205402374268
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 350301

DEFAULT_PATHS = {
    "repo": "/home/__compress_data/xushijie/OG_ea_v3_s3_mechanism",
    "s2_repo": "/home/__compress_data/xushijie/OG_ea_v3_s2_pool",
    "s2_experiment_rel": "experiments/execution_aligned/ea_v3_s2_root_candidate_pool_v1",
    "s2_root_store": "/home/__compress_data/xushijie/ea_v3_root_store/ea_v3_s2_root_candidate_pool_v1",
    "q0_repo": "/home/__compress_data/xushijie/OG_ea_v3_runtime_qualification",
    "q0_experiment_rel": "experiments/execution_aligned/ea_v3_runtime_qualification_v1",
    "dataset_dir": "/home/__compress_data/xushijie/ea_v2_cube_data",
    "train_file": "/home/__compress_data/xushijie/ea_v2_cube_data/cube-double-play-v0.npz",
    "val_file": "/home/__compress_data/xushijie/ea_v2_cube_data/cube-double-play-v0-val.npz",
    "official_source": "/home/__compress_data/xushijie/og_runs/ea_v2_v1/src/ogbench",
    "checkpoint_dir": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0",
    "checkpoint_file": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0/params_1000000.pkl",
    "normalizer_s2": "experiments/execution_aligned/ea_v2_cube_mechanism_v1/manifests/normalizer_train_only.npz",
    "experiment_rel": "experiments/execution_aligned/ea_v3_s3_mechanism_v1",
}

EXPECTED_SHA = {
    "checkpoint": "12f7bd7ed9f909039dd6e3e3c44cd226bf765cf7d959396001e469dbbcbc3d87",
    "normalizer": "d4f912c0e458a031114cce0c084157780e3a527d26e21e4e1b7aa0745dd2c0c5",
    "ogbench_commit": "1d4140997f60c52c6fb0702ec100dc988b18c548",
    "train": "a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9",
    "validation": "b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e",
    "s2_zip": "eca9c4a922752563f9068ab79009574ea4591709430bf68922bc92defef1337d",
    "candidates_npz": "23ab9c91674f5c08d269452dc8a13589b8fb99de2ffb3af93dd8b52c2a977028",
}

SENTINELS = (720000, 720016, 720032, 720048, 720064)

CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "JAX_ENABLE_X64": "0",
}


def experiment_dir(repo: Path | None = None) -> Path:
    repo = Path(repo or DEFAULT_PATHS["repo"])
    return repo / DEFAULT_PATHS["experiment_rel"]


def s2_experiment() -> Path:
    return Path(DEFAULT_PATHS["s2_repo"]) / DEFAULT_PATHS["s2_experiment_rel"]


def load_frozen_protocol(exp: Path | None = None) -> dict:
    exp = exp or experiment_dir()
    return {
        "s3_protocol": load_json(exp / "protocol" / "s3_protocol.json"),
        "execution_matrix": load_json(exp / "protocol" / "execution_matrix.json"),
        "analysis_plan": load_json(exp / "protocol" / "analysis_plan.json"),
        "status_priority": load_json(exp / "protocol" / "status_priority.json"),
        "s2_deviation_register": load_json(exp / "protocol" / "s2_deviation_register.json"),
    }


def candidate_key_label(root_id: int, candidate_id: int, step: int) -> str:
    return f"ea35:candidate:{int(root_id)}:{int(candidate_id)}:{int(step)}"


def tail_key_label(root_id: int, candidate_id: int, step: int) -> str:
    return f"ea35:tail:{int(root_id)}:{int(candidate_id)}:{int(step)}"


def direct_key_label(root_id: int, step: int) -> str:
    return f"ea35:direct:{int(root_id)}:{int(step)}"


def sentinel_key_label(root_id: int, step: int) -> str:
    return f"ea35:sentinel:{int(root_id)}:{int(step)}"


def _json_default(value):
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unserializable type: {type(value)!r}")


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
