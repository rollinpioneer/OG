
"""CPU S1 qualification protocol. Does not rewrite S1 protocol_v1 or Q0/Q2 artifacts."""

from __future__ import annotations

PROTOCOL_ID = "ea_v3_cpu_s1_qualification_v1"
PROTOCOL_NAME = "EA-V3 CPU S1 Qualification v1"
BASELINE_COMMIT = "7b9698cbae9807d9cbb61b44cbcbf872a24331b4"
CHECKPOINT_SHA256 = "12f7bd7ed9f909039dd6e3e3c44cd226bf765cf7d959396001e469dbbcbc3d87"
OGBENCH_COMMIT = "1d4140997f60c52c6fb0702ec100dc988b18c548"
TRAIN_SHA256 = "a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9"
Q0_COMMIT = "78ed06e720bdf3634b564bf52b0f86f97d3cae8e"
S1_GPU_COMMIT = "6fea6044d3ad12b9d77b75702d881c2eedd6ce64"

DEFAULT_PATHS = {
    "repo": "/home/__compress_data/xushijie/OG_ea_v3_cpu_s1_qualification",
    "q0_repo": "/home/__compress_data/xushijie/OG_ea_v3_runtime_qualification",
    "q0_experiment_rel": "experiments/execution_aligned/ea_v3_runtime_qualification_v1",
    "q2_repo": "/home/__compress_data/xushijie/OG_ea_v3_cpu_runtime_adoption",
    "s1_repo": "/home/__compress_data/xushijie/OG_ea_v3_materialized_roots",
    "s1_experiment_rel": "experiments/execution_aligned/ea_v3_materialized_roots_v1",
    "dataset_dir": "/home/__compress_data/xushijie/ea_v2_cube_data",
    "official_source": "/home/__compress_data/xushijie/og_runs/ea_v2_v1/src/ogbench",
    "checkpoint_dir": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0",
    "checkpoint_file": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0/params_1000000.pkl",
    "experiment_rel": "experiments/execution_aligned/ea_v3_cpu_s1_qualification_v1",
    "root_store": "/home/__compress_data/xushijie/ea_v3_root_store/ea_v3_cpu_s1_qualification_v1",
    "compare_v2": "execution_aligned_rl/v3/runtime_qual/compare_v2.py",
}

CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "JAX_ENABLE_X64": "0",
}

# Frozen from GPU S1 engineering_root_manifest.json
REGRESSION_D3_INDEX = {
    690000: 169551,
    690001: 503979,
    690002: 636637,
    690003: 746779,
    690004: 134916,
    690005: 188033,
    690006: 500535,
    690007: 793569,
    690008: 701708,
    690009: 951344,
    690010: 561938,
    690011: 209653,
    690012: 347568,
    690013: 390915,
    690014: 629870,
}

STEPS = (0, 125, 250)
BUDGET = 60000

PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "scope": ["R0", "S1-CPU"],
    "forbidden": ["S2", "S3", "Phase_C", "P_1", "P_MH", "EXEC", "GPU_CURRENT", "train", "modify_checkpoint"],
    "canonical_runtime": "CPU_SINGLE_THREAD",
    "restore": "BOOTSTRAP_PREFIX_REPLAY",
    "comparator": "comparator_contract_v2",
    "cpu_env": CPU_ENV,
    "no_new_xla_flags": True,
    "budget_external_control_steps": BUDGET,
    "regression": {
        "root_ids": list(range(710000, 710015)),
        "reset_seeds": list(range(690000, 690015)),
        "d3_target_index_source": "frozen_gpu_s1_manifest",
    },
    "holdout": {
        "root_ids": list(range(711000, 711015)),
        "reset_seeds": list(range(691000, 691015)),
        "d3_target_index_formula": "stable_int('ea3-cpu-s1-v1:d3-endpoint:{root_id}') % n_train",
    },
    "gates": {
        "regression_legal_min": 10,
        "holdout_legal_min": 10,
        "all_five_tasks_each_set": True,
        "holdout_requires_legal_125_and_250": True,
        "negative_tests": 9,
        "aba_sets": ["regression", "holdout"],
    },
}
