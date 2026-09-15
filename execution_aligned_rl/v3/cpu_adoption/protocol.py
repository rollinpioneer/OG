
"""Frozen Q2 CPU runtime adoption protocol. Does not modify Q0/Q1 or S1 artifacts."""

from __future__ import annotations

PROTOCOL_ID = "ea_v3_cpu_runtime_adoption_v1"
PROTOCOL_NAME = "EA-V3 CPU Runtime Adoption v1"
BASELINE_COMMIT = "78ed06e720bdf3634b564bf52b0f86f97d3cae8e"
Q0_BRANCH = "exp/ea-v3-runtime-qualification-v1"
S1_BRANCH = "exp/ea-v3-materialized-roots-v1"
S1_COMMIT = "6fea6044d3ad12b9d77b75702d881c2eedd6ce64"
CHECKPOINT_SHA256 = "12f7bd7ed9f909039dd6e3e3c44cd226bf765cf7d959396001e469dbbcbc3d87"
OGBENCH_COMMIT = "1d4140997f60c52c6fb0702ec100dc988b18c548"
TRAIN_SHA256 = "a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9"

DEFAULT_PATHS = {
    "repo": "/home/__compress_data/xushijie/OG_ea_v3_cpu_runtime_adoption",
    "q0_repo": "/home/__compress_data/xushijie/OG_ea_v3_runtime_qualification",
    "q0_experiment_rel": "experiments/execution_aligned/ea_v3_runtime_qualification_v1",
    "s1_repo": "/home/__compress_data/xushijie/OG_ea_v3_materialized_roots",
    "dataset_dir": "/home/__compress_data/xushijie/ea_v2_cube_data",
    "official_source": "/home/__compress_data/xushijie/og_runs/ea_v2_v1/src/ogbench",
    "checkpoint_dir": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0",
    "checkpoint_file": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0/params_1000000.pkl",
    "experiment_rel": "experiments/execution_aligned/ea_v3_cpu_runtime_adoption_v1",
}

CANONICAL_CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "JAX_ENABLE_X64": "0",
}

EXPECTED = {
    "python_version_prefix": "3.11.16",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "numpy": "2.4.6",
    "flax": "0.12.8",
    "mujoco": "3.13.0",
    "ogbench": "1.2.1",
    "gymnasium": "1.3.0",
    "jax_enable_x64": False,
    "jax_default_backend": "cpu",
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "ogbench_commit": OGBENCH_COMMIT,
}

GATES = {
    "overall_success_min": 0.26,
    "min_tasks_with_nonzero_success": 4,
    "n_tasks": 5,
    "episodes_per_task": 50,
    "n_episodes_total": 250,
    "action_finite_rate": 1.0,
    "action_bounds_rate": 1.0,
    "pre_post_pairwise_action_max_abs": 0.0,
    "pre_post_pairwise_value_max_abs": 0.0,
    "gpu_reference_overall_success": 0.36,
}

PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "protocol_name": PROTOCOL_NAME,
    "scope": ["Q2"],
    "forbidden": ["S1_retry", "S2", "S3", "Phase_C", "Phase_D", "Phase_E", "Phase_F", "P_1", "P_MH", "EXEC", "new_xla_flags_without_q0"],
    "baseline_commit": BASELINE_COMMIT,
    "canonical_cpu_env": CANONICAL_CPU_ENV,
    "jax_enable_x64": False,
    "no_new_xla_flags": True,
    "q0_corpus_readonly": True,
    "inference_recheck_processes": 2,
    "evaluator": "official ogbench impls.utils.evaluation.evaluate",
    "eval": {
        "dataset_id": "cube-double-play-v0",
        "tasks": [1, 2, 3, 4, 5],
        "episodes_per_task": 50,
        "temperature": 0.0,
        "eval_gaussian": None,
        "numpy_seed_before_eval": 0,
        "checkpoint_rule": "fixed_final_checkpoint",
        "training": False,
    },
    "gates": GATES,
    "gpu_reference_overall_success": 0.36,
    "gpu_reference_role": "engineering_qualification_only",
}
