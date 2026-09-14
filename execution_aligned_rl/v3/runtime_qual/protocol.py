
"""Frozen Q0/Q1 protocol. Does not rewrite EA-V3 S1 protocol_v1.json."""

from __future__ import annotations

PROTOCOL_ID = "ea_v3_runtime_qualification_v1"
PROTOCOL_NAME = "EA-V3 Runtime Qualification v1"
BASELINE_COMMIT = "6fea6044d3ad12b9d77b75702d881c2eedd6ce64"
S1_PROTOCOL_ID = "ea_v3_materialized_roots_v1"
S1_BRANCH = "exp/ea-v3-materialized-roots-v1"
HISTORICAL_COMMIT = "7714e33958dd6ab7d8a1ddf894dca1eb88a69b17"
OGBENCH_COMMIT = "1d4140997f60c52c6fb0702ec100dc988b18c548"
CHECKPOINT_SHA256 = "12f7bd7ed9f909039dd6e3e3c44cd226bf765cf7d959396001e469dbbcbc3d87"
TRAIN_SHA256 = "a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9"

DEFAULT_PATHS = {
    "repo": "/home/__compress_data/xushijie/OG_ea_v3_runtime_qualification",
    "s1_repo": "/home/__compress_data/xushijie/OG_ea_v3_materialized_roots",
    "s1_experiment_rel": "experiments/execution_aligned/ea_v3_materialized_roots_v1",
    "s1_root_store": "/home/__compress_data/xushijie/ea_v3_root_store/ea_v3_materialized_roots_v1/ea_v3_materialized_roots_v1",
    "dataset_dir": "/home/__compress_data/xushijie/ea_v2_cube_data",
    "official_source": "/home/__compress_data/xushijie/og_runs/ea_v2_v1/src/ogbench",
    "checkpoint_dir": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0",
    "checkpoint_file": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0/params_1000000.pkl",
    "experiment_rel": "experiments/execution_aligned/ea_v3_runtime_qualification_v1",
}

THRESHOLDS = {
    "rtol": 0.0,
    "action_max_abs": 1e-7,
    "value_max_abs": 1e-5,
    "observation_max_abs": 1e-6,
    "integration_continuous_max_abs": 1e-6,
    "qpos_max_abs": 1e-6,
    "qvel_max_abs": 1e-6,
    "act_max_abs": 1e-6,
    "ctrl_max_abs": 1e-6,
    "warmstart_max_abs": 1e-6,
    "proxy_max_abs": 1e-5,
    "goal_encoding_max_abs": 0.0,
    "robot_obs_0_19_max_abs": 1e-6,
    "cube_obs_19_37_max_abs": 1e-6,
}

PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "protocol_name": PROTOCOL_NAME,
    "scope": ["Q0", "Q1"],
    "forbidden": [
        "S1_env_retry",
        "S2",
        "S3",
        "Phase_C",
        "Phase_D",
        "Phase_E",
        "Phase_F",
        "P_1",
        "P_MH",
        "EXEC",
        "rewrite_protocol_v1",
        "modify_old_traces",
        "relax_thresholds",
        "auto_switch_dist_mode",
        "modify_checkpoint",
    ],
    "s1_readonly": {
        "commit": BASELINE_COMMIT,
        "branch": S1_BRANCH,
        "protocol_id": S1_PROTOCOL_ID,
        "status": "EA3_HOLD_TRACE_MISMATCH",
        "must_not_modify": True,
    },
    "q0": {
        "n_fresh_processes": 8,
        "runtimes": {
            "GPU_CURRENT": {
                "description": "current sample_actions(..., temperature=0.0) on the S1 GPU runtime",
                "jax_platforms": None,
                "cuda_visible_devices": "3",
                "xla_python_client_preallocate": "false",
            },
            "CPU_SINGLE_THREAD": {
                "description": "same sample_actions interface on CPU, single thread",
                "jax_platforms": "cpu",
                "cuda_visible_devices": "",
                "omp_num_threads": "1",
                "mkl_num_threads": "1",
                "openblas_num_threads": "1",
            },
        },
        "actor_call": "agent.sample_actions(observation, goals=d3_target, seed=key, temperature=0.0)",
        "value_call": "agent.network.select('value')(observation, exact_task_goal)",
        "probe_key_formula": "jax.random.PRNGKey(stable_uint32('ea3-mr-v1:probe:{root_id}:{step}') & 0xFFFFFFFF)",
        "saved_key_schedule_note": "S1 original_live_probe.key_schedule was not persisted; keys are reconstructed from the frozen S1 formula",
        "thresholds": {"action_max_abs": 1e-7, "value_max_abs": 1e-5, "rtol": 0.0},
        "env_step_calls": 0,
    },
    "q1": {
        "comparator_contract": "v2",
        "d0_terminated_truncated": "explicit_agree_or_unified_NA",
        "d1_full_proxy": "NOT_APPLICABLE",
        "d1_step_proxy": "NOT_APPLICABLE",
        "d1_still_compares_action_and_state": True,
        "d2_d3_compare_full_proxy": True,
        "fail_closed": ["missing", "nan", "length", "identity"],
        "does_not_modify_s1_compare_py": True,
    },
    "thresholds": THRESHOLDS,
}
