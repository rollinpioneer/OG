"""Frozen EA-V3 protocol constants. S1+ must read protocol_v1.json; this module is the S0 source."""

from __future__ import annotations

PROTOCOL_ID = "ea_v3_materialized_roots_v1"
PROTOCOL_NAME = "EA-V3 Materialized Roots v1"
SCHEMA_VERSION = "mr-ep-1.0"
HISTORICAL_COMMIT = "7714e33958dd6ab7d8a1ddf894dca1eb88a69b17"
OGBENCH_COMMIT = "1d4140997f60c52c6fb0702ec100dc988b18c548"
DATASET_ID = "cube-double-play-v0"
ENV_ID = "cube-double-v0"

TRAIN_SHA256 = "a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9"
VAL_SHA256 = "b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e"
CHECKPOINT_SHA256 = "12f7bd7ed9f909039dd6e3e3c44cd226bf765cf7d959396001e469dbbcbc3d87"
NORMALIZER_SHA256 = "d4f912c0e458a031114cce0c084157780e3a527d26e21e4e1b7aa0745dd2c0c5"

DEFAULT_PATHS = {
    "repo": "/home/__compress_data/xushijie/OG_ea_v3_materialized_roots",
    "historical_repo": "/home/__compress_data/xushijie/OG",
    "dataset_dir": "/home/__compress_data/xushijie/ea_v2_cube_data",
    "official_source": "/home/__compress_data/xushijie/og_runs/ea_v2_v1/src/ogbench",
    "checkpoint_dir": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0",
    "checkpoint_file": "/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0/params_1000000.pkl",
    "normalizer_file": "/home/__compress_data/xushijie/OG/experiments/execution_aligned/ea_v2_cube_mechanism_v1/manifests/normalizer_train_only.npz",
    "v2_experiment_dir": "/home/__compress_data/xushijie/OG/experiments/execution_aligned/ea_v2_cube_mechanism_v1",
    "conda_env": "og-ea-v2",
    "root_store": "/home/__compress_data/xushijie/ea_v3_root_store/ea_v3_materialized_roots_v1",
    "experiment_dir_rel": "experiments/execution_aligned/ea_v3_materialized_roots_v1",
}

THRESHOLDS = {
    "rtol": 0.0,
    "action_max_abs": 1e-7,
    "observation_max_abs": 1e-6,
    "robot_obs_0_19_max_abs": 1e-6,
    "cube_obs_19_37_max_abs": 1e-6,
    "integration_continuous_max_abs": 1e-6,
    "qpos_max_abs": 1e-6,
    "qvel_max_abs": 1e-6,
    "act_max_abs": 1e-6,
    "ctrl_max_abs": 1e-6,
    "warmstart_max_abs": 1e-6,
    "proxy_max_abs": 1e-5,
    "goal_encoding_max_abs": 0.0,
}

ENGINEERING_ROOTS = []
for task_id in range(1, 6):
    for step in (0, 125, 250):
        root_id = 690000 + (task_id - 1) * 3 + (0 if step == 0 else 1 if step == 125 else 2)
        ENGINEERING_ROOTS.append(
            {
                "root_id": root_id,
                "task_id": task_id,
                "planned_decision_step": step,
                "reset_seed": root_id,
            }
        )

PROTOCOL_V1 = {
    "protocol_id": PROTOCOL_ID,
    "protocol_name": PROTOCOL_NAME,
    "schema_version": SCHEMA_VERSION,
    "status_machine": {
        "phase_c_unlocked": False,
        "s2_unlocked": False,
        "s3_unlocked": False,
        "s4_unlocked": False,
        "human_review": None,
        "agent_may_self_approve_human_review": False,
        "default_on_missing_field": "HOLD",
        "default_on_asset_mismatch": "EA3_HOLD_ASSET_MISMATCH",
        "finalizer_may_rewrite_protocol": False,
    },
    "scope": {
        "this_handoff": "S0_S1",
        "forbidden": [
            "S2_formal_root_pool",
            "S3_mechanism",
            "S4",
            "S5",
            "Phase_C",
            "Phase_D",
            "Phase_E",
            "Phase_F",
            "P_1_training",
            "P_MH",
            "EXEC",
            "gap_statistics",
            "wrong_selection",
            "oracle_ranking",
            "research_success_rate",
            "overwrite_ea_v2_outputs",
        ],
        "result_source": "ENV_EVALUATED",
        "training_eligible": False,
    },
    "historical": {
        "repository_commit": HISTORICAL_COMMIT,
        "ea_v2_closure_status": "HOLD_CUBE_ROOT_RECONSTRUCTION_MISMATCH",
        "ea_v2_output_dir": DEFAULT_PATHS["v2_experiment_dir"],
        "read_only": True,
    },
    "assets": {
        "dataset_id": DATASET_ID,
        "env_id": ENV_ID,
        "ogbench_commit": OGBENCH_COMMIT,
        "train_sha256": TRAIN_SHA256,
        "validation_sha256": VAL_SHA256,
        "actor_and_value_checkpoint_sha256": CHECKPOINT_SHA256,
        "normalizer_sha256": NORMALIZER_SHA256,
        "normalizer_applied_to_actor_value_inputs": False,
        "normalizer_role": "offline_split_and_future_retrieval_distance_only",
        "backbone": {
            "algorithm": "GCIQL",
            "alpha": 1.0,
            "actor_p_randomgoal": 0.0,
            "actor_p_trajgoal": 1.0,
            "actor_p_curgoal": 0.0,
            "seed": 0,
            "updates": 1000000,
            "checkpoint_rule": "fixed_final_checkpoint",
            "shared_actor_value_checkpoint": True,
            "explicit_remaining_horizon_input": False,
            "controller_interface": "pi(a|o,z)",
            "k_protocol": 20,
            "k_is_controller_horizon_input": False,
            "inference_temperature": 0.0,
        },
        "do_not_retrain_or_replace_on_mismatch": True,
    },
    "environment": {
        "mode": "task",
        "observation_type": "states",
        "observation_dim": 37,
        "action_dim": 5,
        "action_low": -1.0,
        "action_high": 1.0,
        "max_episode_steps": 500,
        "terminate_at_goal": True,
        "success_timing": "post",
        "task_count": 5,
        "permute_blocks": True,
        "control_substeps_field": "_n_steps",
        "reset_internal_random_steps_at_goal_state": 2,
        "reset_internal_steps_are_not_external_control_steps": True,
        "official_reset_distribution_unmodified": True,
        "observation_xyz_center": [0.425, 0.0, 0.0],
        "observation_xyz_scaler": 10.0,
        "gripper_opening_scaler": 3.0,
        "cube_success_position_tolerance_m": 0.04,
        "quaternion_order": "wxyz",
        "public_slices": {
            "robot_0_19": [0, 19],
            "cube_19_37": [19, 37],
            "effector_pos_scaled_12_15": [12, 15],
            "cube0_pos_scaled_19_22": [19, 22],
            "cube0_quat_22_26": [22, 26],
            "cube0_yaw_cs_26_28": [26, 28],
            "cube1_pos_scaled_28_31": [28, 31],
            "cube1_quat_31_35": [31, 35],
            "cube1_yaw_cs_35_37": [35, 37],
        },
    },
    "restore": {
        "primary_mode": "BOOTSTRAP_PREFIX_REPLAY",
        "direct_decision_restore_for_s3": False,
        "direct_decision_restore_allowed_in_s1_as_engineering_control": True,
        "new_reset_is_not_root_identity": True,
        "must_restore_exact_37d_goal_and_physical_targets": True,
        "must_replay_saved_prefix_actions": True,
        "must_not_regenerate_closed_loop_prefix": True,
        "observation_stage": {
            "after_external_control_step": [
                "set_control(action)",
                "pre_step",
                "mj_step(nstep=_n_steps)",
                "mj_rnePostConstraint",
                "post_step",
                "compute_observation from env.step return",
            ],
            "after_direct_restore": [
                "mj_setState(mjSTATE_INTEGRATION)",
                "restore python_state",
                "restore observation_stage_state onto mjData",
                "do_not_call_mj_forward",
                "first_observation=compute_observation()",
            ],
            "rollout_uses_env_step_returns_only": True,
            "arrays_must_be_copied": True,
        },
        "integration_spec_name": "mjSTATE_INTEGRATION",
    },
    "unique_executor": {
        "load_root": "execution_aligned_rl.v3.root_loader.load_root",
        "rollout_segment": "execution_aligned_rl.v3.rollout.rollout_segment",
        "compute_proxy": "execution_aligned_rl.v3.rollout.compute_proxy",
        "compare_traces": "execution_aligned_rl.v3.compare.compare_traces",
        "no_audit_only_shortcuts": True,
        "closed_loop_d3": "recompute_action_from_current_observation_each_step",
    },
    "reward_value_contract": {
        "discount": 0.99,
        "environment_reward_success": 1.0,
        "environment_reward_failure": 0.0,
        "gciql_training_reward_goal": 0.0,
        "gciql_training_reward_non_goal": -1.0,
        "planning_conversion": "r_training = r_environment - 1",
        "absorbing_tail": "tail value set to zero after success/termination",
        "proxy_formula": "sum_{i=0}^{T-1} gamma^i * (r_env_i - 1) + gamma^T * tail_value",
        "tail_value": "0 if terminated or truncated else V(o_T, exact_goal)",
        "value_inputs": "raw public observation and exact 37D goal; no extra normalizer",
    },
    "thresholds": THRESHOLDS,
    "engineering": {
        "root_ids": [row["root_id"] for row in ENGINEERING_ROOTS],
        "roots": ENGINEERING_ROOTS,
        "max_roots": 15,
        "probe_steps": 5,
        "d1_steps": 1,
        "min_legal_roots": 10,
        "require_all_five_tasks": True,
        "require_legal_125_and_250": True,
        "ineligible_prefix_policy": "INELIGIBLE_PREFIX_TERMINAL_record_and_do_not_replace_seed",
        "d3_target": "one_deterministic_training_observation_endpoint_per_root",
        "d3_target_formula": "index = sha256('ea3-mr-v1:d3-endpoint:{root_id}')[:16] % n_train",
        "prefix_policy_goal": "exact_task_goal_observation",
        "external_control_step_budget": 30000,
        "max_implementation_fix_versions": 2,
        "fresh_workers": ["A", "B"],
        "workers_must_load_from_files_only": True,
        "collector_process_must_exit_before_workers": True,
        "aba_required": True,
    },
    "python_state_fields": [
        "cur_task_id",
        "cur_task_info",
        "_cur_goal_ob",
        "_success",
        "_reset_next_step",
        "_prev_qpos",
        "_prev_qvel",
        "_prev_ob_info",
        "_render_goal",
        "_target_block",
        "_mode",
        "_success_timing",
        "_terminate_at_goal",
        "_use_oracle_rep",
        "elapsed_steps_by_wrapper",
        "env_np_random",
        "action_space_np_random",
        "global_numpy_rng",
        "python_random_state",
    ],
    "python_state_not_applicable": {
        "jax_global_key": "NOT_APPLICABLE_explicit_per_step_keys_only",
        "cur_goal_xy": "NOT_APPLICABLE_cube_uses__cur_goal_ob_and_mocap",
    },
    "observation_stage_fields": [
        "cfrc_ext",
        "site_xpos",
        "site_xmat",
        "xpos",
        "xmat",
        "xquat",
        "mocap_pos",
        "mocap_quat",
        "sensordata",
    ],
    "integration_units": {
        "qpos": "generalized_position_model_units",
        "qvel": "generalized_velocity_model_units",
        "act": "actuator_activation",
        "time": "seconds",
        "qacc_warmstart": "generalized_acceleration_warmstart",
        "ctrl": "actuator_control",
        "discrete_switches": "exact_equality",
    },
    "budget": {
        "s0_external_control_steps": 0,
        "s1_external_control_step_limit": 30000,
        "count_env_step_of_5d_action_as_external": True,
        "do_not_mix_reset_internal_substeps_or_disk_into_external_count": True,
    },
}


def protocol_payload() -> dict:
    return PROTOCOL_V1