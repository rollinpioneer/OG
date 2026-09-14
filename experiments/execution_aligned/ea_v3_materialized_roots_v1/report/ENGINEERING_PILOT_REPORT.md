# EA-V3 Engineering Pilot Report

Status: `EA3_HOLD_TRACE_MISMATCH`

This report covers S0+S1 only. Formal root pool (S2) was not created. No model was trained. Phase C remains locked.

## Coverage

{
  "legal": 14,
  "planned": 15,
  "tasks": [
    1,
    2,
    3,
    4,
    5
  ],
  "decision_steps_present": [
    0,
    125,
    250
  ],
  "ok": true,
  "ineligible": [
    690005
  ]
}

## Resource

{
  "external_control_steps": 5706,
  "reset_internal_steps": 98,
  "budget": 30000,
  "legal_roots": 14,
  "planned_roots": 15,
  "wall_seconds_acquisition": 9.177256345748901,
  "gpu_used": true,
  "training_performed": false,
  "result_source": "ENV_EVALUATED",
  "training_eligible": false
}

## Per-root pass

- root 690000 task 1 step 0: FAIL
- root 690001 task 1 step 125: FAIL
- root 690002 task 1 step 250: FAIL
- root 690003 task 2 step 0: FAIL
- root 690004 task 2 step 125: FAIL
- root 690006 task 3 step 0: FAIL
- root 690007 task 3 step 125: FAIL
- root 690008 task 3 step 250: FAIL
- root 690009 task 4 step 0: FAIL
- root 690010 task 4 step 125: FAIL
- root 690011 task 4 step 250: FAIL
- root 690012 task 5 step 0: FAIL
- root 690013 task 5 step 125: FAIL
- root 690014 task 5 step 250: FAIL

## Failed roots

[690000, 690001, 690002, 690003, 690004, 690006, 690007, 690008, 690009, 690010, 690011, 690012, 690013, 690014]

## Negative tests

{
  "status": "PASS",
  "cases": [
    {
      "name": "byte_flip_root_file",
      "expected": "hash_failure",
      "detected": true,
      "detail": "ValueError: hash mismatch for goal_observation.npy: c4c73a6cd113c46d368166c7315b1c78dd2d1e86b7cff108e79b8d4139585008 != f5bff6c74c3f349395ea5b8f8f2f013b4047736a290ef1116bbf2b66735eb58d"
    },
    {
      "name": "replace_37d_goal_keep_task",
      "expected": "GOAL_ENCODING",
      "detected": true,
      "status": "FAIL",
      "reasons": [
        "GOAL_ENCODING"
      ]
    },
    {
      "name": "elapsed_plus_one",
      "expected": "ELAPSED_START",
      "detected": true,
      "status": "FAIL",
      "reasons": [
        "ELAPSED_START"
      ]
    },
    {
      "name": "drop_last_step",
      "expected": "LENGTH_MISMATCH",
      "detected": true,
      "status": "FAIL",
      "reasons": [
        "LENGTH_MISMATCH"
      ]
    },
    {
      "name": "nan_observation",
      "expected": "NONFINITE",
      "detected": true,
      "status": "FAIL",
      "first_divergent_step": 4
    },
    {
      "name": "action_bias",
      "expected": "action threshold",
      "detected": true,
      "status": "FAIL",
      "first_divergent_step": 0
    },
    {
      "name": "continue_after_terminal",
      "expected": "TERMINATION_PROTOCOL",
      "detected": true,
      "status": "FAIL"
    },
    {
      "name": "wrong_root_identity",
      "expected": "IDENTITY_root_id",
      "detected": true,
      "status": "FAIL",
      "reasons": [
        "IDENTITY_root_id"
      ]
    }
  ],
  "n_cases": 8,
  "passed_cases": 8
}

## A-B-A

{
  "A1_vs_A2_d3": {
    "first_divergent_step": null,
    "full_proxy": {
      "status": "PASS"
    },
    "n_steps_a": 5,
    "n_steps_b": 5,
    "reasons": [],
    "rtol": 0.0,
    "status": "PASS",
    "steps": [
      {
        "act": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "action": {
          "error": null,
          "limit": 1e-07,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "ctrl": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "cube_obs_19_37": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "elapsed_steps": {
          "agree": true,
          "status": "PASS"
        },
        "failures": [],
        "integration": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "observation": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "proxy": {
          "error": null,
          "limit": 1e-05,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qpos": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qvel": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "reward": {
          "error": null,
          "limit": 0.0,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "robot_obs_0_19": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "step": 0,
        "success": {
          "agree": true,
          "status": "PASS"
        },
        "terminated": {
          "agree": true,
          "status": "PASS"
        },
        "truncated": {
          "agree": true,
          "status": "PASS"
        },
        "warmstart": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        }
      },
      {
        "act": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "action": {
          "error": null,
          "limit": 1e-07,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "ctrl": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "cube_obs_19_37": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "elapsed_steps": {
          "agree": true,
          "status": "PASS"
        },
        "failures": [],
        "integration": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "observation": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "proxy": {
          "error": null,
          "limit": 1e-05,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qpos": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qvel": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "reward": {
          "error": null,
          "limit": 0.0,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "robot_obs_0_19": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "step": 1,
        "success": {
          "agree": true,
          "status": "PASS"
        },
        "terminated": {
          "agree": true,
          "status": "PASS"
        },
        "truncated": {
          "agree": true,
          "status": "PASS"
        },
        "warmstart": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        }
      },
      {
        "act": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "action": {
          "error": null,
          "limit": 1e-07,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "ctrl": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "cube_obs_19_37": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "elapsed_steps": {
          "agree": true,
          "status": "PASS"
        },
        "failures": [],
        "integration": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "observation": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "proxy": {
          "error": null,
          "limit": 1e-05,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qpos": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qvel": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "reward": {
          "error": null,
          "limit": 0.0,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "robot_obs_0_19": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "step": 2,
        "success": {
          "agree": true,
          "status": "PASS"
        },
        "terminated": {
          "agree": true,
          "status": "PASS"
        },
        "truncated": {
          "agree": true,
          "status": "PASS"
        },
        "warmstart": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        }
      },
      {
        "act": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "action": {
          "error": null,
          "limit": 1e-07,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "ctrl": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "cube_obs_19_37": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "elapsed_steps": {
          "agree": true,
          "status": "PASS"
        },
        "failures": [],
        "integration": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "observation": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "proxy": {
          "error": null,
          "limit": 1e-05,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qpos": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qvel": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "reward": {
          "error": null,
          "limit": 0.0,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "robot_obs_0_19": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "step": 3,
        "success": {
          "agree": true,
          "status": "PASS"
        },
        "terminated": {
          "agree": true,
          "status": "PASS"
        },
        "truncated": {
          "agree": true,
          "status": "PASS"
        },
        "warmstart": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        }
      },
      {
        "act": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "action": {
          "error": null,
          "limit": 1e-07,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "ctrl": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "cube_obs_19_37": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "elapsed_steps": {
          "agree": true,
          "status": "PASS"
        },
        "failures": [],
        "integration": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "observation": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "proxy": {
          "error": null,
          "limit": 1e-05,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qpos": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "qvel": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "reward": {
          "error": null,
          "limit": 0.0,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "robot_obs_0_19": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        },
        "step": 4,
        "success": {
          "agree": true,
          "status": "PASS"
        },
        "terminated": {
          "agree": true,
          "status": "PASS"
        },
        "truncated": {
          "agree": true,
          "status": "PASS"
        },
        "warmstart": {
          "error": null,
          "limit": 1e-06,
          "max_abs_diff": 0.0,
          "status": "PASS"
        }
      }
    ]
  },
  "A1_vs_B_d3_should_fail_if_goals_differ": {
    "first_divergent_step": 0,
    "full_proxy": {
      "status": "FAIL"
    },
    "n_steps_a": 5,
    "n_steps_b": 5,
    "reasons": [
      "IDENTITY_root_id",
      "IDENTITY_goal_sha256",
      "IDENTITY_probe_goal_sha256",
      "GOAL_ENCODING",
      "ELAPSED_START"
    ],
    "rtol": 0.0,
    "status": "FAIL",
    "steps": []
  },
  "A_id": 690000,
  "B_id": 690001
}

