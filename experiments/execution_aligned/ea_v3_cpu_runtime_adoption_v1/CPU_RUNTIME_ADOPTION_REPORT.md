# EA-V3 CPU Runtime Adoption Report

Status: `EA32_CPU_CAPABILITY_PASS`

- Canonical CPU runtime SHA256: `a5cf86cbb83a40d6f20872616797f9595f075e56c5537b52eeb4f9a1ab59531a`
- Pre-eval inference check: `PASS` actor pairwise `0.0`, value pairwise `0.0`, matches Q0 `True`
- CPU backbone overall success: `0.356` (gate >= 0.26; GPU reference 0.36 engineering-only)
- Nonzero-success tasks: `4` / 5 (gate >= 4)
- Action finite/bounds: `1.0` / `1.0`
- Post-eval inference check: `PASS` actor pairwise `0.0`, value pairwise `0.0`

No new XLA flags. No S1 retry. No S2. Checkpoint and thresholds unchanged.

{
  "decision": {
    "stage": "Q2",
    "status": "EA32_CPU_CAPABILITY_PASS",
    "holds": [],
    "pre_eval": "PASS",
    "post_eval": "PASS",
    "overall_success": 0.356,
    "gpu_reference_overall_success": 0.36,
    "nonzero_success_tasks": 4,
    "finite_action_rate": 1.0,
    "bounds_action_rate": 1.0,
    "cpu_runtime_environment_sha256": "a5cf86cbb83a40d6f20872616797f9595f075e56c5537b52eeb4f9a1ab59531a",
    "env_step_calls": "official_evaluator_250_episodes",
    "s1_retry_run": false,
    "s2_unlocked": false,
    "phase_c_unlocked": false,
    "training_performed": false,
    "human_review": null,
    "protocol_id": "ea_v3_cpu_runtime_adoption_v1"
  },
  "evaluation_tasks": {
    "task1": {
      "n_episodes": 50,
      "n_success_episodes": 30,
      "official_stats_success": 0.6,
      "result_source": "ENV_EVALUATED",
      "success": 0.6
    },
    "task2": {
      "n_episodes": 50,
      "n_success_episodes": 28,
      "official_stats_success": 0.56,
      "result_source": "ENV_EVALUATED",
      "success": 0.56
    },
    "task3": {
      "n_episodes": 50,
      "n_success_episodes": 20,
      "official_stats_success": 0.4,
      "result_source": "ENV_EVALUATED",
      "success": 0.4
    },
    "task4": {
      "n_episodes": 50,
      "n_success_episodes": 0,
      "official_stats_success": 0.0,
      "result_source": "ENV_EVALUATED",
      "success": 0.0
    },
    "task5": {
      "n_episodes": 50,
      "n_success_episodes": 11,
      "official_stats_success": 0.22,
      "result_source": "ENV_EVALUATED",
      "success": 0.22
    }
  }
}
