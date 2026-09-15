# Offline Supervision Feasibility

S3 environment labels are not eligible for pure offline training.

{
  "public_obs_goal_success_reconstruction": {
    "n_steps": 90680,
    "agree": 90680,
    "disagree": 0,
    "agreement_rate": 1.0,
    "examples": []
  },
  "recorded_window_is_not_target_policy_rollout": true,
  "s3_env_labels_training_eligible": false,
  "true_terminal_vs_truncation_vs_budget": {
    "cube_success_does_not_set_terminated": true,
    "s3_deep_tail_stop_on_success": true,
    "s3_prefix_does_not_stop_on_success": true,
    "time_limit_truncated_at_500": true
  },
  "actor_support_diagnostic": {
    "n": 511,
    "max_abs_vs_recorded_action_mean": 0.15037058348222518,
    "max_abs_vs_recorded_action_p90": 0.2383008599281311,
    "note": "distance to behavior action at the same index, not proof of coverage"
  },
  "future_fqe_supervision": {
    "requires": "target actor bootstrap on train (o,a,o') plus remaining budget R and reconstructible success",
    "cannot_use": "S3 ENV_EVALUATED labels as train y",
    "cannot_treat_recorded_later_actions_as_pi_g": true
  }
}

| Quantity | Available supervision | Must not be impersonated |
|---|---|---|
| One-step env transition | train `(o,a,o')` | current actor outcome of an arbitrary new action |
| Recorded window endpoint | behavior-executed action sequence | frozen actor q-step real endpoint |
| Relabeled success | official xyz-threshold 0.04 if reconstructible from public fields | arbitrary Euclidean substitute |
| Target-actor continuation value | requires target-policy OPE / FQE with actor queries | recorded later actions as if they were pi_z or pi_g |
| Finite-budget success | needs remaining-step R and a success predicate | unlimited discounted IQL value |

True terminal, data-collection truncation, and remaining-budget exhaustion are distinct. Cube S3 traces typically end by TimeLimit truncation or stop-on-success in the tail, not a separate irreversible failure flag.

