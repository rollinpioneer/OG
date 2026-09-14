# EA-V3 Runtime Qualification Report

Status: `EA31_HOLD_POLICY_RUNTIME_NUMERICS`

- GPU: `EA31_HOLD_POLICY_RUNTIME_NUMERICS` actor pairwise max-abs `5.960464477539062e-07` (limit 1e-07); value `3.0517578125e-05` (limit 1e-05)
- CPU: `EA31_CPU_RUNTIME_QUALIFIED` actor pairwise max-abs `0.0` (limit 1e-07); value `0.0` (limit 1e-05)
- Comparator contract v2 tests: `PASS` (8 tests)

S1 traces, protocol_v1.json, and thresholds were not modified. No env.step was called. S2/Phase C remain locked.

## Corpus

{
  "n_legal_roots": 14,
  "n_queries": 154,
  "n_actor_queries": 70,
  "n_value_queries": 84,
  "env_step_calls": 0
}

## Decision

{
  "stage": "Q0_Q1",
  "status": "EA31_HOLD_POLICY_RUNTIME_NUMERICS",
  "gpu_status": "EA31_HOLD_POLICY_RUNTIME_NUMERICS",
  "cpu_status": "EA31_CPU_RUNTIME_QUALIFIED",
  "comparator_status": "PASS",
  "holds": [],
  "env_step_calls": 0,
  "s1_retry_run": false,
  "s2_unlocked": false,
  "phase_c_unlocked": false,
  "human_review": null,
  "training_performed": false,
  "thresholds_relaxed": false,
  "dist_mode_used": false,
  "protocol_id": "ea_v3_runtime_qualification_v1"
}

## GPU pairwise detail

GPU actor pairwise max-abs is 5.960464477539062e-07 against the frozen 1e-7 action threshold; all 70 actor items have SHA256 mismatch across 8 fresh GPU processes.

GPU value pairwise max-abs is 3.0517578125e-05 against the frozen 1e-5 value threshold; 75 of 84 value items mismatch.

CPU actor and value pairwise max-abs are 0.0 across 8 fresh CPU processes (bit-identical SHA256).

Thresholds were not relaxed. dist.mode() was not used. The checkpoint was not modified. S1 was not re-run.

Primary status is `EA31_HOLD_POLICY_RUNTIME_NUMERICS` because GPU actor fails the frozen action threshold. GPU value also fails independently.
