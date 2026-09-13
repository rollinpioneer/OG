# Mechanism Gate Report

**Status: `GAP_PRESENT_PROVISIONAL`.** Execution stopped after phases A, B, and 0. Phases C-F, `P_1`, `P_MH`, and the full EXEC system were not run.

## Frozen basis

- Task: `antmaze-large-stitch-v0`; training seed 0; `k=20`, `m=5`, `N=8`.
- OGBench commit: `1d4140997f60c52c6fb0702ec100dc988b18c548`; experiment code commit: `77910c3f01cee37d1f54158ae53cbb95562a9248`.
- Offline data: 1,000,000 train transitions in 5,000 episodes; 900,000 legal `k=20` windows.
- Phase A: `PASS`; deterministic resets and full-state restore both passed on 8/8 engineering resets.
- Phase B: `PASS`; official GCIQL final-goal value/controller plus a physical-state conditional endpoint prior. The controller is goal-conditioned but not explicitly remaining-horizon-conditioned.
- Frozen controller evaluation success: 0.075; candidate finite rate: 1.000; distinct rate: 1.000.

## Phase 0 evidence

- Legal roots: 32/32; pre-registered deep roots: 8/8.
- Offline-only value tolerance: 0.874826 (`5%` of value standard deviation 17.4965).
- Ideal top-choice proxy error rate beyond tolerance: 0.250 (threshold `0.10`).
- Full-result gap rate on deep roots: 0.125.
- Proxy indistinguishable roots: 0.000; full-result indistinguishable deep roots: 0.875.
- Mean within-root ideal/proxy Spearman: 0.28125000000000006.

All candidate branches are `ENV_EVALUATED`, evaluation-only, and excluded from training. Offline windows and value-scale calculations are `DATA_REAL`. No `MODEL_SIMULATED` phase-0 result was fabricated; dynamics models are outside the authorized scope.

## Decision and next cost

The frozen rule yields `GAP_PRESENT_PROVISIONAL` without changing task, seed, candidate count, or threshold. The next authorized unit would be phase C only: estimated 0.8 A100 GPU-hours, at most 20,480 diagnostic environment steps, and about 2 GiB storage. It has not been started and requires human review.
