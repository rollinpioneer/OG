# Cube Branch Determinism Audit

Status: `SNAPSHOT_V2_REPRODUCIBLE`.

The frozen diagnostic set contained 8 previously non-reproducible and 8 previously reproducible root/candidate pairs. D0 (no step), D1 (fixed single action), D2 (fixed five-action open-loop), and D3 (five-step closed-loop GCIQL) were run twice after restoring the same snapshot and RNG key. All 16 samples passed every recorded comparison: first divergent step was absent; qpos/qvel/act/ctrl/warmstart and observation differences were zero within the recorded checks; success/terminated/truncated agreed.

The `snapshot_v2_audit.json` and `reset_replay_audit.json` both report 16/16 PASS. No averaging, tolerance inflation, model retraining, or Phase 0-R rerun was used. The earlier 51.72% wrong-selection statistic remains non-confirmatory and is not reinterpreted.
