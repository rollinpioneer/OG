# Cube Branch Determinism Audit v2

Status: `SNAPSHOT_V2_REPRODUCIBLE`.

The diagnostic set contains 8 unique roots previously marked non-reproducible and 8 unique roots previously marked reproducible. D0 used capture → mutate/step → restore → compare. D1 replayed one fixed action, D2 replayed a fixed five-action open-loop sequence, and D3 recomputed GCIQL actions from each current observation on every step.

All 16 samples passed every frozen threshold. Each level recorded action, observation, MuJoCo integration state, qpos/qvel/act/ctrl/warmstart, robot observation slice 0:19, cube slice 19:37, success/terminated/truncated agreement, and proxy value differences. No action or proxy differences were hard-coded. Snapshot-v2 and reset-replay both passed 16/16. No Phase 0-R2 rerun or later phase was started.
