# CUBE PHASE 0-R2 REPORT

Status: **HOLD_CUBE_BRANCH_REPRODUCIBILITY**

R2 used frozen Cube GCIQL/V assets, DATA_REAL legal k=20 train windows, roots 4000-4031, k=20, m=5, N=8, and the shared MuJoCo `snapshot_v2` integration-state interface. Candidate branches were evaluation-only (`ENV_EVALUATED`) and no result entered training data.

## Coverage

- Legal roots: 28/32
- Deep roots evaluated: 15/16
- Proxy wrong-selection rate: 0.1786
- Full distinguishable roots: 10
- IDEAL not best deep roots: 2

## Engineering Gate

Repeated branches were compared from restored `mjSTATE_INTEGRATION` snapshots with Python mutable state, RNG, task state, and wrapper elapsed state restored. `all_reproducible` was **False**, so the mandatory engineering gate failed. Per protocol, the numerical mechanism thresholds are not interpreted as a research result and Phase C remains locked.
