# R2 Failure Forensic Audit

Status: **HOLD_CUBE_ROOT_RECONSTRUCTION_MISMATCH**

This engineering-only audit froze all 61 non-reproducible root/candidate pairs from
`branch_reproducibility_r2.json` across 10 unique roots. It generated no roots or candidates
and computed no mechanism or research metrics.

## Protocol Results

- A, same-env `snapshot_v2`: 54/61 branches passed.
- B, two independent envs with reset plus frozen-prefix replay: 54/61 branches passed.
- C, full `mjData` copied to two independent env/data instances: 54/61 branches passed.

All three protocols passed every one of the 54 fully evaluable branches. Their per-step
actions, integration states, observations, proxy values, and termination signals met the
frozen thresholds. No policy-action or post-branch simulator divergence was observed.

The remaining 7 candidate branches all belong to root 4022. In this forensic process, the
frozen closed-loop prefix terminated at step 176, although R2 recorded root 4022 as legal.
Consequently its R2 decision snapshot could not be reconstructed and none of A/B/C could be
validly run for those 7 candidates. This is classified as
`HOLD_CUBE_ROOT_RECONSTRUCTION_MISMATCH`, not simulator contact non-determinism: policy
action divergence was zero and simulator state divergence was zero on all 54 evaluable
branches.

Independent envs produced identical reset observations and task metadata but different 37D
goal observation encodings. Goal encoding was retained as a non-gate diagnostic because the
preregistered branch gate compares initial/step state, action, proxy, and termination signals.

No R2-RP execution was run. Phase C remains locked.
