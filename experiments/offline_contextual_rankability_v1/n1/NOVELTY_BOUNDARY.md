# Novelty Boundary

A later paper cannot be “FQE + uncertainty + fallback.” EA-V4 already
ran that combination and it produced no selector signal.

## Not novel (baselines only)

- ordinary or finite-horizon FQE;
- ensemble LCB / bootstrap CI on Q;
- SPIBB-style baseline bootstrap;
- HCOPE lower bounds on a single policy;
- OPCC as “which policy is better from s0,” if copied without change;
- ranking candidates by V(z, g) (this is IDEAL / B0);
- conformal intervals on absolute return.

## Potentially novel only as a *joint* object

All three must be necessary in ablations:

1. **Composite switched policies** π_z^m ⊕ π_g, not two unrelated π.
2. **Pairwise sign identification** of Δ under a shared trunk, not
   absolute V.
3. **Contrastive pathwise coverage** as the license to output a
   non-zero sign, with abstention and certificate coverage as primary
   metrics.

If any one of these is removed and the method still “works,” it is
not a contribution of this line.

## Overlap tripwires

Declare `RANKABILITY_DIRECTION_OVERLAPS_PRIOR_WORK` if an implementation:

- is OPCC with Cube skins;
- is SPIBB with candidate IDs;
- selects z by argmax FQE;
- uses S3 labels to fit the abstention threshold.

## What EA evidence licenses

EA-V3: a complete-task gap exists among DATA_REAL candidates.  
EA-V4: absolute composite FQE did not recover that gap as a selector.

That licenses a *weaker* question (sign identification + abstention),
not a stronger claim (we have a better evaluator).

## Paper sentence that would be rejected

“We propose FQE with ensembles and only replace IDEAL when LCB > 0.”

## Paper sentence that would still need N2 evidence

“At a decision root, offline data identify a partial order over
composite finite-budget policies exactly on pairs whose contrastive
paths are covered, and otherwise return a certified abstention.”
