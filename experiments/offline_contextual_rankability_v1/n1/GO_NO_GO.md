# Go / No-Go

## Decision (N1)

**`RANKABILITY_DIRECTION_GO`**

This is a go for a *synthetic identifiability program* (future N2),
not a go for Cube training, FQE retuning, or environment confirmation.

## Why not HOLD

The question is well-posed: sign(Δ) of root-conditioned composite
policies, with identified intervals and abstention. Assumptions and
metrics are stated. Literature neighbours are named. There is no
missing definition that blocks writing a synthetic spec.

## Why not NOT_IDENTIFIABLE

EA-V4 showed that *one estimator class* (absolute FQE of success,
greedy or LCB replacement) failed on 35 frozen roots. That is not a
proof that pairwise signs are unidentifiable. Identifiability is a
property of (x, pair, data, assumptions), and Cube contrastive
coverage has not been measured as C_{ij}(x).

## Why not OVERLAPS_PRIOR_WORK *yet*

OPCC, HCOPE, and SPIBB are close. The joint object
(composite switch + pairwise sign + contrastive coverage certificates)
is not the same paper as any one of them. If N2 implements only OPCC
or only SPIBB, this status must be revised downward.

## Binding conditions on the GO

1. N2, if authorized separately, is synthetic only. `env.step` on Cube
   remains 0.
2. No FQE retraining on Cube. No beta search. No extra EA-V4 seeds.
3. S3 labels remain unread for fitting.
4. A method that outputs confident signs under empty contrastive
   coverage fails N2 regardless of accuracy on covered pairs.
5. Success is not “beat IDEAL 15/35.”

## What would flip the status later

- N2 shows signs are identified only when we already have on-policy
  rollouts of both composites → `NOT_IDENTIFIABLE` for offline D.
- N2 is indistinguishable from OPCC tables → `OVERLAPS_PRIOR_WORK`.
- Formulation needs a new confounding term we cannot write → HOLD.

## Locks preserved

```text
n2_unlocked = false
fresh_confirmation_unlocked = false
policy_training_authorized = false
new_environment_steps = 0
```
