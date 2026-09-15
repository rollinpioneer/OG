# Problem Formulation

## 1. Question

Given a context x and a finite set of candidate composite policies, can
offline data identify the *sign* of pairwise value differences well enough
to support a partial order with abstention?

Formally, x is a decision root:

```text
x = (o, g, R, π)
```

where o is the public observation, g the exact 37D task goal, R the remaining
budget, and π the frozen GCIQL actor (query-only).

Each candidate z induces

```text
π^{(z)} = π(· | o, z)^m ⊕ π(· | o, g)^{R-m}
```

with m = 5 unless a later protocol changes m *before* any experiment.

## 2. Estimand

For a pair {i, j} at context x:

```text
Δ_{ij}(x) = V_R(x, π^{(z_i)}) - V_R(x, π^{(z_j)})
```

where V_R is the finite-budget success probability (or undiscounted success
indicator expectation) under official Cube success, not a proxy return.

The *rankability estimand* is not Δ itself. It is

```text
s_{ij}(x) ∈ {+1, -1, 0}
```

with 0 meaning “sign not identified.” Identification of 0 is a first-class
outcome, not a failure of an estimator.

## 3. Identified interval

An identification procedure returns a set I_{ij}(x) ⊆ ℝ such that, under
stated assumptions,

```text
Δ_{ij}(x) ∈ I_{ij}(x)
```

Decision:

```text
s_{ij}(x) =
  +1  if inf I_{ij}(x) > 0
  -1  if sup I_{ij}(x) < 0
   0  otherwise
```

Point estimates, greedy argmax, and LCB-of-absolute-Q are not this object.
EA-V4 already tested those.

## 4. Why pairwise and why contextual

Candidates share the same o, g, R, actor, and the first-stage dynamics
until they diverge. Absolute values share a large common trunk. Pairwise
Δ can be identified when absolute V is not. Context x is part of the
estimand: rankability is a property of (x, candidate set, data), not a
global “OPE is accurate” bit.

## 5. Outputs allowed

- a partial order over candidates at x;
- an abstention set;
- a certificate that a replacement of a baseline candidate (e.g. IDEAL)
  has identified positive Δ;
- coverage diagnostics for those certificates.

Outputs forbidden in this line’s first experimental stage (N2, if later
authorized): a new control policy, a new FQE selector, environment
confirmation.

## 6. Non-goals

- beating IDEAL on the frozen 35 S3 roots by refitting FQE;
- using S3 success labels to tune thresholds;
- claiming novelty for ensemble uncertainty or baseline fallback.
