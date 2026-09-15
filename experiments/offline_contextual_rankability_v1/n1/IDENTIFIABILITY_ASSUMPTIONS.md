# Identifiability Assumptions

These are assumptions for analysis, not facts about Cube. N2, if ever
authorized, must be able to violate them one at a time.

## A1. Known frozen actor

π is known and queryable. Actions a ~ π(o, z) or π(o, g) are not a
learning problem. EA-V3 CPU runtime remains the query implementation.

## A2. Known success functional

c(o, g) is the official Cube predicate from public obs/goal, tolerance
0.04. Reward is the finite-budget success indicator. No learned proxy.

## A3. Offline data as a measure on trajectories

D is official train DATA_REAL. It induces, for each composite policy
π^{(z)}, a *pathwise occupancy* of (state, action, remaining budget,
phase j). Identification of Δ_{ij} depends on occupancy of the
*symmetric difference* of those path measures, not only on each
policy’s marginal coverage.

## A4. Contrastive pathwise coverage

Define the contrastive support of pair {i, j} at x as the set of
transitions that occur under one of {π^{(z_i)}, π^{(z_j)}} with
materially different action or next-state law. Rankability requires
that this contrastive set is covered by D at a stated density, or
that a smoothness/partial-identification assumption replaces coverage.

Without A4, sign(Δ) is not point-identified. An honest method must
output 0.

## A5. No unobserved confounder in the MDP sense

The environment is an MDP with public o. Hidden-state confounding is
out of scope. Goal-conditioning is observed.

## A6. Finite horizon and bounded value

R ≤ 500, V ∈ [0, 1]. Intervals are subsets of [-1, 1].

## A7. Shared-trunk covariance is real

Because π^{(z_i)} and π^{(z_j)} share the actor and often share prefix
dynamics, Var(Δ) ≪ Var(V_i) + Var(V_j). Estimators that ignore pairing
are misspecified for this estimand.

## A8. What is *not* assumed

- overlap of π^{(z)} with behavior in the IS sense for the full horizon;
- correctness of FQE function approximation;
- that IDEAL is the unique baseline;
- that S3 labels identify anything beyond historical illustration.

## Implication

If A4 fails for a pair, the identified interval must contain 0.
Reporting a confident sign in that case is a method failure, not a
Cube failure.
