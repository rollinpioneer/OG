# Synthetic Benchmark Spec

This is a specification only. N1 does **not** run it. N2 remains locked.

## Purpose

Test whether sign(Δ) is identifiable when coverage and pairing are
controlled, before touching Cube.

## Environment class

A small finite-horizon MDP family with:

- horizon H ∈ {8, 16, 32};
- discrete states |S| ≤ 40 or a 2D continuous toy with known density;
- binary success terminal, so V ∈ [0, 1];
- a frozen actor π(a | s, g) analogue with two goal channels z and g;
- composite policies π^{(z)} = π_z^m ⊕ π_g with m ∈ {1, 2, 3}.

No Cube, no MuJoCo, no env.step on the official task.

## Factors (orthogonal)

1. Contrastive coverage of pair {i, j}: high / medium / none.
2. Shared-trunk length: long shared prefix vs immediate divergence.
3. Pairwise |Δ|: large (≥ 0.2), small (0.05), zero (true tie).
4. Behavior overlap with the contrastive set: full, partial, empty.
5. Sample size n of D.
6. Function-approximation misspecification: tabular oracle vs small MLP.

## Queries

For each world, emit a set of pairwise queries (x, i, j) with known
true Δ. A method must output I_{ij} and s_{ij}.

## Pass criteria for a method class (N2, if authorized)

On cells with high contrastive coverage and |Δ| ≥ 0.2:

- P(wrong sign | non-abstain) ≤ 0.05;
- abstention ≤ 0.20;
- wrong-replacement risk ≤ 0.05.

On cells with empty contrastive coverage:

- abstention ≥ 0.90;
- wrong-replacement risk ≤ 0.02.

A method that is confident on empty coverage fails, even if it is
accurate on high coverage.

## Forbidden in this spec

- training a Cube FQE;
- reading S3 jsonl;
- using IDEAL/ORACLE 15/21/10/14 as training targets;
- declaring success from a single random seed.

## Deliverable of a future N2

A table: factor combination × sign-error × abstention × interval length.
No environment confirmation on Cube.
