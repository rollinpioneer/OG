# Rankability Metrics

All metrics are defined on pairs or on contexts. None may be optimized
on S3 labels in N0–N1. S3 numbers below are frozen historical constants
only.

## 1. Root-conditioned pairwise delta

```text
Δ_{ij}(x) = V_R(x, π^{(z_i)}) - V_R(x, π^{(z_j)})
```

On a synthetic MDP, V_R is known. On Cube, V_R is not identified from
D alone; only I_{ij}(x) is.

## 2. Identified interval / confidence region

Primary object: I_{ij}(x) ⊆ [-1, 1].  
Secondary: length |I_{ij}|, and whether 0 ∈ I_{ij}.

A method that returns a point Δ̂ without I is incomplete for this line.

## 3. Sign decision and abstention

```text
s_{ij} ∈ {+1, -1, 0}
```

Abstention rate at x:

```text
α(x) = fraction of pairs with s_{ij} = 0
```

Abstention is scored as correct when 0 ∈ true Δ neighborhood, and as
a coverage loss when the true sign is identified by an oracle but the
method abstains.

## 4. Partial order

From non-zero signs, build a DAG. Metrics:

- acyclicity (must be 1; cycles are engineering failure);
- number of comparable pairs;
- width of the remaining antichain;
- whether the baseline candidate (IDEAL) is strictly dominated.

## 5. Wrong-replacement risk

Let b be the baseline candidate at x (historical IDEAL, not refit).
A replacement i ≠ b is *issued* iff s_{ib} = +1.

```text
wrong_replacement = 1[issued and V(π^{(z_i)}) < V(π^{(z_b)})]
```

Target: P(wrong_replacement | issued) ≤ ε, with ε declared before
looking at outcomes. Issued-empty is allowed and scores as no
replacement, not as success.

## 6. Certificate coverage

```text
certificate_coverage =
  (# contexts with at least one issued replacement whose I_{ib} excludes 0)
  / (# contexts)
```

and the stricter

```text
true_certificate_rate =
  (# issued replacements with true Δ_{ib} > 0) / (# issued)
```

High coverage with uncontrolled wrong-replacement is a fail.

## 7. Contrastive pathwise coverage diagnostic

For each pair, report a scalar C_{ij}(x) ∈ [0, 1] measuring occupancy
of the contrastive support in D (e.g. minimum density along a
divergence tree, or a kernelized analogue). Rankability claims are
only licensed on pairs with C_{ij} ≥ c_min, c_min frozen before
evaluation.

## 8. Explicitly not metrics for this line

- overall success of a greedy FQE selector on 35 S3 roots;
- Brier of absolute success probabilities;
- proxy Spearman;
- any quantity that requires fitting a threshold on S3 labels.
