# Literature Map

This map positions *offline contextual rankability* against existing work.
It does not claim that the new line already exceeds these papers.

## 1. Direct neighbours (must not be rebranded)

### Offline policy comparison with confidence (OPCC)
Koul, Phielipp, Fern, 2022. Uses historical data to answer: which of two
policies is better from a given start state and horizon, with a confidence
value. Benchmarks exist. Model-based ensembles are a published baseline.

Overlap: pairwise comparison, confidence, offline data.  
Gap we may still own: OPCC compares two *fully specified policies*. It does
not treat a *root-conditioned family of composite finite-budget policies*
sharing a frozen actor and differing only by a short intermediate goal, nor
does it treat *sign identification of Δ under shared-trunk covariance* as
the primary estimand.

### High-confidence OPE / policy improvement
Thomas, Theocharous, Ghavamzadeh, AAAI 2015 (HCOPE) and ICML 2015 (HCPI).
Concentration bounds on importance-weighted returns; safe improvement over
a baseline.

Overlap: abstention / refuse-to-deploy when the bound is weak.  
Gap: HCOPE bounds *absolute* value of one policy, not a *partial order*
over many composite candidates at a context x.

### SPIBB and Soft-SPIBB
Laroche, Trichelair, Tachet des Combes, ICML 2019; Nadjahi et al. 2019.
Bootstrap to the baseline in under-covered state-action pairs.

Overlap: knows-what-it-knows, fallback.  
Gap: SPIBB is a *policy improvement algorithm*. Rankability is an
*identification statement* about pairwise signs, which can later justify
abstention without producing a new control policy.

### FQE, Z-estimation, distributional OPE
Le, Voloshin, Yue 2019; Uehara, Huang, Jiang ICML 2022; distributional OPE
ICML 2023. Point or interval estimates of Q^π.

Overlap: the numerical engine one might use.  
Gap: EA-V4 already showed that using FQE as an *absolute success-probability
regressor* then greedily selecting z failed. Rankability asks whether the
*sign of a paired difference* is identified, which is a different functional.

## 2. Partial identification and coverage

Ben-Michael et al. / Kallus-style partial identification for OPE without
overlap; smoothness or boundedness assumptions replace common support.
These papers are the right language for “identified interval” when
importance weights do not exist.

Conformal OPE (Taufiq et al. 2022; Faccio et al. 2023; Zhang, Shi, Luo)
gives prediction sets for returns, usually marginally, not pairwise signs
of composite policies.

## 3. Ranking from pairwise comparisons

Bradley–Terry / Luce ranking; partial identification of rankings from
incomplete tournaments (e.g. 2024–2025 econometric ranking papers).
These treat *observed pairwise outcomes*, not *offline simulated policy
values*.

Learning-to-rank OPE (slate / ranking policies) is a different object:
the policy *is* a ranking over documents, not a ranking over control
policies.

## 4. What EA-V4 already spent

EA-V4 implemented ordinary FQE, finite-budget FQE, no-R ablation, ensemble
LCB, and IDEAL fallback. Those are baselines, not a novelty claim. Repeating
them on Cube is forbidden.

## 5. Working conclusion for N1

The *closest* published problem is OPCC. A new line is only justified if
the estimand is strictly:

```text
sign( V(π_{z_i}|x) - V(π_{z_j}|x) )
```

for composite policies that share a frozen actor, a common trunk, a finite
remaining budget, and a candidate set retrieved from DATA_REAL, together
with an abstention symbol when the identified interval for the difference
contains 0.

If a later experiment reduces to “run OPCC ensembles on Cube,” the status
must become `RANKABILITY_DIRECTION_OVERLAPS_PRIOR_WORK`.
