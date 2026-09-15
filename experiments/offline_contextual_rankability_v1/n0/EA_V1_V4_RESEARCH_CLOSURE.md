# EA-V1–V4 Research Closure

**Status:** CLOSED  
**Accepted terminal status:** `EA41_EVALUATOR_NO_SIGNAL`  
**Closure commit:** `b4829ad31c292eda82712af782ea3b203b386005`  
**Date of closure:** 2026-09-15

This document seals the subgoal-correction / proxy-alignment / outcome-evaluator
method chain. It is not a paper claim. It is a research-line stop.

## 1. What the line attempted

The line asked whether a frozen GCIQL actor on official Cube-double-play data
could be improved, at decision roots, by replacing the IDEAL candidate with
another DATA_REAL intermediate target for a short composite prefix and then
returning to the true goal.

The intended object was a composite policy

```text
π_z^m ⊕ π_g
```

with m = 5, remaining budget R, and selection among eight retrieved endpoints.

## 2. What was established

1. A complete-task opportunity exists among locked candidates:
   `ORACLE_FULL = 21/35 > IDEAL = 15/35`.
2. The five-step proxy is not a usable decision target:
   `ORACLE_PROXY = 10/35`, with harm dominating rescue versus IDEAL.
3. Public observation/goal reconstruct the official Cube success predicate
   (90,680/90,680). That is a data fact, not a method.
4. `GPU_CURRENT` is not cross-process deterministic for actor/value.
   `CPU_SINGLE_THREAD` is bit-identical and is the canonical runtime.
5. Formal roots and DATA_REAL candidates can be reconstructed under
   BOOTSTRAP_PREFIX_REPLAY and comparator contract v2.
6. An offline finite-budget composite FQE (EA-V4 B2–B5) did not yield a
   deployable selector: B4 ensemble 11/35, B5 15/35 with net 0 versus IDEAL.

## 3. What is closed

The following are closed as *methods*, not as software artifacts:

- proxy-aligned candidate selection;
- absolute FQE of composite success probability as a ranking/selection rule;
- epsilon / LCB replacement of IDEAL using that FQE;
- further FQE seeds, beta search, or checkpoint-rule edits on EA-V4;
- fresh-environment confirmation of EA-V4 selectors;
- Phase C–F, P_1, P_MH, EXEC.

Software, hashes, and frozen pools remain readable. They must not be
retrained or overwritten.

## 4. What is not closed

The existence of an oracle gap is not closed. The failure of one estimator
class does not prove that pairwise signs of composite values are
unidentifiable. That weaker question is moved to a new line:
offline contextual rankability. That line is not authorized to train models
or call `env.step` in N0–N1.

## 5. Binding locks

```json
{
  "fresh_confirmation_unlocked": false,
  "new_environment_evaluation_authorized": false,
  "policy_training_authorized": false,
  "s4_unlocked": false,
  "fqe_retraining_authorized": false,
  "n2_unlocked": false,
  "new_environment_steps": 0
}
```

No agent following this closure may treat EA-V4 B4/B5 as a candidate for
environment confirmation.
