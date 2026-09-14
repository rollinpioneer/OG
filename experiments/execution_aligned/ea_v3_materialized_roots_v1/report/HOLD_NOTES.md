# S1 HOLD notes

Final status: `EA3_HOLD_TRACE_MISMATCH`.

Two implementation fixes were used after protocol freeze:
1. `ea1cffee` JSON-encode ndarray `task_state`.
2. `ea8ff2fa` empty `act` arrays compare equal; bootstrap terminal flags recorded.

No third executor/threshold change was applied.

## What passed
- S0 asset locks and unit tests.
- Coverage: 14/15 legal roots; all 5 tasks; both 125 and 250 present. Root `690005` is `INELIGIBLE_PREFIX_TERMINAL` at prefix length 137 and was not replaced.
- External control steps: 5706 / 30000.
- Negative tests: 8/8 injected faults rejected.
- A-B-A in worker A: reloading root A after B reproduces A (`A1_vs_A2` PASS).
- Worker B vs original live probe: D2 and D3 PASS on all legal roots.
- D0 physics fields (observation, 37D goal, integration, qpos/qvel/ctrl/warmstart, robot 0:19, cube 19:37) PASS. The recorded D0 FAIL is only `terminated`/`truncated` MISSING in the D0 measure call, not a physics mismatch.
- Prefix replay (`BOOTSTRAP_PREFIX_REPLAY`) recovered decision snapshots under `strict=True`.

## What failed
- Worker A closed-loop D3 vs original live probe fails at step 1. Example root 690000: action max abs `1.71e-7` (limit `1e-7`), integration/warmstart `3.67e-6` (limit `1e-6`).
- Therefore A vs B D3 also fails. Open-loop D2 mostly passes, so the saved prefix/probe actions replay, but recomputed policy actions in the longer-lived worker A process do not stay inside threshold.
- D1 comparisons fail only on `FULL_PROXY` bookkeeping for 1-step slices, not on step-level action/state fields.

Phase C remains locked. S2 was not started.
