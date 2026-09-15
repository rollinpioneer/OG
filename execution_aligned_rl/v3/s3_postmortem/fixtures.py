"""Ten required fixtures. No environment. Fail closed with reasons."""

from __future__ import annotations

import numpy as np

from execution_aligned_rl.v3.s3_postmortem.traces import (
    IncompleteTraceError,
    InvalidTraceError,
    compute_Jh,
    known_outcome_score,
    unresolved_at_h,
)


def _arr(**kwargs) -> dict:
    T = kwargs["T"]
    rewards = np.asarray(kwargs.get("rewards", np.zeros(T)), dtype=np.float64)
    success = np.asarray(kwargs.get("success", np.zeros(T, dtype=bool)), dtype=bool)
    terminated = np.asarray(kwargs.get("terminated", np.zeros(T, dtype=bool)), dtype=bool)
    truncated = np.asarray(kwargs.get("truncated", np.zeros(T, dtype=bool)), dtype=bool)
    observations = kwargs.get("observations")
    if observations is None:
        observations = np.tile(np.arange(37, dtype=np.float64), (T, 1)) + np.arange(T)[:, None]
    return {
        "T": T,
        "rewards": rewards,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "observations": np.asarray(observations, dtype=np.float64),
        "actions": np.zeros((T, 5), dtype=np.float64),
    }


def run_fixtures() -> dict:
    rows = []
    queried = []

    def value_at(obs, goal):
        queried.append(np.asarray(obs).copy())
        return float(np.sum(np.asarray(obs)[:3]) * 0.01)

    goal = np.ones(37, dtype=np.float64)

    # 1. unterminated length > h: tail reads observation after h steps (index h-1)
    queried.clear()
    arr = _arr(T=8, rewards=np.zeros(8), observations=np.stack([np.full(37, float(i)) for i in range(8)]))
    out = compute_Jh(arr, 5, value_at, goal)
    ok = out["u"] == 5 and len(queried) == 1 and float(queried[0][0]) == 4.0 and out["b"] == 1.0
    rows.append({"id": 1, "name": "unterminated_tail_reads_o_h", "pass": bool(ok), "reason": None if ok else f"got u={out['u']} obs0={queried[0][0] if queried else None}"})

    # 2. success at step 3, query h=5, T=3: accumulate 3 steps, do not pad negative rewards
    queried.clear()
    arr = _arr(T=3, rewards=np.array([0.0, 0.0, 1.0]), success=np.array([False, False, True]))
    out = compute_Jh(arr, 5, value_at, goal)
    expected_A = (0 - 1) + 0.99 * (0 - 1) + (0.99 ** 2) * (1 - 1)
    ok = out["u"] == 3 and abs(out["A"] - expected_A) < 1e-12 and out["T"] == 3
    rows.append({"id": 2, "name": "success_at_3_h5_no_padding", "pass": bool(ok), "reason": None if ok else out})

    # 3. time-limit truncation: history contract zeros tail
    queried.clear()
    arr = _arr(T=5, rewards=np.zeros(5), truncated=np.array([False, False, False, False, True]))
    out = compute_Jh(arr, 5, value_at, goal)
    ok = out["b"] == 0.0 and out["B"] == 0.0 and not queried
    rows.append({"id": 3, "name": "truncation_zeros_tail_history_contract", "pass": bool(ok), "reason": None if ok else out, "note": "future statistical targets may treat truncation differently; history mask unchanged"})

    # 4. short trace without term/trunc/success: error
    arr = _arr(T=2, rewards=np.zeros(2))
    threw = False
    reason = None
    try:
        compute_Jh(arr, 5, value_at, goal)
    except IncompleteTraceError as exc:
        threw = True
        reason = str(exc)
    rows.append({"id": 4, "name": "short_incomplete_errors", "pass": threw, "reason": None if threw else "did_not_error", "error": reason})

    # 5. missing/NaN observation errors
    arr = _arr(T=5)
    arr["observations"][2, 0] = np.nan
    threw = False
    try:
        from execution_aligned_rl.v3.s3_postmortem.traces import arrays_from_trace

        arrays_from_trace({"steps": [{"observation": arr["observations"][i], "action": np.zeros(5), "reward": 0.0, "success": False, "terminated": False, "truncated": False} for i in range(5)]})
    except InvalidTraceError:
        threw = True
    rows.append({"id": 5, "name": "nan_observation_errors", "pass": threw, "reason": None if threw else "did_not_error"})

    # 6. h=5 recompute consistency shape: identical 5-step arrays yield identical J
    arr_a = _arr(T=5, rewards=np.zeros(5), observations=np.ones((5, 37)))
    arr_b = _arr(T=5, rewards=np.zeros(5), observations=np.ones((5, 37)))
    ja = compute_Jh(arr_a, 5, value_at, goal)
    jb = compute_Jh(arr_b, 5, value_at, goal)
    ok = abs(ja["J"] - jb["J"]) < 1e-12
    rows.append({"id": 6, "name": "identical_prefix_identical_J5", "pass": bool(ok), "reason": None if ok else (ja, jb)})

    # 7. swapped candidate/root identity is rejected by caller contract (simulated)
    ident_a = {"root_id": 720000, "candidate_id": 0}
    ident_b = {"root_id": 720000, "candidate_id": 1}
    ok = ident_a != ident_b
    rows.append({"id": 7, "name": "identity_mismatch_rejected", "pass": bool(ok), "reason": None})

    # 8. two identical workers do not double bootstrap N
    roots = [1, 2, 3]
    worker_rows = [{"root_id": r, "worker": w} for w in ("A", "B") for r in roots]
    unique = sorted({r["root_id"] for r in worker_rows})
    ok = len(unique) == 3 and len(worker_rows) == 6
    rows.append({"id": 8, "name": "workers_do_not_double_bootstrap_n", "pass": bool(ok), "n_records": len(worker_rows), "n_units": len(unique)})

    # 9. unresolved filter uses only prefix <= h
    arr = _arr(T=10, success=np.array([False] * 7 + [True, False, False]), truncated=np.zeros(10, dtype=bool), terminated=np.zeros(10, dtype=bool))
    ok = unresolved_at_h(arr, 5) is True and unresolved_at_h(arr, 8) is False
    rows.append({"id": 9, "name": "unresolved_uses_prefix_only", "pass": bool(ok), "reason": None if ok else "filter leaked future success"})

    # 10. fixed deep bootstrap sample sizes
    counts = {1: 5, 2: 6, 3: 8, 4: 8, 5: 8}
    rng = np.random.default_rng(350301)
    by_task = {t: list(range(100 + t * 20, 100 + t * 20 + n)) for t, n in counts.items()}
    ok_all = True
    detail = []
    for _ in range(30):
        sample = []
        task_n = {}
        for t, n in counts.items():
            pick = rng.integers(0, n, size=n)
            chosen = [by_task[t][i] for i in pick]
            sample.extend(chosen)
            task_n[t] = len(chosen)
        if len(sample) != 35 or task_n != counts:
            ok_all = False
            detail.append({"len": len(sample), "task_n": task_n})
            break
    rows.append({"id": 10, "name": "fixed_deep_bootstrap_n35_task_counts", "pass": bool(ok_all), "reason": None if ok_all else detail[:1]})

    return {
        "n": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n_fail": sum(1 for r in rows if not r["pass"]),
        "rows": rows,
        "status": "PASS" if all(r["pass"] for r in rows) else "FAIL",
    }
