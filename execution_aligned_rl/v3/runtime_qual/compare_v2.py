
"""Comparator contract v2. Does not replace or rewrite S1 compare.py."""

from __future__ import annotations

from typing import Any

import numpy as np

from execution_aligned_rl.v3.runtime_qual.protocol import THRESHOLDS

NOT_APPLICABLE = "NOT_APPLICABLE"


def _as_array(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    return np.asarray(value)


def _finite(array: np.ndarray) -> bool:
    if array.size == 0:
        return False
    return bool(np.isfinite(array).all())


def max_abs_diff(a, b) -> tuple[str | float, str | None]:
    aa = _as_array(a)
    bb = _as_array(b)
    if aa.size == 0 and bb.size == 0:
        if aa.shape != bb.shape:
            return float("nan"), "SHAPE_MISMATCH"
        return 0.0, None
    if aa.size == 0 or bb.size == 0:
        return float("nan"), "EMPTY"
    if aa.shape != bb.shape:
        return float("nan"), "SHAPE_MISMATCH"
    if not _finite(aa) or not _finite(bb):
        return float("nan"), "NONFINITE"
    return float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64)))), None


def _pass_numeric(diff: str | float, error: str | None, limit: float) -> bool:
    if error is not None:
        return False
    return bool(diff <= limit)


def _identity(trace: dict, key: str):
    ident = (trace or {}).get("identity") or {}
    return ident.get(key, (trace or {}).get(key))


def _missing(value) -> bool:
    return value is None


def compare_states_v2(state_a: dict, state_b: dict, thresholds: dict | None = None, *, phase: str = "D0") -> dict:
    th = dict(THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    if float(th["rtol"]) != 0.0:
        raise ValueError("protocol requires rtol=0")
    phase = str(phase).upper()
    fields = {}
    failures = []

    def measure(name: str, left, right, limit: float, discrete: bool = False):
        if left is None and right is None:
            fields[name] = {"status": NOT_APPLICABLE, "reason": "both_none"}
            return
        if left is None or right is None:
            fields[name] = {"status": "FAIL", "reason": "MISSING"}
            failures.append(name)
            return
        if discrete:
            ok = left == right
            fields[name] = {"status": "PASS" if ok else "FAIL", "agree": bool(ok), "left": left, "right": right}
            if not ok:
                failures.append(name)
            return
        diff, error = max_abs_diff(left, right)
        ok = _pass_numeric(diff, error, limit)
        fields[name] = {
            "status": "PASS" if ok else "FAIL",
            "max_abs_diff": diff if error is None else None,
            "error": error,
            "limit": limit,
            "rtol": 0.0,
        }
        if not ok:
            failures.append(name)

    measure("integration", state_a.get("integration"), state_b.get("integration"), th["integration_continuous_max_abs"])
    measure("qpos", state_a.get("qpos"), state_b.get("qpos"), th["qpos_max_abs"])
    measure("qvel", state_a.get("qvel"), state_b.get("qvel"), th["qvel_max_abs"])
    measure("act", state_a.get("act"), state_b.get("act"), th["act_max_abs"])
    measure("ctrl", state_a.get("ctrl"), state_b.get("ctrl"), th["ctrl_max_abs"])
    measure("warmstart", state_a.get("warmstart"), state_b.get("warmstart"), th["warmstart_max_abs"])
    measure("observation", state_a.get("observation"), state_b.get("observation"), th["observation_max_abs"])
    obs_a, obs_b = state_a.get("observation"), state_b.get("observation")
    if obs_a is not None and obs_b is not None:
        aa, bb = _as_array(obs_a), _as_array(obs_b)
        if aa.size >= 37 and bb.size >= 37:
            measure("robot_obs_0_19", aa[:19], bb[:19], th["robot_obs_0_19_max_abs"])
            measure("cube_obs_19_37", aa[19:37], bb[19:37], th["cube_obs_19_37_max_abs"])
        else:
            fields["robot_obs_0_19"] = {"status": "FAIL", "reason": "OBS_DIM"}
            fields["cube_obs_19_37"] = {"status": "FAIL", "reason": "OBS_DIM"}
            failures.extend(["robot_obs_0_19", "cube_obs_19_37"])
    measure("goal_observation", state_a.get("goal_observation"), state_b.get("goal_observation"), th["goal_encoding_max_abs"])
    measure("elapsed_steps", state_a.get("elapsed_steps"), state_b.get("elapsed_steps"), 0.0, discrete=True)
    measure("success", state_a.get("success"), state_b.get("success"), 0.0, discrete=True)
    measure("task_id", state_a.get("task_id"), state_b.get("task_id"), 0.0, discrete=True)

    if phase == "D0":
        fields["action"] = {"status": NOT_APPLICABLE, "reason": "D0_no_action"}
        fields["proxy"] = {"status": NOT_APPLICABLE, "reason": "D0_no_rollout"}
        fields["full_proxy"] = {"status": NOT_APPLICABLE, "reason": "D0_no_rollout"}
        for name in ("terminated", "truncated"):
            left, right = state_a.get(name), state_b.get(name)
            if _missing(left) or _missing(right):
                fields[name] = {"status": NOT_APPLICABLE, "reason": "D0_unified_NA_when_not_both_explicit", "left": left, "right": right}
            else:
                ok = bool(left) == bool(right) if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)) else left == right
                # explicit agreement: keep original equality, also True==1.0
                ok = left == right
                fields[name] = {"status": "PASS" if ok else "FAIL", "agree": bool(ok), "left": left, "right": right}
                if not ok:
                    failures.append(name)
    else:
        measure("action", state_a.get("action"), state_b.get("action"), th["action_max_abs"])
        measure("terminated", state_a.get("terminated"), state_b.get("terminated"), 0.0, discrete=True)
        measure("truncated", state_a.get("truncated"), state_b.get("truncated"), 0.0, discrete=True)
        if phase == "D1":
            fields["proxy"] = {"status": NOT_APPLICABLE, "reason": "D1_step_proxy_unified_NA"}
            fields["full_proxy"] = {"status": NOT_APPLICABLE, "reason": "D1_full_proxy_unified_NA"}
        else:
            measure("proxy", state_a.get("step_proxy", state_a.get("proxy")), state_b.get("step_proxy", state_b.get("proxy")), th["proxy_max_abs"])

    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "phase": phase,
        "contract": "comparator_contract_v2",
        "failures": failures,
        "fields": fields,
        "first_divergent_step": None if status == "PASS" else 0,
        "rtol": 0.0,
    }


def compare_traces_v2(trace_a: dict, trace_b: dict, thresholds: dict | None = None, *, phase: str = "D3") -> dict:
    th = dict(THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    if float(th["rtol"]) != 0.0:
        raise ValueError("protocol requires rtol=0")
    phase = str(phase).upper()
    reasons = []
    if not trace_a or not trace_b:
        reasons.append("EMPTY_TRACE_OBJECT")
    steps_a = list((trace_a or {}).get("steps") or [])
    steps_b = list((trace_b or {}).get("steps") or [])
    if (trace_a or {}).get("status") == "EMPTY" or (trace_b or {}).get("status") == "EMPTY":
        reasons.append("EMPTY_TRACE_STATUS")
    if len(steps_a) == 0 or len(steps_b) == 0:
        reasons.append("EMPTY_STEPS")
    if len(steps_a) != len(steps_b):
        reasons.append("LENGTH_MISMATCH")

    for key in ("root_id", "task_id", "goal_sha256", "probe_goal_sha256", "protocol_id"):
        va, vb = _identity(trace_a or {}, key), _identity(trace_b or {}, key)
        if va is not None and vb is not None and va != vb:
            reasons.append(f"IDENTITY_{key}")

    goal_a = (trace_a or {}).get("goal_observation")
    goal_b = (trace_b or {}).get("goal_observation")
    if goal_a is not None and goal_b is not None:
        diff, error = max_abs_diff(goal_a, goal_b)
        if error is not None or diff > th["goal_encoding_max_abs"]:
            reasons.append("GOAL_ENCODING")

    elapsed_a = (trace_a or {}).get("elapsed_steps_start")
    elapsed_b = (trace_b or {}).get("elapsed_steps_start")
    if elapsed_a is not None and elapsed_b is not None and elapsed_a != elapsed_b:
        reasons.append("ELAPSED_START")

    if (trace_a or {}).get("continued_after_terminal") or (trace_b or {}).get("continued_after_terminal"):
        reasons.append("TERMINATION_PROTOCOL")

    measured_steps = []
    first = None
    comparable = len(steps_a) > 0 and len(steps_b) > 0 and len(steps_a) == len(steps_b) and not reasons
    if comparable:
        for idx, (sa, sb) in enumerate(zip(steps_a, steps_b)):
            row_fail = []
            row = {"step": idx}

            def take(name, left, right, limit, discrete=False):
                if left is None or right is None:
                    row[name] = {"status": "FAIL", "reason": "MISSING"}
                    row_fail.append(name)
                    return
                if discrete:
                    ok = bool(left == right)
                    row[name] = {"status": "PASS" if ok else "FAIL", "agree": ok}
                    if not ok:
                        row_fail.append(name)
                    return
                diff, error = max_abs_diff(left, right)
                ok = _pass_numeric(diff, error, limit)
                row[name] = {"status": "PASS" if ok else "FAIL", "max_abs_diff": None if error else diff, "error": error, "limit": limit}
                if not ok:
                    row_fail.append(name)

            take("action", sa.get("action"), sb.get("action"), th["action_max_abs"])
            take("observation", sa.get("observation"), sb.get("observation"), th["observation_max_abs"])
            obs_a, obs_b = _as_array(sa.get("observation")), _as_array(sb.get("observation"))
            if obs_a.size >= 37 and obs_b.size >= 37:
                take("robot_obs_0_19", obs_a[:19], obs_b[:19], th["robot_obs_0_19_max_abs"])
                take("cube_obs_19_37", obs_a[19:37], obs_b[19:37], th["cube_obs_19_37_max_abs"])
            else:
                row["robot_obs_0_19"] = {"status": "FAIL", "reason": "OBS_DIM"}
                row["cube_obs_19_37"] = {"status": "FAIL", "reason": "OBS_DIM"}
                row_fail.extend(["robot_obs_0_19", "cube_obs_19_37"])
            take("integration", sa.get("integration"), sb.get("integration"), th["integration_continuous_max_abs"])
            take("qpos", sa.get("qpos"), sb.get("qpos"), th["qpos_max_abs"])
            take("qvel", sa.get("qvel"), sb.get("qvel"), th["qvel_max_abs"])
            take("act", sa.get("act"), sb.get("act"), th["act_max_abs"])
            take("ctrl", sa.get("ctrl"), sb.get("ctrl"), th["ctrl_max_abs"])
            take("warmstart", sa.get("warmstart"), sb.get("warmstart"), th["warmstart_max_abs"])
            take("reward", sa.get("reward"), sb.get("reward"), 0.0)
            take("success", sa.get("success"), sb.get("success"), 0.0, discrete=True)
            take("terminated", sa.get("terminated"), sb.get("terminated"), 0.0, discrete=True)
            take("truncated", sa.get("truncated"), sb.get("truncated"), 0.0, discrete=True)
            take("elapsed_steps", sa.get("elapsed_steps"), sb.get("elapsed_steps"), 0.0, discrete=True)
            if phase == "D1":
                row["proxy"] = {"status": NOT_APPLICABLE, "reason": "D1_step_proxy_unified_NA"}
            else:
                if sa.get("step_proxy") is None and sb.get("step_proxy") is None:
                    row["proxy"] = {"status": NOT_APPLICABLE, "reason": "not_stored_per_step"}
                else:
                    take("proxy", sa.get("step_proxy"), sb.get("step_proxy"), th["proxy_max_abs"])
            row["failures"] = row_fail
            measured_steps.append(row)
            if row_fail and first is None:
                first = idx

        if phase == "D1":
            full_proxy_field = {"status": NOT_APPLICABLE, "reason": "D1_full_proxy_unified_NA"}
        else:
            proxy_a = (trace_a or {}).get("full_proxy")
            proxy_b = (trace_b or {}).get("full_proxy")
            if proxy_a is None and proxy_b is None:
                full_proxy_field = {"status": NOT_APPLICABLE, "reason": "both_none"}
            elif proxy_a is None or proxy_b is None:
                full_proxy_field = {"status": "FAIL", "reason": "MISSING"}
                reasons.append("FULL_PROXY")
                if first is None:
                    first = 0
            else:
                diff, error = max_abs_diff(proxy_a, proxy_b)
                ok = _pass_numeric(diff, error, th["proxy_max_abs"])
                full_proxy_field = {"status": "PASS" if ok else "FAIL", "max_abs_diff": None if error else diff, "error": error, "limit": th["proxy_max_abs"]}
                if not ok:
                    reasons.append("FULL_PROXY")
                    if first is None:
                        first = 0
    else:
        first = 0
        full_proxy_field = {"status": "FAIL", "reason": "not_comparable"}

    passed = comparable and first is None and not reasons
    if not passed and first is None:
        first = 0
    return {
        "status": "PASS" if passed else "FAIL",
        "phase": phase,
        "contract": "comparator_contract_v2",
        "reasons": reasons,
        "n_steps_a": len(steps_a),
        "n_steps_b": len(steps_b),
        "first_divergent_step": None if passed else first,
        "steps": measured_steps,
        "full_proxy": full_proxy_field,
        "rtol": 0.0,
    }
