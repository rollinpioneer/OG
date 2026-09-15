"""Load S3 traces and compute history-compatible J_h. No environment interaction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from execution_aligned_rl.v3.root_bundle import load_trace
from execution_aligned_rl.v3.s3_postmortem.protocol import GAMMA, S3_TRACES


class IncompleteTraceError(ValueError):
    pass


class InvalidTraceError(ValueError):
    pass


def trace_path(worker: str, kind: str, root_id: int, cand_id: int | None = None) -> Path:
    if kind in ("short", "deep", "deep_prefix"):
        return S3_TRACES / worker / kind / f"{int(root_id)}_{int(cand_id)}.npz"
    if kind == "direct":
        return S3_TRACES / worker / "direct" / f"{int(root_id)}.npz"
    raise ValueError(kind)


def load_branch(worker: str, kind: str, root_id: int, cand_id: int | None = None) -> dict:
    path = trace_path(worker, kind, root_id, cand_id)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return load_trace(path)


def arrays_from_trace(trace: dict) -> dict:
    steps = trace.get("steps") or []
    if not steps:
        return {
            "T": 0,
            "rewards": np.zeros((0,), dtype=np.float64),
            "success": np.zeros((0,), dtype=bool),
            "terminated": np.zeros((0,), dtype=bool),
            "truncated": np.zeros((0,), dtype=bool),
            "observations": np.zeros((0, 37), dtype=np.float64),
            "actions": np.zeros((0, 5), dtype=np.float64),
        }
    obs = []
    acts = []
    rewards = []
    success = []
    terminated = []
    truncated = []
    for i, step in enumerate(steps):
        if step.get("observation") is None or step.get("action") is None or step.get("reward") is None:
            raise InvalidTraceError(f"missing field at step {i}")
        o = np.asarray(step["observation"], dtype=np.float64)
        a = np.asarray(step["action"], dtype=np.float64)
        if o.size == 0 or a.size == 0 or not np.isfinite(o).all() or not np.isfinite(a).all():
            raise InvalidTraceError(f"empty/NaN array at step {i}")
        if not np.isfinite(float(step["reward"])):
            raise InvalidTraceError(f"nonfinite reward at step {i}")
        obs.append(o)
        acts.append(a)
        rewards.append(float(step["reward"]))
        success.append(bool(step.get("success")))
        terminated.append(bool(step.get("terminated")))
        truncated.append(bool(step.get("truncated")))
    return {
        "T": len(steps),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "success": np.asarray(success, dtype=bool),
        "terminated": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "observations": np.stack(obs),
        "actions": np.stack(acts),
        "identity": trace.get("identity") or {},
        "full_proxy": trace.get("full_proxy"),
    }


def compute_Jh(arr: dict, h: int, value_at, goal: np.ndarray | None, gamma: float = GAMMA) -> dict:
    """History-compatible J_h: replicate S3 compute_proxy tail mask (term/trunc only)."""
    T = int(arr["T"])
    if h < 0:
        raise ValueError("h must be >= 0")
    if T < h:
        if T == 0:
            raise IncompleteTraceError("empty trace shorter than h")
        ended = bool(arr["terminated"][T - 1] or arr["truncated"][T - 1] or bool(arr["success"].any()))
        if not ended:
            raise IncompleteTraceError("trace shorter than h without success/terminated/truncated")
        u = T
    else:
        u = h
    rewards = arr["rewards"][:u]
    A = float(np.sum((gamma ** np.arange(u)) * (rewards - 1.0))) if u else 0.0
    if u == 0:
        terminated = truncated = False
        o_u = None
    else:
        terminated = bool(arr["terminated"][u - 1])
        truncated = bool(arr["truncated"][u - 1])
        o_u = arr["observations"][u - 1]
    b = 0.0 if (terminated or truncated) else 1.0
    success_without_done = bool(u and arr["success"][u - 1] and not terminated and not truncated)
    B = 0.0
    queried = False
    if b == 1.0:
        if o_u is None or goal is None:
            raise InvalidTraceError("tail value required but observation/goal missing")
        B = float((gamma ** u) * float(value_at(o_u, goal)))
        queried = True
    return {
        "h": int(h),
        "u": int(u),
        "T": T,
        "A": A,
        "B": B,
        "J": A + B,
        "b": b,
        "terminated": terminated,
        "truncated": truncated,
        "any_success_prefix": bool(arr["success"][:u].any()) if u else False,
        "success_without_done": success_without_done,
        "value_queried": queried,
        "contract": "history_compute_proxy_term_trunc_only",
    }


def known_outcome_score(arr: dict, h: int) -> int:
    T = int(arr["T"])
    u = min(h, T)
    if u <= 0:
        return 0
    if bool(arr["success"][:u].any()):
        return 1
    if bool(arr["terminated"][:u].any()) or bool(arr["truncated"][:u].any()):
        return -1
    return 0


def unresolved_at_h(arr: dict, h: int) -> bool:
    T = int(arr["T"])
    u = min(h, T)
    if u <= 0:
        return True
    if bool(arr["success"][:u].any()) or bool(arr["terminated"][:u].any()) or bool(arr["truncated"][:u].any()):
        return False
    if T < h:
        return False
    return True
