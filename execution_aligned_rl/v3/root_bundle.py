"""Materialized root bundle IO. Paths are the source of truth; hashes are verified on load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from execution_aligned_rl.v3.hashing import sha256_array, sha256_file
from execution_aligned_rl.v3.serialization import decode_nested, dump_json, encode_nested, load_json, load_npy, save_npy

ARRAY_FILES = (
    "initial_observation.npy",
    "goal_observation.npy",
    "bootstrap_integration.npy",
    "prefix_actions.npy",
    "decision_observation.npy",
    "decision_integration.npy",
    "d3_target_observation.npy",
)


def root_dir(store: Path, protocol_id: str, root_id: int) -> Path:
    return Path(store) / protocol_id / f"{int(root_id):06d}"


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key].copy() for key in handle.files}


def save_python_state(directory: Path, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
            meta[key] = {"kind": "ndarray"}
        else:
            meta[key] = {"kind": "json", "value": encode_nested(value)}
    dump_json(directory / "state.json", meta)
    if arrays:
        _write_npz(directory / "arrays.npz", arrays)


def load_python_state(directory: Path) -> dict[str, Any]:
    meta = load_json(directory / "state.json")
    arrays = _read_npz(directory / "arrays.npz") if (directory / "arrays.npz").exists() else {}
    out = {}
    for key, spec in meta.items():
        if spec.get("kind") == "ndarray":
            out[key] = arrays[key]
        else:
            out[key] = decode_nested(spec.get("value"))
    return out


def save_trace_npz(path: Path, trace: dict) -> None:
    steps = trace.get("steps") or []
    payload = {
        "n_steps": np.asarray([len(steps)]),
        "full_proxy": np.asarray([np.nan if trace.get("full_proxy") is None else float(trace["full_proxy"])]),
    }
    if steps:
        payload["actions"] = np.stack([np.asarray(s["action"]) for s in steps])
        payload["observations"] = np.stack([np.asarray(s["observation"]) for s in steps])
        payload["rewards"] = np.asarray([s["reward"] for s in steps], dtype=np.float64)
        payload["success"] = np.asarray([s["success"] for s in steps], dtype=np.float64)
        payload["terminated"] = np.asarray([s["terminated"] for s in steps], dtype=np.bool_)
        payload["truncated"] = np.asarray([s["truncated"] for s in steps], dtype=np.bool_)
        payload["elapsed_steps"] = np.asarray([
            (s.get("elapsed_steps") if s.get("elapsed_steps") is not None else -1) for s in steps
        ], dtype=np.int64)
        if steps[0].get("integration") is not None:
            payload["integration"] = np.stack([np.asarray(s["integration"]) for s in steps])
        payload["qpos"] = np.stack([np.asarray(s["qpos"]) for s in steps])
        payload["qvel"] = np.stack([np.asarray(s["qvel"]) for s in steps])
        payload["ctrl"] = np.stack([np.asarray(s["ctrl"]) for s in steps])
        payload["warmstart"] = np.stack([np.asarray(s["warmstart"]) for s in steps])
        if steps[0].get("act") is not None:
            payload["act"] = np.stack([np.asarray(s["act"]) for s in steps])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    dump_json(path.with_suffix(".json"), {
        "identity": trace.get("identity"),
        "status": trace.get("status"),
        "n_steps": len(steps),
        "full_proxy": trace.get("full_proxy"),
        "elapsed_steps_start": trace.get("elapsed_steps_start"),
        "continued_after_terminal": trace.get("continued_after_terminal", False),
        "key_schedule": trace.get("key_schedule"),
        "goal_sha256": trace.get("goal_sha256"),
        "probe_goal_sha256": trace.get("probe_goal_sha256"),
    })


def load_trace(path: Path) -> dict:
    path = Path(path)
    meta = load_json(path.with_suffix(".json")) if path.with_suffix(".json").exists() else {}
    with np.load(path, allow_pickle=False) as handle:
        n = int(handle["n_steps"][0]) if "n_steps" in handle.files else 0
        steps = []
        for i in range(n):
            step = {
                "action": handle["actions"][i].copy() if "actions" in handle.files else None,
                "observation": handle["observations"][i].copy() if "observations" in handle.files else None,
                "reward": float(handle["rewards"][i]) if "rewards" in handle.files else None,
                "success": handle["success"][i].item() if "success" in handle.files else None,
                "terminated": bool(handle["terminated"][i]) if "terminated" in handle.files else None,
                "truncated": bool(handle["truncated"][i]) if "truncated" in handle.files else None,
                "elapsed_steps": int(handle["elapsed_steps"][i]) if "elapsed_steps" in handle.files and handle["elapsed_steps"].shape[0] == n else None,
                "integration": handle["integration"][i].copy() if "integration" in handle.files else None,
                "qpos": handle["qpos"][i].copy() if "qpos" in handle.files else None,
                "qvel": handle["qvel"][i].copy() if "qvel" in handle.files else None,
                "act": handle["act"][i].copy() if "act" in handle.files else None,
                "ctrl": handle["ctrl"][i].copy() if "ctrl" in handle.files else None,
                "warmstart": handle["warmstart"][i].copy() if "warmstart" in handle.files else None,
            }
            if "elapsed_steps" in handle.files and handle["elapsed_steps"].shape[0] == n:
                val = int(handle["elapsed_steps"][i])
                step["elapsed_steps"] = None if val < 0 else val
            steps.append(step)
        full_proxy = float(handle["full_proxy"][0]) if "full_proxy" in handle.files else None
        if full_proxy is not None and np.isnan(full_proxy):
            full_proxy = None
    out = dict(meta)
    out["steps"] = steps
    out["full_proxy"] = full_proxy if out.get("full_proxy") is None else out.get("full_proxy")
    return out


def write_root_bundle(directory: Path, bundle: dict) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_npy(directory / "initial_observation.npy", bundle["initial_observation"])
    save_npy(directory / "goal_observation.npy", bundle["goal_observation"])
    save_npy(directory / "bootstrap_integration.npy", bundle["bootstrap_integration"])
    save_npy(directory / "prefix_actions.npy", bundle["prefix_actions"])
    save_npy(directory / "decision_observation.npy", bundle["decision_observation"])
    save_npy(directory / "decision_integration.npy", bundle["decision_integration"])
    save_npy(directory / "d3_target_observation.npy", bundle["d3_target_observation"])
    save_python_state(directory / "bootstrap_python_state", bundle["bootstrap_python_state"])
    save_python_state(directory / "decision_python_state", bundle["decision_python_state"])
    _write_npz(directory / "bootstrap_observation_stage.npz", bundle["bootstrap_observation_stage"])
    _write_npz(directory / "decision_observation_stage.npz", bundle["decision_observation_stage"])
    dump_json(directory / "task_state.json", bundle["task_state"])
    dump_json(directory / "root_manifest.json", bundle["manifest"])
    if bundle.get("prefix_trace") is not None:
        save_trace_npz(directory / "prefix_trace.npz", bundle["prefix_trace"])
    if bundle.get("original_live_probe") is not None:
        save_trace_npz(directory / "original_live_probe.npz", bundle["original_live_probe"])
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "file_hashes.json":
            hashes[str(path.relative_to(directory)).replace("\\", "/")] = sha256_file(path)
    dump_json(directory / "file_hashes.json", hashes)
    hashes["file_hashes.json"] = sha256_file(directory / "file_hashes.json")
    return hashes


def read_root_bundle(directory: Path, *, verify: bool = True) -> dict:
    directory = Path(directory)
    recorded = load_json(directory / "file_hashes.json")
    if verify:
        for rel, digest in recorded.items():
            if rel == "file_hashes.json":
                continue
            path = directory / rel
            if not path.exists():
                raise FileNotFoundError(f"missing artifact {rel}")
            actual = sha256_file(path)
            if actual != digest:
                raise ValueError(f"hash mismatch for {rel}: {actual} != {digest}")
    bundle = {
        "directory": directory,
        "manifest": load_json(directory / "root_manifest.json"),
        "task_state": load_json(directory / "task_state.json"),
        "file_hashes": recorded,
        "initial_observation": load_npy(directory / "initial_observation.npy"),
        "goal_observation": load_npy(directory / "goal_observation.npy"),
        "bootstrap_integration": load_npy(directory / "bootstrap_integration.npy"),
        "prefix_actions": load_npy(directory / "prefix_actions.npy"),
        "decision_observation": load_npy(directory / "decision_observation.npy"),
        "decision_integration": load_npy(directory / "decision_integration.npy"),
        "d3_target_observation": load_npy(directory / "d3_target_observation.npy"),
        "bootstrap_python_state": load_python_state(directory / "bootstrap_python_state"),
        "decision_python_state": load_python_state(directory / "decision_python_state"),
        "bootstrap_observation_stage": _read_npz(directory / "bootstrap_observation_stage.npz"),
        "decision_observation_stage": _read_npz(directory / "decision_observation_stage.npz"),
    }
    if (directory / "prefix_trace.npz").exists():
        bundle["prefix_trace"] = load_trace(directory / "prefix_trace.npz")
    if (directory / "original_live_probe.npz").exists():
        bundle["original_live_probe"] = load_trace(directory / "original_live_probe.npz")
    bundle["goal_sha256"] = sha256_array(bundle["goal_observation"])
    bundle["d3_target_sha256"] = sha256_array(bundle["d3_target_observation"])
    return bundle