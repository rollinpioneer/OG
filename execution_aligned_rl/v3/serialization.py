"""Typed JSON + NumPy serialization for root artifacts. Pickle is not the snapshot format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        raise TypeError("ndarray must be stored in .npy/.npz, not JSON")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unserializable type: {type(value)!r}")


def dump_json(path: Path, payload: dict, *, sort_keys: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=sort_keys, default=json_default, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))


def load_npy(path: Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def encode_rng_state(state: Any) -> dict:
    """Encode Gymnasium/NumPy bit_generator or MT19937 get_state()."""
    if isinstance(state, dict) and "bit_generator" in state:
        inner = state.get("state", {})
        encoded_inner = {}
        for key, value in inner.items():
            if isinstance(value, (np.ndarray, list, tuple)):
                encoded_inner[key] = {
                    "kind": "ndarray",
                    "dtype": str(np.asarray(value).dtype),
                    "shape": list(np.asarray(value).shape),
                    "data": np.asarray(value).reshape(-1).tolist(),
                }
            else:
                encoded_inner[key] = {"kind": "scalar", "value": int(value) if isinstance(value, (int, np.integer)) else value}
        return {
            "kind": "bit_generator",
            "bit_generator": state.get("bit_generator"),
            "has_uint32": state.get("has_uint32"),
            "uinteger": None if state.get("uinteger") is None else int(state.get("uinteger")),
            "state": encoded_inner,
        }
    if isinstance(state, tuple) and len(state) >= 3 and state[0] == "MT19937":
        keys = np.asarray(state[1])
        return {
            "kind": "mt19937",
            "keys": keys.astype(np.uint32).tolist(),
            "pos": int(state[2]),
            "has_gauss": int(state[3]) if len(state) > 3 else 0,
            "cached_gaussian": float(state[4]) if len(state) > 4 else 0.0,
        }
    raise TypeError(f"unsupported RNG state: {type(state)!r}")


def decode_rng_state(payload: dict) -> Any:
    if payload["kind"] == "bit_generator":
        inner = {}
        for key, value in payload["state"].items():
            if value["kind"] == "ndarray":
                inner[key] = np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])
            else:
                inner[key] = value["value"]
        state = {
            "bit_generator": payload["bit_generator"],
            "state": inner,
        }
        if "has_uint32" in payload:
            state["has_uint32"] = payload["has_uint32"]
        if "uinteger" in payload:
            state["uinteger"] = payload["uinteger"]
        return state
    if payload["kind"] == "mt19937":
        keys = np.asarray(payload["keys"], dtype=np.uint32)
        return ("MT19937", keys, int(payload["pos"]), int(payload.get("has_gauss", 0)), float(payload.get("cached_gaussian", 0.0)))
    raise ValueError(f"unknown RNG payload kind {payload.get('kind')}")


def encode_nested(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.reshape(-1).tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode_nested(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_nested(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"__repr__": repr(value), "__type__": type(value).__name__}


def decode_nested(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__"):
        return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])
    if isinstance(value, dict):
        return {k: decode_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_nested(v) for v in value]
    return value