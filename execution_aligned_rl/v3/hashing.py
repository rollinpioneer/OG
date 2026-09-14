"""Byte hashing and stable integer helpers for EA-V3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_jsonable(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    header = f"{arr.dtype.str}|{arr.shape}".encode("utf-8")
    return sha256_bytes(header + b"\n" + arr.tobytes())


def stable_uint32(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def sha256_files(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}