
"""Collect and hash the canonical CPU runtime lock. No extra XLA flags."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.cpu_adoption.protocol import (
    CANONICAL_CPU_ENV,
    CHECKPOINT_SHA256,
    DEFAULT_PATHS,
    EXPECTED,
    OGBENCH_COMMIT,
    TRAIN_SHA256,
)
from execution_aligned_rl.v3.serialization import dump_json


def apply_canonical_cpu_env() -> None:
    for key, value in CANONICAL_CPU_ENV.items():
        if key == "JAX_PLATFORMS" or key == "CUDA_VISIBLE_DEVICES" or key.endswith("THREADS") or key == "JAX_ENABLE_X64":
            os.environ[key] = value
    os.environ.pop("XLA_FLAGS", None)


def _pkg(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cmd(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def collect_lock(*, include_jax: bool = True) -> dict:
    lock = {
        "canonical_cpu_env": dict(CANONICAL_CPU_ENV),
        "applied_env": {k: os.environ.get(k) for k in list(CANONICAL_CPU_ENV) + ["XLA_FLAGS"]},
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "glibc": None,
        "lscpu": _cmd(["lscpu"]),
        "packages": {name: _pkg(name) for name in ("python", "jax", "jaxlib", "numpy", "flax", "mujoco", "ogbench", "gymnasium", "dm-control")},
        "checkpoint_file": DEFAULT_PATHS["checkpoint_file"],
        "checkpoint_sha256": sha256_file(Path(DEFAULT_PATHS["checkpoint_file"])),
        "train_sha256": sha256_file(Path(DEFAULT_PATHS["dataset_dir"]) / "cube-double-play-v0.npz"),
        "ogbench_commit": _cmd(["git", "-C", DEFAULT_PATHS["official_source"], "rev-parse", "HEAD"]),
        "no_xla_flags": os.environ.get("XLA_FLAGS") in (None, ""),
    }
    try:
        import os as _os

        lock["glibc"] = _os.confstr("CS_GNU_LIBC_VERSION")
    except Exception:
        lock["glibc"] = platform.libc_ver()
    if include_jax:
        import jax

        lock["jax_default_backend"] = jax.default_backend()
        lock["jax_devices"] = [str(d) for d in jax.devices()]
        lock["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
        lock["jax_default_matmul_precision"] = str(jax.config.jax_default_matmul_precision)
        lock["jax_version"] = jax.__version__
        try:
            import jaxlib

            lock["jaxlib_version"] = jaxlib.__version__
        except Exception:
            lock["jaxlib_version"] = None
    return lock


def canonical_lock_for_hash(lock: dict) -> dict:
    drop = {"lscpu"}  # keep lscpu in file but hash a stable subset plus lscpu text
    payload = {
        "canonical_cpu_env": lock["canonical_cpu_env"],
        "python": lock["python"],
        "platform": lock["platform"],
        "glibc": lock["glibc"],
        "packages": lock["packages"],
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "train_sha256": lock["train_sha256"],
        "ogbench_commit": lock["ogbench_commit"],
        "jax_default_backend": lock.get("jax_default_backend"),
        "jax_devices": lock.get("jax_devices"),
        "jax_enable_x64": lock.get("jax_enable_x64"),
        "jax_version": lock.get("jax_version"),
        "jaxlib_version": lock.get("jaxlib_version"),
        "no_xla_flags": lock.get("no_xla_flags"),
        "lscpu": lock.get("lscpu"),
        "executable": lock.get("executable"),
    }
    return payload


def lock_sha256(lock: dict) -> str:
    text = json.dumps(canonical_lock_for_hash(lock), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_lock(lock: dict) -> list[str]:
    holds = []
    if lock.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        holds.append("checkpoint_sha256_mismatch")
    if lock.get("ogbench_commit") != OGBENCH_COMMIT:
        holds.append("ogbench_commit_mismatch")
    if lock.get("train_sha256") != TRAIN_SHA256:
        holds.append("train_sha256_mismatch")
    if lock.get("jax_enable_x64") not in (False, None) and lock.get("jax_enable_x64") is True:
        holds.append("jax_enable_x64_true")
    if lock.get("jax_default_backend") not in (None, "cpu"):
        holds.append(f"backend_{lock.get('jax_default_backend')}")
    env = lock.get("applied_env") or {}
    for key, expected in CANONICAL_CPU_ENV.items():
        if key == "JAX_ENABLE_X64":
            continue
        if str(env.get(key, "")) != str(expected):
            holds.append(f"env_{key}")
    if env.get("XLA_FLAGS") not in (None, ""):
        holds.append("unexpected_xla_flags")
    pkgs = lock.get("packages") or {}
    for name, expected in (("jax", EXPECTED["jax"]), ("jaxlib", EXPECTED["jaxlib"]), ("numpy", EXPECTED["numpy"]), ("flax", EXPECTED["flax"]), ("mujoco", EXPECTED["mujoco"]), ("ogbench", EXPECTED["ogbench"]), ("gymnasium", EXPECTED["gymnasium"])):
        if pkgs.get(name) != expected:
            holds.append(f"package_{name}:{pkgs.get(name)}!={expected}")
    if not str(lock.get("python", "")).startswith(EXPECTED["python_version_prefix"]):
        holds.append("python_version")
    return holds
