
"""Stable runtime identity schema v2. Dynamic CPU frequency is diagnostics-only."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.cpu_s1.protocol import (
    CHECKPOINT_SHA256,
    CPU_ENV,
    DEFAULT_PATHS,
    OGBENCH_COMMIT,
    PROTOCOL_ID,
    TRAIN_SHA256,
)


DYNAMIC_LSCPU_KEYS = (
    "CPU(s) scaling MHz",
    "CPU MHz",
    "BogoMIPS",
)


def apply_cpu_env() -> None:
    for key, value in CPU_ENV.items():
        os.environ[key] = value
    os.environ.pop("XLA_FLAGS", None)


def _pkg(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_lscpu(text: str) -> tuple[dict, str]:
    stable = {}
    keep = {
        "Architecture",
        "CPU op-mode(s)",
        "Byte Order",
        "CPU(s)",
        "Vendor ID",
        "Model name",
        "CPU family",
        "Model",
        "Stepping",
        "Thread(s) per core",
        "Core(s) per socket",
        "Socket(s)",
        "L1d cache",
        "L1i cache",
        "L2 cache",
        "L3 cache",
        "NUMA node(s)",
        "Virtualization",
        "Flags",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key in DYNAMIC_LSCPU_KEYS:
            continue
        if key in keep:
            stable[key] = value
    return stable, text


def collect_identity(*, role: str, include_jax: bool = True) -> tuple[dict, dict]:
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
    cpu_stable, lscpu_raw = parse_lscpu(lscpu)
    try:
        glibc = os.confstr("CS_GNU_LIBC_VERSION")
    except Exception:
        glibc = list(platform.libc_ver())
    compare_v2 = Path(DEFAULT_PATHS["repo"]) / DEFAULT_PATHS["compare_v2"]
    if not compare_v2.exists():
        compare_v2 = Path(__file__).resolve().parents[2] / "runtime_qual" / "compare_v2.py"
    identity = {
        "schema": "runtime_identity_v2",
        "protocol_id": PROTOCOL_ID,
        "role": role,
        "canonical_env": dict(CPU_ENV),
        "applied_env": {k: os.environ.get(k) for k in list(CPU_ENV) + ["XLA_FLAGS"]},
        "no_xla_flags": os.environ.get("XLA_FLAGS") in (None, ""),
        "cpu_stable": cpu_stable,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "glibc": glibc,
        "packages": {
            name: _pkg(name)
            for name in ("jax", "jaxlib", "numpy", "flax", "mujoco", "ogbench", "gymnasium", "dm-control")
        },
        "checkpoint_sha256": sha256_file(Path(DEFAULT_PATHS["checkpoint_file"])),
        "train_sha256": sha256_file(Path(DEFAULT_PATHS["dataset_dir"]) / "cube-double-play-v0.npz"),
        "ogbench_commit": subprocess.run(
            ["git", "-C", DEFAULT_PATHS["official_source"], "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "comparator_contract_v2_sha256": sha256_file(compare_v2),
        "comparator_contract": "v2",
    }
    if include_jax:
        import jax

        identity["jax_default_backend"] = jax.default_backend()
        identity["jax_devices"] = [str(d) for d in jax.devices()]
        identity["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
        identity["jax_default_matmul_precision"] = str(jax.config.jax_default_matmul_precision)
        identity["jax_version"] = jax.__version__
        try:
            import jaxlib

            identity["jaxlib_version"] = jaxlib.__version__
        except Exception:
            identity["jaxlib_version"] = None
    diagnostics = {
        "schema": "runtime_diagnostics_v2",
        "role": role,
        "pid": os.getpid(),
        "lscpu_raw": lscpu_raw,
        "note": "pid, lscpu_raw (including scaling MHz), and timestamps are NOT part of identity SHA256",
    }
    return identity, diagnostics


def identity_payload_for_hash(identity: dict) -> dict:
    skip = {"role"}  # roles differ; hash must match across collector/workers/finalizer
    return {k: v for k, v in identity.items() if k not in skip}


def identity_sha256(identity: dict) -> str:
    text = json.dumps(identity_payload_for_hash(identity), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_identity(path: Path, role: str) -> dict:
    apply_cpu_env()
    identity, diagnostics = collect_identity(role=role, include_jax=True)
    identity["runtime_identity_sha256"] = identity_sha256(identity)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from execution_aligned_rl.v3.serialization import dump_json

    dump_json(path, identity)
    dump_json(path.with_name(path.stem + "_diagnostics.json") if path.suffix == ".json" else path.parent / f"{role}_diagnostics.json", diagnostics)
    return identity


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    apply_cpu_env()
    ident = write_identity(Path(args.out), args.role)
    print(json.dumps({"role": args.role, "sha256": ident["runtime_identity_sha256"], "backend": ident.get("jax_default_backend")}))


if __name__ == "__main__":
    main()
