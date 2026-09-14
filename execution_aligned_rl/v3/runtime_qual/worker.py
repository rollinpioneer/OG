
"""Fresh-process actor/value worker. Runtime env vars must be set before importing JAX."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def _maybe_configure_runtime(runtime: str) -> dict:
    applied = {}
    if runtime == "CPU_SINGLE_THREAD":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        applied = {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    elif runtime == "GPU_CURRENT":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.pop("JAX_PLATFORMS", None)
        applied = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        }
    else:
        raise ValueError(runtime)
    return applied


def capture_runtime_env():
    import jax

    info = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }
    try:
        import jax.extend.backend as jb  # type: ignore

        cache = getattr(jb, "get_compilation_cache_dir", None)
        info["jax_compilation_cache_dir_api"] = None if cache is None else str(cache())
    except Exception as exc:
        info["jax_compilation_cache_dir_api"] = f"UNAVAILABLE:{type(exc).__name__}"
    try:
        import jaxlib

        info["jaxlib_version"] = getattr(jaxlib, "__version__", None)
    except Exception:
        info["jaxlib_version"] = None
    try:
        import jax as jax_mod

        info["jax_version"] = jax_mod.__version__
    except Exception:
        info["jax_version"] = None
    try:
        import subprocess

        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        info["nvidia_smi"] = smi.stdout.strip() if smi.returncode == 0 else smi.stderr.strip()
    except Exception as exc:
        info["nvidia_smi"] = f"UNAVAILABLE:{type(exc).__name__}"
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, choices=["GPU_CURRENT", "CPU_SINGLE_THREAD"])
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--official-source", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()
    applied = _maybe_configure_runtime(args.runtime)

    import numpy as np
    import jax

    from execution_aligned_rl.v3.hashing import sha256_array
    from execution_aligned_rl.v3.policy import action_for, load_agent, value_for
    from execution_aligned_rl.v3.runtime_qual.protocol import DEFAULT_PATHS
    from execution_aligned_rl.v3.serialization import dump_json

    corpus_dir = Path(args.corpus_dir)
    actor = np.load(corpus_dir / "actor_queries.npz", allow_pickle=False)
    value = np.load(corpus_dir / "value_queries.npz", allow_pickle=False)
    env = capture_runtime_env()
    env["applied_env"] = applied
    agent, config, _train = load_agent(
        args.official_source or DEFAULT_PATHS["official_source"],
        args.checkpoint_dir or DEFAULT_PATHS["checkpoint_dir"],
        args.dataset_dir or DEFAULT_PATHS["dataset_dir"],
    )
    actor_rows = []
    for i in range(len(actor["observation"])):
        obs = np.asarray(actor["observation"][i])
        goal = np.asarray(actor["goal"][i])
        key = jax.random.PRNGKey(int(actor["key_uint32"][i]))
        action = np.asarray(action_for(agent, obs, goal, key), dtype=np.float64)
        actor_rows.append(
            {
                "index": i,
                "root_id": int(actor["root_id"][i]),
                "step": int(actor["step"][i]),
                "action": action.tolist(),
                "action_sha256": sha256_array(action),
                "finite": bool(np.isfinite(action).all()),
                "in_bounds": bool(np.all(action >= -1.0) and np.all(action <= 1.0)),
                "max_abs": float(np.max(np.abs(action))),
            }
        )
    value_rows = []
    for i in range(len(value["observation"])):
        obs = np.asarray(value["observation"][i])
        goal = np.asarray(value["goal"][i])
        val = np.asarray(value_for(agent, obs, goal), dtype=np.float64).reshape(-1)
        value_rows.append(
            {
                "index": i,
                "root_id": int(value["root_id"][i]),
                "value": val.tolist(),
                "value_sha256": sha256_array(val),
                "finite": bool(np.isfinite(val).all()),
                "max_abs": float(np.max(np.abs(val))),
            }
        )
    out = {
        "runtime": args.runtime,
        "process_id": int(args.process_id),
        "pid": os.getpid(),
        "env": env,
        "actor": actor_rows,
        "value": value_rows,
        "n_actor": len(actor_rows),
        "n_value": len(value_rows),
        "env_step_calls": 0,
        "dist_mode_used": False,
        "checkpoint_modified": False,
    }
    dump_json(Path(args.out), out)


if __name__ == "__main__":
    main()
