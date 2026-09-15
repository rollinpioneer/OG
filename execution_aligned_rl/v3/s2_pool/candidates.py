
"""Exact DATA_REAL k=20 candidate retrieval. No ANN, prior, value, or env rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import episode_bounds, sha256_file
from execution_aligned_rl.v3.hashing import sha256_array
from execution_aligned_rl.v3.root_bundle import read_root_bundle, root_dir
from execution_aligned_rl.v3.s2_pool.protocol import DEFAULT_PATHS, PROTOCOL_ID
from execution_aligned_rl.v3.serialization import dump_json, load_json


def legal_starts(terminals: np.ndarray, k: int = 20) -> np.ndarray:
    bounds = episode_bounds(terminals)
    starts = []
    for s, e in bounds:
        n = e - s
        for i in range(max(0, n - k)):
            starts.append(s + i)
    return np.asarray(starts, dtype=np.int64)


def retrieve(exp: Path) -> dict:
    repo = Path(DEFAULT_PATHS["repo"])
    train = np.load(Path(DEFAULT_PATHS["dataset_dir"]) / "cube-double-play-v0.npz", allow_pickle=False)
    obs = np.asarray(train["observations"])
    starts = legal_starts(train["terminals"], 20)
    endpoints = starts + 20
    norm = np.load(repo / DEFAULT_PATHS["normalizer"])
    mean = np.asarray(norm["observation_mean"], dtype=np.float64)
    std = np.asarray(norm["observation_std"], dtype=np.float64)
    std = np.where(std == 0, 1.0, std)
    rows = load_json(exp / "manifests" / "all_roots.json")["roots"]
    store = Path(DEFAULT_PATHS["root_store"])
    legal = [r for r in rows if r.get("legal")]
    all_cands = []
    audit = []
    chunk = 65536
    start_obs = obs[starts]
    for rec in legal:
        bundle = read_root_bundle(root_dir(store, PROTOCOL_ID, rec["root_id"]), verify=True)
        q = (np.asarray(bundle["decision_observation"], dtype=np.float64) - mean) / std
        dists = np.empty(len(starts), dtype=np.float64)
        for i in range(0, len(starts), chunk):
            sl = slice(i, i + chunk)
            x = (np.asarray(start_obs[sl], dtype=np.float64) - mean) / std
            dists[sl] = np.sqrt(np.mean((x - q) ** 2, axis=1))
        order = np.lexsort((starts, dists))  # dist primary, start index secondary
        seen = set()
        picked = []
        for idx in order:
            ep = np.asarray(obs[int(endpoints[idx])])
            key = (str(ep.dtype), ep.shape, ep.tobytes())
            if key in seen:
                continue
            seen.add(key)
            picked.append(
                {
                    "rank": len(picked),
                    "start_index": int(starts[idx]),
                    "endpoint_index": int(endpoints[idx]),
                    "distance_rms": float(dists[idx]),
                    "endpoint_sha256": sha256_array(ep),
                    "result_source": "DATA_REAL",
                }
            )
            all_cands.append(np.asarray(ep, dtype=np.float64).copy())
            if len(picked) == 8:
                break
        audit.append(
            {
                "root_id": rec["root_id"],
                "n_unique_considered": len(seen),
                "n_picked": len(picked),
                "candidates": picked,
                "decision_obs_sha256": sha256_array(bundle["decision_observation"]),
            }
        )
        print({"candidates": rec["root_id"], "n": len(picked)}, flush=True)
        if len(picked) != 8:
            return {"status": "FAIL", "reason": "insufficient_unique_endpoints", "audit": audit}
    arr = np.stack(all_cands).reshape(len(legal), 8, 37)
    np.savez_compressed(exp / "candidates.npz", candidates=arr, root_ids=np.asarray([r["root_id"] for r in legal], dtype=np.int64))
    dump_json(exp / "candidate_retrieval_audit.json", {"n_legal_roots": len(legal), "k": 20, "n_legal_starts": int(len(starts)), "rows": audit})
    dump_json(
        exp / "candidate_manifest.json",
        {
            "protocol_id": PROTOCOL_ID,
            "n_legal_roots": len(legal),
            "n_candidates_per_root": 8,
            "k": 20,
            "result_source": "DATA_REAL",
            "training_eligible": False,
            "candidates_sha256": sha256_file(exp / "candidates.npz"),
            "root_ids": [r["root_id"] for r in legal],
        },
    )
    dump_json(
        exp / "candidate_index_manifest.json",
        {
            "starts_count": int(len(starts)),
            "endpoint_offset": 20,
            "tie_break": "start_index_ascending",
            "dedup": "endpoint_dtype_shape_tobytes",
        },
    )
    dump_json(
        exp / "normalizer_manifest.json",
        {
            "file": DEFAULT_PATHS["normalizer"],
            "sha256": sha256_file(repo / DEFAULT_PATHS["normalizer"]),
            "source": "official_training_only",
            "applied_to_actor_value": False,
            "applied_to_retrieval_distance": True,
        },
    )
    return {"status": "PASS", "n_roots": len(legal), "shape": list(arr.shape)}
