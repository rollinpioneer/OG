"""K=64 nearest-state action-support audit. Does not load S3 success labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.hashing import sha256_array
from execution_aligned_rl.v3.serialization import load_json
from execution_aligned_rl.v4.outcome_pilot.const import (
    CKPT_FILE,
    CKPT_SHA,
    K_NN,
    NORM_FILE,
    NORM_SHA,
    OGBENCH,
    OFFICIAL,
    S2_EXP,
    S2_PROTOCOL,
    S2_STORE,
    TRAIN_FILE,
    TRAIN_SHA,
    VAL_FILE,
    VAL_SHA,
)


def asset_lock() -> dict:
    import subprocess

    og = subprocess.check_output(["git", "-C", OFFICIAL, "rev-parse", "HEAD"], text=True).strip()
    hashes = {
        "train": sha256_file(Path(TRAIN_FILE)),
        "validation": sha256_file(Path(VAL_FILE)),
        "checkpoint": sha256_file(Path(CKPT_FILE)),
        "normalizer": sha256_file(Path(NORM_FILE)),
        "ogbench": og,
    }
    expected = {
        "train": TRAIN_SHA,
        "validation": VAL_SHA,
        "checkpoint": CKPT_SHA,
        "normalizer": NORM_SHA,
        "ogbench": OGBENCH,
    }
    fail = [k for k, v in hashes.items() if v != expected[k]]
    return {"hashes": hashes, "expected": expected, "failures": fail, "status": "PASS" if not fail else "FAIL"}


def load_s3_query_states():
    ident = load_json(
        Path("/home/__compress_data/xushijie/OG_ea_v3_s3_postmortem/experiments/execution_aligned/ea_v3_s3_mechanism_v1/root_pool_identity.json")
    )
    audit = {int(r["root_id"]): r for r in load_json(Path(S2_EXP) / "candidate_retrieval_audit.json")["rows"]}
    train = np.load(TRAIN_FILE, allow_pickle=False)
    tobs = np.asarray(train["observations"])
    legal = [int(x) for x in ident["legal_root_ids"]]
    deep = set(int(x) for x in ident["legal_deep_root_ids"])
    rows = []
    for rid in legal:
        d = Path(S2_STORE) / S2_PROTOCOL / f"{rid:06d}"
        o = np.load(d / "decision_observation.npy")
        g = np.load(d / "goal_observation.npy")
        cands = []
        for c in audit[rid]["candidates"]:
            z = np.asarray(tobs[int(c["endpoint_index"])])
            cands.append({"id": int(c["rank"]), "z": z, "endpoint_index": int(c["endpoint_index"])})
        rows.append({"root_id": rid, "is_deep": rid in deep, "obs": o, "goal": g, "cands": cands})
    return rows


def chunked_topk_idx(queries: np.ndarray, corpus: np.ndarray, k: int = K_NN, q_chunk: int = 32, c_chunk: int = 8192) -> np.ndarray:
    Q = int(queries.shape[0])
    out = np.empty((Q, k), dtype=np.int32)
    corpus = np.asarray(corpus, dtype=np.float32)
    for qs in range(0, Q, q_chunk):
        q = np.asarray(queries[qs : qs + q_chunk], dtype=np.float32)
        B = q.shape[0]
        best_d = np.full((B, k), np.inf, dtype=np.float32)
        best_i = np.zeros((B, k), dtype=np.int32)
        for cs in range(0, corpus.shape[0], c_chunk):
            c = corpus[cs : cs + c_chunk]
            d = ((q[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            idx = np.arange(c.shape[0], dtype=np.int32) + np.int32(cs)
            comb_d = np.concatenate([best_d, d], axis=1)
            comb_i = np.concatenate([best_i, np.broadcast_to(idx, d.shape)], axis=1)
            sel = np.argpartition(comb_d, kth=k - 1, axis=1)[:, :k]
            best_d = np.take_along_axis(comb_d, sel, axis=1)
            best_i = np.take_along_axis(comb_i, sel, axis=1)
        order = np.argsort(best_d, axis=1)
        out[qs : qs + B] = np.take_along_axis(best_i, order, axis=1)
        print({"nn_q": int(qs + B), "Q": Q}, flush=True)
    return out


def min_action_l2(actions_q, neighbor_idx, corpus_act) -> np.ndarray:
    neigh = np.asarray(corpus_act, dtype=np.float32)[neighbor_idx]
    diff = neigh - np.asarray(actions_q, dtype=np.float32)[:, None, :]
    return np.sqrt((diff ** 2).sum(-1)).min(axis=1)


def summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    qs = np.quantile(arr, [0.5, 0.9, 0.95, 0.99])
    return {
        "n": int(arr.size),
        "median": float(qs[0]),
        "p90": float(qs[1]),
        "p95": float(qs[2]),
        "p99": float(qs[3]),
        "mean": float(arr.mean()),
    }
