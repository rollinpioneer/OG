"""O1 assets, episode audit, support Gate A. No S3 success labels."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.pop("JAX_PLATFORMS", None)

from pathlib import Path

import numpy as np

from execution_aligned_rl.v3.policy import load_agent
from execution_aligned_rl.v3.serialization import dump_json, load_json
from execution_aligned_rl.v4.outcome_pilot.const import (
    ACTION_HIGH,
    ACTION_LOW,
    CKPT_DIR,
    EXP_REL,
    K_NN,
    NORM_FILE,
    OFFICIAL,
    P99_FRAC_MAX,
    REPO,
    TRAIN_FILE,
    VAL_FILE,
)
from execution_aligned_rl.v4.outcome_pilot.data import load_split
from execution_aligned_rl.v4.outcome_pilot.success import success_from_obs_goal
from execution_aligned_rl.v4.outcome_pilot.support import (
    asset_lock,
    load_s3_query_states,
    min_action_l2,
    summarize,
)


def exp_dir() -> Path:
    return Path(REPO) / EXP_REL


def episode_audit(ds, name: str) -> dict:
    n = ds["n"]
    last = ds["is_last"]
    ep = ds["ep_id"]
    cross = int(np.sum((~last[:-1]) & (ep[:-1] != ep[1:])))
    return {
        "split": name,
        "n_transitions": int(n),
        "n_episodes": int(ds["n_ep"]),
        "n_last": int(last.sum()),
        "cross_episode_nonlast": cross,
        "episode_cover_ok": int(sum(e - s for s, e in ds["bounds"])) == n,
    }


def jax_topk_idx(queries: np.ndarray, corpus: np.ndarray, k: int = K_NN, q_chunk: int = 256) -> np.ndarray:
    """Exact L2 top-k via GEMM. Avoids giant broadcast kernels."""
    import jax
    import jax.numpy as jnp

    queries = np.asarray(queries, dtype=np.float32)
    corpus = np.asarray(corpus, dtype=np.float32)
    Q = int(queries.shape[0])
    D = int(queries.shape[1])
    out = np.empty((Q, k), dtype=np.int32)
    corpus_g = jnp.asarray(corpus)
    c2 = jnp.sum(corpus_g * corpus_g, axis=1)

    @jax.jit
    def topk_q(q):
        d2 = jnp.sum(q * q, axis=1, keepdims=True) + c2[None, :] - 2.0 * (q @ corpus_g.T)
        _vals, idx = jax.lax.top_k(-d2, k)
        return idx

    pad = np.zeros((q_chunk, D), dtype=np.float32)
    n_blocks = (Q + q_chunk - 1) // q_chunk
    for bi, qs in enumerate(range(0, Q, q_chunk)):
        q = queries[qs : qs + q_chunk]
        B = int(q.shape[0])
        if B < q_chunk:
            pad[:B] = q
            pad[B:] = q[-1]
            q_in = pad
        else:
            q_in = q
        idx = np.asarray(topk_q(jnp.asarray(q_in)))
        out[qs : qs + B] = idx[:B]
        if bi % 5 == 0 or B < q_chunk:
            print(
                {
                    "nn_q": int(qs + B),
                    "Q": Q,
                    "block": bi + 1,
                    "n_blocks": n_blocks,
                    "backend": jax.default_backend(),
                },
                flush=True,
            )
    return out


def actor_batch(agent, obs, goals):
    import jax

    out = []
    bs = 1024
    key0 = jax.random.PRNGKey(0)
    n = len(obs)
    for i in range(0, n, bs):
        o = np.asarray(obs[i : i + bs], dtype=np.float32)
        g = np.asarray(goals[i : i + bs], dtype=np.float32)
        a = np.asarray(agent.sample_actions(o, goals=g, seed=key0, temperature=0.0))
        out.append(np.asarray(a, dtype=np.float32))
        if (i // bs) % 20 == 0:
            print({"actor": int(i + len(o)), "n": int(n)}, flush=True)
    return np.concatenate(out, axis=0)


def finite_bounds(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    finite = bool(np.isfinite(a).all())
    inb = bool((a >= ACTION_LOW - 1e-6).all() and (a <= ACTION_HIGH + 1e-6).all())
    return {"finite": finite, "in_bounds": inb, "n": int(a.shape[0]), "pass": finite and inb}


def hold_payload(status: str, extra=None) -> dict:
    d = {
        "status": status,
        "s4_unlocked": False,
        "fresh_confirmation_unlocked": False,
        "new_environment_evaluation_authorized": False,
        "policy_training_authorized": False,
        "human_review": None,
        "new_environment_steps": 0,
        "training_performed": False,
    }
    if extra:
        d.update(extra)
    return d


def main() -> None:
    exp = exp_dir()
    (exp / "audit").mkdir(parents=True, exist_ok=True)
    lock = asset_lock()
    dump_json(exp / "asset_lock.json", lock)
    if lock["status"] != "PASS":
        dump_json(exp / "decision.json", hold_payload("EA41_HOLD_ASSET_MISMATCH", {"lock": lock}))
        raise SystemExit(2)

    train = load_split(TRAIN_FILE)
    val = load_split(VAL_FILE)
    ea = {"train": episode_audit(train, "train"), "val": episode_audit(val, "val")}
    dump_json(exp / "episode_boundary_audit.json", ea)
    if ea["train"]["cross_episode_nonlast"] or ea["val"]["cross_episode_nonlast"]:
        dump_json(exp / "decision.json", hold_payload("EA41_HOLD_ENGINEERING", {"reason": "cross_episode"}))
        raise SystemExit(2)

    pm = load_json(
        Path(
            "/home/__compress_data/xushijie/OG_ea_v3_s3_postmortem/experiments/execution_aligned/ea_v3_s3_postmortem_v1/analysis/offline_supervision_audit.json"
        )
    )
    o = train["obs"][:1000]
    self_ok = bool(np.all(success_from_obs_goal(o, o)))
    far = np.array(o, copy=True)
    far[:, 19:22] += 5.0
    far_ok = bool(np.all(~success_from_obs_goal(o, far)))
    pred = {
        "postmortem_n_steps": pm["public_obs_goal_success_reconstruction"]["n_steps"],
        "postmortem_agreement_rate": pm["public_obs_goal_success_reconstruction"]["agreement_rate"],
        "self_success_1000": self_ok,
        "far_failure_1000": far_ok,
        "status": "PASS"
        if pm["public_obs_goal_success_reconstruction"]["agreement_rate"] == 1.0 and self_ok and far_ok
        else "FAIL",
        "s3_raw_labels_loaded_for_training": False,
        "note": "90680-step agreement reused from sealed postmortem audit JSON, not S3 jsonl success fields",
    }
    dump_json(exp / "success_predicate_audit.json", pred)
    if pred["status"] != "PASS":
        dump_json(exp / "decision.json", hold_payload("EA41_HOLD_ENGINEERING", {"reason": "success_predicate"}))
        raise SystemExit(2)

    norm = np.load(NORM_FILE)
    mean = np.asarray(norm["observation_mean"], dtype=np.float32)
    std = np.asarray(norm["observation_std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)

    def nrm(x):
        return (np.asarray(x, dtype=np.float32) - mean) / std

    print("NN_BEHAVIOR", flush=True)
    vobs = val["obs"]
    vact = val["act"].astype(np.float32)
    tobs_n = nrm(train["obs"])
    vobs_n = nrm(vobs)
    neigh = jax_topk_idx(vobs_n, tobs_n, k=K_NN)
    np.save(exp / "audit" / "val_behavior_neighbor_idx.npy", neigh)
    beh = min_action_l2(vact, neigh, train["act"])
    np.save(exp / "audit" / "val_behavior_min_action_l2.npy", beh)
    beh_sum = summarize(beh)
    dump_json(exp / "audit" / "behavior_reference.json", {"neighbor_k": K_NN, **beh_sum})
    p95 = beh_sum["p95"]
    p99 = beh_sum["p99"]

    print("LOAD_ACTOR", flush=True)
    agent, _config, _tr = load_agent(OFFICIAL, CKPT_DIR, str(Path(TRAIN_FILE).parent))

    print("ACTOR_VAL", flush=True)
    g_idx = np.minimum(np.arange(val["n"]) + 20, val["ep_end"])
    z_idx = np.minimum(np.arange(val["n"]) + 5, val["ep_end"])
    pi_g_val = actor_batch(agent, vobs, val["obs"][g_idx])
    pi_z_val = actor_batch(agent, vobs, val["obs"][z_idx])
    fb_g = finite_bounds(pi_g_val)
    fb_z = finite_bounds(pi_z_val)
    d_g = min_action_l2(pi_g_val, neigh, train["act"])
    d_z = min_action_l2(pi_z_val, neigh, train["act"])

    print("S3_QUERIES_NO_LABELS", flush=True)
    s3 = load_s3_query_states()
    s3_o = np.stack([r["obs"] for r in s3]).astype(np.float32)
    s3_g = np.stack([r["goal"] for r in s3]).astype(np.float32)
    s3_z = np.stack([np.stack([c["z"] for c in r["cands"]]) for r in s3]).astype(np.float32)
    pi_g_s3 = actor_batch(agent, s3_o, s3_g)
    o_rep = np.repeat(s3_o, 8, axis=0)
    z_flat = s3_z.reshape(-1, 37)
    pi_z_s3 = actor_batch(agent, o_rep, z_flat)
    s3n = nrm(s3_o)
    s3n_z = nrm(o_rep)
    neigh_s3 = jax_topk_idx(s3n, tobs_n, k=K_NN)
    neigh_s3z = jax_topk_idx(s3n_z, tobs_n, k=K_NN)
    d_g_s3 = min_action_l2(pi_g_s3, neigh_s3, train["act"])
    d_z_s3 = min_action_l2(pi_z_s3, neigh_s3z, train["act"])

    def pack(name, arr):
        s = summarize(arr)
        s["frac_over_p95"] = float(np.mean(arr > p95)) if arr.size else None
        s["frac_over_p99"] = float(np.mean(arr > p99)) if arr.size else None
        s["set"] = name
        return s

    report = {
        "behavior_reference_val": beh_sum,
        "behavior_p95": p95,
        "behavior_p99": p99,
        "val_pi_g": pack("val_pi_g", d_g),
        "val_pi_z": pack("val_pi_z", d_z),
        "s3_pi_g": pack("s3_pi_g", d_g_s3),
        "s3_pi_z": pack("s3_pi_z", d_z_s3),
        "finite_bounds": {
            "val_pi_g": fb_g,
            "val_pi_z": fb_z,
            "s3_pi_g": finite_bounds(pi_g_s3),
            "s3_pi_z": finite_bounds(pi_z_s3),
        },
        "s3_labels_loaded": False,
        "k_neighbors": K_NN,
        "nn_backend": "jax_gemm_l2",
    }
    dump_json(exp / "target_action_support_audit.json", report)

    over = [
        report["val_pi_g"]["frac_over_p99"],
        report["val_pi_z"]["frac_over_p99"],
        report["s3_pi_g"]["frac_over_p99"],
        report["s3_pi_z"]["frac_over_p99"],
    ]
    bounds_ok = all(v["pass"] for v in report["finite_bounds"].values())
    support_hold = any(x is not None and x > P99_FRAC_MAX for x in over)
    status = "EA40_HOLD_OFFLINE_SUPPORT" if (support_hold or not bounds_ok) else "EA40_SUPPORT_AND_DATA_PASS"
    dump_json(
        exp / "gate_a.json",
        {
            "status": status,
            "over_p99": over,
            "bounds_ok": bounds_ok,
            "p99_max": P99_FRAC_MAX,
            "high_risk": [x is not None and x > 0.10 for x in over],
        },
    )
    if status != "EA40_SUPPORT_AND_DATA_PASS":
        dump_json(
            exp / "decision.json",
            hold_payload(
                "EA40_HOLD_OFFLINE_SUPPORT" if support_hold else "EA41_HOLD_ENGINEERING",
                {"gate_a": status, "over_p99": over},
            ),
        )
        print(status, flush=True)
        raise SystemExit(2)
    print({"gate_a": status, "over_p99": over}, flush=True)


if __name__ == "__main__":
    main()
