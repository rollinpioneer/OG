"""Offline FQE trainer. Never reads S3 labels or S3 outcome jsonl."""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.serialization import dump_json
from execution_aligned_rl.v4.outcome_pilot.const import (
    BATCH,
    CKPT_DIR,
    CKPT_STEPS,
    CLIP,
    EXP_REL,
    G_OFFSETS,
    HIDDEN,
    K,
    LR,
    MONO_MAX,
    OFFICIAL,
    R_VALUES,
    REPO,
    TAU,
    TRAIN_FILE,
    UPDATES,
    VAL_FILE,
    Z_OFFSETS,
)
from execution_aligned_rl.v4.outcome_pilot.data import legal_k_starts, load_split
from execution_aligned_rl.v4.outcome_pilot.features import feature_dim, load_norm, pack_features
from execution_aligned_rl.v4.outcome_pilot.models import Critic
from execution_aligned_rl.v4.outcome_pilot.success import success_from_obs_goal

METHODS = ("B1", "B2", "B3", "B4")
VAL_BATCHES = 32
MAX_H = 500


def exp_dir() -> Path:
    return Path(REPO) / EXP_REL


class Sampler:
    def __init__(self, ds, seed: int):
        self.ds = dict(ds)
        self.ds["obs"] = np.asarray(ds["obs"], dtype=np.float32)
        self.ds["act"] = np.asarray(ds["act"], dtype=np.float32)
        self.rng = np.random.default_rng(int(seed))
        self.starts = legal_k_starts(ds["term"], K)
        if self.starts.size == 0:
            raise RuntimeError("no legal k=20 starts")
        self.n = int(ds["n"])
        self.nonlast = np.where(~ds["is_last"])[0]
        self.g_off = np.asarray(G_OFFSETS, dtype=np.int32)
        self.z_off = np.asarray(Z_OFFSETS, dtype=np.int32)
        self.r_vals = np.asarray(R_VALUES, dtype=np.int32)
        self.z_end = np.clip(self.starts + K, 0, self.n - 1)

    def _future(self, idx, offsets):
        rem = self.ds["ep_end"][idx] - idx
        off = np.asarray(offsets, dtype=np.int32)
        legal = off[None, :] <= rem[:, None]
        scores = self.rng.random(size=legal.shape)
        scores = np.where(legal, scores, -1.0)
        pick = scores.argmax(axis=1)
        chosen = off[pick]
        chosen = np.where(legal.any(axis=1), chosen, 0)
        return idx + chosen

    def sample(self, batch: int = BATCH, with_behavior_window: bool = False) -> dict:
        idx = self.rng.choice(self.nonlast, size=batch, replace=True).astype(np.int32)
        o = self.ds["obs"][idx]
        a = self.ds["act"][idx]
        nxt = np.minimum(idx + 1, self.n - 1)
        o2 = self.ds["obs"][nxt]
        next_is_last = self.ds["is_last"][nxt]
        use_g_fut = self.rng.random(batch) < 0.75
        g_fut = self._future(idx, self.g_off)
        g_rnd = self.rng.integers(0, self.n, size=batch, dtype=np.int32)
        g_idx = np.where(use_g_fut, g_fut, g_rnd)
        use_z_fut = self.rng.random(batch) < 0.75
        z_fut = self._future(idx, self.z_off)
        z_rnd = self.z_end[self.rng.integers(0, len(self.z_end), size=batch)]
        z_idx = np.where(use_z_fut, z_fut, z_rnd)
        g = self.ds["obs"][g_idx]
        z = self.ds["obs"][z_idx]
        j = self.rng.integers(0, 6, size=batch).astype(np.float32)
        R = self.r_vals[self.rng.integers(0, len(self.r_vals), size=batch)].astype(np.int32)
        R = np.clip(R + self.rng.integers(-5, 6, size=batch), 1, 500).astype(np.float32)
        c2 = success_from_obs_goal(o2, g).astype(np.float32)
        censored = ((c2 < 0.5) & next_is_last).astype(np.float32)
        bw = np.zeros(batch, dtype=np.float32)
        bw_mask = np.ones(batch, dtype=np.float32)
        if with_behavior_window:
            rem = (self.ds["ep_end"][idx] - idx).astype(np.int32)
            horizon = np.minimum(R.astype(np.int32), rem)
            max_h = int(np.clip(horizon.max(), 1, MAX_H))
            offsets = np.arange(1, max_h + 1, dtype=np.int32)
            pos = np.clip(idx[:, None] + offsets[None, :], 0, self.n - 1)
            valid = offsets[None, :] <= horizon[:, None]
            fut = self.ds["obs"][pos]
            hit = success_from_obs_goal(fut.reshape(-1, 37), np.repeat(g, max_h, axis=0)).reshape(batch, max_h)
            hit = hit & valid
            bw = hit.any(axis=1).astype(np.float32)
            bw_mask = np.where((bw < 0.5) & (rem < R.astype(np.int32)), 0.0, 1.0).astype(np.float32)
        return {
            "o": np.asarray(o, dtype=np.float32),
            "a": np.asarray(a, dtype=np.float32),
            "o2": np.asarray(o2, dtype=np.float32),
            "z": np.asarray(z, dtype=np.float32),
            "g": np.asarray(g, dtype=np.float32),
            "j": j,
            "R": R,
            "c2": c2,
            "censored": censored,
            "bw": bw,
            "bw_mask": bw_mask,
        }


def hash_batch(batch: dict) -> str:
    h = sha256()
    for k in sorted(batch):
        h.update(k.encode())
        h.update(np.ascontiguousarray(batch[k]).tobytes())
    return h.hexdigest()


def train_one(method: str, seed: int, gpu: str, updates=None, ckpt_steps=None) -> dict:
    if method not in METHODS:
        raise ValueError(method)
    updates = int(UPDATES if updates is None else updates)
    ckpt_steps = tuple(CKPT_STEPS if ckpt_steps is None else ckpt_steps)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.pop("JAX_PLATFORMS", None)

    import jax
    import jax.numpy as jnp
    import optax
    from flax.serialization import to_bytes
    from flax.training import train_state

    print({"phase": "start", "method": method, "seed": seed, "gpu": gpu}, flush=True)
    mean, std, amean, astd = load_norm()
    mean_j, std_j = jnp.asarray(mean), jnp.asarray(std)
    amean_j, astd_j = jnp.asarray(amean), jnp.asarray(astd)
    train = load_split(TRAIN_FILE)
    val = load_split(VAL_FILE)
    samp = Sampler(train, seed)
    vsamp = Sampler(val, seed + 10000)
    need_bw = method == "B1"
    fixture = samp.sample(BATCH, with_behavior_window=need_bw)
    outdir = exp_dir() / "training" / f"{method}_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    dump_json(
        outdir / "sampling_fixture.json",
        {"method": method, "seed": seed, "batch": BATCH, "sha256": hash_batch(fixture), "keys": sorted(fixture)},
    )

    actor_fn = None
    if method != "B1":
        from execution_aligned_rl.v3.policy import load_agent

        print("LOAD_ACTOR", flush=True)
        agent, _c, _t = load_agent(OFFICIAL, CKPT_DIR, str(Path(TRAIN_FILE).parent))

        @jax.jit
        def actor_fn(obs, goals):
            return agent.sample_actions(obs, goals=goals, seed=jax.random.PRNGKey(0), temperature=0.0)

        _ = actor_fn(jnp.asarray(fixture["o"][:8]), jnp.asarray(fixture["g"][:8]))

    print({"backend": str(jax.default_backend()), "devices": [str(d) for d in jax.devices()]}, flush=True)
    print("INIT_CRITIC", flush=True)
    dim = feature_dim(method)
    model = Critic(hidden=HIDDEN)
    rng = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, dim), dtype=jnp.float32)
    params = model.init(rng, dummy)["params"]
    tx = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    tgt = params

    def pack_jax(o, a, z, g, j, R):
        o_n = (o - mean_j) / std_j
        a_n = (a - amean_j) / astd_j
        g_n = (g - mean_j) / std_j
        j_n = jnp.reshape(j, (-1, 1)) / 5.0
        r_n = jnp.reshape(R, (-1, 1)) / 500.0
        if method == "B2":
            return jnp.concatenate([o_n, a_n, g_n, r_n], axis=-1)
        z_n = (z - mean_j) / std_j
        if method == "B3":
            return jnp.concatenate([o_n, a_n, z_n, g_n, j_n], axis=-1)
        return jnp.concatenate([o_n, a_n, z_n, g_n, j_n, r_n], axis=-1)

    @jax.jit
    def supervised_step(state, tgt, x, y, mask):
        def loss_fn(p):
            pred = jax.nn.sigmoid(state.apply_fn({"params": p}, x))
            se = (pred - y) ** 2
            return (se * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        tgt = jax.tree_util.tree_map(lambda t, p: (1.0 - TAU) * t + TAU * p, tgt, state.params)
        return state, tgt, loss

    @jax.jit
    def fqe_step(state, tgt, o, a, z, g, j, R, o2, c2, drop):
        if method == "B2":
            j = jnp.zeros_like(j)
        j2 = jnp.maximum(j - 1.0, 0.0)
        R2 = jnp.maximum(R - 1.0, 0.0)
        goal_next = jnp.where((j2 > 0.0)[:, None], z, g)
        a2 = actor_fn(o2, goal_next)
        x = pack_jax(o, a, z, g, j, R)
        x2 = pack_jax(o2, a2, z, g, j2, R2)
        qbar = jax.lax.stop_gradient(jax.nn.sigmoid(state.apply_fn({"params": tgt}, x2)))
        y = c2 + (1.0 - c2) * (R2 > 0.0).astype(jnp.float32) * qbar
        mask = 1.0 - drop

        def loss_fn(p):
            pred = jax.nn.sigmoid(state.apply_fn({"params": p}, x))
            se = (pred - y) ** 2
            return (se * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        tgt = jax.tree_util.tree_map(lambda t, p: (1.0 - TAU) * t + TAU * p, tgt, state.params)
        return state, tgt, loss

    @jax.jit
    def apply_prob(params, x):
        return jax.nn.sigmoid(state.apply_fn({"params": params}, x))

    def pack_np(o, a, z, g, j, R):
        return pack_features(o, a, z, g, j, R, method, mean, std, amean, astd)

    def next_actions_np(o2, z, g, j):
        if actor_fn is None:
            return np.zeros((o2.shape[0], 5), dtype=np.float32)
        j2 = np.maximum(np.asarray(j) - 1.0, 0.0)
        goal_next = np.where((j2 > 0.0)[:, None], z, g)
        return np.asarray(actor_fn(jnp.asarray(o2), jnp.asarray(goal_next)), dtype=np.float32)

    def val_metrics(state, tgt):
        mses = []
        preds = []
        saved = vsamp.rng.bit_generator.state
        vsamp.rng = np.random.default_rng(seed + 77777)
        for _ in range(VAL_BATCHES):
            b = vsamp.sample(BATCH, with_behavior_window=need_bw)
            j = np.zeros_like(b["j"]) if method == "B2" else b["j"]
            x = pack_np(b["o"], b["a"], b["z"], b["g"], j, b["R"])
            p = np.asarray(apply_prob(state.params, jnp.asarray(x)))
            if method == "B1":
                y, mask = b["bw"], b["bw_mask"]
            else:
                a2 = next_actions_np(b["o2"], b["z"], b["g"], j)
                j2 = np.maximum(j - 1.0, 0.0)
                R2 = np.maximum(b["R"] - 1.0, 0.0)
                x2 = pack_np(b["o2"], a2, b["z"], b["g"], j2, R2)
                qn = np.asarray(apply_prob(tgt, jnp.asarray(x2)))
                y = b["c2"] + (1.0 - b["c2"]) * (R2 > 0).astype(np.float32) * qn
                drop = (b["censored"] > 0) & (b["c2"] < 0.5)
                mask = np.where(drop, 0.0, 1.0).astype(np.float32)
            mse = float((((p - y) ** 2) * mask).sum() / max(float(mask.sum()), 1.0))
            mses.append(mse)
            preds.append(p)
        vsamp.rng.bit_generator.state = saved
        pcat = np.concatenate(preds)
        finite = bool(np.isfinite(pcat).all())
        inrange = bool((pcat >= -1e-6).all() and (pcat <= 1.0 + 1e-6).all())
        mono = 0.0
        if method in ("B1", "B2", "B4"):
            b = vsamp.sample(BATCH, with_behavior_window=False)
            j = np.zeros_like(b["j"]) if method == "B2" else b["j"]
            ps = []
            for rv in (80.0, 250.0, 500.0):
                x = pack_np(b["o"], b["a"], b["z"], b["g"], j, np.full((BATCH,), rv, dtype=np.float32))
                ps.append(np.asarray(apply_prob(state.params, jnp.asarray(x))))
            mono = float(((ps[0] > ps[1] + 1e-4) | (ps[1] > ps[2] + 1e-4)).mean())
        return {
            "val_bellman_mse": float(np.mean(mses)),
            "finite": finite,
            "in_range": inrange,
            "monotonic_violation": mono,
        }

    print("LOOP_START", {"updates": updates, "ckpt": ckpt_steps}, flush=True)
    metrics = []
    for u in range(1, updates + 1):
        b = samp.sample(BATCH, with_behavior_window=need_bw)
        if method == "B1":
            x = pack_np(b["o"], b["a"], b["z"], b["g"], b["j"], b["R"])
            state, tgt, loss = supervised_step(
                state, tgt, jnp.asarray(x), jnp.asarray(b["bw"]), jnp.asarray(b["bw_mask"])
            )
        else:
            drop = ((b["censored"] > 0) & (b["c2"] < 0.5)).astype(np.float32)
            state, tgt, loss = fqe_step(
                state,
                tgt,
                jnp.asarray(b["o"]),
                jnp.asarray(b["a"]),
                jnp.asarray(b["z"]),
                jnp.asarray(b["g"]),
                jnp.asarray(b["j"]),
                jnp.asarray(b["R"]),
                jnp.asarray(b["o2"]),
                jnp.asarray(b["c2"]),
                jnp.asarray(drop),
            )
        if u % 1000 == 0:
            print({"method": method, "seed": seed, "u": u, "loss": float(loss)}, flush=True)
        if u in ckpt_steps:
            path = outdir / f"params_{u}.msgpack"
            path.write_bytes(to_bytes(state.params))
            rec = {
                "method": method,
                "seed": seed,
                "update": int(u),
                "sha256": sha256_file(path),
                "path": str(path),
            }
            rec.update(val_metrics(state, tgt))
            metrics.append(rec)
            dump_json(outdir / f"val_{u}.json", rec)
            print(rec, flush=True)

    ok = [m for m in metrics if m["finite"] and m["in_range"] and m["monotonic_violation"] <= MONO_MAX]
    if not ok:
        selected = min(metrics, key=lambda m: (not m["finite"], m["val_bellman_mse"], m["update"]))
        unstable = True
    else:
        selected = min(ok, key=lambda m: (m["val_bellman_mse"], m["update"]))
        unstable = False
    dump_json(outdir / "selected.json", {"selected": selected, "unstable": unstable, "all": metrics})
    return {"selected": selected, "unstable": unstable, "n_ckpt": len(metrics)}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--updates", type=int, default=None)
    args = parser.parse_args()
    updates = UPDATES if args.updates is None else int(args.updates)
    ckpt_steps = tuple(x for x in CKPT_STEPS if x <= updates) or (updates,)
    print(json.dumps(train_one(args.method, args.seed, args.gpu, updates=updates, ckpt_steps=ckpt_steps), default=str), flush=True)


if __name__ == "__main__":
    main()
