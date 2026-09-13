"""Train the frozen physical-state k-step candidate prior for phase B."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax

from execution_aligned_rl.data.audit_assets import episode_bounds, sha256_file


class CandidatePrior(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, state):
        x = state
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        mean = nn.Dense(self.output_dim)(x)
        log_std = self.param("log_std", nn.initializers.constant(-1.0), (self.output_dim,))
        return mean, jnp.clip(log_std, -5.0, 1.0)


def valid_window_indices(terminals: np.ndarray, horizon: int) -> np.ndarray:
    indices = []
    for start, end in episode_bounds(terminals):
        indices.extend(range(start, end - horizon))
    return np.asarray(indices, dtype=np.int64)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="antmaze-large-stitch-v0")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--updates", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, train, val = ogbench.make_env_and_datasets(args.dataset, dataset_dir=args.dataset_dir)
    train_obs = np.asarray(train["observations"], dtype=np.float32)
    val_obs = np.asarray(val["observations"], dtype=np.float32)
    train_idxs = valid_window_indices(train["terminals"], args.horizon)
    val_idxs = valid_window_indices(val["terminals"], args.horizon)
    obs_mean = train_obs.mean(axis=0)
    obs_std = np.maximum(train_obs.std(axis=0), 1e-6)
    target_min = train_obs.min(axis=0)
    target_max = train_obs.max(axis=0)

    model = CandidatePrior((256, 256), train_obs.shape[-1])
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros((1, train_obs.shape[-1]), dtype=jnp.float32))["params"]
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, states, targets):
        def loss_fn(current):
            mean, log_std = model.apply({"params": current}, states)
            inv_var = jnp.exp(-2.0 * log_std)
            nll = 0.5 * ((targets - mean) ** 2 * inv_var + 2.0 * log_std)
            return nll.mean(), {"nll": nll.mean(), "mse": jnp.mean((targets - mean) ** 2)}

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, info

    rng_np = np.random.default_rng(args.seed)
    started = time.time()
    last = {}
    for step in range(1, args.updates + 1):
        batch_idxs = train_idxs[rng_np.integers(0, len(train_idxs), size=args.batch_size)]
        states = (train_obs[batch_idxs] - obs_mean) / obs_std
        targets = (train_obs[batch_idxs + args.horizon] - obs_mean) / obs_std
        params, opt_state, loss, info = train_step(params, opt_state, states, targets)
        if step == 1 or step % 1000 == 0:
            last = {"step": step, "loss": float(loss), "mse": float(info["mse"]), "wall_seconds": time.time() - started}
            print(json.dumps(last), flush=True)

    eval_rng = np.random.default_rng(20260913)
    audit_count = min(4096, len(val_idxs))
    audit_idxs = val_idxs[eval_rng.choice(len(val_idxs), size=audit_count, replace=False)]
    audit_states = (val_obs[audit_idxs] - obs_mean) / obs_std
    audit_targets = (val_obs[audit_idxs + args.horizon] - obs_mean) / obs_std
    means, log_stds = model.apply({"params": params}, jnp.asarray(audit_states))
    means = np.asarray(means)
    log_stds = np.asarray(log_stds)
    samples = []
    for sample_id in range(8):
        noise = eval_rng.standard_normal(means.shape)
        sample = (means + np.exp(log_stds) * noise) * obs_std + obs_mean
        samples.append(np.clip(sample, target_min, target_max))
    samples = np.stack(samples, axis=1)
    distinct_rate = float(np.mean(np.linalg.norm(samples[:, 1:] - samples[:, :-1], axis=-1) > 1e-6))
    finite_rate = float(np.isfinite(samples).all(axis=-1).mean())

    payload = {
        "params": jax.device_get(params),
        "config": {
            "dataset_id": args.dataset,
            "seed": args.seed,
            "horizon": args.horizon,
            "hidden_dims": (256, 256),
            "updates": args.updates,
            "batch_size": args.batch_size,
            "learning_rate": 3e-4,
            "representation": "physical_observation",
            "checkpoint_rule": "fixed_final_checkpoint",
        },
        "normalization": {
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "target_min": target_min,
            "target_max": target_max,
        },
    }
    checkpoint = out / "candidate_prior.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    audit = {
        "status": "PASS" if finite_rate == 1.0 and distinct_rate > 0.95 else "HOLD_BACKBONE_OR_CANDIDATES",
        "dataset_id": args.dataset,
        "result_source": "DATA_REAL",
        "training_windows": int(len(train_idxs)),
        "validation_windows": int(len(val_idxs)),
        "audit_windows": audit_count,
        "candidate_count": 8,
        "finite_candidate_rate": finite_rate,
        "adjacent_sample_distinct_rate": distinct_rate,
        "normalized_endpoint_mse": float(np.mean((means - audit_targets) ** 2)),
        "mean_predicted_std": float(np.exp(log_stds).mean()),
        "final_training_record": last,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "candidate_set_is_reachability_label": False,
    }
    write_json(out / "candidate_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

