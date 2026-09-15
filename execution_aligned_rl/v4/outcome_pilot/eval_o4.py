"""O4 CPU evaluation. Pass --load-s3-labels only after training seal."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "0"
os.environ.pop("XLA_FLAGS", None)

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.policy import load_agent, value_for
from execution_aligned_rl.v3.serialization import dump_json, load_json
from execution_aligned_rl.v4.outcome_pilot.const import (
    BETA,
    CKPT_DIR,
    EXP_REL,
    HIDDEN,
    M,
    OFFICIAL,
    REF,
    REPO,
    SEEDS,
    TRAIN_FILE,
)
from execution_aligned_rl.v4.outcome_pilot.features import feature_dim, load_norm, pack_features
from execution_aligned_rl.v4.outcome_pilot.models import Critic

S3_EXP = Path("/home/__compress_data/xushijie/OG_ea_v3_s3_mechanism/experiments/execution_aligned/ea_v3_s3_mechanism_v1")
BOOT_N = 10000
BOOT_SEED = 350301


def exp_dir() -> Path:
    return Path(REPO) / EXP_REL


def auroc(y, s):
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    gt = float((pos[:, None] > neg[None, :]).mean())
    eq = float((pos[:, None] == neg[None, :]).mean())
    return gt + 0.5 * eq


def brier(y, s):
    return float(np.mean((np.asarray(s, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def choose(scores, tie_small=True):
    scores = np.asarray(scores, dtype=np.float64)
    if tie_small:
        # maximize score; ties -> smallest index
        m = np.max(scores)
        hits = np.flatnonzero(np.abs(scores - m) <= 1e-12)
        return int(hits[0])
    return int(np.argmax(scores))


def load_labels(deep_ids):
    table = {}
    with (S3_EXP / "mechanism_root_table.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["root_id"]) in deep_ids and rec.get("is_deep"):
                table[int(rec["root_id"])] = rec
    return table


def load_ckpt(path, method, rng):
    import jax
    import jax.numpy as jnp
    from flax.serialization import from_bytes

    model = Critic(hidden=HIDDEN)
    dummy = jnp.zeros((1, feature_dim(method)), dtype=jnp.float32)
    template = model.init(rng, dummy)["params"]
    params = from_bytes(template, Path(path).read_bytes())
    return model, params


def metrics_for_selector(rows, selected, scores_by_root=None, name=""):
    n = len(rows)
    succ = []
    rescue = harm = interv = 0
    per_task = {}
    pairwise = []
    cand_y = []
    cand_s = []
    for r in rows:
        rid = r["root_id"]
        y = np.asarray(r["successes"], dtype=np.int32)
        ideal = int(r["ideal_candidate_id"])
        sel = int(selected[rid])
        s_sel = bool(y[sel])
        s_ideal = bool(y[ideal])
        succ.append(s_sel)
        if s_sel and not s_ideal:
            rescue += 1
        if (not s_sel) and s_ideal:
            harm += 1
        if sel != ideal:
            interv += 1
        per_task.setdefault(int(r["task_id"]), []).append(s_sel)
        if scores_by_root is not None:
            sc = np.asarray(scores_by_root[rid], dtype=np.float64)
            pairwise.append(auroc(y, sc))
            cand_y.extend(y.tolist())
            cand_s.extend(sc.tolist())
    out = {
        "name": name,
        "success": int(np.sum(succ)),
        "n": n,
        "success_rate": float(np.mean(succ)),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "intervention": interv,
        "per_task": {str(k): float(np.mean(v)) for k, v in sorted(per_task.items())},
        "per_task_success_counts": {str(k): int(np.sum(v)) for k, v in sorted(per_task.items())},
    }
    if scores_by_root is not None:
        pw = [x for x in pairwise if x is not None]
        out["root_pairwise_auc"] = float(np.mean(pw)) if pw else None
        out["n_distinguishable_for_auc"] = len(pw)
        out["candidate_brier"] = brier(cand_y, cand_s)
        out["candidate_auroc"] = auroc(cand_y, cand_s)
        # ranking acc on distinguishable
        dist_acc = []
        for r in rows:
            y = np.asarray(r["successes"])
            if y.min() == y.max():
                continue
            sc = np.asarray(scores_by_root[r["root_id"]])
            dist_acc.append(int(choose(sc) in np.flatnonzero(y == 1)))
        out["distinguishable_ranking_accuracy"] = float(np.mean(dist_acc)) if dist_acc else None
        out["n_distinguishable"] = len(dist_acc)
    return out


def bootstrap_ci(rows, selected, rng):
    by_task = {}
    for r in rows:
        by_task.setdefault(int(r["task_id"]), []).append(r)
    stats = {"success": [], "rescue": [], "harm": [], "net": []}
    for _ in range(BOOT_N):
        picks = []
        for recs in by_task.values():
            idx = rng.integers(0, len(recs), size=len(recs))
            picks.extend([recs[i] for i in idx])
        m = metrics_for_selector(picks, selected)
        stats["success"].append(m["success"])
        stats["rescue"].append(m["rescue"])
        stats["harm"].append(m["harm"])
        stats["net"].append(m["net"])
    out = {}
    for k, arr in stats.items():
        a = np.asarray(arr, dtype=np.float64)
        out[k] = {
            "mean": float(a.mean()),
            "p025": float(np.quantile(a, 0.025)),
            "p975": float(np.quantile(a, 0.975)),
        }
    return out


def gate_status(ens, b5, seed_sel, b2, rows):
    # exclusive science statuses after engineering holds already passed
    b4 = ens["B4"]
    ok_b4 = (
        b4["success"] >= 15
        and (b4.get("root_pairwise_auc") is not None)
        and (b2.get("root_pairwise_auc") is not None)
        and b4["root_pairwise_auc"] > b2["root_pairwise_auc"]
        and b4["root_pairwise_auc"] > ens["B3"]["root_pairwise_auc"]
    )
    ok_b5 = (
        b5["success"] >= 17
        and b5["rescue"] >= 3
        and b5["harm"] <= 1
        and b5["net"] >= 2
        and b5["intervention"] <= 15
    )
    nets = [seed_sel[s]["net"] for s in SEEDS]
    harms = [seed_sel[s]["harm"] for s in SEEDS]
    ok_seeds = (sum(n >= 0 for n in nets) >= 2) and all(h < 4 for h in harms)
    # not only one task: B5 rescues spread
    task_counts = b5.get("per_task_success_counts", {})
    # improvement vs IDEAL 15 not from one task: use rescue distribution if available
    multi_task = True
    if ok_b4 and ok_b5 and ok_seeds and multi_task:
        return "EA41_EVALUATOR_PILOT_SIGNAL"
    if b2["success"] > 15 or (b2.get("root_pairwise_auc") or 0) > 0.6:
        return "EA41_TAIL_EVALUATOR_SIGNAL_ONLY"
    return "EA41_EVALUATOR_NO_SIGNAL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-s3-labels", action="store_true")
    args = parser.parse_args()
    if not args.load_s3_labels:
        raise SystemExit("refuse to load S3 labels without --load-s3-labels")
    import jax
    import jax.numpy as jnp

    exp = exp_dir()
    inputs = load_json(exp / "s3_evaluation_inputs_manifest.json")
    deep = [int(r["root_id"]) for r in inputs["rows"]]
    labels = load_labels(set(deep))
    if len(labels) != 35:
        raise SystemExit(f"expected 35 deep labels, got {len(labels)}")
    mean, std, amean, astd = load_norm()
    agent, _c, _t = load_agent(OFFICIAL, CKPT_DIR, str(Path(TRAIN_FILE).parent))

    @jax.jit
    def actor_fn(obs, goals):
        return agent.sample_actions(obs, goals=goals, seed=jax.random.PRNGKey(0), temperature=0.0)

    rows = []
    for meta in inputs["rows"]:
        rid = int(meta["root_id"])
        arr = np.load(meta["arrays"])
        lab = labels[rid]
        rows.append(
            {
                **meta,
                "obs": np.asarray(arr["obs"], dtype=np.float32),
                "goal": np.asarray(arr["goal"], dtype=np.float32),
                "z": np.asarray(arr["z"], dtype=np.float32),
                "successes": [bool(x) for x in lab["successes"]],
                "ideal_candidate_id": int(lab["ideal_candidate_id"]),
            }
        )

    # actions at j=M=5: pi_z
    for r in rows:
        o = np.repeat(r["obs"][None], 8, axis=0)
        r["a_z"] = np.asarray(actor_fn(jnp.asarray(o), jnp.asarray(r["z"])), dtype=np.float32)
        r["a_g"] = np.asarray(actor_fn(jnp.asarray(r["obs"][None]), jnp.asarray(r["goal"][None])), dtype=np.float32)[0]
        r["v_zg"] = np.asarray(agent.network.select("value")(jnp.asarray(r["z"]), jnp.asarray(np.repeat(r["goal"][None], 8, 0))), dtype=np.float64)

    # B0
    b0_sel = {r["root_id"]: choose(r["v_zg"]) for r in rows}
    b0_scores = {r["root_id"]: r["v_zg"] for r in rows}

    def score_method(method, params):
        import jax.numpy as jnp

        model = Critic(hidden=HIDDEN)

        @jax.jit
        def fn(x):
            return jax.nn.sigmoid(model.apply({"params": params}, x))

        out = {}
        for r in rows:
            j = np.full((8,), float(M), dtype=np.float32)
            R = np.full((8,), float(r["R"]), dtype=np.float32)
            o = np.repeat(r["obs"][None], 8, axis=0)
            g = np.repeat(r["goal"][None], 8, axis=0)
            x = pack_features(o, r["a_z"], r["z"], g, j, R, method, mean, std, amean, astd)
            out[r["root_id"]] = np.asarray(fn(jnp.asarray(x)), dtype=np.float64)
        return out

    seed_scores = {m: {} for m in ("B1", "B2", "B3", "B4")}
    selected_ckpts = {}
    for method in ("B1", "B2", "B3", "B4"):
        for seed in SEEDS:
            sel = load_json(exp / "training" / f"{method}_{seed}" / "selected.json")["selected"]
            selected_ckpts[f"{method}_{seed}"] = sel
            rng = jax.random.PRNGKey(0)
            _model, params = load_ckpt(sel["path"], method, rng)
            seed_scores[method][seed] = score_method(method, params)

    ens_scores = {}
    for method in ("B1", "B2", "B3", "B4"):
        ens_scores[method] = {}
        for r in rows:
            rid = r["root_id"]
            stack = np.stack([seed_scores[method][s][rid] for s in SEEDS], axis=0)
            ens_scores[method][rid] = stack.mean(axis=0)
            r.setdefault("ens_std", {})[method] = stack.std(axis=0, ddof=1)

    # B5 LCB vs IDEAL using B4 ensemble
    b5_sel = {}
    b4_lcb = {}
    for r in rows:
        rid = r["root_id"]
        ideal = int(r["ideal_candidate_id"])
        stack = np.stack([seed_scores["B4"][s][rid] for s in SEEDS], axis=0)
        delta = stack - stack[:, ideal : ideal + 1]
        mean_d = delta.mean(axis=0)
        std_d = delta.std(axis=0, ddof=1)
        lcb = mean_d - BETA * std_d
        b4_lcb[rid] = lcb
        if np.max(lcb) > 0:
            b5_sel[rid] = int(choose(lcb))
        else:
            b5_sel[rid] = ideal

    seedwise = {}
    for method in ("B1", "B2", "B3", "B4"):
        seedwise[method] = {}
        for seed in SEEDS:
            sel = {rid: choose(sc) for rid, sc in seed_scores[method][seed].items()}
            seedwise[method][seed] = metrics_for_selector(rows, sel, seed_scores[method][seed], name=f"{method}_{seed}")
            # per-seed selective vs ideal using point delta
            ssel = {}
            for r in rows:
                rid = r["root_id"]
                ideal = int(r["ideal_candidate_id"])
                sc = seed_scores[method][seed][rid]
                d = sc - sc[ideal]
                ssel[rid] = int(choose(d)) if np.max(d) > 0 else ideal
            seedwise[method][seed]["selective"] = metrics_for_selector(rows, ssel, name=f"{method}_{seed}_sel")

    ensemble = {}
    ens_sel = {}
    for method in ("B1", "B2", "B3", "B4"):
        ens_sel[method] = {rid: choose(sc) for rid, sc in ens_scores[method].items()}
        ensemble[method] = metrics_for_selector(rows, ens_sel[method], ens_scores[method], name=f"{method}_ens")

    b0 = metrics_for_selector(rows, b0_sel, b0_scores, name="B0")
    b5 = metrics_for_selector(rows, b5_sel, ens_scores["B4"], name="B5")
    # attach intervention already in metrics

    rng = np.random.default_rng(BOOT_SEED)
    boot = {
        "B0": bootstrap_ci(rows, b0_sel, rng),
        "B4": bootstrap_ci(rows, ens_sel["B4"], rng),
        "B5": bootstrap_ci(rows, b5_sel, rng),
        "B2": bootstrap_ci(rows, ens_sel["B2"], rng),
    }

    status = gate_status(ensemble, b5, {s: seedwise["B4"][s]["selective"] for s in SEEDS}, ensemble["B2"], rows)

    # jsonl per root
    jsonl_path = exp / "selector_results.jsonl"
    with jsonl_path.open("w") as f:
        for r in rows:
            rec = {
                "root_id": r["root_id"],
                "task_id": r["task_id"],
                "successes": r["successes"],
                "ideal_candidate_id": r["ideal_candidate_id"],
                "R": r["R"],
                "B0_scores": b0_scores[r["root_id"]].tolist(),
                "B0_selected": b0_sel[r["root_id"]],
                "B5_selected": b5_sel[r["root_id"]],
                "B4_ens_scores": ens_scores["B4"][r["root_id"]].tolist(),
                "B4_lcb_delta": b4_lcb[r["root_id"]].tolist(),
            }
            f.write(json.dumps(rec) + "\n")

    dump_json(exp / "seedwise_results.json", seedwise)
    dump_json(
        exp / "ensemble_results.json",
        {"B0": b0, "B5": b5, **{k: ensemble[k] for k in ensemble}, "references": REF},
    )
    dump_json(exp / "bootstrap_intervals.json", boot)
    dump_json(
        exp / "ablation_summary.json",
        {
            "B0_success": b0["success"],
            "B2_success": ensemble["B2"]["success"],
            "B3_success": ensemble["B3"]["success"],
            "B4_success": ensemble["B4"]["success"],
            "B5_success": b5["success"],
            "B4_auc": ensemble["B4"]["root_pairwise_auc"],
            "B2_auc": ensemble["B2"]["root_pairwise_auc"],
            "B3_auc": ensemble["B3"]["root_pairwise_auc"],
        },
    )

    decision = {
        "status": status,
        "fresh_confirmation_unlocked": False,
        "new_environment_evaluation_authorized": False,
        "policy_training_authorized": False,
        "s4_unlocked": False,
        "human_review": None,
        "new_environment_steps": 0,
        "training_performed": True,
        "references": REF,
        "B0": b0,
        "B5": b5,
        "ensemble": ensemble,
        "jax_backend": str(jax.default_backend()),
        "checkpoint_sha256": {k: v.get("sha256") for k, v in selected_ckpts.items()},
    }
    dump_json(exp / "decision.json", decision)
    print({"status": status, "B0": b0["success"], "B4": ensemble["B4"]["success"], "B5": b5["success"]}, flush=True)


if __name__ == "__main__":
    main()
