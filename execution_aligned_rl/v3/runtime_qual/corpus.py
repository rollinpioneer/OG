
"""Read-only extraction of the S1 legal-root actor/value inference corpus. No env.step."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.hashing import sha256_array, stable_uint32
from execution_aligned_rl.v3.root_bundle import read_root_bundle
from execution_aligned_rl.v3.runtime_qual.protocol import DEFAULT_PATHS, PROTOCOL_ID, S1_PROTOCOL_ID
from execution_aligned_rl.v3.serialization import dump_json


def probe_key_uint32(root_id: int, step: int) -> int:
    return stable_uint32(f"ea3-mr-v1:probe:{root_id}:{step}") & 0xFFFFFFFF


def pre_action_observations(decision_obs: np.ndarray, probe_steps: list[dict]) -> list[np.ndarray]:
    obs = [np.asarray(decision_obs, dtype=np.float64).copy()]
    for step in probe_steps[:-1]:
        obs.append(np.asarray(step["observation"], dtype=np.float64).copy())
    if len(obs) != len(probe_steps):
        raise ValueError(f"pre-action observation count {len(obs)} != probe steps {len(probe_steps)}")
    return obs


def extract_corpus(s1_repo: Path | None = None, root_store: Path | None = None, out_dir: Path | None = None) -> dict:
    s1_repo = Path(s1_repo or DEFAULT_PATHS["s1_repo"])
    root_store = Path(root_store or DEFAULT_PATHS["s1_root_store"])
    out_dir = Path(out_dir or Path(DEFAULT_PATHS["repo"]) / DEFAULT_PATHS["experiment_rel"] / "corpus")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = s1_repo / DEFAULT_PATHS["s1_experiment_rel"] / "manifests" / "engineering_root_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = []
    roots = []
    for row in raw["roots"]:
        if not row.get("legal"):
            continue
        root_id = int(row["root_id"])
        bundle = read_root_bundle(root_store / f"{root_id:06d}", verify=True)
        if not bundle["manifest"]["legal"]:
            raise RuntimeError(f"manifest/store legal mismatch for {root_id}")
        goal = np.asarray(bundle["goal_observation"], dtype=np.float64).copy()
        d3 = np.asarray(bundle["d3_target_observation"], dtype=np.float64).copy()
        decision = np.asarray(bundle["decision_observation"], dtype=np.float64).copy()
        probe = bundle["original_live_probe"]
        steps = list(probe["steps"])
        pre_obs = pre_action_observations(decision, steps)
        saved_keys = probe.get("key_schedule")
        recon_keys = [probe_key_uint32(root_id, i) for i in range(len(steps))]
        root_rec = {
            "root_id": root_id,
            "task_id": bundle["manifest"]["task_id"],
            "store_path": str(bundle["directory"]),
            "store_sha256_goal": sha256_array(goal),
            "store_sha256_d3": sha256_array(d3),
            "store_sha256_decision": sha256_array(decision),
            "n_probe_steps": len(steps),
            "saved_key_schedule": saved_keys,
            "reconstructed_key_uint32": recon_keys,
            "file_hashes_sha256": sha256_file(bundle["directory"] / "file_hashes.json"),
        }
        roots.append(root_rec)
        queries.append(
            {
                "query_id": f"{root_id}:decision:value",
                "kind": "value",
                "root_id": root_id,
                "role": "decision_observation",
                "observation": decision,
                "goal": goal,
                "goal_role": "exact_task_goal",
            }
        )
        for i, obs in enumerate(pre_obs):
            queries.append(
                {
                    "query_id": f"{root_id}:probe{i}:actor",
                    "kind": "actor",
                    "root_id": root_id,
                    "role": "probe_pre_action",
                    "step": i,
                    "observation": obs,
                    "goal": d3,
                    "goal_role": "frozen_d3_target",
                    "key_uint32": recon_keys[i],
                    "key_formula": f"ea3-mr-v1:probe:{root_id}:{i}",
                }
            )
            queries.append(
                {
                    "query_id": f"{root_id}:probe{i}:value",
                    "kind": "value",
                    "root_id": root_id,
                    "role": "probe_pre_action",
                    "step": i,
                    "observation": obs,
                    "goal": goal,
                    "goal_role": "exact_task_goal",
                }
            )
        # original actions are stored only as reference; workers do not need them to call the policy
        np.savez_compressed(
            out_dir / f"root_{root_id}_arrays.npz",
            goal=goal,
            d3_target=d3,
            decision=decision,
            **{f"pre_obs_{i}": obs for i, obs in enumerate(pre_obs)},
            **{f"orig_action_{i}": np.asarray(steps[i]["action"], dtype=np.float64) for i in range(len(steps))},
        )

    slim_queries = []
    for q in queries:
        item = {k: v for k, v in q.items() if k not in {"observation", "goal"}}
        item["observation_sha256"] = sha256_array(q["observation"])
        item["goal_sha256"] = sha256_array(q["goal"])
        item["observation_shape"] = list(np.asarray(q["observation"]).shape)
        item["goal_shape"] = list(np.asarray(q["goal"]).shape)
        slim_queries.append(item)

    corpus = {
        "protocol_id": PROTOCOL_ID,
        "s1_protocol_id": S1_PROTOCOL_ID,
        "n_legal_roots": len(roots),
        "n_queries": len(slim_queries),
        "n_actor_queries": sum(1 for q in slim_queries if q["kind"] == "actor"),
        "n_value_queries": sum(1 for q in slim_queries if q["kind"] == "value"),
        "env_step_calls": 0,
        "roots": roots,
        "queries": slim_queries,
        "s1_manifest_sha256": sha256_file(manifest_path),
        "array_dir": str(out_dir),
    }
    dump_json(out_dir / "inference_corpus_manifest.json", corpus)
    # compact arrays for workers
    actor = [q for q in queries if q["kind"] == "actor"]
    value = [q for q in queries if q["kind"] == "value"]
    np.savez_compressed(
        out_dir / "actor_queries.npz",
        observation=np.stack([q["observation"] for q in actor]),
        goal=np.stack([q["goal"] for q in actor]),
        key_uint32=np.asarray([q["key_uint32"] for q in actor], dtype=np.uint32),
        root_id=np.asarray([q["root_id"] for q in actor], dtype=np.int64),
        step=np.asarray([q["step"] for q in actor], dtype=np.int64),
    )
    np.savez_compressed(
        out_dir / "value_queries.npz",
        observation=np.stack([q["observation"] for q in value]),
        goal=np.stack([q["goal"] for q in value]),
        root_id=np.asarray([q["root_id"] for q in value], dtype=np.int64),
    )
    corpus["actor_queries_sha256"] = sha256_file(out_dir / "actor_queries.npz")
    corpus["value_queries_sha256"] = sha256_file(out_dir / "value_queries.npz")
    dump_json(out_dir / "inference_corpus_manifest.json", corpus)
    return corpus
