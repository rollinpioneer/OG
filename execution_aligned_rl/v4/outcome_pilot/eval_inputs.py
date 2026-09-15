"""Build S3 evaluation inputs without outcome labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from execution_aligned_rl.v3.serialization import dump_json, load_json
from execution_aligned_rl.v4.outcome_pilot.const import (
    EXP_REL,
    M,
    REPO,
    S2_EXP,
    S2_PROTOCOL,
    S2_STORE,
    TRAIN_FILE,
)

S3_EXP = Path("/home/__compress_data/xushijie/OG_ea_v3_s3_mechanism/experiments/execution_aligned/ea_v3_s3_mechanism_v1")
MAX_H = 500


def main() -> None:
    ident = load_json(S3_EXP / "root_pool_identity.json")
    deep = [int(x) for x in ident["legal_deep_root_ids"]]
    audit_rows = load_json(Path(S2_EXP) / "candidate_retrieval_audit.json")["rows"]
    audit = {int(r["root_id"]): r for r in audit_rows}
    ranking = {int(r["root_id"]): r for r in load_json(S3_EXP / "ideal_ranking_manifest.json")["roots"]}
    train = np.load(TRAIN_FILE, allow_pickle=False)
    tobs = np.asarray(train["observations"], dtype=np.float32)
    rows = []
    for rid in deep:
        man = load_json(Path(S2_STORE) / S2_PROTOCOL / f"{rid:06d}" / "root_manifest.json")
        o = np.load(Path(S2_STORE) / S2_PROTOCOL / f"{rid:06d}" / "decision_observation.npy")
        g = np.load(Path(S2_STORE) / S2_PROTOCOL / f"{rid:06d}" / "goal_observation.npy")
        elapsed = int(man["decision_elapsed_steps"])
        R = int(MAX_H - elapsed)
        cands = []
        for c in audit[rid]["candidates"]:
            idx = int(c["endpoint_index"])
            cands.append(
                {
                    "candidate_id": int(c["rank"]),
                    "endpoint_index": idx,
                    "z": np.asarray(tobs[idx], dtype=np.float32),
                }
            )
        rows.append(
            {
                "root_id": rid,
                "task_id": int(man["task_id"]),
                "decision_elapsed_steps": elapsed,
                "R": R,
                "j": int(M),
                "obs": np.asarray(o, dtype=np.float32),
                "goal": np.asarray(g, dtype=np.float32),
                "candidates": cands,
                "ideal_candidate_id": int(ranking[rid]["ideal_candidate_id"]),
                "ideal_value": float(ranking[rid]["ideal_value"]),
                "s3_success_loaded": False,
            }
        )
    out = Path(REPO) / EXP_REL / "s3_evaluation_inputs_manifest.json"
    serial = []
    arr_dir = Path(REPO) / EXP_REL / "eval_inputs"
    arr_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        rid = r["root_id"]
        np.savez(
            arr_dir / f"{rid:06d}.npz",
            obs=r["obs"],
            goal=r["goal"],
            z=np.stack([c["z"] for c in r["candidates"]]),
        )
        serial.append(
            {
                "root_id": rid,
                "task_id": r["task_id"],
                "decision_elapsed_steps": r["decision_elapsed_steps"],
                "R": r["R"],
                "j": r["j"],
                "ideal_candidate_id": r["ideal_candidate_id"],
                "ideal_value": r["ideal_value"],
                "candidate_ids": [c["candidate_id"] for c in r["candidates"]],
                "endpoint_indices": [c["endpoint_index"] for c in r["candidates"]],
                "arrays": str(arr_dir / f"{rid:06d}.npz"),
                "s3_success_loaded": False,
            }
        )
    dump_json(
        out,
        {
            "n_deep": len(serial),
            "n_candidates": 8,
            "max_horizon": MAX_H,
            "s3_labels_loaded": False,
            "rows": serial,
        },
    )
    print({"n": len(serial), "out": str(out)}, flush=True)


if __name__ == "__main__":
    main()
