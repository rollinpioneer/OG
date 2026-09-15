"""Training-seal manifest. Must run before any S3 outcome jsonl is loaded."""
from __future__ import annotations

from pathlib import Path

from execution_aligned_rl.data.audit_assets import sha256_file
from execution_aligned_rl.v3.serialization import dump_json, load_json
from execution_aligned_rl.v4.outcome_pilot.const import EXP_REL, REPO, SEEDS

METHODS = ("B1", "B2", "B3", "B4")


def main() -> None:
    exp = Path(REPO) / EXP_REL
    runs = []
    unstable = []
    for method in METHODS:
        for seed in SEEDS:
            p = exp / "training" / f"{method}_{seed}" / "selected.json"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            rec = load_json(p)
            sel = rec["selected"]
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "unstable": rec["unstable"],
                    "update": sel["update"],
                    "val_bellman_mse": sel["val_bellman_mse"],
                    "finite": sel["finite"],
                    "in_range": sel["in_range"],
                    "monotonic_violation": sel["monotonic_violation"],
                    "sha256": sel["sha256"],
                    "path": sel["path"],
                }
            )
            if rec["unstable"]:
                unstable.append(f"{method}_{seed}")
    by_method = {m: [r for r in runs if r["method"] == m] for m in METHODS}
    method_all_unstable = [m for m, rs in by_method.items() if all(r["unstable"] for r in rs)]
    leak = load_json(exp / "data_leakage_audit.json")
    gate_a = load_json(exp / "gate_a.json")
    status = "TRAINING_SEALED"
    if leak.get("status") != "PASS":
        status = "EA41_HOLD_DATA_LEAKAGE"
    elif method_all_unstable:
        status = "EA41_HOLD_TRAINING_INSTABILITY"
    manifest = {
        "status": status,
        "s3_labels_not_loaded": True,
        "s3_labels_readable_before_seal": False,
        "n_runs": len(runs),
        "runs": runs,
        "unstable_runs": unstable,
        "methods_all_unstable": method_all_unstable,
        "gate_a": gate_a.get("status"),
        "leakage": leak.get("status"),
        "checkpoint_files": [r["path"] for r in runs],
        "checkpoint_sha256": {f"{r['method']}_{r['seed']}": r["sha256"] for r in runs},
    }
    dump_json(exp / "training_complete_manifest.json", manifest)
    dump_json(exp / "checkpoint_manifest.json", {"runs": runs})
    dump_json(exp / "training_manifest.json", {"n_runs": len(runs), "seeds": list(SEEDS), "methods": list(METHODS)})
    print(manifest["status"], len(runs), "unstable", unstable, flush=True)
    if status != "TRAINING_SEALED":
        dump_json(
            exp / "decision.json",
            {
                "status": status,
                "fresh_confirmation_unlocked": False,
                "new_environment_evaluation_authorized": False,
                "policy_training_authorized": False,
                "s4_unlocked": False,
                "human_review": None,
                "new_environment_steps": 0,
                "training_performed": True,
            },
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
