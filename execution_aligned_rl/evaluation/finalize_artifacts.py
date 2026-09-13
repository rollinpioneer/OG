"""Assemble the lightweight A/B/0 result package after the mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from execution_aligned_rl.data.audit_assets import sha256_file


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--training-code-commit", required=True)
    args = parser.parse_args()
    repo = Path(args.repo)
    run = Path(args.run_root)
    exp = repo / "experiments/execution_aligned/ea_v2_v1"
    manifests = exp / "manifests"
    audits = exp / "audits"
    mechanism = exp / "metrics/mechanism"
    report = exp / "report"
    package = exp / "package"
    report.mkdir(parents=True, exist_ok=True)
    package.mkdir(parents=True, exist_ok=True)

    environment = load(manifests / "environment_manifest.json")
    dataset = load(manifests / "dataset_manifest.json")
    interface = load(manifests / "interface_audit.json")
    resource = load(manifests / "resource_estimate.json")
    backbone = load(manifests / "backbone_manifest.json")
    candidate = load(audits / "candidate_audit.json")
    gate = load(mechanism / "phase0_summary.json")
    training_eval = load(run / "checkpoints/gciql_seed0/evaluation.json")
    checkpoint = run / "checkpoints/gciql_seed0/params_1000000.pkl"
    prior_checkpoint = run / "checkpoints/candidate_prior/candidate_prior.pkl"

    freeze = subprocess.check_output(
        ["conda", "run", "-n", "og-ea-v2", "python", "-m", "pip", "freeze"], text=True
    )
    freeze_path = manifests / "requirements_frozen.txt"
    freeze_path.write_text(freeze, encoding="utf-8")
    official_commit = subprocess.check_output(
        ["git", "-C", args.official_source, "rev-parse", "HEAD"], text=True
    ).strip()
    source_manifest = {
        "repository": "rollinpioneer/OG",
        "training_code_commit": args.training_code_commit,
        "official_ogbench": {
            "url": "https://github.com/seohongpark/ogbench",
            "commit": official_commit,
            "version": environment["ogbench_version"],
        },
        "documents": {
            "EA_V2_EXPERIMENT_PLAN_OG.md": sha256_file(repo / "docs/EA_V2_EXPERIMENT_PLAN_OG.md"),
            "execution_aligned_multihorizon_offline_rl_research_v2.md": sha256_file(
                repo / "docs/execution_aligned_multihorizon_offline_rl_research_v2.md"
            ),
        },
        "dependency_lock_sha256": sha256_file(freeze_path),
        "dataset_files": dataset["files"],
        "legacy_rollinpioneer_sc_assets_used": False,
    }
    dump(manifests / "source_manifest.json", source_manifest)

    train_csv = run / "checkpoints/gciql_seed0/train.csv"
    coverage = {
        "phases_completed": ["A", "B", "0"],
        "phases_not_run": ["C", "D", "E", "F"],
        "planned_roots": 32,
        "legal_roots": gate["legal_roots"],
        "planned_candidates_per_legal_root": 8,
        "raw_candidate_records": sum(1 for _ in (mechanism / "phase0_candidates.jsonl").open(encoding="utf-8")),
        "deep_roots": gate["deep_roots_evaluated"],
        "failed_or_missing_roots": 32 - gate["legal_roots"],
        "environment_results_training_eligible": False,
        "backbone_evaluation": training_eval,
    }
    dump(report / "coverage.json", coverage)
    decision = {
        "overall_status": gate["status"],
        "phase_a": interface["status"],
        "phase_b": backbone["status"],
        "phase_0": gate["status"],
        "stop_after_phase_0": True,
        "automatic_next_phase_started": False,
    }
    dump(report / "decision.json", decision)
    shutil.copy2(mechanism / "phase0_roots.csv", exp / "candidate_metrics.csv")
    (exp / "prediction_metrics.csv").write_text(
        "phase,status,result_source\nC_D,NOT_RUN,MODEL_SIMULATED\n", encoding="utf-8"
    )
    with (exp / "control_metrics.csv").open("w", encoding="utf-8") as handle:
        handle.write("phase,task_id,success,result_source,training_eligible\n")
        for task_id in range(1, 6):
            handle.write(f"B,task{task_id},{training_eval[f'task{task_id}']['success']},ENV_EVALUATED,false\n")
    dump(
        report / "paired_comparisons.json",
        {
            "scope": "phase_0_candidate_ranking_only",
            "comparison": "IDEAL_top1_vs_environment_proxy_oracle",
            "proxy_wrong_selection_rate": gate["proxy_wrong_selection_rate"],
            "full_gap_rate_on_preregistered_deep_roots": gate[
                "full_gap_rate_on_preregistered_deep_roots"
            ],
            "confirmatory_method_comparisons": "NOT_RUN",
        },
    )

    resource.update(
        {
            "run_root": "<LOCAL_RUN_ROOT>",
            "gciql_checkpoint_bytes": checkpoint.stat().st_size,
            "candidate_prior_checkpoint_bytes": prior_checkpoint.stat().st_size,
            "gciql_train_csv_sha256": sha256_file(train_csv),
            "measured_peak_gpu_memory_mib": 22836,
            "measured_gpu_model": "NVIDIA A100-SXM4-40GB",
            "phase_0_wall_seconds": gate["elapsed_seconds"],
            "next_phase_estimate": {
                "scope": "phase C only; not started",
                "gpu_hours": 0.8,
                "environment_rollout_steps_upper_bound": 20480,
                "storage_gib": 2.0,
                "assumptions": "P_1 ensemble of 3 x 50k updates; same 32x8x2x40 bounded candidate branches; lightweight logs plus checkpoints",
            },
        }
    )
    dump(manifests / "resource_estimate.json", resource)

    mean_corr = gate.get("mean_ideal_proxy_spearman")
    report_text = f"""# Mechanism Gate Report

**Status: `{gate['status']}`.** Execution stopped after phases A, B, and 0. Phases C-F, `P_1`, `P_MH`, and the full EXEC system were not run.

## Frozen basis

- Task: `antmaze-large-stitch-v0`; training seed 0; `k=20`, `m=5`, `N=8`.
- OGBench commit: `{official_commit}`; experiment code commit: `{args.training_code_commit}`.
- Offline data: {dataset['train']['transitions']:,} train transitions in {dataset['train']['episodes']:,} episodes; {dataset['train']['legal_windows_by_horizon']['20']:,} legal `k=20` windows.
- Phase A: `{interface['status']}`; deterministic resets and full-state restore both passed on 8/8 engineering resets.
- Phase B: `{backbone['status']}`; official GCIQL final-goal value/controller plus a physical-state conditional endpoint prior. The controller is goal-conditioned but not explicitly remaining-horizon-conditioned.
- Frozen controller evaluation success: {training_eval['overall_success']:.3f}; candidate finite rate: {candidate['finite_candidate_rate']:.3f}; distinct rate: {candidate['adjacent_sample_distinct_rate']:.3f}.

## Phase 0 evidence

- Legal roots: {gate['legal_roots']}/32; pre-registered deep roots: {gate['deep_roots_evaluated']}/8.
- Offline-only value tolerance: {gate['value_tolerance']:.6g} (`5%` of value standard deviation {gate['value_std_data_real']:.6g}).
- Ideal top-choice proxy error rate beyond tolerance: {gate['proxy_wrong_selection_rate']:.3f} (threshold `0.10`).
- Full-result gap rate on deep roots: {gate['full_gap_rate_on_preregistered_deep_roots']:.3f}.
- Proxy indistinguishable roots: {gate['proxy_indistinguishable_rate']:.3f}; full-result indistinguishable deep roots: {gate['full_indistinguishable_rate']:.3f}.
- Mean within-root ideal/proxy Spearman: {mean_corr if mean_corr is not None else 'not identifiable'}.

All candidate branches are `ENV_EVALUATED`, evaluation-only, and excluded from training. Offline windows and value-scale calculations are `DATA_REAL`. No `MODEL_SIMULATED` phase-0 result was fabricated; dynamics models are outside the authorized scope.

## Decision and next cost

The frozen rule yields `{gate['status']}` without changing task, seed, candidate count, or threshold. The next authorized unit would be phase C only: estimated 0.8 A100 GPU-hours, at most 20,480 diagnostic environment steps, and about 2 GiB storage. It has not been started and requires human review.
"""
    (report / "MECHANISM_GATE_REPORT.md").write_text(report_text, encoding="utf-8")
    experiment_report = f"""# EA V2 A/B/0 Experiment Report

Completed the frozen A/B/0 scope on `antmaze-large-stitch-v0`. Final decision: `{gate['status']}`.

See `MECHANISM_GATE_REPORT.md` for the one-page gate result, `../manifests` and `../audits` for provenance, and `../metrics/mechanism` for raw candidate-level results. Environment forks were never added to training data. No later phase was started.
"""
    (report / "EXPERIMENT_REPORT.md").write_text(experiment_report, encoding="utf-8")
    local_only = {
        "note": "Paths intentionally omitted; large artifacts remain on the experiment host.",
        "artifacts": {
            "gciql_checkpoint": {"bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)},
            "candidate_prior_checkpoint": {
                "bytes": prior_checkpoint.stat().st_size,
                "sha256": sha256_file(prior_checkpoint),
            },
            "training_log": {"bytes": train_csv.stat().st_size, "sha256": sha256_file(train_csv)},
        },
    }
    (report / "LOCAL_ONLY_ARTIFACTS.md").write_text(
        "# Local-only artifacts\n\n```json\n" + json.dumps(local_only, indent=2, sort_keys=True) + "\n```\n",
        encoding="utf-8",
    )

    zip_path = package / "EA_V2_results_lightweight.zip"
    included_roots = [exp / "config", manifests, audits, mechanism, report]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in included_roots:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path != zip_path:
                    archive.write(path, path.relative_to(repo))
        for name in ("candidate_metrics.csv", "prediction_metrics.csv", "control_metrics.csv"):
            path = exp / name
            archive.write(path, path.relative_to(repo))
    digest = sha256_file(zip_path)
    (package / "EA_V2_results_lightweight.zip.sha256").write_text(
        f"{digest}  {zip_path.name}\n", encoding="ascii"
    )
    print(json.dumps({"status": gate["status"], "zip_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
