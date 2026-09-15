"""Write the EA-V4 evaluator pilot report from sealed evaluation artifacts."""
from __future__ import annotations

from pathlib import Path

from execution_aligned_rl.v3.serialization import load_json
from execution_aligned_rl.v4.outcome_pilot.const import EXP_REL, REPO


def main() -> None:
    exp = Path(REPO) / EXP_REL
    dec = load_json(exp / "decision.json")
    ens = load_json(exp / "ensemble_results.json")
    gate_a = load_json(exp / "gate_a.json")
    leak = load_json(exp / "data_leakage_audit.json")
    seal = load_json(exp / "training_complete_manifest.json")
    boot = load_json(exp / "bootstrap_intervals.json")
    lines = []
    lines.append("# EA-V4 Outcome Evaluator Pilot Report")
    lines.append("")
    lines.append(f"Status: `{dec['status']}`")
    lines.append("")
    lines.append("This is an exploratory offline evaluator pilot, not a preregistered confirmation.")
    lines.append("")
    lines.append("## Locks")
    lines.append("")
    lines.append("- new_environment_steps: 0")
    lines.append("- fresh_confirmation_unlocked: false")
    lines.append("- new_environment_evaluation_authorized: false")
    lines.append("- policy_training_authorized: false")
    lines.append("- s4_unlocked: false")
    lines.append(f"- Gate A: {gate_a.get('status')}")
    lines.append(f"- leakage: {leak.get('status')}")
    lines.append(f"- training seal: {seal.get('status')}")
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append("- IDEAL 15/35")
    lines.append("- ORACLE_FULL 21/35")
    lines.append("- ORACLE_PROXY 10/35")
    lines.append("- DIRECT_35 14/35")
    lines.append("")
    lines.append("## Ensemble")
    lines.append("")
    for name in ("B0", "B1", "B2", "B3", "B4", "B5"):
        rec = ens.get(name, dec.get(name, {}))
        if not rec:
            continue
        lines.append(
            f"- {name}: success={rec.get('success')}/35 rescue={rec.get('rescue')} harm={rec.get('harm')} net={rec.get('net')} interv={rec.get('intervention')} pairwise_auc={rec.get('root_pairwise_auc')}"
        )
    lines.append("")
    lines.append("## Bootstrap (task-stratified, 10000)")
    lines.append("")
    lines.append("```json")
    import json

    lines.append(json.dumps(boot, indent=2)[:4000])
    lines.append("```")
    lines.append("")
    lines.append("## Decision locks")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({k: dec[k] for k in dec if k in (
        'status','fresh_confirmation_unlocked','new_environment_evaluation_authorized','policy_training_authorized','s4_unlocked','human_review','new_environment_steps')}, indent=2))
    lines.append("```")
    (exp / "EA_V4_EVALUATOR_PILOT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote report", flush=True)


if __name__ == "__main__":
    main()
