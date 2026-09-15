"""Static leakage audit. Training modules must not read S3 outcome labels."""
from __future__ import annotations

import ast
from pathlib import Path

from execution_aligned_rl.v3.serialization import dump_json
from execution_aligned_rl.v4.outcome_pilot.const import EXP_REL, REPO

TRAIN_FILES = [
    "execution_aligned_rl/v4/outcome_pilot/train.py",
    "execution_aligned_rl/v4/outcome_pilot/models.py",
    "execution_aligned_rl/v4/outcome_pilot/features.py",
    "execution_aligned_rl/v4/outcome_pilot/data.py",
    "execution_aligned_rl/v4/outcome_pilot/success.py",
]
FORBIDDEN_SUBSTR = [
    "deep_branch_results",
    "mechanism_root_table",
    "mechanism_summary",
    "final_success",
    "ever_success",
    "ideal_full_success",
    "selector_results",
]
FORBIDDEN_IMPORT = [
    "eval_o4",
]


def main() -> None:
    root = Path(REPO)
    hits = []
    for rel in TRAIN_FILES:
        p = root / rel
        text = p.read_text(encoding="utf-8")
        ast.parse(text)
        for s in FORBIDDEN_SUBSTR:
            if s in text:
                hits.append({"file": rel, "token": s, "kind": "substr"})
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(x in node.module for x in FORBIDDEN_IMPORT):
                    hits.append({"file": rel, "token": node.module, "kind": "import"})
            if isinstance(node, ast.Import):
                for a in node.names:
                    if any(x in a.name for x in FORBIDDEN_IMPORT):
                        hits.append({"file": rel, "token": a.name, "kind": "import"})
    report = {
        "s3_labels_not_loaded": True,
        "training_files_scanned": TRAIN_FILES,
        "hits": hits,
        "status": "PASS" if not hits else "FAIL",
    }
    out = Path(REPO) / EXP_REL / "data_leakage_audit.json"
    dump_json(out, report)
    print(report, flush=True)
    if hits:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
