"""Assess one complete three-arm replication with the requested training seed."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from assess_baselines import compare
from train_baseline import assess, node_check, write_json


def main():
    node_check(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    batch = json.loads(Path(args.batch_gate).read_text())
    for arm, folder in batch.items():
        if (Path(folder) / "assessment_predictions.npz").exists():
            raise RuntimeError("Assessment already exists; do not silently rerun")
        assess(SimpleNamespace(output=folder, batch_gate=args.batch_gate,
                               seed=args.seed, workers=4))
    compare(batch, args.output)
    output = Path(args.output)
    result = json.loads(output.read_text())
    result["status"] = f"seed-{args.seed} development assessment, not external evidence"
    result["training_seed"] = args.seed
    write_json(output, result)
    markdown = output.with_suffix(".md")
    text = markdown.read_text().replace("Seed-0 baseline", f"Seed-{args.seed} baseline")
    markdown.write_text(text)


if __name__ == "__main__":
    main()
