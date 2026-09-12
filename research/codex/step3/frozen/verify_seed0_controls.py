"""Verify Step-3 seed-0 controls and consolidate validation-only results."""
import hashlib
import json
import math
import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step3"
RUNS = WORK / "runs"
ARMS = ("C1_equal_domain", "C2_target_label", "C3_equal_domain_target_label")
LABELS = ("diabetic_retinopathy", "macular_edema")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def domain_label_counts(ledger, domain):
    records = {r["stratum"]: r["draws"] for r in ledger["records"] if r["domain"] == domain}
    return [sum(records.values()), records["10"] + records["11"], records["01"] + records["11"]]


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Verification requires an allocated compute node")
    preflight = load(WORK / "preflight.json")
    reference = load(ROOT / "research/codex/step2/full_pool/runs/B1_seed0/selection_summary.json")
    result = {"status": "verified", "seed": 0, "selection_split": "mBRSET val, 725 images",
              "assessment_performed": False, "reference": {}, "controls": {},
              "screen_rule": "Replicate if absolute DR F1 delta >= 0.01 or absolute ME F1 delta >= 0.01; final interpretation remains a human decision.",
              "limitations": ["Validation F1 thresholds are selected on the same validation split.",
                              "The test split was not used for Step-3 screening.",
                              "Sampler stratification changes draw composition and within-batch composition variance."]}
    for label in LABELS:
        m = reference["metrics"][label]
        result["reference"][label] = {key: m[key] for key in ("f1_positive", "auroc", "average_precision")}

    for arm in ARMS:
        folder = RUNS / f"{arm}_seed0"
        complete = load(folder / "complete.json")
        summary = load(folder / "selection_summary.json")
        ledger = load(folder / "sampling_ledger.json")
        if not complete["training_complete"] or complete["smoke"] or complete["updates"] != 5775:
            raise AssertionError(f"Incomplete run: {arm}")
        contract = complete["contract"]
        if contract["arm"] != arm or contract["seed"] != 0 or contract["single_changed_factor"] != "training sampler":
            raise AssertionError(f"Contract mismatch: {arm}")
        if summary["metadata"]["checkpoint_sha256"] != sha(folder / "best.pth"):
            raise AssertionError(f"Checkpoint hash mismatch: {arm}")
        expected = preflight["ledgers"][arm]
        for key in ("total_draws", "domain_draws", "positive_draws", "records", "declared_group_quotas"):
            if ledger[key] != expected[key]:
                raise AssertionError(f"Ledger mismatch {arm}/{key}")
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        phase = [event for event in events if event["event"] == "phase_complete"]
        if len(phase) != 1:
            raise AssertionError(f"Missing phase completion: {arm}")
        expected_exposure = [domain_label_counts(ledger, domain) for domain in ("BRSET", "mBRSET")]
        if phase[0]["exposure"] != expected_exposure:
            raise AssertionError(f"Actual exposure mismatch: {arm}")
        if (folder / "assessment_summary.json").exists() or (folder / "assessment_predictions.npz").exists():
            raise AssertionError(f"Unexpected test assessment: {arm}")
        control = {"checkpoint_sha256": summary["metadata"]["checkpoint_sha256"],
                   "checkpoint_update": summary["metadata"]["checkpoint_update"],
                   "thresholds": summary["metadata"]["thresholds"], "domain_draws": ledger["domain_draws"],
                   "metrics": {}, "differences_from_B1": {}}
        for label in LABELS:
            m = summary["metrics"][label]
            if m["n_images"] != 725 or any(not math.isfinite(float(m[k])) for k in ("f1_positive", "auroc", "average_precision")):
                raise AssertionError(f"Invalid validation metric: {arm}/{label}")
            control["metrics"][label] = {key: m[key] for key in ("f1_positive", "auroc", "average_precision")}
            control["differences_from_B1"][label] = {key: m[key] - reference["metrics"][label][key]
                                                        for key in ("f1_positive", "auroc", "average_precision")}
        dr = abs(control["differences_from_B1"][LABELS[0]]["f1_positive"])
        me = abs(control["differences_from_B1"][LABELS[1]]["f1_positive"])
        control["replication_screen"] = {"absolute_DR_F1_delta_at_least_0.01": dr >= .01,
                                         "absolute_ME_F1_delta_at_least_0.01": me >= .01,
                                         "passes_numeric_screen": dr >= .01 or me >= .01}
        result["controls"][arm] = control

    out = WORK / "seed0_validation_summary.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Step-3 seed-0 validation-only comparison", "",
             "These values screen sampling controls on the 725-image mBRSET validation split. No Step-3 test assessment was performed.", "",
             "| Arm | DR F1 | Δ vs B1 | DR AUROC | ME F1 | Δ vs B1 | ME AUROC | Replication screen |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    ref_dr, ref_me = result["reference"][LABELS[0]], result["reference"][LABELS[1]]
    lines.append(f"| B1 existing | {ref_dr['f1_positive']:.4f} | — | {ref_dr['auroc']:.4f} | {ref_me['f1_positive']:.4f} | — | {ref_me['auroc']:.4f} | reference |")
    for arm in ARMS:
        c=result["controls"][arm]; dr=c["metrics"][LABELS[0]]; me=c["metrics"][LABELS[1]]
        ddr=c["differences_from_B1"][LABELS[0]]["f1_positive"]; dme=c["differences_from_B1"][LABELS[1]]["f1_positive"]
        lines.append(f"| {arm.split('_')[0]} | {dr['f1_positive']:.4f} | {ddr:+.4f} | {dr['auroc']:.4f} | {me['f1_positive']:.4f} | {dme:+.4f} | {me['auroc']:.4f} | {'pass' if c['replication_screen']['passes_numeric_screen'] else 'no numeric trigger'} |")
    lines.extend(["", "Passing the numeric screen prioritizes replication; it is not a test-set, significance, clinical, or novelty claim."])
    (WORK / "seed0_validation_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
