"""Verify C1 seeds 0/1/2 and summarize validation-only differences from B1."""
import hashlib
import json
import math
import os
import socket
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "research/codex/step3"
LABELS = ("diabetic_retinopathy", "macular_edema")
METRICS = ("f1_positive", "auroc", "average_precision")


def load(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def domain_label_counts(ledger, domain):
    records={r["stratum"]:r["draws"] for r in ledger["records"] if r["domain"]==domain}
    return [sum(records.values()),records["10"]+records["11"],records["01"]+records["11"]]


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Verification requires an allocated compute node")
    result={"status":"verified","seeds":[0,1,2],"selection_split":"mBRSET val, 725 images",
            "assessment_performed":False,"per_seed":{},"aggregate":{},
            "interpretation_limits":["All model and threshold choices use the same validation split.",
              "Three seeds estimate limited training variability and do not provide external validation.",
              "No Step-3 test assessment was performed."]}
    for seed in result["seeds"]:
        c1_folder=WORK/f"runs/C1_equal_domain_seed{seed}"
        b1_folder=ROOT/f"research/codex/step2/full_pool/runs/B1_seed{seed}"
        complete=load(c1_folder/"complete.json"); ledger=load(c1_folder/"sampling_ledger.json")
        c1=load(c1_folder/"selection_summary.json"); b1=load(b1_folder/"selection_summary.json")
        if not complete["training_complete"] or complete["smoke"] or complete["updates"]!=5775:
            raise AssertionError(f"Incomplete C1 seed {seed}")
        contract=complete["contract"]
        if contract["arm"]!="C1_equal_domain" or contract["seed"]!=seed or contract["single_changed_factor"]!="training sampler":
            raise AssertionError(f"Contract mismatch seed {seed}")
        if c1["metadata"]["checkpoint_sha256"]!=sha(c1_folder/"best.pth"):
            raise AssertionError(f"Checkpoint mismatch seed {seed}")
        if ledger["total_draws"]!=369600 or ledger["domain_draws"]!={"BRSET":184800,"mBRSET":184800}:
            raise AssertionError(f"Domain ledger mismatch seed {seed}")
        events=[json.loads(line) for line in (c1_folder/"events.jsonl").read_text().splitlines()]
        phase=[e for e in events if e["event"]=="phase_complete"]
        expected=[domain_label_counts(ledger,d) for d in ("BRSET","mBRSET")]
        if len(phase)!=1 or phase[0]["exposure"]!=expected:
            raise AssertionError(f"Actual exposure mismatch seed {seed}")
        if (c1_folder/"assessment_summary.json").exists() or (c1_folder/"assessment_predictions.npz").exists():
            raise AssertionError(f"Unexpected test assessment seed {seed}")
        record={"C1":{},"B1":{},"C1-B1":{},"checkpoint_sha256":c1["metadata"]["checkpoint_sha256"],
                "domain_draws":ledger["domain_draws"]}
        for label in LABELS:
            record["C1"][label]={};record["B1"][label]={};record["C1-B1"][label]={}
            for metric in METRICS:
                cv=float(c1["metrics"][label][metric]);bv=float(b1["metrics"][label][metric])
                if not math.isfinite(cv) or not math.isfinite(bv) or c1["metrics"][label]["n_images"]!=725:
                    raise AssertionError(f"Invalid metric seed {seed}/{label}")
                record["C1"][label][metric]=cv;record["B1"][label][metric]=bv;record["C1-B1"][label][metric]=cv-bv
        result["per_seed"][str(seed)]=record
    for label in LABELS:
        result["aggregate"][label]={}
        for metric in METRICS:
            c1=[result["per_seed"][str(s)]["C1"][label][metric] for s in result["seeds"]]
            b1=[result["per_seed"][str(s)]["B1"][label][metric] for s in result["seeds"]]
            delta=[a-b for a,b in zip(c1,b1)]
            result["aggregate"][label][metric]={"C1_values":c1,"B1_values":b1,"difference_values":delta,
                "C1_mean":statistics.mean(c1),"C1_sample_sd":statistics.stdev(c1),
                "B1_mean":statistics.mean(b1),"B1_sample_sd":statistics.stdev(b1),
                "difference_mean":statistics.mean(delta),"difference_sample_sd":statistics.stdev(delta),
                "positive_seeds":sum(x>0 for x in delta)}
    result["engineering_interpretation"]={
        "C1_mean_DR_F1_improvement_at_least_0.01":result["aggregate"][LABELS[0]]["f1_positive"]["difference_mean"]>=.01,
        "C1_mean_ME_F1_not_worse_than_minus_0.01":result["aggregate"][LABELS[1]]["f1_positive"]["difference_mean"]>=-.01}
    (WORK/"c1_validation_three_seeds.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lines=["# C1 equal-domain validation results across three seeds","",
      "All values use the 725-image mBRSET validation split. No Step-3 test assessment was performed.","",
      "| Seed | B1 DR F1 | C1 DR F1 | Difference | B1 ME F1 | C1 ME F1 | Difference |",
      "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in result["seeds"]:
        r=result["per_seed"][str(seed)];
        lines.append(f"| {seed} | {r['B1'][LABELS[0]]['f1_positive']:.4f} | {r['C1'][LABELS[0]]['f1_positive']:.4f} | {r['C1-B1'][LABELS[0]]['f1_positive']:+.4f} | {r['B1'][LABELS[1]]['f1_positive']:.4f} | {r['C1'][LABELS[1]]['f1_positive']:.4f} | {r['C1-B1'][LABELS[1]]['f1_positive']:+.4f} |")
    for label,name in ((LABELS[0],"DR"),(LABELS[1],"ME")):
        a=result["aggregate"][label]["f1_positive"]
        lines.append(f"- {name}: C1 {a['C1_mean']:.4f} ± {a['C1_sample_sd']:.4f}; B1 {a['B1_mean']:.4f} ± {a['B1_sample_sd']:.4f}; mean paired difference {a['difference_mean']:+.4f}; positive in {a['positive_seeds']}/3 seeds.")
    lines.extend(["","This is validation evidence for choosing a sampling baseline, not a test-set, clinical, significance, or novelty claim."])
    (WORK/"c1_validation_three_seeds.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__": main()
