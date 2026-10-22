"""Pick the cascade setting by the rule fixed in D091 (no test data involved).

Among the pilot folders of one gate run: keep settings whose copy rate is <= 1 %, then take the
highest DR AUROC of the real-mBRSET judge against the conditioning label. Refuses (exit 1) if
no setting qualifies, or if the best judge AUROC is below --floor (the generator would not be
carrying the label; stop and look rather than train classifiers on it).
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="gate tag (results/copycheck_<tag>.json, labelgate_<tag>.json)")
    ap.add_argument("--floor", type=float, default=0.80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cc = json.loads((ROOT / f"results/copycheck_{a.tag}.json").read_text())["sets"]
    lg = json.loads((ROOT / f"results/labelgate_{a.tag}.json").read_text())
    rows = []
    for name, c in cc.items():
        j = lg.get(name, {}).get("mbrset_real", {})
        m = re.match(r"pilot_(.+)_w([0-9.]+)_s([0-9.]+)$", name)
        if not m or j.get("dr_auroc") is None:
            continue
        rows.append(dict(folder=name, base_tag=m.group(1), w=float(m.group(2)), strength=float(m.group(3)),
                         copy_rate=c["copy_rate"], judge_auroc=j["dr_auroc"],
                         judge_auroc_brset=lg[name].get("brset_only", {}).get("dr_auroc"),
                         by_grade=j.get("mean_dr_score_by_grade")))
    for r in sorted(rows, key=lambda r: -r["judge_auroc"]):
        print(f"{r['folder']:45s} copy {r['copy_rate']:.3f}  judge AUROC {r['judge_auroc']:.4f} "
              f"(BRSET judge {r['judge_auroc_brset']:.4f})  by grade {json.dumps(r['by_grade'])}", flush=True)
    ok = [r for r in rows if r["copy_rate"] <= 0.01]
    if not ok:
        sys.exit("no setting has copy rate <= 1 %; stopping")
    best = max(ok, key=lambda r: r["judge_auroc"])
    if best["judge_auroc"] < a.floor:
        sys.exit(f"best judge AUROC {best['judge_auroc']:.4f} < floor {a.floor}; stopping")
    Path(a.out).write_text(json.dumps({"chosen": best, "all": rows}, indent=1))
    print("CHOSEN", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
