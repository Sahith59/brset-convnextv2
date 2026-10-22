"""D097 gate for the lesion painter pilots. Reads results/labelgate_<tag>.json (scripts/134) for each pilot folder.

Pass (all): real-phone judge mean DR score  painted grade 1 >= 0.40, painted grade 2 >= 0.60, grade-0 control <= 0.20;
            BRSET-only judge: painted grade 1 minus grade-0 control >= 0.15.
Choice: smallest w_grade that passes, then smaller strength. Exit 1 if none passes (stop: no classifier).
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", default="abs", choices=["abs", "rel", "rel_ctrl"])
    a = ap.parse_args()
    lg = json.loads((ROOT / f"results/labelgate_{a.tag}.json").read_text())
    rows = []
    for name, r in lg.items():
        m = re.search(r"paintpilot_w([0-9.]+)_s([0-9.]+)$", name)
        if not m:
            continue
        ph, br = r["mbrset_real"]["mean_dr_score_by_grade"], r["brset_only"]["mean_dr_score_by_grade"]
        if a.rule == "rel_ctrl":   # D101: D098 rule + unpainted controls must stay near v1 level (no spots everywhere)
            ok = (ph["1"] - ph["0"] >= 0.15 and ph["2"] - ph["0"] >= 0.30 and br["1"] - br["0"] >= 0.15 and ph["0"] <= 0.35)
        elif a.rule == "rel":   # D098: painted minus unpainted control of the SAME eyes
            ok = (ph["1"] - ph["0"] >= 0.15 and ph["2"] - ph["0"] >= 0.30 and br["1"] - br["0"] >= 0.15)
        else:                 # D097 original
            ok = (ph["1"] >= 0.40 and ph["2"] >= 0.60 and ph["0"] <= 0.20 and br["1"] - br["0"] >= 0.15)
        rows.append(dict(folder=name, w=float(m.group(1)), s=float(m.group(2)), phone_g0=ph["0"], phone_g1=ph["1"],
                         phone_g2=ph["2"], brset_g0=br["0"], brset_g1=br["1"], passed=ok))
        print(f"{name}: phone judge g0 {ph['0']:.3f} g1 {ph['1']:.3f} g2 {ph['2']:.3f} | BRSET judge g0 {br['0']:.3f} "
              f"g1 {br['1']:.3f} (rise {br['1'] - br['0']:+.3f}) -> {'PASS' if ok else 'fail'}", flush=True)
    ok = sorted([r for r in rows if r["passed"]], key=lambda r: (r["w"], r["s"]))
    Path(a.out).write_text(json.dumps({"chosen": ok[0] if ok else None, "all": rows}, indent=1))
    if not ok:
        sys.exit("no pilot setting passes the D097 gate: STOP (no classifier)")
    print("CHOSEN", json.dumps(ok[0]), flush=True)


if __name__ == "__main__":
    main()
