"""Manifest of EXTERNAL graded fundus images (DDR, Aptos, Messidor2).

Why this exists. The measured bottleneck is mild disease: BRSET's training split
holds only ~115 ICDR grade-1 images, so every intervention that merely reweights
or re-supervises those images shows the model the same pictures more often. The
oracle test confirmed the test was underpowered rather than the idea wrong --
only 59 images (0.51% of training data) differed between its arms.

These public datasets are already on the cluster and carry the same ICDR 0-4
scale, giving 7,093 real mild cases against our 115.

Only the DR grade is imported. DDR/Aptos/Messidor2 carry no macular-oedema
label, so the manifest marks that label INVALID rather than writing a 0, which
would otherwise teach the model that every imported eye is oedema-free.
"""
import argparse, json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
DS = Path("/data/users4/nshaik3/Datasets")

SOURCES = {
    "ddr": dict(csv=DS / "DDR/DR_grading.csv", img=DS / "DDR/DR_grading",
                idcol="id_code", gcol="diagnosis"),
    "aptos": dict(csv=DS / "Aptos/train.csv", img=DS / "Aptos",
                  idcol="id_code", gcol="diagnosis"),
    "messidor2": dict(csv=DS / "Messidor2/messidor_data.csv",
                      img=DS / "Messidor2/messidor-2",
                      idcol="id_code", gcol="diagnosis"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="ddr")
    ap.add_argument("--out", default=str(ROOT / "results/extra_manifest.csv"))
    a = ap.parse_args()

    rows, report = [], {}
    for name in a.sources.split(","):
        s = SOURCES[name]
        if not Path(s["csv"]).exists():
            print(f"  {name}: csv missing, skipped"); continue
        df = pd.read_csv(s["csv"])
        index = {p.name: p for p in Path(s["img"]).rglob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        n_hit = 0
        for iid, g in zip(df[s["idcol"]].astype(str), df[s["gcol"]]):
            if pd.isna(g):
                continue
            p = index.get(iid) or index.get(Path(iid).name)
            if p is None:
                continue
            rows.append(dict(source=name, file=str(p.resolve()), grade=int(g)))
            n_hit += 1
        report[name] = dict(csv_rows=int(len(df)), files_indexed=int(len(index)),
                            matched=int(n_hit))
        print(f"  {name}: {n_hit} of {len(df)} rows matched a file "
              f"({len(index)} files indexed)", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no external images matched; manifest not written")
    out.to_csv(a.out, index=False)
    print(f"\n{'grade':>6} {'n':>7}")
    for g, n in out.grade.value_counts().sort_index().items():
        print(f"{g:>6} {n:>7}")
    mild = int(out.grade.isin([1, 2]).sum()); sev = int(out.grade.isin([3, 4]).sum())
    print(f"\n  mild (1-2)   {mild}")
    print(f"  severe (3-4) {sev}   <- caps any matched-count design")
    print(f"  usable matched arm size: {min(mild, sev)}")
    Path(str(a.out).replace(".csv", ".json")).write_text(json.dumps(
        dict(sources=report, grade_hist={str(k): int(v) for k, v in
             out.grade.value_counts().sort_index().items()},
             mild=mild, severe=sev, matched_arm_cap=min(mild, sev)), indent=2))
    print(f"\nwrote {a.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
