"""D105a experiment B: build data/syn_xmild2 and data/syn_xnat2.

train = the 15,372 C1-prev rows UNCHANGED (data/syn_c1prev/train/labels.csv: C1's Translator-v2 twins + 4,000 repeats of
its DR+ twins) + the translated external photos of brset_gen/ext2/{xmild,xnat}.csv, written with domain = 2 so the
trainer masks their oedema label (scripts/30 masks oedema only for domain-2 rows, with --extra_manifest given).
Images flagged by the copy check (results/copycheck_ext2.json full record in brset_gen/copycheck/ext2.json) are dropped.
Refuses if any training path is outside the allow-list, missing, or duplicated. val/test -> the real mBRSET splits.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
G = Path("/data/users3/sthummala2/brset_gen")
TW = G / "synth/ext2_dc_matfix_s0.4_w1.5"
ALLOW = [G / "synth/full_gen_both_dc_phys_s0.4_w1.5", TW]


def main():
    base = pd.read_csv(ROOT / "data/syn_c1prev/train/labels.csv")
    assert len(base) == 15372 and set(base.domain) <= {0, 1}, (len(base), set(base.domain))
    flagged = set()
    rec = G / "copycheck/ext2.json"
    for s in json.loads(rec.read_text())["sets"].values():
        flagged |= set(s.get("copies", []))
    for arm in ("xmild", "xnat"):
        x = pd.read_csv(G / f"ext2/{arm}.csv")
        n0 = len(x)
        x = x[~(x.name + ".png").isin(flagged)]
        add = pd.DataFrame(dict(file=[str(TW / f"{n}.png") for n in x.name], diabetic_retinopathy=x.dr.values,
                                macular_edema=0, domain=2))
        df = pd.concat([base, add], ignore_index=True)
        bad = [f for f in df.file if not any(Path(f).resolve().is_relative_to(a) for a in ALLOW) or not Path(f).exists()]
        if bad or df.file[df.domain == 2].duplicated().any():
            raise SystemExit(f"refusing {arm}: {len(bad)} bad paths, e.g. {bad[:3]}")
        out = ROOT / "data" / f"syn_{arm}2"
        (out / "train").mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "train/labels.csv", index=False)
        for s in ("val", "test"):
            if not (out / s).exists():
                (out / s).symlink_to(ROOT / "data/finetune_joint" / s)
        print(f"{out}: {len(df)} rows = 15,372 C1-prev + {len(add)} external (of {n0}; {n0 - len(add)} copy-flagged); "
              f"DR+ {int(df.diabetic_retinopathy.sum())} ({df.diabetic_retinopathy.mean():.4f}); domain-2 rows {int((df.domain == 2).sum())}", flush=True)


if __name__ == "__main__":
    main()
