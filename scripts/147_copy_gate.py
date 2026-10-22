"""D105 copy gate for one synthetic folder: reads results/copycheck_<tag>.json (scripts/132, --ref mbrset) and applies
the D094 full-set stop rule: FAIL if any image has vessel-pattern similarity > 0.50 to a real mBRSET training photo,
or if the flag rate exceeds 3x the real new-patient OR-flag rate. Exit 0 = pass, 3 = fail."""
import json, sys
from pathlib import Path
tag, folder = sys.argv[1], sys.argv[2]
d = json.loads((Path(__file__).parent.parent / f"results/copycheck_{tag}.json").read_text())
s, ref = d["sets"][folder], d["reference_val"]["or_flag_rate"]
ok = s["max"]["struct"] <= 0.50 and s["copy_rate"] <= 3 * ref
print(f"{folder}: max vessel sim {s['max']['struct']:.3f} (limit 0.50), flag rate {s['copy_rate']:.4f} "
      f"(limit 3 x {ref:.4f} = {3 * ref:.4f}) -> {'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 3)
