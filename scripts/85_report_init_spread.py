"""Identifiability report across random restarts of the acquisition fit.

A fitted parameter value is only a claim about the camera if the data pin it
down. If several restarts reach comparable held-out distance with that parameter
in very different places, then the statistics do not constrain it and every
sentence about its value, including "it pins at its bound so the physical model
is exhausted", is unsupported. That exact claim was made for the 9-parameter fit
and was never checked this way.

Criterion, fixed in advance: spread across restarts, measured as
(max - min) / (hi - lo) in each parameter's own bounded range. Above 0.25 the
parameter is reported as NOT IDENTIFIED. The threshold is a convention, not a
test, so the full table is printed and the raw fraction is kept in the output so
a reader can apply a different one.

Reads the json files written by script 79. No images, no GPU, so it runs on a
login node.
"""
import argparse, json
import numpy as np
from pathlib import Path

THRESHOLD = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=[
        "results/physics_v2_fit.json",
        "results/physics_v2_fit_init1.json",
        "results/physics_v2_fit_init2.json",
        "results/physics_v2_fit_init3.json"])
    ap.add_argument("--out", default="results/physics_v2_identifiability.json")
    a = ap.parse_args()

    runs = []
    for f in a.files:
        if not Path(f).exists():
            print(f"missing {f}, skipping")
            continue
        runs.append((f, json.load(open(f))))
    if len(runs) < 3:
        raise SystemExit(f"need at least 3 completed fits, have {len(runs)}")

    ranges = runs[0][1]["ranges"]
    keys = list(runs[0][1]["params"].keys())

    print(f"{len(runs)} fits")
    print(f"{'file':44s} {'init':>5s} {'z2_fit':>9s} {'var2_fit':>9s} {'noise_share_gap':>16s}")
    for f, r in runs:
        print(f"{Path(f).name:44s} {r['init_seed']:5d} {r['z2_fitted']:9.3f} "
              f"{r['var2_fitted']:9.3f} {100*r['noise_share_of_gap_closed']:15.1f}%")
    z2 = [r["z2_fitted"] for _, r in runs]
    print(f"held-out z^2 across restarts: min {min(z2):.3f} max {max(z2):.3f} "
          f"spread {max(z2)-min(z2):.3f}")

    print(f"\n{'param':16s} {'min':>12s} {'max':>12s} {'mean':>12s} "
          f"{'spread/range':>13s}  verdict")
    rows, unidentified = {}, []
    for k in keys:
        v = np.array([r["params"][k] for _, r in runs], dtype=float)
        lo, hi = ranges[k]
        frac = float((v.max() - v.min()) / (hi - lo))
        bad = frac > THRESHOLD
        if bad:
            unidentified.append(k)
        rows[k] = {"values": v.tolist(), "min": float(v.min()), "max": float(v.max()),
                   "mean": float(v.mean()), "spread_frac_of_range": frac,
                   "identified": not bad}
        print(f"{k:16s} {v.min():+12.6f} {v.max():+12.6f} {v.mean():+12.6f} "
              f"{frac:13.3f}  {'NOT IDENTIFIED' if bad else 'identified'}")

    print(f"\nNOT IDENTIFIED at spread > {THRESHOLD:.0%} of range "
          f"({len(unidentified)} of {len(keys)}):")
    print(f"  {unidentified if unidentified else 'none'}")
    print("Do not make claims about the values of these parameters, in either "
          "direction, including that they pin at a bound.")
    ident = [k for k in keys if k not in unidentified]
    print(f"identified ({len(ident)}): {ident if ident else 'none'}")

    json.dump({"threshold": THRESHOLD, "n_runs": len(runs),
               "files": [f for f, _ in runs],
               "z2_fitted": z2, "params": rows,
               "unidentified": unidentified, "identified": ident},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
