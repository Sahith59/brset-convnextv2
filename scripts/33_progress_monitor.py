"""Emit a verdict whenever either strong-baseline run crosses a milestone epoch.

Compares each run's best-so-far validation macro-F1 against the best prior
BRSET run (convnextv2_large_BRSET_multilabel_512, best val macro_f1 0.8435) at
the SAME epoch, so "is it improving" is answered against a real reference
trajectory rather than a vibe. Also reports if a run dies.
"""
import json
import os
import subprocess
import sys
import time

RES = "/home/users/sthummala2/brset-convnextv2/results"
RUNS = {
    "ASL(two-dial)": (f"{RES}/convnextv2_large_BRSET_strong_baseline/log.txt", "3950186"),
    "focal(plain)": (f"{RES}/convnextv2_large_BRSET_strong_baseline_focal/log.txt", "3950187"),
}
BENCH = f"{RES}/convnextv2_large_BRSET_multilabel_512/log.txt"
BENCH_FINAL_F1 = 0.8435
MILESTONES = [5, 10, 15, 20, 25, 30, 35, 39]


def best_so_far(path, upto):
    if not os.path.exists(path):
        return None, 0
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return None, 0
    keep = [r["macro_f1"] for r in rows if r["epoch"] <= upto]
    return (max(keep) if keep else None), len(rows)


def job_alive(jobid):
    try:
        out = subprocess.run(["squeue", "-j", jobid, "-h"], capture_output=True,
                             text=True, timeout=30).stdout
        return jobid in out
    except Exception:
        return True  # don't cry wolf on a squeue hiccup


def main():
    fired = {name: set() for name in RUNS}
    dead = set()
    print("monitor armed: milestones at epochs " + ",".join(map(str, MILESTONES)), flush=True)

    while True:
        all_done = True
        for name, (path, jobid) in RUNS.items():
            alive = job_alive(jobid)
            if alive:
                all_done = False
            elif name not in dead:
                dead.add(name)
                _, n = best_so_far(path, 10**9)
                print(f"[{name}] JOB {jobid} LEFT QUEUE after {n} epochs "
                      f"(finished or died - check final results)", flush=True)

            cur, n_epochs = best_so_far(path, 10**9)
            if cur is None:
                continue
            last_epoch = n_epochs - 1

            for ms in MILESTONES:
                if ms in fired[name] or last_epoch < ms:
                    continue
                fired[name].add(ms)
                mine, _ = best_so_far(path, ms)
                ref, _ = best_so_far(BENCH, ms)
                if mine is None:
                    continue
                if ref is None:
                    print(f"[{name}] epoch {ms}: best val macro_f1={mine:.4f} "
                          f"(no reference at this epoch)", flush=True)
                    continue
                delta = mine - ref
                if delta >= 0.005:
                    verdict = "AHEAD of prior best run"
                elif delta > -0.005:
                    verdict = "level with prior best run"
                elif delta > -0.02:
                    verdict = "slightly behind prior best run"
                else:
                    verdict = "BEHIND prior best run - may not beat 0.869 F1"
                print(f"[{name}] epoch {ms}: best val macro_f1={mine:.4f} vs "
                      f"prior-run {ref:.4f} at same epoch (delta {delta:+.4f}) -> {verdict}. "
                      f"Prior run's FINAL best was {BENCH_FINAL_F1:.4f}.", flush=True)

        if all_done and len(dead) == len(RUNS):
            print("both runs have left the queue - monitor exiting", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
