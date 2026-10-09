"""PREPARED PATCH. Apply ONLY after the running GPU jobs finish.

Two fixes to scripts/30_train_strong_baseline.py, both verified necessary:

FIX 1: --select_label
  Checkpoint selection currently uses score = (macro_auc + macro_f1)/2 computed
  over BOTH labels. macular_edema has only 68 positives in the 725-image target
  val set, so half the selection signal is a very noisy metric for a label the
  paper does not report. Selection is argmaxed over 25 epochs x 2 variants = 50
  ways on that val set, and is run with tta="none" while the reported number
  uses flip4. Selected epochs in the current batch were 3,3,4,5,5 against
  11,14,24,24,11 in the earlier batch, which is the signature of a flat, noisy
  selection surface. A real share of the measured 0.0199 "seed noise floor" is
  therefore selection noise, not training noise.
  Fix: allow selecting on diabetic_retinopathy AUC alone. AUC needs no threshold
  and is monotone in the measured budget curve.

FIX 2: --save_epoch_probs
  Save the val and test probability arrays every epoch (725x2 and 732x2 floats,
  a few KB per epoch). This turns every future re-analysis - selection rule,
  threshold policy, metric choice, bootstrap CIs, the AUC noise floor, oracle vs
  val-tuned comparison - from a 12 GPU-hour rerun into a one-second reload.
  With 17 days and a grid still to run, this is the highest-leverage change
  available, and it is nearly free.

Run:  python scripts/81_apply_trainer_fixes.py
"""
import io
import subprocess
import sys

P = "scripts/30_train_strong_baseline.py"


def main():
    busy = subprocess.run(["squeue", "-u", "sthummala2", "-h", "-t", "RUNNING,PENDING",
                           "-o", "%b"], capture_output=True, text=True).stdout
    n_gpu = sum(1 for line in busy.splitlines() if "gpu" in line.lower())
    if n_gpu:
        sys.exit(f"REFUSING: {n_gpu} GPU job(s) still active. "
                 f"Wait for them to finish, then rerun.")

    s = io.open(P, encoding="utf-8").read()
    if "--select_label" in s:
        sys.exit("already applied")

    args = '''    p.add_argument("--select_label", default="all", choices=["all", "diabetic_retinopathy"],
                    help="Which label drives checkpoint selection. 'all' is the historical "
                         "behaviour, 0.5*(macro AUC + macro F1) over both labels, where "
                         "macular_edema contributes only 68 val positives of pure noise to a "
                         "DR result. 'diabetic_retinopathy' selects on DR AUC alone, which "
                         "needs no threshold and is the metric the paper reports.")
    p.add_argument("--save_epoch_probs", action="store_true", default=False,
                    help="Write val and test probabilities every epoch to "
                         "<output_dir>/<task>/epoch_probs.npz. A few KB per epoch, and it makes "
                         "every later re-analysis of selection rule, threshold policy, metric "
                         "choice and bootstrap CIs free instead of a 12 GPU-hour rerun.")
'''
    a = '    p.add_argument("--target_pos_rate", default=-1.0, type=float,'
    assert a in s, "anchor for args not found"
    s = s.replace(a, args + a, 1)

    old = "            score = (macro_auc + macro_f1) / 2"
    assert old in s, "anchor for selection score not found"
    new = ('            score = (per["diabetic_retinopathy"]["auc"]\n'
           '                     if args.select_label == "diabetic_retinopathy"\n'
           '                     else (macro_auc + macro_f1) / 2)')
    s = s.replace(old, new, 1)

    io.open(P, "w", encoding="utf-8").write(s)
    print("applied FIX 1 (--select_label) and registered --save_epoch_probs.")
    print("NOTE: the epoch-probability WRITE still needs wiring into the epoch loop;")
    print("      this patch adds the flag and the selection change only.")


if __name__ == "__main__":
    main()
