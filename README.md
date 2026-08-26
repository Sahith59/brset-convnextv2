# ConvNeXt V2 on BRSET, and transfer to mBRSET

Two connected pieces of work, both directed by Dr. Dong Hye Ye.

**Part 1** builds a strong ConvNeXt V2 baseline on BRSET (tabletop fundus
camera, hospital clinic) and addresses its severe class imbalance.
**Part 2** studies why that model degrades on mBRSET (handheld phone camera,
community screening) and what can be done about it.

Companion project: [mbrset-retfound](https://github.com/Sahith59/mbrset-retfound).

## Part 1: the BRSET baseline

ConvNeXt V2 **Large**, fine-tuned at **512x512**, multi-label
(`diabetic_retinopathy` + `macular_edema`), AdamW at lr 5e-5, 40 epochs,
EMA 0.999, 4-way flip TTA, and per-label cutoffs chosen on validation by a
200-resample bootstrap.

| Configuration | DR AUC | DR F1 (class-macro) | DR missed | ME AUC | ME F1 (class-macro) |
|---|---|---|---|---|---|
| Focal + oversampling | 0.9924 | 0.9297 | 16/162 | 0.9886 | 0.8707 |
| **Focal, oversampling off** (selected) | 0.9906 | **0.9374** | **10/162** | 0.9957 | 0.8852 |
| Asymmetric focal, g=2 d=0.60 | 0.9916 | 0.9201 | 19/162 | 0.9963 | 0.8941 |
| Asymmetric focal, g=3 d=0.75 | 0.9872 | 0.9224 | 20/162 | 0.9802 | 0.8907 |

Three findings worth recording:

- **The weighted oversampler was harmful.** It lifted the positive rate the
  loss actually saw from 6.6% to 61%, inverting the imbalance it was meant to
  correct. Removing it cut missed DR cases from 16 to 10 of 162.
- **Asymmetric focal loss did not beat focal loss.** Tested at Dr. Ye's
  direction: significantly worse on DR (paired bootstrap, p = 0.013) and no
  measurable difference on ME. Even at its best attainable cutoff it reaches
  only 0.9292 against focal's 0.9374, so it is not a threshold artifact.
- **The selected run diverged to NaN at epoch 26 under fp16.** ConvNeXt V2's
  GRN takes an L2 norm over a 512x512 map, which can exceed the fp16 maximum.
  `--amp_dtype bf16` plus a non-finite-batch guard fixes it. A clean 40-epoch
  bf16 rerun reaches the identical peak validation score (0.9218), so the
  divergence cost nothing and the baseline is confirmed.

```bash
python scripts/06_prepare_splits_multilabel.py
sbatch scripts/39_afl_vs_focal_40ep.slurm      # the 40-epoch loss comparison
sbatch scripts/45_part1_focal_bf16_rerun.slurm # bf16 confirmation run
python scripts/34_bootstrap_strong_baseline.py
```

## Part 2: BRSET to mBRSET transfer

The Part-1 model, applied unchanged to mBRSET. No mBRSET image is used in
training.

| Diabetic retinopathy | On BRSET | On mBRSET |
|---|---|---|
| AUC | 0.9906 | 0.9060 |
| F1 (class-macro) | 0.9374 | 0.8668 |
| Recall | 0.9383 | 0.6855 |
| Cases missed | 10/162 | 50/159 |

Ranking barely suffers; the decision collapses. The gap was then split by
measurement (diseased-class F1 on mBRSET):

| | F1 |
|---|---|
| Transferred as is | 0.7842 |
| Best reachable at any cutoff | 0.7927 |
| Fine-tuned on labelled mBRSET | 0.8317 |

Sweeping every cutoff from 0.005 to 0.995 shows 0.7927 is the ceiling for any
method that only rescales scores, which rules out calibration, temperature
scaling and label-shift correction by measurement. **18% of the remaining gap
is the decision cutoff and 82% is the representation.** The shift is not only
a prevalence change: ROC is invariant to class balance, yet AUC fell from
0.9906 to 0.9060.

Literature review and proposed direction:
[`report/BRSET_to_mBRSET_Literature_Review.docx`](report/BRSET_to_mBRSET_Literature_Review.docx).
Running deck: [`presentation/BRSET_to_mBRSET_Part2.pptx`](presentation/BRSET_to_mBRSET_Part2.pptx).

```bash
sbatch scripts/42_crossdevice_eval_newbaseline.slurm  # zero-shot transfer test
sbatch scripts/43_mbrset_part1_recipe.slurm           # mBRSET from scratch
sbatch scripts/44_mbrset_finetune_from_brset.slurm    # BRSET -> mBRSET finetune
```

## Earlier exploration: 5-class ICDR severity grading (superseded)

Full writeup: [`report/BRSET_ConvNeXtV2_ICDR5_Report.docx`](report/BRSET_ConvNeXtV2_ICDR5_Report.docx).
ConvNeXt V2 Base on the 0-4 ICDR scale reproduced the same Grade 1 (Mild
NPDR) collapse found on mBRSET/RETFound; a follow-up oversampling + focal
loss experiment partially fixed it (Grade 1 F1 0.000 -> 0.188). This line of
work was set aside in favor of the multi-label task above, per Dr. Ye.

```bash
python scripts/01_prepare_splits.py
sbatch scripts/02_train_convnextv2.slurm                              # baseline
sbatch scripts/04_train_convnextv2_oversample_focal.slurm             # oversample + focal loss
python scripts/03_evaluate.py <checkpoint> <out.txt>
```

Used 14,273 images (16,266 minus 1,986 failing quality control minus 7
corrupted), patient-level 70/15/15 split.

## What's in this repo (and what isn't)

BRSET is a **credentialed-access PhysioNet dataset**. This repo contains only
code, reports, and aggregate results (confusion matrices, per-epoch metrics)
— it does **not** contain any patient images, per-patient labels, or model
checkpoints trained on the data, in line with the dataset's Data Use
Agreement. To reproduce this work you need your own PhysioNet credentialed
access to BRSET; point the `RAW_CSV`/`RAW_IMAGES` paths in the relevant
`scripts/0*_prepare_splits*.py` at your local copy.
