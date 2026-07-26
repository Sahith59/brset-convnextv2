# ConvNeXt V2 Fine-Tuning on BRSET

Fine-tuning ConvNeXt V2 (via `timm`) on the full BRSET dataset, assigned by
Nagur Shareef Shaik as a companion project to the
[mbrset-retfound](https://github.com/Sahith59/mbrset-retfound) RETFound work.

## Current model: multi-label diabetic_retinopathy + macular_edema (primary result)

Per Dr. Ye's direction, the primary task is now independent binary
classification of two BRSET findings — `diabetic_retinopathy` and
`macular_edema` — rather than the 5-class ICDR severity scale used in the
earlier exploration below. ConvNeXt V2 **Large**, fine-tuned at **512x512**
(above its native 384px pretraining), with weighted oversampling + focal
loss and per-label threshold tuning. Full writeup:
[`report/BRSET_ConvNeXtV2_Multilabel_Report.docx`](report/BRSET_ConvNeXtV2_Multilabel_Report.docx).

**Test results** (epoch 9, TTA): `diabetic_retinopathy` AUC 0.9872 / F1 0.8452;
`macular_edema` AUC 0.9926 / F1 0.7869. The paper's own binary DR benchmark is
AUC 0.97 / F1 0.89 — we exceed their AUC, sit slightly below on F1. No paper
baseline exists for macular_edema specifically.

```bash
python scripts/06_prepare_splits_multilabel.py
sbatch scripts/08_train_convnextv2_multilabel.slurm
python scripts/09_evaluate_multilabel.py <checkpoint> <out.txt>
python scripts/10_generate_figures.py
python scripts/11_generate_multilabel_report.py
```

Uses all 16,258 usable images (16,266 minus 8 corrupted/missing — no quality
filter, matching the paper's own inclusion criteria).

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
