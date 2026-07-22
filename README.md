# ConvNeXt V2 Fine-Tuning on BRSET

Fine-tuning ConvNeXt V2 (`convnextv2_base.fcmae_ft_in22k_in1k`, via `timm`) on the
full BRSET dataset for 5-class ICDR diabetic retinopathy severity grading
(Grades 0-4), assigned by Nagur Shareef Shaik as a companion project to the
[mbrset-retfound](https://github.com/Sahith59/mbrset-retfound) RETFound work.

This is an independent experiment on the same dataset as Nakayama et al.'s
original BRSET paper (PLOS Digital Health, 2024) — **not a replication of
their task**. The paper's ConvNeXt V2 results cover binary DR (Normal vs. DR)
and a coarser 3-class grouping (Normal / Non-Proliferative / Proliferative),
reporting only macro AUC/F1 for each, with no per-grade breakdown. We instead
trained on the full 5-class ICDR scale (Grades 0-4) with per-class metrics,
which the paper never reports. See the table below for exactly where our
setup matches or diverges from theirs; there is no paper number to compare
our 5-class results against directly.

## What's in this repo (and what isn't)

BRSET is a **credentialed-access PhysioNet dataset**. This repo contains only
code and aggregate results (confusion matrices, per-epoch metrics) — it does
**not** contain any patient images, per-patient labels, or model checkpoints
trained on the data, in line with the dataset's Data Use Agreement. To
reproduce this work you need your own PhysioNet credentialed access to BRSET.

## Setup

Place your own credentialed BRSET download under `data/raw/` (or point
`RAW_CSV`/`RAW_IMAGES` in `scripts/01_prepare_splits.py` at your local copy),
then run:

```bash
python scripts/01_prepare_splits.py
sbatch scripts/02_train_convnextv2.slurm
python scripts/03_evaluate.py <checkpoint> <out.txt>
```

Note: `01_prepare_splits.py` excludes 7 images (out of 16,266) found to be
corrupted/truncated via a full-dataset integrity scan (`results/corrupted_files.txt`,
not included here per the DUA — regenerate it yourself with a PIL verify/load
pass over the image directory if needed). All 7 are Grade 0/2, so exclusion has
no material effect on class balance.

## Recipe vs. the original BRSET paper

| Aspect | Paper (Nakayama et al.) | This repo |
|---|---|---|
| Task / label | Binary DR, and a separate 3-class (Normal/Non-Proliferative/Proliferative) task | Full 5-class ICDR (0-4), single task — **not the same task, no paper baseline exists for this** |
| Per-class metrics | Not reported (macro AUC/F1 only) | Full per-grade precision/recall/F1 |
| Image inclusion | All 16,266 images (no quality filter mentioned) | Quality-passing only (14,280), minus 7 corrupted = 14,273 |
| Model | ConvNeXt V2 (variant unspecified) | Base (ImageNet-22k->1k pretrained) |
| Input pipeline | Resize 256 -> crop 224 | Same |
| Normalization | Raw 0-1 | ImageNet mean/std (required for the pretrained checkpoint) |
| Optimizer | Adam, lr=1e-5 | AdamW, lr=1e-4 |
| Loss | Weighted cross-entropy | Unweighted (first baseline run) |
| Epochs | 50, early-stopping patience 7 | 100, best-checkpoint selection (best epoch stayed at 5 for the full 100) |
| Split | 70% train (20% of that as val) / 30% test | Patient-level 70/15/15 (matches the mBRSET/RETFound methodology) |
| Metrics | AUC-ROC, macro F1 | Superset: accuracy, hamming loss, macro F1/AUC/precision/recall/jaccard/avg-precision, kappa |

## Data

14,273 usable images (16,266 total minus 1,986 failing quality control minus 7
corrupted files) from 8,069 patients, patient-level split: train 9,980 / val
2,150 / test 2,143 images.
