**The three running baselines: explanation and exact counts — September 10, 2026**

Source = BRSET. Target = mBRSET. All three classifiers predict the existing any-DR and ME image labels. These are supervised-transfer baselines, not new degradation experiments. Each starts from the same ImageNet-pretrained ConvNeXt V2 Large seed-0 state with a newly initialized two-output head.

| Experiment | Training | Selection and assessment | Question |
|---|---|---|---|
| B0 target-only | Real mBRSET fitting images only | Real mBRSET | How well can target supervision alone perform? |
| B1 joint / aggregated | Real BRSET and mBRSET fitting images mixed from the start | Real mBRSET | Does joint training provide a useful reference under the update budget? |
| B2 source→target fine-tuning | First real BRSET, then continue the learned model on real mBRSET | Real mBRSET; selection only during the target phase | Does sequential transfer help under the same total update budget? |

Training on BRSET and evaluating on mBRSET is source-only training, not target-only. If target labels choose checkpoints or thresholds, that is target-informed selection/calibration rather than strict zero-shot evaluation. No new standalone source-only evaluation is part of this three-arm batch.

Fine-tuning changes model weights by continuing training on another dataset. It does not transform BRSET images into mBRSET-like images. The PDF describes fitting an image simulator and checking appearance statistics. None of B0/B1/B2 imports or applies that fitted degradation. They do retain the common ordinary augmentation recipe below.

**Which splits are being used now?**

For development, the previous training pool was divided into five patient groups: three for fitting, one for selection and one for assessment (approximately 60/20/20% of patients). All images from one dataset-specific patient remain in one role. The original validation/test partitions are excluded from these runs. This is not a new claim that historical test exposure has disappeared.

Fit means actual training; selection plays the validation role and chooses the saved model and yes/no thresholds; assessment scores those frozen choices after the whole batch finishes. Assessment has the scoring role a test usually has, but is called development assessment because it can inform later method choices. Do not compare it directly with old test scores on different patients.

**Current counts and DR prevalence**

A patient is counted in the DR-positive column if at least one of their images carries DR label 1. This is a descriptive aggregation of image labels, not a separately verified patient diagnosis. Joint counts sum dataset-specific patient IDs; cross-dataset person identity is not established. Image percentage = positive images / all images; patient percentage = patients with any positive image / all patient IDs.

| Pool | Images | DR-positive images | Patients | Patients with a DR-positive image |
|---|---:|---:|---:|---:|
| BRSET fit | 6,826 | 442 (6.48%) | 3,579 | 262 (7.32%) |
| BRSET selection | 2,261 | 147 (6.50%) | 1,193 | 87 (7.29%) |
| BRSET assessment | 2,285 | 159 (6.96%) | 1,194 | 88 (7.37%) |
| mBRSET fit | 2,043 | 481 (23.54%) | 539 | 166 (30.80%) |
| mBRSET selection | 686 | 170 (24.78%) | 180 | 55 (30.56%) |
| mBRSET assessment | 673 | 146 (21.69%) | 180 | 55 (30.56%) |
| joint fit | 8,869 | 923 (10.41%) | 4,118 | 428 (10.39%) |

B0 uses mBRSET fit. B1 uses joint fit. B2 uses BRSET fit for its source phase and mBRSET fit for its target phase. All three use the same mBRSET selection (686 images/180 patients) and assessment (673/180). BRSET selection/assessment groups exist in the manifest but are not used to choose/score the primary target baselines. The source phase of B2 ends at its fixed budget and transfers its final raw weights.

The larger patient-positive fraction does not contradict the image-positive fraction: a patient can have a positive photograph and other negative photographs. Counts remain image-level for classifier metrics. This class-mixture difference is one reason later balance/exposure controls matter.

**Hyperparameters actually used**

| Setting | Value / interpretation |
|---|---|
| Model | ConvNeXt V2 Large, `convnextv2_large.fcmae_ft_in22k_in1k_384`, two independent sigmoid outputs |
| Initialization | Same saved pretrained seed-0 state for all arms; not random initialization of the whole network |
| Training precision | bf16 forward; fp32 model/optimizer weights |
| Microbatch / accumulation | 16 images × 4 microbatches = 64 images per weight update |
| Total budget | 5,775 optimizer updates per experiment; 369,600 image draws |
| Optimizer | AdamW, betas (0.9,0.999), epsilon 1e-8, weight decay 0.1 |
| Loss | Historical soft-target focal-style loss, gamma 2, no positive-class weighting |
| Exact loss | b = BCEWithLogits(z, y'); average (1−exp(−b))² b; y' = 0.9 y_mix + 0.05 |
| Mixup | Beta(0.2,0.2) mixing coefficient; mixes images and labels inside each microbatch |
| Label smoothing | 0.1; moves targets toward 0.5, reducing absolute certainty |
| Drop path | 0.3; stochastic branch dropping during training, off at evaluation |
| Gradient clipping | Maximum gradient norm 1.0; stops a single large gradient from dominating |
| EMA | Decay 0.999; a moving average of model weights, selected/evaluated consistently |
| Sampling | Repeated shuffled passes; no rare-class oversampling; joint uses natural image proportions |
| Training geometry | RGB, bilinear 560×560 resize, random 512×512 crop, horizontal flip p=0.5, rotation ±15° with nearest interpolation and black fill |
| Ordinary color augmentation | Brightness jitter 0.2, contrast jitter 0.2, saturation jitter 0.1, hue 0 |
| Normalization | Mean [0.485,0.456,0.406], standard deviation [0.229,0.224,0.225] |
| Evaluation geometry | Same bilinear 560 resize, center 512 crop, same normalization |
| Four-view evaluation | Average probabilities from original, horizontal flip, vertical flip and both flips |
| Checkpoint choice | Highest mean DR/ME AUROC on mBRSET selection; exact ties choose earlier update; EMA only |
| Decision thresholds | Separately maximize each label's positive F1 on selection; grid 0..1 by 0.001; ties closest to 0.5, then lower |
| Assessment | Thresholds fixed; DR positive F1 primary; ME, AUROC, average precision, sensitivity/specificity, confusion matrices and other predefined metrics retained |
| Uncertainty | 2,000 paired patient-cluster bootstrap draws, seed 20260911; training-seed variability reported separately |

An update is one adjustment of the model weights after four microbatches. An epoch is one pass through a dataset. Equal epochs would give the larger dataset more weight updates, so these runs use equal update budgets. Repeated draws are not new independent images.

| Experiment / phase | Updates | Peak learning rate | Warmup updates |
|---|---:|---:|---:|
| B0 target-only | 5,775 | 0.00003 | 693 |
| B1 joint | 5,775 | 0.00003 | 693 |
| B2 source phase | 2,887 | 0.00003 | 346 |
| B2 target phase | 2,888 | 0.00001 | 347 |

Warmup gradually raises the learning rate from one tenth of its peak; cosine decay then reduces it to zero. B2 transfers the final source raw model weights, resets optimizer/scheduler, and initializes the target EMA from those weights. It is one two-stage run, not two independently replicated experiments.

B0/B1 have selection checks every 231 updates (25 opportunities). B2 has 25 evenly spaced selection checks only in its target phase, at ceil(j×2888/25), j=1..25. Its source phase never selects on assessment images.

At this fixed budget, B0 repeats its target pool about 181 times, B1 its joint pool about 42 times, and B2 has about 27 source passes then 90 target passes. Thus equal updates do not mean equal target-image exposure. These compare complete training strategies; they do not isolate every effect of pooling. Future augmentation comparisons must also hold sampler and source/target exposure fixed.

**What runs are active?**

Three runs total, one per strategy, all seed 0. The seed is a reproducible randomization identifier, not a claim that randomness is absent. Each run learns different weights because its data/training order differ, despite matched starting weights. Relevant comparisons will be repeated with seeds 1 and 2; those are not running yet. Repeating all three strategies at all three seeds would be nine runs, but only the current three are submitted.

Array 4372453: B0 on arctrdagn031, B1 on arctrdagn035, B2 on arctrdagn036. User explicitly authorized concurrent execution; throttle changed from 1 to 3 without restarting B0, changing training settings, or submitting duplicate runs. Each takes one A40, eight CPU cores and 64 GB RAM, with no exclusive node reservation. Nodes 035/036 were idle at inspection and each contains two A40s. Assessment job 4372454 still waits for all successes. Approximately 7.4 GPU-hours per experiment remains the measured estimate; concurrency reduces elapsed waiting, not total GPU work. Queue/I/O variability remain.

**How this connects to the final paper**

These baselines establish what ordinary target supervision and transfer can do. Then balance/exposure controls address Dong's class-distribution question. Only after that can a matched augmentation study establish whether fitted degradation adds diagnostic value. A later validated diagnostic-preservation mechanism is a candidate contribution, not an existing result. A positive appearance-distance result alone cannot establish better diagnosis or novelty. We should change direction if the controlled evidence does not support the proposed mechanism.

**Historical split counts (not the active evaluation splits)**

The earlier roughly 70/15/15 train/validation/test partitions are retained. These counts explain why prior documents show 3,402 target training images, 725 validation images and 732 test images rather than the current development-role counts.

| Historical pool | Images | DR-positive images | Patients | Patients with a DR-positive image |
|---|---:|---:|---:|---:|
| BRSET legacy train | 11,372 | 748 (6.58%) | 5,966 | 437 (7.32%) |
| BRSET legacy val | 2,451 | 159 (6.49%) | 1,279 | 94 (7.35%) |
| BRSET legacy test | 2,435 | 162 (6.65%) | 1,279 | 94 (7.35%) |
| mBRSET legacy train | 3,402 | 797 (23.43%) | 899 | 276 (30.70%) |
| mBRSET legacy val | 725 | 176 (24.28%) | 193 | 60 (31.09%) |
| mBRSET legacy test | 732 | 159 (21.72%) | 193 | 59 (30.57%) |

Evidence: `prevalence_explanation.json` (generated from manifests by allocated CPU job 4372512), `../protocol/v1.json`, frozen trainer/core source, run contracts/events, and `scheduling_update_20260910.json`. No final baseline assessment results are claimed.
