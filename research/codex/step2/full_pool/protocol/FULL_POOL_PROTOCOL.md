# Step 2: full original training pools — protocol v2

Approved September 10, 2026 following the user's request to stop reduced-pool runs. This supersedes v1's nested partition for new Step-2 baselines. Previous artifacts remain historical, not completed baseline results.

| Arm | Fitting images | Validation/selection | Test/assessment |
|---|---|---|---|
| B0 target-only | mBRSET 3,402 (899 patients) | mBRSET 725 (193 patients) | mBRSET 732 (193 patients) |
| B1 joint | BRSET 11,372 (5,966 patients) + mBRSET 3,402 (899 patients) | same 725 | same 732 |
| B2 sequential fine-tuning | BRSET 11,372, then mBRSET 3,402 | same 725, during target phase only | same 732 |

Use original patient-separated manifests without reshuffling. BRSET validation/test are not used in these target-outcome comparisons. Inclusion criteria remain historical: BRSET 16,258 eligible images; mBRSET 4,859 quality-eligible labeled images. Full original training pools does not mean all raw images are training data. Patient identifiers are namespaced by dataset; cross-dataset identity is not established.

Selection means validation: evaluate saved training candidates, choose EMA checkpoint maximizing mean DR/ME AUROC, then choose one decision threshold per label maximizing positive-class validation F1 on grid 0:.001:1, ties nearest .5 then lower. Validation never contributes gradients. Test/assessment means apply that fixed checkpoint and thresholds after all three runs complete; no test-based checkpoint, threshold, epoch, or hyperparameter selection.

**Historical test limitation:** original mBRSET validation/test were used during previous project experimentation. These results are a transparent reused benchmark, not untouched external confirmation. Further method choices should use validation (or training-only inner validation); do not tune against this test. A stronger generalization claim needs separate independent confirmation, such as additional external data, rather than relabeling reused data as unseen. Full-pool training alone does not establish publication readiness.

Keep the v1 numerical recipe unchanged: ConvNeXtV2-Large external pretrained seed-0 initialization shared across arms; two sigmoid labels (any-DR and ME); 5,775 total updates, effective batch 64 (16 x accumulation 4), bf16, AdamW weight decay .1, gradient cap 1, EMA .999, drop path .3. B0/B1 LR 3e-5 with 693 warmup updates; B2 source 2,887 updates LR 3e-5/warmup346 followed by target 2,888 LR 1e-5/warmup347; transfer final raw source weights and reset optimizer/scheduler/EMA. Initial LR .1 of peak, cosine to zero. Focal gamma2 with Mixup alpha .2 and smoothing .1, as explicitly defined in v1.json. No class oversampling.

Training: bilinear 560x560 resize, random 512 crop, horizontal flip .5, rotation +/-15 degrees, color jitter brightness .2/contrast .2/saturation .1, ImageNet normalization. Evaluation: bilinear 560 resize, center512 crop, average sigmoid probabilities over four flips. These ordinary augmentations are present; no fitted synthetic degradation, routing, consistency loss, or private encoder. One common backbone with a two-label head per model; B2 transfers the backbone sequentially.

Twenty-five checkpoint selection opportunities per arm; B2 selects only in target phase. Same total compute updates, not same number of target-image exposures or same learning-rate schedule across transfer stages. Nominal draws: B0 target369600 (~108.64 passes); B1 joint369600 (~25.02 passes); B2 source184768 (~16.25 passes) then target184832 (~54.33 passes). These differences motivate Step 3 controls and limit causal interpretation of a baseline ranking.

Report both labels: positive F1, class-macro F1, AUROC, average precision, precision, sensitivity/recall, specificity, confusion matrix, and F1 at .5. Patient-cluster paired bootstrap 2,000 draws with fixed thresholds; uncertainty conditional on trained models. Seed0 is an initial run, not replication. Relevant seeds1/2 follow baseline sanity review; do not declare a winner or novelty from one seed.

Step 2 completes after valid baseline outcomes, provenance review and needed replication. Step 3 investigates class balance and source/target exposure; preparation can overlap. Step 4 compares ordinary augmentation, faithful FundusAug and fitted degradation. These three are the primary training-strategy controls, not all literature competitors or ablations needed for a final paper.

New code isolated under step2/full_pool/frozen; protocol retains local filename v1.json for compatibility but internal version is 2.0-full-original-pools. Split/role guard changed explicitly, not disabled: original train->fit, val->selection, test->assessment. Allocated CPU preflight, exact-pixel duplicate audit across all included images, tests and real-image GPU resume checks precede full training. No image/classifier computation on login nodes. At most three training GPUs on distinct nodes, no whole-node reservation.
