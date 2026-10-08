# WORKING MEMORY

Live state of the ISBI push. Plan of record is `report/PLAN_v2.md`.
**Last updated: 8 October 2026.** Update this file whenever a run lands or a
decision is made; it is the first thing to read when resuming work.

**Deadline: ISBI 2027, 26 October 2026. 18 days out.**

---

## Currently running

| Job | Task | Budget | Started | Expected |
|---|---|---|---|---|
| 4572237 | lebase_b0_s0 | 0 | 8 Oct | ~11h, tonight |
| 4572238 | lebase_b250_s0 | 250 | 8 Oct | ~11h, tonight |
| 4572239 | lebase_b500_s0 | 500 | 8 Oct | ~11h, tonight |
| 4572240 | lebase_b1000_s0 | 1000 | 8 Oct | ~11h, tonight |
| 4572241 | lebase_b2000_s0 | 2000 | 8 Oct | ~11h, tonight |

Check with `squeue -u sthummala2`. Results land in `results/lebase_b<N>_s0/`.

## Done and verified

- **Saturation measured.** Pooling 0.8455 vs best target-trained 0.8456 at matched
  best-cutoff. Explains every prior negative. The pivot rests on this.
- **Seed noise floor.** 0.0199 DR F1, 9 patients, 3 seeds. Nothing smaller is claimable.
- **Appearance gap measured.** 400 images/dataset, 4 resolutions. Target is noisier,
  higher contrast, dimmer, more unevenly lit. NOT blurrier.
- **GDRNet is worse than nothing.** Distance 7.75 vs untouched 5.13 vs fitted 1.36.
- **`--target_budget` in the trainer.** Stratified on DR, nested across budgets.
  Offline-tested: stratification, nesting, seed independence, no-op path.
- **`scripts/72_physical_layer.py`.** Differentiable acquisition model. Statistics
  agree with numpy reference to 0.2%; 9/9 parameters receive live gradients.
- **Bounding is load-bearing.** Unbounded fitting drives spot/hole amplitude to
  2.98/1.95 on a [0,1] image (blob painting). Bounded: 91.8% distance reduction,
  both amplitudes pinned at ceiling, contrast and saturation still short.
  This is claim 2 of the paper and it is already evidence.

## Next up, in order

1. **Residual refiner + discriminator + joint training loop** (stage 2). Critical path.
2. **Lesion-preservation metric.** Classifier trained on real target data only;
   score agreement between source label and prediction on translated image.
3. **CycleGAN/CUT baseline**, trained in parallel as insurance and as the
   comparison reviewers will expect.
4. **Generate and audit synthetic sets**, qualitative figure.
5. **Downstream grid**: budgets 250/500/1000 first, arms B4 and B3.

## Open decisions

- Refiner capacity and residual bound: start small and tight, loosen only if the
  appearance gap does not close. Over-capacity is the lesion-erasure risk.
- Whether to translate at 512 (matches downstream) or train the translator at 256
  and upsample. 256 is faster; must verify it does not blur lesions.

## Standing constraints

- BRSET/mBRSET are credentialed PhysioNet data under a DUA. The repo holds code,
  reports and AGGREGATE results only. No patient images, no per-patient label
  files, no checkpoints. `results/degradation_examples*.png` and the status deck
  are gitignored for this reason.
- Do not monopolise the cluster. Five to six concurrent A40 jobs is the ceiling.
- Every comparison carries seeds and is scored against the 0.0199 noise floor.

## Numbers to quote without re-deriving

```
test set              732 images, 159 DR+, 63 ME+
zero-shot             DR F1 0.7842, 50/159 missed
pooling (3 seeds)     DR F1 0.8250 / 0.8449 / 0.8396, mean 0.8365 at tuned thr
pooling best-cutoff   mean 0.8455
target-trained best   0.8456
noise floor           0.0199 DR F1, 9 patients, 0.0424 ME F1
BRSET train           11,372 images, 748 DR+ (6.6%)
mBRSET train          3,402 images, 797 DR+ (23.4%)
appearance distance   untouched 5.13, GDRNet 7.75, fitted 1.36
```
