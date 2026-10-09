# WORKING MEMORY

Live state of the ISBI push. Plan of record is `report/PLAN_v2.md`.
**Last updated: 8 October 2026.** Update this file whenever a run lands or a
decision is made; it is the first thing to read when resuming work.

**Deadline: ISBI 2027, 26 October 2026. 18 days out.**

---

## Currently running

Seeds 1 and 2 at budgets 0/250/500/1000, jobs 4588222-4588229. Eight runs, the
left half of the curve where synthesis has to act. Single-seed points cannot be
claimed against a 0.0199 floor.

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

## Label-efficiency curve, seed 0 (first read)

    budget   DR AUC   best-cut F1   missed/159
         0   0.9104        0.7826           42
       250   0.9278        0.7785           36
       500   0.9422        0.8167           32
      1000   0.9424        0.8456           33
      2000   0.9453        0.8350           35
      3402   0.9377        0.8383           32   (seed 0)
                           0.8455                (3-seed mean)

**The curve plateaus at roughly 500 to 1000 real images, not 3402.** AUC is the
cleaner read: 0.9104 -> 0.9422 by budget 500, then flat. Steps 250->500 (+0.0382)
and 500->1000 (+0.0289) clear the noise floor; 1000->2000 (-0.0106) and
2000->3402 (+0.0033) do not.

So the target for synthesis is sharp: reach ~0.84 F1 / ~0.942 AUC using FEWER
than 500 real images. Caveat: single seed, seeds 1 and 2 now running.

## Next up, in order

1. **Residual refiner + discriminator + joint training loop** (stage 2). Critical path.
2. **Lesion-preservation metric.** Classifier trained on real target data only;
   score agreement between source label and prediction on translated image.
3. **CycleGAN/CUT baseline**, trained in parallel as insurance and as the
   comparison reviewers will expect.
4. **Generate and audit synthetic sets**, qualitative figure.
5. **Downstream grid**: budgets 250/500/1000 first, arms B4 and B3.

## TOP RISK: RESOLVED 9 Oct, IN OUR FAVOUR

**The label-balance confound. This can kill the headline claim.**

BRSET train is 6.6 percent DR-positive, mBRSET is 23.4 percent. Pooling does two
things at once: it adds handheld appearance AND it more than doubles the absolute
positive count, 748 to 1545. Every result so far confounds them.

Synthetic handheld images are made from BRSET and INHERIT BRSET'S LABELS at 6.6
percent. They add appearance but not one new positive. So if the pooling gain is
mostly class balance, synthesis cannot reproduce it and the label-efficiency
claim fails regardless of translator quality.

**RESULT, 2 seeds, best-cutoff DR F1:**

    budget 0, no rebalancing      0.7826   (42 missed)
    budget 0, rebalanced to 23.4% 0.7843   (45.5 missed)
    pooling, all 3402 real        0.8455

    total gap                     +0.0629
    closed by class balance alone +0.0017   = 3 percent of the gap
    remaining for device          +0.0612   = 3x the noise floor
    noise floor                    0.0199

**Class balance explains essentially none of it.** Dr. Ye's label-distribution
hypothesis is tested and rejected, which is itself a citable result: it rules
out the main alternative explanation for why pooling works. Device adaptation
is real and synthesis has a genuine job. Stage 2 is cleared to build.

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
