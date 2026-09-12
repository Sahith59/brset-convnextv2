**How to interpret the upcoming baselines**

No new baseline result is claimed here. This note is written before the seed-0 baseline batch.

B0 teaches the model using mBRSET fitting patients only. B1 teaches it using both BRSET and mBRSET fitting patients. B2 first teaches it using BRSET and then continues with mBRSET. All start from the same externally pretrained seed-0 state. “Target-only” does not mean random network initialization.

All three have 5,775 weight updates and an effective batch of 64: 369,600 image draws. A draw can repeat a photograph. Equal total training updates do not give each strategy equal target-image exposure. In particular, target-only repeats its smaller dataset many more times. Thus these compare training strategies under an update budget; they do not isolate a causal effect of source images by themselves.

The fitting pools contain 6,826 BRSET images (442 DR-positive, 165 ME-positive) and 2,043 mBRSET images (481 DR-positive, 180 ME-positive). Joint training therefore uses 8,869 images (923 DR-positive, 345 ME-positive). These counts are from the fixed development manifest, not the full old training splits. Their class mixtures differ. Step 3 must control sampling/exposure before attributing an augmentation gain to device effects.

B0's 369,600 draws correspond to about 181 fitting-pool passes. B1's draws correspond to about 42 joint-pool passes. B2 has 184,768 source draws and 184,832 target draws, about 27 source passes and 90 target passes. These are approximate pass equivalents, not new epoch settings. Actual per-domain and positive-label exposures are saved in phase-complete events and checkpoints.

Selection chooses one saved EMA model using the mean of DR/ME ranking scores and then sets a yes/no threshold separately for each label. Selection performance is optimistic because those images made the choices. Assessment uses the chosen model and thresholds only after all three strategies finish. We will show each label's F1, ranking metrics, sensitivity/specificity and uncertainty; a gain in DR does not erase a loss in ME.

The old test scores are not directly comparable with these new assessment scores: the patients, fitting pool and selection rules differ. Retain old scores as historical evidence; do not plot them as matched controls against the new protocol. The new assessment is still development evidence if it guides later method choices.

The first batch uses seed 0. It can identify errors and candidates worth repeating. It cannot establish that a strategy reliably wins across training randomness. Paired patient intervals condition on these trained models; seed replication remains necessary. A successful smoke test proves only that the implementation passed specific execution checks, not that the disease problem has been solved.

If no proposed augmentation eventually improves on a strong relevant baseline, we must change or narrow the claim. The final paper must describe the mechanism and supported result, not promise novelty on the basis of an elaborate pipeline.
