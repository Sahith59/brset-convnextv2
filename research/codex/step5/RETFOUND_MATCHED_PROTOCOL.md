# Protocol-matched RETFound comparator

**Status:** Frozen feasibility contract before GPU smoke. This is a published-backbone comparator, not the proposed contribution.

## Scientific question

Does retinal-domain pretraining on the published RETFound DINOv2 ViT-Large backbone materially improve the same supervised BRSET+mBRSET task under B1's data, labels, optimization budget and validation rule?

The existing `mbrset-retfound` results cannot answer this question. They predict referable DR only, omit ME, use a different patient split for 2,262 of 4,859 mBRSET images, use a different DR label for 271 images, use 224-pixel inputs and follow a different training/selection procedure.

## Matched contract

- Fitting pool: the exact B1 joint pool, 11,372 BRSET plus 3,402 mBRSET images.
- Selection: the exact 725-image mBRSET validation split. No assessment image or old companion test metric may influence implementation or selection.
- Outputs: two independent sigmoid logits for image-level any DR and ME.
- Updates: 5,775 optimizer steps, effective batch 64, AdamW, peak learning rate 3e-5, 693 warm-up steps, cosine decay, weight decay 0.1, gradient cap 1.0 and EMA 0.999.
- Objective/augmentation: B1 focal loss, Mixup and ordinary spatial/color augmentation.
- Selection/evaluation: four-flip mean probabilities; checkpoint by mean DR/ME AUROC; one validation-selected F1 threshold per label.
- Initialization: official RETFound DINOv2 Shanghai CFP ViT-Large weight, local SHA256 recorded by the compatibility audit; replace only the classifier head.

## Resolution feasibility gate

The first candidate uses B1's 560-resize/512-crop pixels with interpolated ViT positional embeddings. Because RETFound ViT-Large has about 303 million parameters and was released for 224-pixel input, this candidate must pass a one-A40 feasibility smoke before a full run:

1. official weight loads with only expected classifier/position differences;
2. real fitting images produce finite loss and gradients;
3. real validation images produce finite four-flip probabilities;
4. peak allocated memory is at most 38 GiB;
5. extrapolated training time is at most 60 A40 hours for one seed.

Gradient checkpointing is a computational accommodation, not a method change. A conservative microbatch-1 smoke first passed; a complete effective-batch calibration then showed that microbatch 16 with accumulation 4 fits and exactly matches B1's batch structure. Job 4410120 measured full-backbone gradients, finite real-image optimization, 7.02 GiB peak allocation and an estimated 7.03 A40-hours. Passing permits implementation of a resumable full trainer; it is not a performance result and does not automatically establish novelty.

If 512 pixels fails memory or runtime, a 224-pixel native-resolution comparator may be considered only as an explicitly resolution-confounded published-backbone baseline. It must receive a separate frozen contract and cannot be described as isolating retinal pretraining from resolution.

## Decision boundary

- Do not run the old companion scripts unchanged.
- Do not access the 732-image assessment during feasibility, implementation or seed-0 validation screening.
- A seed-0 validation gain must meet the existing materiality gate before replication.
- A stronger RETFound result improves the baseline; it is not automatic novelty.
