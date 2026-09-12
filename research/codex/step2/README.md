**Step 2 — baseline implementation and execution**

Current status is maintained in `STATUS.md`. Research outcomes are not inferred from smoke tests or selection scores.

User constraints: compute-intensive checks/training on allocated Slurm compute nodes only; up to three baseline GPUs concurrently, one per distinct node, no exclusive node reservation. Respect shared cluster capacity. Editing, inspecting logs and submitting jobs occur on the login node. User reports no Dong reply yet and tentatively identifies ISBI.

The official ISBI 2027 [conference dates](https://biomedicalimaging.org/2027/) and [author instructions](https://biomedicalimaging.org/2027/papers/) list October 26, 2026, 11:59 p.m. EDT for four-page papers (verified September 10). Keep October 15 as the internal full-draft target. Venue is still tentative until Dong/user confirms it.

**Execution stages**

1. `tools/check_image_duplicates.py`: decode all development images and check exact RGB duplicates across roles, including EXIF-normalized duplicates. This cannot exclude recompressed/near-duplicate eyes or establish cross-dataset person identity.
2. `tools/test_baseline.py`: adversarial CPU checks for metric definitions, threshold tie handling, degenerate labels, nonfinite predictions, sampler wrap/resume, gradient accumulation, augmentation RNG isolation, Mixup, preprocessing role guards and update schedules.
3. `tools/train_baseline.py`: pinned external initialization, fit-only training, selection-only checkpoint/threshold decisions, separate gated assessment. Stateless draw ordering/transform seeds let prefetched batches be reconstructed from the consumed counter. Save raw/EMA/optimizer/scheduler/RNG state at update boundaries.
4. `jobs/smoke.slurm`: real-image GPU smoke; compare uninterrupted and interrupted/resumed training. Repeat across the source→target phase boundary. Short selection scores are not diagnostic results.
5. Freeze tested source files and launch B0/B1/B2 seed 0 with current array concurrency three, after all gates pass and runtime is measured. Assess only after the whole predefined batch finishes. Replication follows observed development results; seed 0 is not a final winner.

**Failure handling**

Stop on corrupt images, cross-role exact duplicates, wrong hashes/roles, nonfinite loss/gradients/probabilities, incompatible checkpoint contracts, or unavailable bf16 GPU. Do not silently skip batches, switch precision or load legacy test. Retain failure logs and amend procedures explicitly. All fitted augmentation and task-specific initialization from old whole-training runs remain excluded under this partition.

**What the user can contribute**

When Dong replies, share methodological feedback and confirm whether ISBI 2027 is the intended venue. Arrange qualified review of representative original/transformed retinal images before any lesion-preservation claim. Agree with Dong on the clinically relevant sensitivity/specificity operating point; current DR-F1 prioritization is an engineering development choice. None of these blocks baseline implementation.
