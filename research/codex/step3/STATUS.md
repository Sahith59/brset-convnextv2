# Step-3 status

## September 12, 2026, 14:22 UTC

Stage 3A is complete. The Step-2 exposure records and full-pool strata counts were audited. Natural B1 receives about 85,108 target draws, fewer than B0's 369,600; the source contribution cannot be explained by additional target exposure. Every domain × DR/ME cell is nonempty, but DR0/ME1 is small at 22 BRSET and 14 mBRSET images.

The fixed-compute sampler protocol is frozen in `SAMPLER_PROTOCOL.md`. Allocated CPU job 4379902 completed exit 0: five unit tests passed, full 369,600-draw quotas were checked for C1–C3, indices stayed in range, and resume suffixes matched uninterrupted schedules. Allocated A40 smoke job 4379912 completed exit 0 with finite loss/gradients, correct 192/192 source/target exposure and validation inference. It performed no test assessment.

Seed-0 wave 4379924 is running on nodes 032–034. Task 0=C1 equal-domain, task 1=C2 target-label-matched and task 2=C3 equal-domain plus target-label-matched. Each task uses one A40, eight CPUs and 64 GiB on a nonexclusive node. The scientific recipe is the Step-2 B1 recipe; only the deterministic sampler changes. There is no dependent test assessment job. Results are screened on the 725-image mBRSET validation split.

Allocated CPU finalization job 4379931 is dependency-pending after successful completion of the full wave. It will verify contracts, selected-checkpoint hashes, exact exposure ledgers, finite 725-image validation metrics and the absence of test-assessment artifacts. It will then write `seed0_validation_summary.*` and regenerate both maintained research documents. A passing numeric screen prioritizes replication but does not declare a winner.

Do not interpret early validation checkpoints or smoke scores as results. After all three controls finish, verify contracts, exposure ledgers and validation summaries. Compare them with existing B1 seed0 validation. Replicate only an interpretation-critical control at seeds 1 and 2. The large repetition counts in the DR0/ME1 strata require a grouped-ME sensitivity check before a label-matched sampler could become the paper baseline.

## September 13, 2026

Seed-0 wave4379924 completed exit0 in7:25:24; all arm/wave stderr files are empty. The original automatic finalizer4379931 failed before reading results because its repository root was one directory too shallow. Corrected only that reporting path and reran allocated finalizer4385144, which completed exit0. This failure did not affect training or predictions.

Verified validation-only results: B1 DR/ME F1 .8703/.8489. C1 .8768/.8636 (differences +.0065/+.0147), C2 .8772/.8504 (+.0069/+.0015), and C3 .8739/.8507 (+.0036/+.0018). C1 crosses the predeclared replication screen through ME F1; C2/C3 do not. All contracts, selected-checkpoint hashes, exact exposure ledgers, finite725-image metrics and absence of test artifacts passed. `seed0_validation_summary.*` is authoritative.

C1-only seeds1/2 replication wave4385146 is running on two nonexclusive A40 nodes033–034, leaving the third previously allowed GPU slot unused. Finalizer4385147 depends afterok on the wave; it will verify the repeated controls, write `c1_validation_three_seeds.*`, and regenerate the living DOCX/PPTX. Step3 remains in progress. Do not choose B1 versus C1 or launch the long exposure control before the repeated validation evidence and final-quarter curve review.
