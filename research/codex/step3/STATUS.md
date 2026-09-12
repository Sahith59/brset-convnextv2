# Step-3 status

## September 12, 2026, 14:22 UTC

Stage 3A is complete. The Step-2 exposure records and full-pool strata counts were audited. Natural B1 receives about 85,108 target draws, fewer than B0's 369,600; the source contribution cannot be explained by additional target exposure. Every domain × DR/ME cell is nonempty, but DR0/ME1 is small at 22 BRSET and 14 mBRSET images.

The fixed-compute sampler protocol is frozen in `SAMPLER_PROTOCOL.md`. Allocated CPU job 4379902 completed exit 0: five unit tests passed, full 369,600-draw quotas were checked for C1–C3, indices stayed in range, and resume suffixes matched uninterrupted schedules. Allocated A40 smoke job 4379912 completed exit 0 with finite loss/gradients, correct 192/192 source/target exposure and validation inference. It performed no test assessment.

Seed-0 wave 4379924 is running on nodes 032–034. Task 0=C1 equal-domain, task 1=C2 target-label-matched and task 2=C3 equal-domain plus target-label-matched. Each task uses one A40, eight CPUs and 64 GiB on a nonexclusive node. The scientific recipe is the Step-2 B1 recipe; only the deterministic sampler changes. There is no dependent test assessment job. Results are screened on the 725-image mBRSET validation split.

Do not interpret early validation checkpoints or smoke scores as results. After all three controls finish, verify contracts, exposure ledgers and validation summaries. Compare them with existing B1 seed0 validation. Replicate only an interpretation-critical control at seeds 1 and 2. The large repetition counts in the DR0/ME1 strata require a grouped-ME sensitivity check before a label-matched sampler could become the paper baseline.
