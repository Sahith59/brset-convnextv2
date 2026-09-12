# Full-pool Step-2 status — September 10, 2026, 15:52 UTC

Reduced-pool array4372453 and dependent4372454 cancelled at user request. Artifacts preserved. No complete baseline results.

Current: allocated preflight4372856 running; manifests/patient separation and14CPU tests pass. Exact-pixel integrity scan16231images ongoing. GPU resume smoke4372877 pending afterok preflight. B0/1/2 jobs4372880/4372881/4372882 pending afterok smoke; each also checks integrity, code/protocol/init hashes and smoke evidence before training. Three distinct nodes035/036/037, one A40 each. Assessment4372883 waits for allthree training successes. Invalid dependencies cancel jobs; failures require diagnosis, not blind resubmission.

Update at 16:03 UTC: preflight 4372856 COMPLETED in 8m10s. All 16,231 images decoded; zero decode errors and zero exact native/EXIF-normalized duplicate groups; 14 CPU tests passed. Smoke 4372877 RUNNING normally on arctrdagn031 with finite logged values and median observed update time about 4.34 seconds. B0/B1/B2 remain dependency-pending, so full training has not started. Current batch is seed0 only; seeds1/2 are not submitted.

See full_pool/launch_seed0.json, protocol/FULL_POOL_PROTOCOL.md, preflight.json and logs. Pinned seed0 initialization reused; cancelled model states are not reused. No seed1/2 launch yet. Ordinary augmentation/Mixup only; no proposed mechanism. No empirical novelty claim.

Conditional timing: prior measured throughput suggests ~7–10 hours per baseline after start, plus current audit/smoke and scheduling. New smoke refines estimate. Jobs continue independently after this conversation; no claim of continuous assistant monitoring.

Update at 16:43 UTC: smoke4372877 completed successfully at16:13:35. Both interruption/resume checks passed without state mismatches. B0/B1/B2 started at16:13:36 on their assigned distinct nodes and are RUNNING. Latest logs: B0=375/5775, B1=375/5775, B2=400/5775 in source phase. Stable around4.34s/update; sampled losses and gradients finite; all stderr empty. Assessment4372883 pending after all three. Early update231 validation is a warm-up diagnostic only. Full-smoke estimate7.38h/run suggests approximately23:36UTC training completion if throughput remains stable, followed by assessment; retain a conservative late-September10/early-September11 window.

Update at 23:28 UTC: B0=5650/5775, B1=5675/5775 and B2=5662/5775 target phase. All three still RUNNING; stderr empty; no complete.json yet. Assessment4372883 remains pending. Seed0 batch is close but not finished. Step2 remains incomplete after seed0 alone: inspect final test/provenance/metrics and run relevant seeds1/2 before the baseline stage supports stable paper claims.

Update September11 02:38UTC: seed0 B0/B1/B2 and gated assessment all COMPLETED exit0; stderr empty. B0 DR/ME F1 .8000/.8571, B1 .8312/.9060, B2 .8041/.7805. Independent prediction/provenance verification4376489 passed. See SEED0_RESULTS_REVIEW.md. Step2 remains in progress because this is one training seed. Replication preflight4376502 passed; init seed1/2 jobs4376505/06 active at last check; seed1 wave4376507 and assessment4376508, then seed2 wave4376509 and assessment4376510 queued. Maximum three GPUs, one per three distinct nodes. Cross-seed audit is the Step2 completion gate.

At02:41UTC initialization jobs4376505/06 completed successfully. Seed1 wave4376507 is pending for resources; replication training has not started. Dependencies for assessments and seed2 remain intact.

Update September11 03:20UTC: seed1 wave4376507 RUNNING on three distinct nodes033/036/039 since02:45:54; B0/B1/B2 contracts aligned and approximately450/5775 updates with finite values, stable speed and empty stderr. Seed1 assessment4376508 pending afterok. Seed2 wave4376509 correctly pending after seed1 assessment; seed2 assessment4376510 pending after its wave. No dependency/config mismatch found.

Update September11 15:09UTC: original wave4376507 failed06:49 due transient FileNotFoundError for mBRSET507.1.jpg on node039; old downstream4376508/09/10 cancelled by dependency. No seed1 result. Diagnosis4378211 on same node passed full target decode and checkpoint audit; see seed1_failure_diagnosis.json. Resume4378228 RUNNING nodes032–034 from B0/B1 update3003 and B2 target231/global3118, gates passed and stderr empty. New chain: assessment4378229 -> seed2 wave4378230 -> assessment4378231. Preserve old failure artifacts; do not resume/submit duplicates.

Update September11 20:35UTC: seed1 resume4378228 and assessment4378229 COMPLETED exit0; independent verification4378661 passed. B0/B1/B2 DR F1 .8051/.8310/.8239; ME F1 .8000/.8224/.8430. Full review SEED1_RESULTS_REVIEW.md. Seed2 wave4378230 RUNNING nodes032–034 since18:47, around24–26%, stable finite logs and empty stderr. Assessment4378231 pending. Step2 awaits seed2 and cross-seed audit.

## Final status — September 12, 2026

**Step 2 is complete.** Seed2 wave 4378230 completed at 02:11:34 UTC and its gated assessment 4378231 completed at 02:15:31 UTC, both exit 0 with empty stderr. Independent recomputation job 4379338 and three-seed summary job 4379339 also completed exit 0.

Seed2 test results: B0 DR/ME F1 .8264/.8148; B1 .8393/.8571; B2 .8137/.8319. Mean ± sample SD across seeds 0/1/2: B0 DR/ME F1 .8105±.0140/.8240±.0297; B1 .8338±.0048/.8619±.0420; B2 .8139±.0099/.8184±.0333. B1−B0 F1 is positive for both labels in every seed; mean differences are DR +.0233 and ME +.0379.

The prespecified engineering screen passes and selects B1 as the reference for Step 3. It does not prove novelty, clinical significance, or external generalization. The 732-image mBRSET test split was historically reused and cannot be described as untouched. See `STEP_2_FINAL_REVIEW.md`, `baseline_assessment_three_seeds.json`, and `seed2_independent_verification.json`.
