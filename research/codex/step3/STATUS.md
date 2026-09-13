# Step-3 status

## Complete — September 13, 2026

Stage 3A exposure and stratum auditing, Stage 3B fixed-compute sampler controls, the interpretation-critical C1 replication, and the conditional Stage 3C decision are complete. No Step-3 test assessment was performed.

Allocated checks and runs all passed: CPU sampler preflight `4379902`, A40 smoke `4379912`, seed-0 C1/C2/C3 wave `4379924`, corrected result verifier `4385144`, C1 seed-1/2 replication wave `4385146`, and cross-seed finalizer `4385147`. The initial automatic seed-0 finalizer `4379931` failed before reading results because its repository root was one directory too shallow; the reporting path was corrected, the allocated verifier passed, and training artifacts were unaffected.

Seed-0 validation screening against B1 gave DR/ME F1 differences of C1 `+.0065/+.0147`, C2 `+.0069/+.0015`, and C3 `+.0036/+.0018`. Only C1 crossed the predeclared absolute 0.01 replication screen.

Across seeds 0/1/2, C1−B1 mean validation differences were:

- DR F1 `+0.0035`, positive in 2/3 seeds; DR AUROC `−0.0071`; DR average precision `−0.0039`.
- ME F1 `+0.0079`, positive in 3/3 seeds; ME AUROC `−0.0009`; ME average precision `+0.0022`.

All six B1/C1 runs reached their best mean-DR/ME-AUROC selection score before the final quarter, and all six selection scores declined from the first final-quarter checkpoint to update 5,775. The conditional 11,550-update exposure comparison is therefore not justified under the frozen protocol.

**Decision:** retain natural joint training B1 as the simplest strong Step-4 reference. Equal-domain sampling gives small, metric-dependent tradeoffs and does not establish a stronger baseline. This is a validation-based engineering decision, not a significance, clinical, external-generalization, or novelty claim.

Authoritative evidence: `seed0_validation_summary.json`, `c1_validation_three_seeds.json`, `final_curve_review.json`, and `STEP_3_FINAL_REVIEW.md`. Step 4 is next: freeze and compare ordinary B1 augmentation, a faithful published fundus augmentation component, and FIT-only fitted degradation, then add a preservation constraint only if the fitted transform is promising.
