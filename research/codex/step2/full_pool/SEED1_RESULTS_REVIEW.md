# Full-pool baseline seed-1 review

Status: completed and independently recomputed September 11, 2026. Seed1 experienced one diagnosed transient shared-filesystem read failure and resumed from exact checkpoints. Exposure totals, contracts and final predictions passed verification. This remains a historically reused test benchmark.

| Arm | DR F1 | DR AUROC | DR AP | ME F1 | ME AUROC | ME AP |
|---|---:|---:|---:|---:|---:|---:|
| B0 target-only | 0.8051 | 0.9442 | 0.8947 | 0.8000 | 0.9888 | 0.9186 |
| B1 joint | 0.8310 | 0.9489 | 0.9089 | 0.8224 | 0.9950 | 0.9592 |
| B2 source then target | 0.8239 | 0.9449 | 0.9009 | 0.8430 | 0.9890 | 0.9344 |

Validation-selected DR/ME thresholds: B0 0.360/0.371, B1 0.554/0.569 and B2 0.482/0.389. Checkpoint updates: B0 1617, B1 2541 and B2 4043.

B1 has the highest seed1 DR F1 and the highest DR/ME AUROC and AP. B2 has the highest ME F1 at its validation-selected threshold. B1 versus B0 differences: DR F1 +0.0259 and ME F1 +0.0224. Their fixed-model patient-bootstrap intervals include zero for both F1 differences: DR [-0.0183,+0.0688], ME [-0.0368,+0.1026]. B1's threshold produces higher DR precision (0.944 vs 0.818) but lower sensitivity (0.742 vs 0.792) than B0; no clinical operating point has been agreed.

Across seeds0/1, B1 DR F1 is 0.8312/0.8310 versus B0 0.8000/0.8051, so the direction has repeated twice. B1 ME F1 is 0.9060/0.8224 versus B0 0.8571/0.8000; direction repeats but magnitude varies. B2 ME F1 changes from 0.7805 to 0.8430, showing that ME conclusions are particularly seed-sensitive. Wait for seed2 and cross-seed analysis.

Allocated verification job4378661 recomputed all metrics from the prediction arrays, verified the common 732-image/193-patient assessment cohort, hashes and exact exposure totals. B0 target draws369600; B1 source/target284498/85102; B2 source/target184768/184832. See seed1_independent_verification.json.
