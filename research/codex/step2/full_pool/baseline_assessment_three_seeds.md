# Step-2 full-pool baseline results across three seeds

Values are mean ± sample standard deviation across training seeds 0, 1 and 2.

| Arm | DR F1 | DR AUROC | ME F1 | ME AUROC |
|---|---:|---:|---:|---:|
| B0 | 0.8105 ± 0.0140 | 0.9395 ± 0.0052 | 0.8240 ± 0.0297 | 0.9835 ± 0.0130 |
| B1 | 0.8338 ± 0.0048 | 0.9467 ± 0.0020 | 0.8619 ± 0.0420 | 0.9945 ± 0.0005 |
| B2 | 0.8139 ± 0.0099 | 0.9416 ± 0.0031 | 0.8184 ± 0.0333 | 0.9882 ± 0.0014 |

Per-seed B1−B0 F1 differences:

- DR: [0.03116883116883118, 0.025874094406695747, 0.012955373406193127]; mean +0.0233; positive in 3/3 seeds.
- ME: [0.04884004884004889, 0.02242990654205601, 0.042328042328042326]; mean +0.0379; positive in 3/3 seeds.

B1 passes the prespecified engineering prioritization screen versus B0. This prioritizes Step-3 controls; it is not a novelty, clinical, significance, or external-validation claim.

The test split was historically reused. Step 3 must use validation for design choices and address unequal data exposure.
