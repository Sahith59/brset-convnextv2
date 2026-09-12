**Degradation work update**

The frozen target-fitted transformation brings the source images closer to the target on five measured appearance statistics in a small new training-only evaluation. Diagnostic improvement has not yet been tested.

The original work measured 400 training images per dataset and fitted parameters using 25 BRSET images over 150 random trials. The transformation includes blur, illumination changes, bright/dark artifacts, a halo, noise and color adjustments. The saved parameters were kept unchanged for the new check. [1]

The new check uses 64 BRSET images from 64 patients and 64 mBRSET images from 59 patients. These patients were excluded from the original measurement/fitting cohorts reconstructed from the saved scripts and seeds. Three stochastic degradation seeds were evaluated on the same images. No diagnosis labels, validation/test images, classifier inference or model training were used. [2]

| Preprocessing | BRSET unchanged | Local generic | Frozen fitted |
|---|---:|---:|---:|
| Legacy 512 resize | 6.742 | 10.092 | 1.261 |
| 560 resize / 512 crop | 8.474 | 12.011 | 4.294 |

Distance is the sum, over five appearance measures, of squared source-target mean differences divided by the target standard deviation squared. Lower means closer on these summaries only. Target means and scales are recomputed for each preprocessing row, so comparisons are within a row; the values are not accuracy, percentages, confidence intervals or full distribution matching.

Fitting improves this descriptive appearance objective on the new cohort under both preprocessing procedures. The fitted examples still show artificial spots and residual differences from real target images. Neither lesion preservation nor DR/ME classification benefit is established. The comparison labeled local generic is our existing handcrafted transform, not published FundusAug or full GDRNet. [2,3]

![Degradation comparison](degradation_examples.png)

First three randomly selected BRSET examples from the new cohort. Columns 1–3 show the same source image and two transformations. Column 4 shows unrelated real mBRSET reference images: these are not paired acquisitions. Display uses 560-pixel bilinear square resize and a 512-pixel center crop. Figure transform seeds are 100–102 and 200–202; quantitative seeds are 0–2. Source files and seeds are recorded in the internal manifest. This figure supports qualitative review, not a clinical certification.

Next, validate representative lesion-bearing pairs, align fitting with the selected classifier preprocessing, and compare ordinary augmentation, a faithful FundusAug component and fitted augmentation under one training protocol. Test augmentation before adding distillation. In parallel, control label/domain balance and training exposure as discussed in the meeting. A fair longer-training routing comparison remains a separate follow-up.

**Sources**

[1] Existing project files: scripts/55_degradation_ops.py, 56_measure_datasets.py, 57_fit_degradation.py; results/fitted_degradation_params.json.
[2] New evaluation: research/codex/tools/degradation_update.py; validation_summary.json, appearance_statistics.csv and appearance_distances.csv, September 10, 2026.
[3] Che et al., Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains, MICCAI 2023. Official augmentation implementation: https://github.com/chehx/DGDR/blob/main/dataset/fundusaug.py
