# Step-5 mechanism deep research

**Review date:** September 15, 2026  
**Question:** What should replace the failed Step-4 image-augmentation mechanism, and is a generative/world-model direction justified?  
**Scope:** Primary implementation anchors are full papers from CVPR, ICCV, NeurIPS, WACV, AAAI and MICCAI. Broader papers and preprints are retained only to prevent an unsupported novelty claim.

## Decision

Do not continue tuning the Step-4 image transforms and do not train a full pixel-generating diffusion/world model next. The recommended novelty-facing experiment is a **target-calibrated, action-conditioned latent-transition objective** added to the natural joint B1 classifier. This is a bounded adaptation of the domain-transition idea in CheXWorld, with explicit DR/ME preservation and controls that distinguish it from ordinary augmentation or blind feature consistency.

Before that GPU experiment, run two smaller checks:

1. add the missing strict source-only BRSET baseline so that “BRSET to mBRSET” has an unambiguous number; and
2. test the published Greedy Feature Pruning (GFP) method on frozen B1 features as a low-cost top-conference comparator, using fitting data for pruning and mBRSET validation only for hyperparameter selection.

The proposed method is a hypothesis. Its novelty and benefit are conditional on controlled validation, replication and closest-work review. There is no evidence yet that it will improve BRSET/mBRSET performance.

## What Step 4 established

Step 4 made BRSET images much closer to mBRSET under five global appearance statistics, but none of A1/A2/A3 met the frozen diagnostic replication gate. The best-looking signal, overlay-free A3, changed validation DR F1 by `+0.0062` and ME F1 by `-0.0004` versus B1. Therefore:

- the transformation implementation worked as an appearance transformation;
- direct use of transformed source images was not a sufficiently strong classifier mechanism; and
- matching brightness, contrast, sharpness, falloff and saturation does not guarantee preservation or better use of small DR/ME lesions.

This rules out “better global appearance matching” as the main paper contribution. It does not rule out using the measured transformation as a controlled probe or action for representation learning.

## The important terminology correction

A **pixel generator** produces a visible image. Diffusion models belong to this category. They can be useful, but a generated fundus image can remove, add or alter a lesion, so visual realism is not enough evidence of diagnostic correctness.

A **latent predictive world model** predicts a feature representation. I-JEPA predicts masked-region representations and is explicitly non-generative. CheXWorld similarly predicts target radiograph features from a transformed context and a known transformation parameter. CheXWorld used a diffusion decoder only for analysis/visualization; the core representation objective does not synthesize pixels.

Thus, the best-supported interpretation of Dong's garbled “generative world model” comment is a CheXWorld-style latent transition model, possibly compared with a diffusion alternative. The exact paper title remains unconfirmed.

## Evidence from the closest top-conference work

| Work | Evidence relevant to us | What it does **not** establish for BRSET/mBRSET | Decision |
|---|---|---|---|
| I-JEPA, CVPR 2023 | Predicts target-block features from context blocks without reconstructing pixels. Establishes latent prediction as a viable self-supervised representation objective. | No medical, retinal or cross-device evidence. | Conceptual foundation only. |
| CheXWorld, CVPR 2025 | Adds a parameter-conditioned domain-variation task using brightness, contrast, gamma and blur. Its ablation says the domain condition contributes to transfer; the complete model combines local, global and domain tasks. | Radiographs, ViT, 224-pixel input, approximately 0.5M pretraining images; no fundus or BRSET/mBRSET result. Official recipe trains 300 epochs with batch 2048 for 16 hours on eight RTX 4090 GPUs. | Use the equation and control logic; independently implement a small ConvNeXt-compatible objective. Do not reproduce the full model. |
| Samba, NeurIPS 2024 | Severity-aware recurrent patch modeling plus EM state recalibration improves average DR grading accuracy by 4.1 points over VMamba in its six-target source-only protocol. It directly supports the importance of small lesion patches. | Different backbone, five-grade task, eight-dataset training protocol and unseen-target setting. It is not directly comparable to our binary DR/ME target-supervised setting. | Strong conceptual/architecture comparator, not the next implementation. |
| DG-ADR, WACV 2025 | Grade-conditioned augmentation and same-class cross-domain feature alignment improve source-only DR generalization. | Does not validate target-calibrated latent transition modeling. | Prevents us from claiming generic augmentation or class alignment as novel. |
| DECO, MICCAI 2024 | Disentangles retinal semantics and domain noise; uses class/domain prototypes and pixel-level alignment. | Does not use an explicit measured device action. | Prevents us from claiming generic semantic/domain disentanglement as novel. |
| Standardized-color MIM, MICCAI 2025 | Learns from original and standardized fundus images with masked modeling and cross-attention; reports up to nearly 4% improvement. | Reviewer record raises content-preservation, label-dependent standardization, compute and reproducibility concerns. | Strong collision with “standardize then fuse”; do not use that as our novelty. |
| GFP, CVPR 2026 | Post-training greedy selection of frozen final-feature groups improves or maintains AUROC/AUPRC in many fundus settings; maximum reported cross-dataset AUROC gain is 2.16 points. It needs no backbone retraining. | Supplementary F1/recall results are mixed. In one EfficientNetV2 Messidor-2 case, recall falls from 81.48% to 18.52%. It does not guarantee our primary F1 improves. | Run as a cheap, faithful published comparator and diagnostic, not as the proposed contribution. |
| Class-conditioned diffusion for DR, MICCAI 2025 | Semantic filtering and class-conditioned diffusion improve balanced grading accuracy from 66.84% to 74.20% in an imbalance problem. | Different objective; it does not establish lesion-safe cross-device synthesis or BRSET/mBRSET benefit. | Evidence that generation can help when semantically filtered, but not the next experiment. |

Additional collision checks matter. DIRL (AAAI 2022) already measures channel sensitivity between original and photometrically transformed features; a 2023 preprint already prunes domain-sensitive channels; CVPR 2026 GFP already prunes final fundus features. Therefore, “prune features that change under degradation” alone is not a safe novelty claim.

## Repository and reproducibility audit of candidate implementations

- The CheXWorld repository was inspected at commit `090102758801dc097f53c49d135b835570c8d173`. It exposes the training recipe and implementation, but the repository root contains no license file. No source should be copied. The method can be independently implemented from the published equations.
- The Samba repository linked by the NeurIPS paper was inspected at commit `de3705a880f997b6ddd5ae613c7237bc0e86d858`. Its current default branch contains only a one-line README reading `Account takeover vulnerability by H1-Shamim`. It is not a trustworthy executable source. Do not run or import it.
- The MICCAI 2025 standardized-color paper lists no public repository on its official page.

These are implementation-risk findings, not judgments about the validity of the published papers.

## Recommended proposed mechanism

Working description: **target-calibrated, action-conditioned latent transition with multi-label preservation**. This is a working description, not a final method name or established novelty claim.

For a BRSET source image `x`, construct an overlay-free degraded view

`x_tilde = T_a(x)`,

where `a` records the sampled strength of the fitted blur, illumination, brightness, contrast, saturation and noise operations. The endpoint is the Step-4 transform fitted using mBRSET training images; intermediate strengths interpolate from identity to that endpoint. Geometry is shared between `x` and `x_tilde` so corresponding retinal locations remain aligned.

Let `f_theta` be the ConvNeXtV2 encoder and `g_phi` a small predictor. The action-conditioned feature prediction loss is

`L_transition = || LN(g_phi(f_theta(x_tilde), a)) - stopgrad(LN(f_theta(x))) ||_2^2 / d`.

This asks the model: “Given this retinal image after a known amount of mobile-like change, what would the clean-source representation be?” The predictor receives `a`, so the encoder need not pretend that all changes are identical.

For label `k` in `{DR, ME}`, add a preservation term between clean and transformed-view logits:

`L_preserve = sum_k w_k * BCE(sigmoid(q_k(x_tilde)), stopgrad(sigmoid(q_k(x))))`.

The ordinary B1 focal classification objective on natural BRSET and mBRSET fitting images remains the main supervised loss. The transition predictor is discarded at inference, so the deployed classifier remains one ConvNeXtV2 pass.

The preservation term is algorithmic evidence only. It does not prove that ophthalmic lesions are clinically preserved; expert review or lesion-level annotations would still be needed for that claim.

## Why this is different from the failed augmentation

Step 4 trained the classifier to treat a transformed image as another labeled training image. The new objective uses a clean/transformed pair and the known transformation strength. It teaches the encoder how the representation should move under the device change and explicitly checks whether DR/ME predictions survive that move. The image transformation becomes a supervised representation-learning signal rather than more data by itself.

## Required falsification controls

Three seed-0 validation arms should be run against the existing B1 reference:

| Arm | Mechanism | Question answered |
|---|---|---|
| W1 | Paired latent consistency without action input | Is simple invariance sufficient? |
| W2 | Action-conditioned latent transition | Does knowing the transformation strength help? |
| W3 | W2 plus DR/ME preservation | Does explicit task preservation prevent the ME/rare-lesion weakness seen in Step 4? |

A proposed-method claim requires W3 to beat both B1 and W1; otherwise any gain cannot be assigned to action conditioning plus preservation. W2 isolates the contribution of the action. A3 remains the direct-image-augmentation control.

Use the existing frozen validation gate: at least one label F1 delta `>= +0.01`, the other label F1 delta `>= -0.01`, and both AUROC deltas `>= -0.01`. This is an engineering screen, not a clinical or statistical significance criterion. Only a qualifying arm is repeated at seeds 1 and 2; test assessment remains closed until the replicated choice is frozen.

## Missing baseline that must be repaired

The current strongest B1 result is **not** pure `BRSET -> mBRSET`. B1 trains on `BRSET train + labeled mBRSET train` and tests on mBRSET. It is supervised cross-device joint training.

For paper clarity, add a matched strict source-only baseline trained on BRSET train, selected and thresholded on BRSET validation, and evaluated on mBRSET only after its choices are frozen. This quantifies the real zero-target-label transfer gap. It does not replace B1 because the current project setting permits labeled target training.

Until Dong confirms otherwise, the primary method remains the target-supervised setting already used in Steps 2–4. The paper must not call it zero-shot domain generalization.

## Why a full generative/diffusion world model is deferred

A full generator is possible, but it is currently a poor next bet:

1. Step 4 already shows that globally target-like images need not improve diagnostic performance.
2. Diffusion generation needs a semantic filter or clinical validation; recent MICCAI work makes that prior-art requirement explicit.
3. A pixel model adds substantially more training, storage and ablation work within a four-page ISBI schedule.
4. It risks hallucinating or erasing DR/ME lesions, especially ME-positive findings, which are rare in the data.

It becomes reasonable only if the latent method shows a signal and we need a qualitative decoder, or if Dong supplies the exact intended paper/model and explicitly prioritizes a diffusion comparison.

## Ranked next steps and schedule

1. **September 15–17:** freeze the strict source-only baseline, feature-extraction contract, faithful GFP comparator and W1/W2/W3 objectives. Implement independent tests and Slurm smoke checks.
2. **September 18:** run source-only seed 0 and frozen-feature GFP validation analysis. These are baselines/diagnostics, not novelty.
3. **September 19–21:** run W1/W2/W3 seed 0 on at most three nonexclusive one-GPU nodes.
4. **September 22–25:** replicate only a qualifying arm at seeds 1 and 2; otherwise stop and document the failed hypothesis.
5. **September 26–30:** one gated test assessment, patient bootstrap, error analysis and lesion-preservation review for the frozen method.
6. **October 1–12:** paper writing, figures, closest-work wording and advisor revision.
7. **October 15:** complete internal draft. **October 19:** hard experimental/content freeze. **October 26:** official ISBI 2027 four-page-paper deadline.

## Conditional novelty statement

If W3 passes validation, replicates and improves the frozen mBRSET test result, the defensible claim is:

> We introduce a target-calibrated action-conditioned latent-transition regularizer for supervised cross-device retinal disease detection. It uses measured mobile-camera appearance changes as a training-time representation-prediction task and preserves separate DR and ME decisions without adding inference-time cost.

Do not claim “first,” clinical benefit, lesion preservation or general domain generalization until the corresponding evidence exists. If W3 fails, the correct conclusion is that measured appearance transitions do not add value beyond strong natural joint training in this setting; the project should then pivot to a stronger published backbone/comparator or an empirical paper framing rather than repeatedly modifying the same degradation family.

## Primary sources

1. Assran et al., I-JEPA, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html
2. Yue et al., CheXWorld, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Yue_CheXWorld_Exploring_Image_World_Modeling_for_Radiograph_Representation_Learning_CVPR_2025_paper.html
3. Bi et al., Samba, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aa0c4d28021c0b273480f9e2aab83a6-Abstract-Conference.html
4. Chokuwa and Khan, DG-ADR, WACV 2025: https://openaccess.thecvf.com/content/WACV2025/html/Chokuwa_Divergent_Domains_Convergent_Grading_Enhancing_Generalization_in_Diabetic_Retinopathy_Grading_WACV_2025_paper.html
5. Che et al., DECO, MICCAI 2024: https://papers.miccai.org/miccai-2024/356-Paper0781.html
6. Jang et al., standardized-color MIM, MICCAI 2025: https://papers.miccai.org/miccai-2025/0784-Paper4143.html
7. Pham et al., GFP, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Pham_Post-training_Feature_Pruning_for_Fundus_Images_Classification_CVPR_2026_paper.html
8. Zhang et al., class-conditioned diffusion for DR, MICCAI 2025: https://papers.miccai.org/miccai-2025/0146-Paper4449.html
9. Galappaththige et al., self-distilled DR DG, WACV 2024: https://openaccess.thecvf.com/content/WACV2024/html/Galappaththige_Generalizing_to_Unseen_Domains_in_Diabetic_Retinopathy_Classification_WACV_2024_paper.html
10. Xu et al., DIRL, AAAI 2022: https://ojs.aaai.org/index.php/AAAI/article/view/20193
11. ISBI 2027 author instructions: https://biomedicalimaging.org/2027/papers/

