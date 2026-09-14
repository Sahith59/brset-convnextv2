**Literature direction for BRSET–mBRSET transfer**

The recommended direction is controlled, target-informed augmentation with explicit validation of diagnostic preservation. Existing research provides the starting operators, baselines and evaluation methods. A new contribution must establish an incremental mechanism or resource advantage beyond those methods. Novelty is unresolved; neither assembling published components nor improving an appearance-distance objective establishes it.

The review is an expanded decision review, updated through September 14, 2026. Access levels are recorded below. It is not a completed systematic review, and methods examined only through abstracts are not ready for implementation-level conclusions. The immediate advisor update concerned degradation validation; Step 4 now requires a deeper closest-work and implementation audit before a method claim.

**The question and information setting**

Current experiments allow BRSET images/labels and mBRSET training images/labels. This is supervised transfer or joint training. Source-only domain generalization, unlabeled-target adaptation and source-free adaptation allow different information. A technique from one setting may be a useful baseline in another, but the paper must disclose the adaptation and avoid comparing published scores as if the protocols were identical.

The candidate question is whether fitting an appearance transformation to target training data produces diagnostic benefit beyond ordinary augmentation and class/domain balancing, without paired camera acquisitions. Synthetic source/altered pairs provide correspondence to the original image. They do not establish faithful target appearance or preservation of small diagnostic findings. The latter must be validated independently.

**Closest work and what it changes**

| Work | Evidence examined | Implication for the project |
|---|---|---|
| GDRNet / FundusAug | Full main paper HTML and official augmentation source | Required close comparator; augmentation, consistency and domain-class balancing are already combined |
| cofe-Net | Primary abstract and prior audit | Useful degradation inspiration; not proof that our implementation reproduces its optics or real target distribution |
| RetSyn | Published abstract and primary bibliographic record | Diffusion plus disease/quality considerations and cross-device evaluation already exist; actual code/paired release remains to be verified |
| SCR-Net and cross-camera application | MICCAI official abstract, author repository, indexed BMJ methods passages | Structural preservation and synthetic/unpaired style transfer are existing ideas; paired evaluation and paired training requirements differ |
| Fourier phase augmentation + distillation | Published abstract | “Frequency augmentation plus teacher-student learning for DR generalization” is already occupied |
| CauDR | Primary preprint abstract; journal lead | Causality-inspired DR generalization exists; stronger claims require its full method and assumptions |
| Confidence-guided multi-image fusion | Abstract plus full-PDF methods/results passages | Same-dataset quality/fusion story is nearby; endpoint is patient-level and includes selective coverage |
| Li et al. foundation-model transfer study | Primary abstract and indexed methods in prior audit; direct retrieval currently blocked | BRSET external evaluation and mBRSET fine-tuning are not an untouched application |
| FRLA vision-language adaptation | Current arXiv record | Withdrawn; exclude performance claims as supporting evidence |
| Recent restoration/diagnosis benchmarks | Primary publication search leads only | Full-text follow-up required; enhancement fidelity and downstream diagnosis must be separated |

**GDRNet is more than random blur.** The method includes FundusAug, a hybrid supervised/contrastive objective and domain-class rebalancing. Its augmentation uses visual adjustments and artifact operations with random application and magnitudes. The paper evaluates grading with ResNet50 across multiple datasets; its benchmark differs from this project's two-output ConvNeXt transfer setup. The local deterministic generic transform is not the official augmentation. Implementing FundusAug alone would be a component baseline, not reproduction of full GDRNet.[^1][^2]

For experimental design, compare the same classifier with ordinary appearance augmentation, official FundusAug and frozen/fitted local augmentation. If the proposed contribution includes balancing and consistency as well, those components need ablations and closer comparison with the full prior framework. A claimed novelty based solely on combining those three ideas would be weak.

**Physical degradation is an inspiration, not an identified camera model.** cofe-Net models several fundus degradation factors and develops enhancement. Its existence supports considering optical/artifact effects, but a local implementation and small statistical fit do not inherit the original method's validation. Source and target images are unpaired and differ in cohort as well as acquisition. Appearance fitting should be described as empirical transformation fitting, not measured recovery of the target camera's physical process.[^3]

**RetSyn is a close competitor.** Its published abstract describes disease/quality-conditioned diffusion, group balancing and alignment involving a small paired smartphone/tabletop set, followed by classifier training with synthetic data. It reports a paired-data release. This constrains broad claims about synthetic images, quality imbalance and device transfer. The actual release, training requirements, classifier protocol and reproducible implementation need verification before deciding which comparison is feasible.[^4]

A lightweight method without paired acquisitions could still have practical value, but that resource distinction is not alone proof of methodological novelty. It needs a clearly different mechanism and a fair measurement of accuracy, label requirements or computational cost.

**Structural preservation and unpaired restoration already have precedents.** The MICCAI SCR-Net work constructs synthetic cataract sets and uses high-frequency components to encourage structural consistency. Its author repository includes simulation and training instructions. The later cross-camera application describes style standardization and evaluates paired camera acquisitions. That does not mean every training approach requires paired BRSET/mBRSET photographs. Distinguish the training data requirement from how a method was evaluated.[^5][^6]

The repository's current black/bright spot simulator should therefore be checked for actual diagnostic preservation, rather than assuming that unchanged label metadata means an unchanged visible diagnosis. A robust new procedure might reject or constrain transformations that erase diagnostic evidence, but such a mechanism needs a validated signal and controls against simple severity limiting.

**Frequency augmentation plus distillation is not an empty area.** Zhang and Liu's published abstract describes Fourier phase-based augmentation, teacher-student distillation and feature fusion for DR generalization. Its six-dataset claim does not directly establish usefulness for our task, and the full implementation remains to be examined. It does establish that a proposal titled “Fourier augmentation with distillation for DR” would need a more specific distinction.[^7]

**Causal language requires assumptions.** CauDR presents a causality-inspired framework for spurious correlations in fundus grading. It is a relevant comparator lead. Its abstract is not sufficient to endorse its identification assumptions or to claim that it solves the current camera/cohort confounding. Similarly, our threshold-oracle calculation does not identify separate causal proportions of a domain gap.[^8]

**Multi-image confidence is a separate candidate, not the next default pivot.** The July fusion preprint uses patient-level maximum disease labels, fixed image counts and RETFoundGreen, comparing simple pooling with learned fusion and confidence filtering. Its preprocessing includes exclusions and same-eye imputation that differ from our cohort. Methods passages describe threshold sweeps and selective test coverage. Its results should not be compared with our full-coverage image-level F1. It nevertheless makes a generic quality/fusion contribution less distinctive.[^9]

This also refines the prior audit's abstract-only assessment: the full PDF has now been accessed and the relevant preprocessing, training and evaluation passages examined. A complete reproduction review is still outstanding. If we revisit patient-level screening, the patient endpoint and acceptance threshold must be selected on development data, with coverage and missed disease reported together.

**Transfer with modern backbones is an existing baseline family.** The April foundation-model study evaluates BRSET training, mBRSET external validation and target fine-tuning. It is relevant to baseline strength and calibration, but does not justify direct comparison of its headline scores with ours. Its published split/input/backbone choices differ. The present task should establish a mechanism beyond simply applying a different existing backbone or adding target labels.[^10]

**Source status matters.** FRLA's current arXiv page marks the paper withdrawn on August 10, 2026 and states that some experimental comparisons may not be fair. Its retrieved abstract still contains positive claims, illustrating why search snippets alone are insufficient. Exclude those performance claims from the evidence supporting our direction.[^11]

A WACV 2026 workshop benchmark explicitly concerns retinal enhancement and downstream diagnosis. The current access is a primary publication lead; full-text retrieval was blocked. It should be examined before treating an enhancement-versus-diagnosis analysis as new. A 2025 unpaired retinal enhancement paper also provides downstream segmentation comparisons in indexed primary tables; it is a useful reminder that the strongest reconstruction metric need not identify the strongest downstream result. Neither has been reproduced here.[^12][^13]

**What to retain, change and defer**

Retain the earlier review's useful operator and distillation references, patient-aware evaluation, and emphasis on a strong simple baseline. Correct blanket exclusions of domain adaptation, unsupported physical claims about blur, and assumptions that synthetic correspondence guarantees fidelity. Extend the review to current published work and clearly labeled preprints. Report access limitations rather than substituting an abstract for a methods audit.

Proceed first with a modest augmentation comparison after the protocol and balance controls are fixed. The new training-only appearance check is useful because it tests the existing frozen fit on other patients. It remains a descriptive first check: it does not measure classifier benefit, semantic preservation or novelty. Its role is to justify the next validation question, not to establish the paper's conclusion.

## Step-4 novelty update — September 14, 2026

The current search makes a broad preservation-aware augmentation claim untenable. Several close ideas are already occupied:

| Work | Primary evidence checked | Constraint on this project |
|---|---|---|
| DG-ADR, WACV 2025 | Open-access paper and official repository | Grade-conditioned, image-conditioned Stable Diffusion augmentation for DR is already published; synthetic diagnostic relevance alone is not new |
| Context-aware OT retinal enhancement, WACV 2025 | Full primary article | Unpaired target distribution matching with deep-context preservation of retinal structures already exists in enhancement |
| EyeBench-V2, WACV Workshops 2026 | Open-access paper | Lesion, vessel, DR-grading and expert downstream evaluation of retinal transformation is an evaluation precedent, not a novel mechanism here |
| MedDiffuseMix, arXiv 2026 | Full preprint; not treated as peer-reviewed evidence | Saliency-guided medical augmentation with an adaptive preservation constraint is already proposed outside fundus imaging |
| CausalFund, medRxiv 2026 | Full preprint; not treated as peer-reviewed evidence | Hospital-to-portable fundus robustness through causal/spurious feature separation is a close problem setting |

The GDRNet repository was inspected at commit `10f339748de69bd2f4b1bec49858eef95082679c`. It uses brightness/contrast/saturation/hue preprocessing and sharpness/halo/hole/spot/blur operations, with the artifact operations applied at probability 0.5 in the released configuration. No license file was visible in the inspected repository root or first two levels. Record the commit and independently implement the paper-described behavior; do not copy or redistribute unlicensed source. The repository's current `GDRNET_GENERIC` dictionary is not this published implementation.

A potentially distinguishable hypothesis is a low-compute, joint-label-composition-aware target-device calibration of a parametric acquisition transform, learned from training data without paired camera images or a generative model and bounded by a validated diagnostic-preservation measure. Its components overlap prior work; novelty would depend on the exact joint formulation, resource setting, ablation and empirical behavior. The first Step-4 audit therefore asks whether global appearance fitting is materially confounded by the different DR/ME mixtures before committing to this direction.

New primary links: [DG-ADR paper](https://openaccess.thecvf.com/content/WACV2025/papers/Chokuwa_Divergent_Domains_Convergent_Grading_Enhancing_Generalization_in_Diabetic_Retinopathy_Grading_WACV_2025_paper.pdf), [DG-ADR code](https://github.com/sharonchokuwa/dg-adr), [context-aware OT](https://pmc.ncbi.nlm.nih.gov/articles/PMC12337797/), [EyeBench-V2](https://openaccess.thecvf.com/content/WACV2026W/P2P/html/Dong_Bridging_Restoration_and_Diagnosis_A_Comprehensive_Benchmark_for_Retinal_Fundus_WACVW_2026_paper.html), [MedDiffuseMix preprint](https://arxiv.org/abs/2606.28419), and [CausalFund preprint](https://www.medrxiv.org/content/10.64898/2026.03.02.26347127v1.full).

Defer large diffusion and world-model development until the closest methods, available paired resources and expected advantage are understood. Defer an architecture stack that mixes fitting, domain separation, routing and distillation before individual contributions can be measured. Keep a bounded longer-budget routing study as an explicit response to Dong, with the train/inference mismatch addressed and a matched baseline.

**Requirements for selecting the proposed new method**

The method-selection memo should answer five questions: What does the closest existing method do? What exact operation or learning objective changes? Why should that change address a failure demonstrated on development data? What additional information or compute does it require? Which controlled result would disprove the proposed benefit?

Candidate distinctions include diagnostic-preservation-aware transformation selection, target-distribution fitting that generalizes beyond a few means, or a demonstrable reduction in target-label requirements. These are research hypotheses, not certified open areas. Each requires searching its closest conceptual equivalents, not just the same dataset names. Do not claim originality from a new acronym.

The extensive review proceeds in three passes. First, establish the closest method families and reliable source status; this document records that pass. Second, obtain full methods, official code and data requirements for the shortlist, including RetSyn, GDRNet and structural/frequency alternatives; produce a resource/comparison matrix. Third, search backward and forward from the chosen mechanism, document the remaining difference, and then freeze the contribution hypothesis before confirmatory experiments. Continue until the material novelty claim is supported or explicitly bounded; paper count is not the completion criterion.

**Sources**

[^1]: Che et al. *Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains*. MICCAI 2023. [Main paper](https://arxiv.org/html/2307.04378v3). Full main-text HTML reviewed.
[^2]: Che et al. [Official FundusAug source](https://github.com/chehx/DGDR/blob/main/dataset/fundusaug.py). Source inspected; not executed in this phase.
[^3]: Shen et al. *Modeling and Enhancing Low-quality Retinal Fundus Images*. [Primary record](https://arxiv.org/abs/2005.05594). Abstract-level evidence here.
[^4]: Shuai et al. *Enhancing AI-based diabetic retinopathy screening in low- and middle-income countries with synthetic data*. JBI 2025. [Published abstract](https://pubmed.ncbi.nlm.nih.gov/41138952/).
[^5]: Li et al. *Structure-consistent Restoration Network for Cataract Fundus Image Enhancement*. MICCAI 2022. [Official conference record](https://conferences.miccai.org/2022/papers/482-Paper0050.html); [author repository](https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement). Abstract and repository README examined.
[^6]: Joseph et al. *Enhancing AI-based diabetic retinopathy diagnosis through universal cross-camera image adaptation*. BMJ Open Ophthalmology 2025. [Primary PDF](https://bmjophth.bmj.com/content/bmjophth/10/1/e002238.full.pdf). Indexed primary methods passages accessed; direct full retrieval blocked.
[^7]: Zhang and Liu. *Domain generalization for diabetic retinopathy grading with phase augmentation framework*. Medical & Biological Engineering & Computing 2026; online 2025. [Published abstract](https://pubmed.ncbi.nlm.nih.gov/41199099/).
[^8]: Wei et al. *CauDR: A Causality-inspired Domain Generalization Framework for Fundus-based Diabetic Retinopathy Grading*. [Primary preprint record](https://arxiv.org/abs/2309.15493). Abstract examined; journal-version reconciliation pending.
[^9]: Raghu et al. *Model Confidence-Guided Multi-Image Fusion of Fundus Images for Diabetic Retinopathy Diagnosis*. July 2026 preprint. [Full PDF](https://arxiv.org/pdf/2607.03643). Relevant methods/results passages examined; not reproduced.
[^10]: Li et al. *Comparison of foundation models and transfer learning strategies for diabetic retinopathy classification*. April 2026 preprint. [Primary record](https://www.medrxiv.org/content/10.64898/2026.04.17.26351092v1). Abstract and indexed methods examined in preceding audit; current direct retrieval blocked.
[^11]: Huai et al. *Forgetting-Resistant and Lesion-Aware Source-Free Domain Adaptive Fundus Image Analysis with Vision-Language Model*. [Withdrawn record](https://arxiv.org/abs/2602.19471), withdrawal verified September 10, 2026. Not supporting evidence.
[^12]: Dong et al. *Bridging Restoration and Diagnosis: A Comprehensive Benchmark for Retinal Fundus Enhancement*. WACV Workshops 2026. [Primary publication lead](https://openaccess.thecvf.com/content/WACV2026W/P2P/papers/Dong_Bridging_Restoration_and_Diagnosis_A_Comprehensive_Benchmark_for_Retinal_Fundus_WACVW_2026_paper.pdf). Search-level lead; full text pending.
[^13]: Dong et al. *CUNSB-RFIE: Context-Aware Unpaired Neural Schrodinger Bridge in Retinal Fundus Image Enhancement*. WACV 2025. [Primary publication](https://openaccess.thecvf.com/content/WACV2025/papers/Dong_CUNSB-RFIE_Context-Aware_Unpaired_Neural_Schrodinger_Bridge_in_Retinal_Fundus_Image_WACV_2025_paper.pdf). Indexed primary result tables examined; full retrieval pending.

## World-model and generative-degradation addendum — September 14, 2026

The user-supplied meeting transcript likely references CheXWorld (CVPR 2025), not a generic video-prediction world model. CheXWorld learns local anatomy, global layout and feature transitions under parameterized blur/brightness/contrast/gamma transformations. Its official example pretrains ViT-Base for 300 epochs on MIMIC, NIH and CheXpert using eight RTX 4090 GPUs. It is a strong conceptual anchor for modeling the same anatomy across acquisition conditions, but it is not a drop-in fundus model and is too large to be the first deadline-facing experiment.

GenDeg (CVPR 2025) is a separate image-space diffusion approach. It conditions on a clean image, text and degradation intensity, and was trained on combined restoration datasets for six weather/low-level degradations. It does not provide a pretrained handheld-fundus degradation model. Applying it unchanged would not answer the current camera-transfer question; adapting it would require new retinal supervision or a defensible unpaired objective plus lesion-preservation validation.

Yang et al. (ICCV 2025) learn a universal content/degradation decomposition with homogeneous and spatially varying components. This further narrows the novelty space: learning or transferring degradation separately from content is occupied. DuDoNet (CVPR 2019) supports Dong's dual-representation analogy for CT, but its sinogram/image relationship uses a known Radon transform that fundus camera domains do not possess.

The operational decision is to finish the controlled Step-4 comparison before a world-model investment. A bounded Step-5 candidate may adapt CheXWorld's parameter-conditioned latent prediction to target-calibrated fundus transformations. Its claim would need to exceed augmentation alone and show either diagnostic preservation, target-label efficiency or another isolated practical advantage. See `STEP4_WORLD_MODEL_DIRECTION_REVIEW.md` and `meeting/DONG_TRANSCRIPT_AUDIT_2026-09.md` for the detailed evidence matrix.
