# Step-4 and world-model direction review

## Decision

Run the controlled Step-4 augmentation experiment first. Keep a **fundus latent-transition world model** as a conditional Step-5 candidate. Do not begin full CheXWorld or GenDeg training before Step 4 establishes that modeled source-to-target acquisition variation improves diagnosis.

This decision is based on scientific isolation and resource evidence. The present project already has an auditable B1 baseline and a training-only appearance fit. Step 4 changes one component and can directly falsify whether that fit helps. Full CheXWorld pretraining uses a ViT-Base for 300 epochs on eight RTX 4090 GPUs in the official example. GenDeg is a Stable-Diffusion-based generator trained on multiple large restoration datasets and uses an additional structure-correction model. Either would introduce a new backbone, objective, data regime and large compute burden at once.

## Venue filter and literature policy

The user asked that implementation reference papers come from top conferences listed by Research.com. Its current computer-science ranking places CVPR first, ICCV second, WACV eighteenth, MICCAI twenty-fourth and ISBI seventy-seventh. The site ranks conferences with a bibliometric score based on estimated h-index and endorsements; it is a useful screening tool, not a scientific quality guarantee.

For this project:

- implementation anchors must be peer-reviewed papers from ranked major conferences, with preference for CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML, WACV and MICCAI;
- official code is inspected for reproducibility, dependency and license constraints;
- all relevant prior art, including journals, workshops and preprints, still enters the novelty audit because ignoring it would create a false novelty claim;
- preprints and workshops cannot be the sole evidence for choosing the main mechanism under the requested filter.

## Four distinct method families

| Family | Top-conference anchor | What it learns | Match to this project | Main limitation here |
|---|---|---|---|---|
| Parametric fundus augmentation | GDRNet/FundusAug, MICCAI 2023 | Hand-specified color, sharpness, halo, hole, spot and blur variation | Direct Step-4 comparator | Not target-calibrated; full GDRNet also changes loss and rebalancing |
| Latent world modeling | CheXWorld, CVPR 2025 | Local anatomy, global layout and predictable feature transitions under known acquisition-like transforms | Strong conceptual match to same eye anatomy under device variation | Published model is for radiographs, uses ViT/I-JEPA and large pretraining |
| Diffusion degradation synthesis | GenDeg, CVPR 2025 | Conditional clean-to-degraded image generation with type/intensity control | Could create diverse BRSET-to-handheld-like samples | Trained for weather/low-level restoration, not retinal diagnosis; large data and compute; generator may alter lesions |
| Learned universal degradation transfer | Yang et al., ICCV 2025 | Separates content from homogeneous and spatially varying degradation for transfer | Relevant to unpaired, complex acquisition effects | Demonstrated on film grain/restoration rather than fundus disease; adaptation burden and clinical preservation remain |

DuDoNet (CVPR 2019) is an additional conceptual analogy. It jointly restores sinogram and CT image domains using a known Radon inversion layer. BRSET and mBRSET are two photographs without a known invertible physical mapping, so calling them a DuDoNet-style dual domain would overstate the analogy.

## CheXWorld in simple technical terms

CheXWorld follows a joint-embedding predictive architecture. An encoder produces features for a visible context, a target encoder produces features for a target view, and a predictor estimates the target features. Its domain-variation task creates a target image \(y\), applies a known transformation \(T_a\) with parameters \(a\) to obtain context \(x=T_a(y)\), and trains

\[
\hat h_y=g_\phi(f_\theta(x);a), \qquad
\mathcal L_{domain}=\|\hat h_y-f'_\theta(y)\|_2^2.
\]

The transformation parameters act like an action: they tell the predictor what appearance change occurred. The model learns features in which acquisition-like appearance changes are predictable while anatomy remains represented.

The published transformation includes Gaussian blur and brightness, contrast and gamma changes. Our existing fitted transform adds fundus-specific illumination, noise and overlays, but those operators alone are not a world model. A fundus adaptation would require a learned latent predictor and explicit transition conditioning.

## Plausible fundus adaptation

A bounded candidate is **target-calibrated latent transition pretraining**:

1. draw a BRSET training image \(y\);
2. sample target-calibrated parameters \(a\) and form \(x=T_a(y)\);
3. encode the altered view with a context encoder and the original view with an exponential-moving-average target encoder;
4. condition a small predictor on \(a\) and minimize latent prediction error;
5. fine-tune the encoder for DR/ME using the same B1 labeled data and evaluation boundary.

This differs from full CheXWorld by focusing on acquisition transitions and the existing ConvNeXt-compatible pipeline rather than recreating three radiograph tasks. It also differs from image-space diffusion because it predicts representations rather than pixels.

This is not yet a novelty claim. CheXWorld already establishes parameter-conditioned domain-transition prediction, and fundus domain-disentanglement methods already separate semantics and domain noise. A defensible contribution would need an additional demonstrated distinction, such as a target-calibrated transformation distribution plus an explicit lesion-preservation constraint and a label-efficiency/resource result.

## Why GenDeg is not the immediate main method

GenDeg conditions Stable Diffusion on a clean image, text and degradation intensity, and was trained by combining datasets for haze, rain, snow, motion blur, low light and raindrops. It generated more than 550,000 synthetic samples for restoration. The official release requires external training data, generator checkpoints, a Stable Diffusion codebase and a NAFNet structure-correction network.

Using its public weather generator unchanged would not model fundus-camera artifacts. Fine-tuning it on BRSET/mBRSET would require credible clean/degraded supervision or an unpaired adaptation objective, and retinal lesions create a stricter preservation requirement than the published restoration setting. It is valuable as a future comparator or generator architecture, but it does not automatically provide novelty or clinical validity.

## What Step 4 will establish

Step 4 asks the minimum useful question: when everything else is held fixed, does target-calibrated alteration of BRSET training views improve validation DR/ME over ordinary augmentation and the published FundusAug artifact component?

- A positive A2/A3 result supports investing in learned transitions or preservation constraints.
- A FundusAug-only gain with no fitted-transform gain suggests generic robustness rather than target calibration.
- No gain rejects the current parametric premise and argues against immediately scaling it into a world model.
- An A2 loss but A3 gain implicates synthetic overlays as harmful and motivates preservation-aware control.

## Candidate novelty ladder

1. **Not novel:** apply blur, color jitter, spots or a diffusion generator.
2. **Weak on its own:** fit several global appearance means to a target set.
3. **Potentially useful empirical contribution:** show that low-compute target-calibrated acquisition augmentation improves handheld fundus diagnosis without paired cameras, across seeds and against FundusAug.
4. **Stronger method claim, still to be audited:** learn a distribution of target-calibrated transformations and constrain it with a validated lesion-preservation budget.
5. **World-model extension:** condition a latent predictor on those fitted transformations and demonstrate better accuracy or target-label efficiency than augmentation alone.

Every level requires ablations and a closest-work comparison. The project should claim only the highest level supported by completed evidence.

## Schedule to be paper-ready early

The official ISBI 2027 four-page deadline is Monday, October 26, 2026 at 11:59 p.m. EDT. Technical content must fit in the first four pages. The working schedule is:

- September 14–18: complete Step 4A and seed-0 screen;
- September 19–27: replicate a promising Step-4 arm and run preservation ablations;
- September 28–October 5: run one bounded Step-5 mechanism only if justified;
- October 6–10: final analyses, figures, limitations and statistical reporting;
- October 11–15: complete four-page draft and advisor revision;
- October 19: hard internal freeze, seven days before the official deadline;
- October 20–26: submission checks and contingency only.

## Primary sources

1. Che et al., [*Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains*](https://arxiv.org/html/2307.04378v3), MICCAI 2023; [official code](https://github.com/chehx/DGDR).
2. Yue et al., [*CheXWorld: Exploring Image World Modeling for Radiograph Representation Learning*](https://openaccess.thecvf.com/content/CVPR2025/html/Yue_CheXWorld_Exploring_Image_World_Modeling_for_Radiograph_Representation_Learning_CVPR_2025_paper.html), CVPR 2025; [official code](https://github.com/LeapLabTHU/CheXWorld).
3. Rajagopalan et al., [*GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration*](https://openaccess.thecvf.com/content/CVPR2025/html/Rajagopalan_GenDeg_Diffusion-based_Degradation_Synthesis_for_Generalizable_All-In-One_Image_Restoration_CVPR_2025_paper.html), CVPR 2025; [official code](https://github.com/sudraj2002/GenDeg).
4. Yang et al., [*Towards a Universal Image Degradation Model via Content-Degradation Disentanglement*](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Towards_a_Universal_Image_Degradation_Model_via_Content-Degradation_Disentanglement_ICCV_2025_paper.html), ICCV 2025.
5. Lin et al., [*DuDoNet: Dual Domain Network for CT Metal Artifact Reduction*](https://openaccess.thecvf.com/content_CVPR_2019/html/Lin_DuDoNet_Dual_Domain_Network_for_CT_Metal_Artifact_Reduction_CVPR_2019_paper.html), CVPR 2019.
6. Xia et al., [*Generalizing to Unseen Domains in Diabetic Retinopathy with Disentangled Representations*](https://papers.miccai.org/miccai-2024/356-Paper0781.html), MICCAI 2024.
7. Chokuwa and Khan, [*Divergent Domains, Convergent Grading*](https://openaccess.thecvf.com/content/WACV2025/html/Chokuwa_Divergent_Domains_Convergent_Grading_Enhancing_Generalization_in_Diabetic_Retinopathy_Grading_WACV_2025_paper.html), WACV 2025.
8. [Research.com 2026 computer-science conference ranking](https://research.com/conference-rankings/computer-science).
9. [ISBI 2027 author instructions](https://biomedicalimaging.org/2027/papers/).

Repository inspection on September 14, 2026 used DGDR commit `10f339748de69bd2f4b1bec49858eef95082679c`, CheXWorld commit `090102758801dc097f53c49d135b835570c8d173`, and GenDeg commit `24656bfd4458e87f0efefae4692e7ad48dd4ca44`. No visible license file was found in the inspected DGDR or CheXWorld trees. Their code can inform reproducibility analysis, but source should not be copied into this repository without clarified permission. GenDeg includes a Stable Diffusion license within its bundled dependency tree; that does not by itself establish a license for every repository file. The Step-4 comparator is therefore an independent implementation of the paper-described component with all adaptations recorded.
