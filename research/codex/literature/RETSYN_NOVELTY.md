**RetSyn and the closest prior work: novelty study for the BRSET→mBRSET generative translator**

Written 14 October 2026 by Claude Code for the ISBI 2027 submission (deadline 26 October 2026). This is a read-only literature study. No experiment, Slurm job or git change was made, and no other file was edited. It extends `DIRECTION_REVIEW.md` and `../protocol/NOVELTY_HYPOTHESIS.md`; it does not replace them.

**Evidence markers.** [full text] = read in the body of the paper. [abstract] = only the abstract or a listing (PubMed/Crossref record, the publisher's reference list as exposed by Crossref, a dataset card, a repository listing). [inferred] = my reasoning, not something the authors state. Our own numbers come from the repository files named beside them.

**Access summary.**
- RetSyn full text could NOT be read. ScienceDirect answers HTTP 403 with a JavaScript bot challenge, as do its archived copies. The article is not in PMC or Europe PMC. There is no arXiv or medRxiv preprint. Elsevier's text-mining API needs an institutional key.
- The article is open access under CC BY 4.0 (Unpaywall, Crossref), so a person with a browser can read it. Doing so is the first open item.
- Everything about RetSyn below is therefore [abstract] or [inferred].
- SCR-Net (with ArcNet and GFE-Net), ContourDiff and FGDM: the full texts and released code were read in this session by a delegated literature pass. I re-checked the most load-bearing claims myself: the BMJ cross-camera paper (from its Europe PMC full-text XML) and the FGDM abstract.
- Residual-CycleGAN: its method section was read from the arXiv PDF.
- All other works are [abstract] unless marked otherwise.

---

## 0. Bottom line

1. **RetSyn occupies our problem statement.** It uses diffusion-generated synthetic retinal images to make DR screening robust across devices, specifically portable smartphone-based cameras versus tabletop cameras in LMICs. It augments classifier training with them. It comes from the group that built mBRSET: Shuai, Wu, Restrepo and Nakayama are all mBRSET co-authors. Its paired data use the same smartphone-camera description as mBRSET's Phelcom Eyer. A claim such as "first diffusion-based synthetic data for tabletop-to-handheld DR" is not available to us.
2. **What RetSyn does not appear to do** (abstract-level only) is our mechanism. RetSyn generates class- and quality-conditioned images and aligns them with DPO on a small paired smartphone–tabletop set. We instead translate each labelled tabletop eye into a handheld rendering without any paired images: a target-fitted physics model gives the starting point, and a diffusion renderer is conditioned on the source eye's lesion-scale structure. The label is inherited through the source anatomy, not asserted by a class token.
3. **None of our components is new on its own.**
   - Structure-conditioned, target-trained diffusion translation: FGDM, ContourDiff.
   - Signed high-frequency structure maps for fundus: SCR-Net, ArcNet.
   - SDEdit-style starts: SDEdit; FGDM's start is SDEdit from the source.
   - Target-fitted camera-effect augmentation: Sensor Transfer, 2019.
   - Simulate-then-refine: SimGAN.
   - Cross-camera fundus translation for DR: Residual-CycleGAN (2020) and the BMJ SCR-Net application (2025).
   - Generative augmentation for distribution shift: Ktena et al., Nature Medicine 2024.
   - Translation plus pseudo-label self-training: BDL (2019), and Ju et al. (2021) for fundus devices.
4. **Verdict: weak-to-moderate novelty as things stand; moderate if the missing baselines (Section 5e) come out in our favour.**
   - The defensible contribution is specific: an unpaired, zero-target-training-label, physics-initialised, lesion-structure-conditioned diffusion translator for tabletop→handheld fundus.
   - The ablation shows that both stages are needed for the ranking gain. The effect is modest (+0.027 AUROC, F1 unchanged) and has not yet been compared with any GAN, UDA or RetSyn-style baseline.

---

## 1. Our method as it actually stands (verified against the repository)

| Component | What the code and results say | Source |
|---|---|---|
| Data | BRSET train 11,372 labelled images. mBRSET train 3,402 images, whose labels are never read by the generator. mBRSET validation 725 and test 732 images (193 patients, 159 DR+). Task: binary DR (ICDR ≥ 1) plus macular oedema; ConvNeXt-V2-L classifier | `MEMORY.md` 12–13 Oct; `results/baselines_per_seed_2026-10-13.json` |
| Stage 1 (physics) | 19 bounded photometric parameters: signed sharpening, illumination falloff, halo, per-channel shot/read noise, per-channel gain/offset, contrast, saturation. Spot/hole artefacts are capped off ("noartifact" checkpoint). Fitted to unpaired mBRSET training-image statistics (13 colour, sharpness and illumination statistics plus a VGG-Gram style loss). The fit was done jointly with a since-discarded 68k-parameter refiner; only stage 1 is used now | `scripts/72_physical_layer.py`, `83_train_translator.py`, `88_synth_augment.py`, `results/xlate_v2_noartifact/` |
| Stage 2 (renderer) | 104.7M-parameter diffusers `UNet2DModel`: base 128, multipliers (1,2,2,4,4), no attention. Input: noisy RGB, structure map and field-of-view (FOV) mask (5 channels). Class embedding over {BRSET, mBRSET, null}. v-prediction, cosine schedule, AdamW, EMA. Trained on 256 px crops and applied to the full 560 px frame. 10% dropout on the structure map and on the device token | `scripts/109_gen_common.py`, `110_train_diffusion.py` |
| Structure map | Signed, locally normalised green-channel contrast: tanh(0.5·(g−μ_local)/(σ_local+ε)), σ_l = 10 px, masked and rim-eroded. During training it is computed from a randomly blurred and noised copy, so source sensor texture is not copied through | `109_gen_common.py` |
| Coupling | DDIM with 30 steps and trailing spacing. Sampling starts from the stage-1 output noised to strength 0.4, so the last 12 steps run. Classifier-free guidance (CFG) on structure, w = 1.5. Device CFG is implemented but was **not used**: w_device = 0 in every reported translation | `111_translate.py`, `111_translate_array.slurm` |
| Variants | gen_tgt: mBRSET images only. gen_both: BRSET and mBRSET with a device token, rendered with the mBRSET token. Neither uses paired eyes | `110_train_diffusion.py` |
| Gates | **Gate 1:** a logistic device classifier on ConvNeXt-V2-nano features scores the translated images. **Gate 2:** a "judge" classifier gives AUROC per ICDR grade on 546 held-out BRSET images, with a Gaussian-blur (σ 4) positive control. The judge is `brset_mbrset_joint`, which **was trained with all mBRSET training labels** | `scripts/112_gates.py` |
| Downstream | Each BRSET training image is replaced by its twin with probability p at every load, inside the unchanged classifier recipe. Validation, test and labels are untouched | `MEMORY.md` 13 Oct 02:55; D069 |

**Results that matter for the novelty claim** (test DR, mean of 3 seeds, patient-cluster bootstrap over 193 test patients; `results/full_table_2026-10-14_0125.json`, `results/f1_why_2026-10-14.json`):

| Arm (0 real mBRSET training labels unless stated) | Test AUROC | Δ vs baseline [95% CI] | F1 | Notes |
|---|---|---|---|---|
| Baseline, BRSET only | 0.9007 | — | 0.7670 | validation 0.9377 |
| **Final configuration (D072): gen_both 40k, 25% swap** | **0.9279** | **+0.0272 [+0.010, +0.049]**, 3/3 seeds | 0.7746 (+0.008 [−0.019, +0.038], n.s.) | chosen on validation (0.9550) among 12 configurations |
| gen_both 60k / 100k, 25% swap | 0.9271 / 0.9262 | +0.027 / +0.026 | 0.7502 / 0.7693 | longer generator training gives no downstream gain |
| Full method used for the ablations: gen_tgt 40k, 50% (physics → diffusion) | 0.9271 | +0.0264 [+0.006, +0.053] | 0.7506 | |
| Ablation: physics only | 0.9039 | +0.0033 [−0.012, +0.019], n.s. | 0.7485 | |
| Ablation: diffusion only, from noise | 0.9176 | +0.0170 [−0.002, +0.040], n.s. | 0.7314 (−0.036 [−0.069, −0.003]) | F1 significantly worse |
| Ablation: diffusion from the plain BRSET image (s 0.4) | 0.9139 | +0.0133 [−0.001, +0.031], n.s. | 0.7295 (−0.037 [−0.067, −0.008]) | F1 significantly worse |
| Paired comparison: full − physics only | | +0.0231 [+0.0036, +0.0488], 2/3 seeds | | validation agrees |
| Paired comparison: full − no-physics start | | +0.0131 [+0.0045, +0.0241], 3/3 seeds | | validation agrees |
| Paired comparison: full − diffusion from noise | | +0.0094 [−0.0000, +0.0193], 3/3 seeds | | not resolved |
| 250 real labels, no synthetic / with synthetic | 0.9219 / ≈ same (±0.001) | | | synthetic adds nothing once labels exist |
| 500 real labels | 0.9360 | | 0.8082 | real labels mainly cut false alarms (25 → 14) |

**Measured caveats a reviewer can find in our own files:**
- **Device realism is partial.** For the final configuration:
  - Mean P(mBRSET) is 0.405; 40.5% of twins are called mBRSET.
  - The feature-mean gap to mBRSET is still 91.9% of its original size.
  - A direct linear probe separates twins from real mBRSET with AUROC 1.000, which is also its value on untouched BRSET (`results/gates_g40000_gen_both_physics_s0.4_w1.5.json`).
- **Label preservation is partial.**
  - Judge AUROC for grade 1 is 0.907, against 0.981 untouched and 0.685 for the blur control.
  - The mean judge score on healthy eyes rises from 0.054 to 0.125, which may be invented lesion evidence or a calibration effect.
- **The diffusion edit is small.** Full-method twins differ from physics-only images by about 3.5/255 per pixel, against 13.4/255 from the source (`MEMORY.md` 13 Oct 16:50, gen_tgt 40k).
- **"Zero labels" means zero target *training* labels.**
  - 725 labelled mBRSET validation images chose the checkpoint, the cut-off and the final configuration.
  - The label-trained judge chose the sampler settings (strength 0.4, w 1.5) through the gate selection rule.
- **Protocol.** The legacy trainer and a historically reused test split were used (D067). Ablations were run at gen_tgt 40k / 50%, not at the final gen_both 40k / 25%.
- **Where the gain sits.** The ranking gain is concentrated in mild DR: grade-1 AUROC rises from 0.759 to 0.815 and sensitivity at 90% specificity from 0.795 to 0.834. At the F1 operating point the gain is about 3 more cases caught and 2 more false alarms.

---

## 2. RetSyn in depth

### 2.1 Citation and status
- **Reference.** Shuai Z, Wu C, Tang Z, Restrepo D, Morley M, Nakayama LF. "Enhancing AI-based diabetic retinopathy screening in low- and middle-income countries with synthetic data." *Journal of Biomedical Informatics* 172:104938, December 2025 (online 24 Oct 2025). DOI 10.1016/j.jbi.2025.104938; PMID 41138952. [abstract] https://doi.org/10.1016/j.jbi.2025.104938 ; PubMed XML via https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=41138952
- **Dates.** Received 27 May 2025, revised 13 Sep 2025, accepted 11 Oct 2025. [abstract] (PubMed history)
- **Licence and access.** CC BY 4.0, "hybrid" open access. No repository copy and not in PMC. [abstract] https://api.unpaywall.org/v2/10.1016/j.jbi.2025.104938 ; https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=10.1016/j.jbi.2025.104938
- **Affiliations.** University of Michigan EECS (Shuai, Wu, Tang); MIT Institute for Medical Engineering and Science (Restrepo, Nakayama); Schepens Eye Research Institute / Harvard (Morley). [abstract]
- **Keywords.** Cross-device generalization; Diabetic retinopathy; Diffusion models; Low-resource settings; Synthetic data. [abstract]
- **Working title.** The associated dataset card cites "RetSyn: Improving Synthetic Samples for Ophthalmology with DPO Using Paired Data", so an earlier version may have circulated under that name. No such paper was found in arXiv, OpenAlex or web search. [abstract] https://huggingface.co/datasets/smartretina2025/paired_retina
- **Code.** None found. A GitHub search for "RetSyn" returns unrelated repositories, and the abstract names no repository. [abstract]

### 2.2 Who wrote it, and why it matters
- **Author overlap with mBRSET.** The mBRSET paper (Wu, Restrepo, Nakayama, …, Shuai, …; *Scientific Data* 2025) shares four authors with RetSyn, including both first authors. [abstract] https://api.crossref.org/works/10.1038/s41597-025-04627-3
- **RetSyn cites mBRSET twice.** Crossref lists it as reference entries 14 and 25. [abstract] https://api.crossref.org/works/10.1016/j.jbi.2025.104938
- **Same smartphone camera.** mBRSET was acquired with "the Phelcom Eyer … a portable, handheld retinal fundus camera that integrates with a Samsung Galaxy S10 running Android 11". [full text] https://pmc.ncbi.nlm.nih.gov/articles/PMC11846882/
  - RetSyn's paired-dataset card describes the portable device as "Samsung Galaxy S10 smartphones running Android 11 with integrated handheld retinal cameras", and the dataset's image folders are named `eyer` and `visucam`. [abstract] https://datasets-server.huggingface.co/info?dataset=smartretina2025/paired_retina
- **Same tasks as the mBRSET benchmark.** RetSyn's tasks are binary and three-class. [abstract] These match the mBRSET benchmark tasks (healthy vs DR; normal / non-proliferative / proliferative) [full text, mBRSET] and the labels on RetSyn's paired card. [abstract]
- **Inference.** RetSyn very probably targets the same handheld device as mBRSET, and probably uses BRSET and/or mBRSET. [inferred] I could not confirm which datasets it trains and tests on.
  - Consequence: a reviewer from this group would know RetSyn intimately, and our paper must cite it and position against it explicitly. [inferred]

### 2.3 Data
- **Abstract wording.** "Portable device data scarcity" is a stated challenge, and the evaluation covers "low-quality tabletop images" and "out-of-domain portable images". [abstract]
- **Paired dataset.** The card describes "327 patients", each with "one image captured using a tabletop device and another using a portable device, taken simultaneously under identical conditions". Labels: DR status (normal, NPDR, PDR) and image quality. Tabletop device: "standard retinal cameras". The card also says: "Will release the whole dataset after accepted." It carries an MIT licence and a Hugging Face DOI 10.57967/hf/5433; the repository was created 2 Jan 2025 and last modified 25 Mar 2025. [abstract] https://huggingface.co/datasets/smartretina2025/paired_retina/raw/main/README.md
- **What is actually public.** The viewer exposes one `train` split of 399 images. Its only label is a two-class field whose names are `eyer` and `visucam`, i.e. device identity. DR or quality labels are not visible. [abstract] (datasets-server info, above)
  - [inferred] The pairs are Phelcom Eyer (handheld) versus Zeiss Visucam (tabletop). BRSET's cameras are Canon CR-2 (65.1%) and Nikon NF5050 (34.9%) [full text] https://pmc.ncbi.nlm.nih.gov/articles/PMC11239107/ , so RetSyn's paired tabletop camera is probably not a BRSET camera.
- **Other cited datasets.** RetSyn cites DeepDRiD, which carries image-quality labels. [abstract] (Crossref reference list) Whether RetSyn uses it is unknown.
- **Not verifiable without full text:**
  - training and test sets and their sizes;
  - whether portable-device labels enter training;
  - how many paired images the DPO step uses;
  - whether unlabelled portable images are used at all.

### 2.4 Generative model and conditioning
- **Abstract wording.** "RetSyn leverages advanced diffusion models to generate synthetic retinal images with diverse device and imaging quality characteristics… uses class and quality-conditioned diffusion for controllable synthesis." [abstract]
- **Backbone clues.** The reference list includes Esser et al. 2024 ("Scaling rectified flow transformers", i.e. the SD3 family), Wallace et al. 2024 (Diffusion-DPO), Qin et al. 2023 (class-balancing diffusion), CLIP, Go et al. 2024 (structurally realistic fundus diffusion), RETFound and a vision–language ophthalmology foundation model. [abstract] (Crossref reference list)
  - [inferred] This suggests a fine-tuned large latent text-to-image model (SD3-class) with class and quality prompts or tokens, aligned with Diffusion-DPO. It is a guess from citations only.
- **Unknown:**
  - whether the device is a conditioning variable or is induced only by DPO;
  - whether generation is image-conditioned (translation of a specific eye) or from noise (new eyes);
  - resolution, and any structure or vessel conditioning.
- **[inferred]** "Class-conditioned … synthesis" reads as generating new images whose label is the conditioning class. That is the main mechanistic contrast with our instance translation.

### 2.5 Training
- **Group-balanced loss.** "a group-balanced loss to increase coverage of minority (quality, disease) pairs". [abstract]
- **DPO.** "a Direct Preference Optimization alignment step with a small paired smartphone-tabletop set". [abstract]
  - How the preference pairs are formed (for example, real portable twin preferred over a generated sample) is not stated in the abstract. [inferred]

### 2.6 Downstream use
- "The synthesized images are then used to augment classifier training"; "training retinal diagnosis models on a combination of real and synthetic data." [abstract]
- Unknown: the classifier architecture, the real:synthetic ratio, and how labels are assigned to synthetic images (presumably the conditioning class [inferred]).

### 2.7 Evaluation and reported results ([abstract], exact numbers)
- **Low-quality tabletop images:** F1 rises from 0.781 to 0.874 (binary) and from 0.607 to 0.703 (three-class). AUROC reaches 0.982 and 0.951 respectively.
- **Out-of-domain portable images:**
  - RetSyn: AUROC 0.813 / F1 0.703 (binary); AUROC 0.804 / F1 0.609 (three-class).
  - GroupDRO: AUROC 0.786 / F1 0.626 (binary); AUROC 0.789 / F1 0.544 (three-class).
- **Not comparable to our 0.928.**
  - The datasets, splits, label definitions, classifiers and selection rules almost certainly differ. Our zero-shot BRSET-only baseline is already 0.901, far above RetSyn's out-of-domain portable figure. [inferred]
  - Never put the two numbers in one table as if they were comparable.

### 2.8 Stated limitations
- The abstract states none. The full text could not be read.

### 2.9 Unlabelled target images and pseudo-labels in RetSyn (coordinator's question)
- The abstract mentions **neither unlabelled target (portable) images nor pseudo-labels nor self-training**. [abstract]
- Its only explicit use of target-device data is the small **paired** smartphone–tabletop set for DPO. [abstract]
- Whether its diffusion model or classifier sees unlabelled or labelled portable images is unknown without the full text. Do not claim either way.

---

## 3. Other closest work

### 3.1 SCR-Net, its follow-ups, and the BMJ cross-camera application

**SCR-Net**
- **Reference.** Li H, Liu H, Fu H, Shu H, Zhao Y, Luo X, Hu Y, Liu J. "Structure-consistent Restoration Network for Cataract Fundus Image Enhancement." MICCAI 2022, LNCS pp. 487–496, DOI 10.1007/978-3-031-16434-7_47; arXiv 2206.04684. Code: https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement [full text] https://arxiv.org/abs/2206.04684
- **Method.** It restores cataract-degraded images to clear ones and is trained only on *synthetic* pairs. [full text]
  - The degradation follows Peli & Peli's imaging model: per-channel blur plus a radial "cataract panel" haze, with randomised parameters (centre, α, β, kernel). 16 simulations are made per clear image. [full text]
  - **No parameter is fitted to target data.** [full text]
- **Generator input.** Only the signed high-frequency components H(I) = I − Gaussian low-pass (r 26, σ 9), FOV-masked. [full text]
- **Architecture and losses.** One encoder and two decoders (one aligns high-frequency components, one restores); L1 alignment, restoration and cycle losses; no adversarial loss. [full text]
- **Evaluation.** SSIM/PSNR, vessel-segmentation IoU and diagnosis F1/kappa on Fundus-iSee (ResNet-50). Diagnosis F1 rises from 0.730 (cataract input) to 0.770, against 0.838 on clear images. [full text]

**Follow-ups from the same group**
- **ArcNet** (TMI 2022, arXiv 2203.07737) uses the same simulation and high-frequency input. It adds output-space adversarial UDA with **unlabelled real cataract images**. [full text] https://arxiv.org/abs/2203.07737
- **GFE-Net** (Medical Image Analysis 2023, arXiv 2309.00885) uses several degradation models, including cofe-Net's, with a self-supervised pretext task that reconstructs the high-frequency map. [full text] https://arxiv.org/abs/2309.00885

**BMJ cross-camera application**
- **Reference.** Joseph S, Chen X, Liu C, Zhu Z, Ramasamy K, Ravilla TD, Ge Z, He M. "Enhancing AI-based diabetic retinopathy diagnosis through universal cross-camera image adaptation." *BMJ Open Ophthalmology* 2025; 10(1):e002238; PMC12658492. [full text] https://pmc.ncbi.nlm.nih.gov/articles/PMC12658492/
- **Model.** It applies SCR-Net, renamed "Style-Consistent Retinal Image Transformation Network". SCR-Net was trained on about 20,000 UK Biobank desktop images to map *any* camera to one canonical style. [full text]
- **Classifier.** The DR model (InceptionNeXt-T) was trained on EyePACS; **zero labels from either test camera** were used. [full text]
- **Test data.** Paired images from 367 patients at Aravind, India, with an Optain Resolve (portable, automated) and a Topcon NW400 (desktop). The pairs are used **only for evaluation**. [full text]
- **Scenarios.** Scenario 3 trains on original + adapted images and tests on adapted images. It gives Optain accuracy 79.2% (95% CI 75.9–82.6), against 75.9% (72.3–79.4) for original/original; the CIs overlap. [full text]
- **Structure check.** PSNR 29.35 between each original and its own transformed image. [full text]

**Overlap with us**
- Like SCR-Net, our stage 1 is a parametric degradation that makes structure-sharing twins. Ours is *fitted to unpaired target statistics*; theirs is randomly drawn. [inferred]
- Our structure map belongs to the same signed high-pass family as SCR-Net's high-frequency components, which are SCR-Net's only generator input. **Cite SCR-Net/ArcNet for the representation; do not claim it.** [inferred]
- The BMJ paper is the closest *application* precedent: portable-versus-desktop DR, structure-preserving translation, mixed training, and zero target-camera labels. [full text]
- **The differences:**
  - They harmonise *target images toward a canonical desktop style* at inference. We translate *labelled training images toward the handheld device* and leave test images untouched.
  - Their generator is deterministic, with a hand-set degradation; ours is stochastic diffusion.
  - Their portable device is an automated camera, not a smartphone. [inferred]

### 3.2 Residual-CycleGAN camera adaptation for DR
- **Reference.** Yang D, Yang Y, Huang T, Wu B, Wang L, Xu Y. "Residual-CycleGAN based Camera Adaptation for Robust Diabetic Retinopathy Screening." MICCAI 2020; arXiv 2007.15874. [full text, method section] https://arxiv.org/abs/2007.15874
- **Setting.** It frames camera-brand shift as UDA: one labelled source brand, several **unlabelled** target brands (EyePACS with inferred brand labels, plus a private set of 55,326 images). [full text]
- **Method.** An unpaired CycleGAN with residual generators and a camera-oriented feature extractor. Target images are translated to the source style **at test time**; the classifier is not retrained. [full text]
- **Comparisons and metrics.** It compares with DANN and CycleGAN, using quadratic kappa and AUC. [full text]
- **Overlap.** Unpaired camera translation for zero-target-label DR classification has existed since 2020. Our direction is the opposite (train-time, source→target) and our generator differs. [inferred]

### 3.3 ContourDiff
- **Reference.** Chen Y, Konz N, Gu H, Dong H, Chen Y, Li L, Lee J, Mazurowski MA. "ContourDiff: Unpaired Medical Image Translation with Structural Consistency." MELBA 3:711–727 (2025), DOI 10.59275/j.melba.2025-79a2; arXiv 2403.10786 (v1 title: "…Contour-Guided Diffusion Models"). Code: https://github.com/mazurowski-lab/ContourDiff [full text] https://arxiv.org/abs/2403.10786
- **Method.** A pixel-space diffusion UNet trained **only on unlabelled target MRI**. At every step it is conditioned on a Canny edge map of the input. [full text]
  - Sampling starts from **pure noise**, with no CFG. [full text, code]
  - The downstream segmenter is trained on translated CT with the CT masks and tested on real MRI. [full text]
- **Results.** Dice, for example lumbar 0.705 against 0.554 for the best GAN; FGDM scored 0.455. [full text]
- **Limitations.** Manual thresholds, and noise-sensitive contours. [full text]
- **Overlap.** Same family as our gen_tgt: target-trained, structure-concatenated diffusion translation used to train a downstream model with source labels. [inferred]
  - **Differences:** a signed continuous lesion-scale map versus binary edges; a physics-initialised start versus pure noise; CFG and a device token; RGB fundus with an appearance-only gap. [inferred]

### 3.4 FGDM
- **Reference.** Li Y, Shao H-C, Liang X, Chen L, Li R, Jiang S, Wang J, Zhang Y. "Zero-shot Medical Image Translation via Frequency-Guided Diffusion Models." IEEE TMI 43(3):980–993 (2024), DOI 10.1109/TMI.2023.3325703; arXiv 2304.02742. Code: https://github.com/Kent0n-Li/FGDM [full text] https://arxiv.org/abs/2304.02742
- **Method.** Trained **only on target-domain** data ("zero-shot"). [abstract, verified]
  - Low frequencies come from the **forward-noised source image** (an SDEdit-style start).
  - High frequencies come from a thresholded Sobel edge map that conditions every step.
  - The backbone is a Denoising Diffusion GAN with at most about 8 steps. [full text]
- **Tasks and results.** CBCT→CT and cross-institution MR. FID 24.0 against 33.2 for SDEdit on head and neck. **No downstream task.** [full text]
- **Stated limitation.** The domains must share low and high frequencies. [full text]
- **Overlap.** The closest sampler-level precedent: an SDEdit start plus a high-frequency condition. Our distinctive change is to noise the **physics-transformed** image instead of the source. This addresses FGDM's own stated limitation, because physics first moves the low-frequency colour and illumination. [inferred]
  - Our "diffusion from the plain BRSET image" ablation is essentially FGDM-style sampling with our renderer. It gave +0.013 AUROC (n.s.) with significantly worse F1, while the full method beat it by +0.013 [+0.0045, +0.024]. This is our best evidence that the physics start is a real, measurable difference. [inferred from our results]

### 3.5 Other 2023–2026 work a reviewer could raise

| Work | What it does | Overlap |
|---|---|---|
| DG-ADR, Chokuwa & Khan, WACV 2025, arXiv 2411.02614, code github.com/sharonchokuwa/dg-adr | "image-to-image diagnostically relevant fundus augmentations conditioned on the grade of the original fundus image", plus alignment loss and focal loss, for DR domain generalisation [abstract] https://arxiv.org/abs/2411.02614 | **High** for "grade-aware diffusion augmentation for DR shift". Its target is not a specific device and it uses no physics start [inferred] |
| Ktena et al., *Nature Medicine* 2024 (arXiv 2304.09218) | Diffusion models conditioned on diagnostic and/or attribute labels, trained using abundant unlabelled data. Synthetic and real images are mixed and improve OOD robustness in histopathology, chest X-ray and dermatology [abstract] https://arxiv.org/abs/2304.09218 | **High** for the general claim "generative augmentation fixes distribution shift". Generation is label-conditioned, not instance translation [inferred] |
| Ju et al., IEEE TMI 2021 (arXiv 2011.13816) | Modified CycleGAN from regular to ultra-widefield fundus generates extra UWF training images, with consistency regularisation and **pseudo-labelling**. Covers DR classification and lesion tasks; uses limited UWF annotations [abstract] https://arxiv.org/abs/2011.13816 | **High** for "cross-device fundus translation plus pseudo-labels". It is a direct precedent for our planned self-training step |
| Ju et al., arXiv 2003.10042 (2020) | Traditional→UWF domain adaptation by **pseudo-labelling plus adversarial learning**, with MixUp against noisy pseudo-labels [abstract] https://arxiv.org/abs/2003.10042 | Medium–high (self-training precedent) |
| Self-supervised DA for DR grading via vessel reconstruction, KI 2021 | A target-domain self-supervised task plus progressively updated **pseudo-labels** for DR grading UDA [abstract, search listing only; authors not verified] https://www.dfki.de/fileadmin/user_upload/import/11715_KI_2021_Self_Supervised_Domain_Adaptation_for_Diabetic_Retinopathy_Grading.pdf | Medium (self-training for DR UDA) |
| BDL, Li, Yuan & Vasconcelos, CVPR 2019 (arXiv 1904.10620) | The image-translation model and the segmentation model are learned alternately, with self-supervised **pseudo-labels** [abstract] https://arxiv.org/abs/1904.10620 | The canonical "translation + self-training" recipe |
| CyCADA, Hoffman et al., ICML 2018 (arXiv 1711.03213) | Pixel- and feature-level cycle-consistent adversarial adaptation; labelled source images are translated to the target style [abstract] https://arxiv.org/abs/1711.03213 | The canonical "translate labelled source, train on it" UDA |
| Go et al., CVPR-W 2024 (arXiv 2305.06813) | Artery/vein-mask-conditioned diffusion fundus synthesis for augmentation [abstract] https://arxiv.org/abs/2305.06813 | Medium: structure-conditioned fundus diffusion, cited by RetSyn |
| RLAD, Fhima et al. 2025 (arXiv 2503.01190; Nakayama co-author) | Diffusion conditioned on layout components (vessels, etc.) extracted from real fundus images, for segmentation generalisation [abstract] https://arxiv.org/abs/2503.01190 | Medium: structure from real images conditions the generator; same author network as BRSET/mBRSET |
| AdaptDiff, Hu et al. 2024 (arXiv 2410.04648) | Source-model **pseudo-labels** on target images train a conditional diffusion model, which then synthesises target-style images from source vessel masks for fine-tuning [abstract] https://arxiv.org/abs/2410.04648 | Medium: pseudo-labels plus target diffusion in retina |
| Diffuse-UDA, Gong et al. 2024 (arXiv 2408.05985) | Target **pseudo-labels**, then a diffusion model trained on image–(pseudo)label pairs from both domains generates target-style image–label pairs [abstract] https://arxiv.org/abs/2408.05985 | Medium (segmentation) |
| Uncertainty-Aware ControlNet, Niemeijer et al., ICCV-W 2025 (arXiv 2510.11346) | Uses **unlabelled** Home-OCT (low-cost device) data to train a ControlNet that synthesises labelled target-like OCT [abstract] https://arxiv.org/abs/2510.11346 | Medium: cheap-device shift handled by controlled diffusion with unlabelled target data |
| Palani et al. 2026 (arXiv 2604.15723) | Diffusion autoencoder trained on tabletop images restores handheld-fundus artefacts [abstract] https://arxiv.org/abs/2604.15723 | Low–medium: handheld fundus with diffusion, but the opposite direction (restoration) |
| FD3, Kim et al., IEEE JBHI (arXiv 2409.12377); LED, Cheng et al. 2023 (arXiv 2303.04603) | A synthetic forward (degradation) model plus diffusion for fundus *enhancement*; LED learns the degradation from unpaired data first [abstract] https://arxiv.org/abs/2409.12377 , https://arxiv.org/abs/2303.04603 | Medium for "physics/degradation + diffusion in fundus"; the direction is enhancement, not device rendering for augmentation |
| Sensor Transfer, Carlson et al., IEEE RA-L 2019 | "learned augmentation network composed of physically-based augmentation functions" that transfers chromatic aberration, blur, exposure, noise and colour temperature **from a real dataset** to synthetic data for detection [abstract] https://doi.org/10.1109/LRA.2019.2896470 | **High for stage 1 alone**: target-fitted camera-effect augmentation is not new |
| SimGAN, Shrivastava et al., CVPR 2017 (arXiv 1612.07828); SDEdit, Meng et al., ICLR 2022 (arXiv 2108.01073) | Simulate then refine with a learned model; diffusion editing that starts from a noised guide image [abstract] https://arxiv.org/abs/1612.07828 , https://arxiv.org/abs/2108.01073 | The generic ancestors of "physics output → diffusion refinement" |
| DRUM, Miyawaki et al., ICRA 2026 | A diffusion prior trained on **unlabelled real** LiDAR translates simulator output by reproducing measurement characteristics, with masked guidance [abstract, listing] https://doi.org/10.1109/ICRA57385.2026.11697452 | Medium: a "simulator + target diffusion prior" analogue outside medicine |
| DuDoNet, Lin et al., CVPR 2019 (arXiv 1907.00273) | Sinogram- and image-domain networks linked by a differentiable Radon inversion layer and trained end to end [abstract] https://arxiv.org/abs/1907.00273 | The name only. Our coupling is a non-differentiable initialisation with no joint training, so "dual-domain" in the DuDoNet sense is an overclaim [inferred] |

**Physics + diffusion for fundus or device translation.** In this session's searches I found no paper that initialises a diffusion translator from a target-fitted acquisition model in order to render labelled fundus images in another *device's* appearance for classifier training. [inferred] The searches covered arXiv, OpenAlex, PubMed and the web, but they were not exhaustive.
- The nearest works are FD3 and LED (degradation model + diffusion, for enhancement), SCR-Net and GFE-Net (degradation model + structure-preserving CNN), Sensor Transfer (target-fitted camera effects, no diffusion), DRUM (simulator + target diffusion prior, LiDAR) and FGDM (SDEdit from the source).

**Pseudo-labels and unlabelled target images: a summary for the planned self-training arm.**
- **Unlabelled target images to train the generator** are standard practice: ContourDiff, FGDM, Residual-CycleGAN, ArcNet, Ktena et al., Uncertainty-Aware ControlNet and our own gen_tgt/gen_both all do it.
- **Pseudo-labels from a source-trained teacher on unlabelled target images** are also well established:
  - Ju et al. 2020 and 2021, the cross-device fundus case, including DR classification;
  - the KI 2021 DR UDA paper;
  - BDL (translation + self-training);
  - AdaptDiff and Diffuse-UDA (pseudo-labels feeding diffusion);
  - source-free denoised pseudo-labelling for fundus, Chen et al., MICCAI 2021 (arXiv 2109.09735). [abstract] https://arxiv.org/abs/2109.09735
- **RetSyn's abstract mentions neither.**
- Adding self-training will place our method in the BDL / Ju et al. family. It adds no novelty by itself, and it needs its own control (Section 5e). [inferred]

---

## 4. Side-by-side comparison

| | **Ours** | **RetSyn** (JBI 2025) | **SCR-Net** (MICCAI 2022) + **BMJ 2025 application** | **Residual-CycleGAN** (MICCAI 2020) | **ContourDiff** (MELBA 2025) | **FGDM** (TMI 2024) |
|---|---|---|---|---|---|---|
| Problem setting | Zero-target-training-label DR (+ME) classification, tabletop → handheld | Robust DR screening across devices and image quality, LMICs [abstract] | Cataract restoration; BMJ: cross-camera style standardisation for DR [full text] | Camera-brand UDA for DR [full text] | Unpaired CT→MRI translation for segmentation [full text] | Zero-shot CBCT→CT, cross-site MR [full text] |
| Devices | Canon CR-2 / Nikon NF5050 (BRSET) → Phelcom Eyer + Galaxy S10 (mBRSET) | Tabletop vs Galaxy S10 handheld; paired set Eyer vs Visucam [abstract]; training datasets unknown | Synthetic cataract → clear; BMJ: any camera → UK Biobank style, tested on Optain Resolve and Topcon NW400 [full text] | EyePACS brands A→B…E, private set [full text] | CT, MRI | CBCT, CT, MR |
| What is synthesised | A twin of **each labelled source eye** in the target device's appearance | New images conditioned on class and quality [abstract]; translation vs generation unknown | Deterministic restoration; BMJ maps target images to canonical style (+ mixed training) | **Target test images** → source style at inference | Translated source images | Translated source images |
| Paired images needed | **No** | **Yes**: a small paired smartphone–tabletop set for DPO [abstract] | No (synthetic pairs) | No | No | No (pairs used only for evaluation) |
| Unlabelled target images used | Yes, for the generator and the physics fit | Unknown [abstract silent] | SCR-Net no; ArcNet yes | Yes, for the translator | Yes (target-only training) | Yes (target-only training) |
| Pseudo-labels / self-training | Not yet (planned) | Not mentioned [abstract] | No | No | No | No |
| Target training labels | **None.** Validation labels were used for selection | Unknown | None (BMJ: classifier trained on EyePACS) | None | None (segmenter) | n/a |
| Generator | 104.7M conv pixel-space U-Net, v-prediction, DDIM | "advanced diffusion models"; cites SD3 and Diffusion-DPO [abstract] | CNN encoder + two decoders | CycleGAN, residual generators | Pixel UNet DDPM/DDIM | Denoising Diffusion GAN, ≤ 8 steps |
| Conditioning | Signed local-contrast map of the **source** + FOV mask + device token; CFG on structure (w 1.5) | DR class and image quality [abstract]; device role unclear | High-frequency map of the degraded image (only input) | Unconditional GAN | Canny edges of the source | Thresholded Sobel edges of the source |
| Physics / acquisition model | 19-parameter photometric model **fitted to unpaired target statistics** | None stated | Hand-set cataract imaging model, random parameters | None | None | None |
| Sampling start | **Physics output** noised to strength 0.4 | Unknown | n/a | n/a | Pure noise | Noised **source** image (SDEdit-like) |
| Label / structure check | Device-classifier gate + per-ICDR-grade judge AUROC with blur positive control | Unknown | SSIM/PSNR, vessel IoU, diagnosis F1/kappa; BMJ: PSNR/SSIM to self | Downstream only | Edge HD95, Dice | FID, PSNR/SSIM to source |
| Downstream use | Swap twins into classifier training with probability p | Real + synthetic classifier training; group-balanced loss [abstract] | Enhancement before diagnosis; BMJ mixed training | Test-time translation | Segmenter trained on translations | None |
| Metrics | DR AUROC, F1, per-grade AUROC, patient bootstrap, 3 seeds | AUROC, F1 [abstract] | SSIM/PSNR/IoU/F1/kappa; BMJ accuracy/kappa | Kappa, AUC | Dice, ASSD, FID | FID, PSNR, SSIM |
| Headline result | AUROC 0.901 → 0.928 (+0.027, CI above 0); F1 n.s. | OOD portable AUROC 0.813 vs GroupDRO 0.786 (binary) [abstract] | F1 0.730 → 0.770; BMJ Optain accuracy 75.9 → 79.2%, CIs overlap | "Improved" (numbers not extracted) | Dice 0.705 vs 0.554 | FID 24.0 vs SDEdit 33.2 |
| Code | Public repository (not packaged) | None found | Yes | Promised (PaddlePaddle) | Yes | Yes |

---

## 5. Novelty analysis

### (a) What is genuinely new, as far as this search can tell
1. **The specific coupling: an SDEdit start from a *target-fitted acquisition model's* output, followed by structure-conditioned diffusion.**
   - FGDM starts from the noised *source*, ContourDiff from noise, SCR-Net has no generator stage, and Sensor Transfer has no diffusion. [inferred]
   - It is the only element with a dedicated controlled test in our data: full method − FGDM-style start (diffusion from the plain BRSET image) = +0.0131 [+0.0045, +0.0241], 3/3 seeds; full method − physics only = +0.0231 [+0.0036, +0.0488].
   - This is the strongest candidate contribution, but it is a design-level novelty, not a new principle.
2. **The resource setting for the tabletop→handheld fundus problem: no paired acquisitions and no target training labels.**
   - RetSyn, by its abstract, needs a paired smartphone–tabletop set. The BMJ and Residual-CycleGAN works adapt *test* images toward a source/canonical style rather than rendering *training* images as the handheld device. [inferred]
   - This is a real difference in practice, but it is a setting, not a mechanism.
3. **Instance translation versus class-conditioned generation.** Our label travels with the source eye's lesion-scale structure; RetSyn (by its abstract) and Ktena et al. assert the label through a conditioning token. [inferred]
   - This is a meaningful conceptual contrast, but we have **not tested it**. The class-conditioned baseline in (e)2 would test it.
4. **A label-preservation gate per ICDR grade, with a validated judge and a destructive positive control,** plus a dose-response and label-budget analysis.
   - Methodological rigour more than novelty; EyeBench-V2 and SCR-Net already evaluate downstream preservation. [inferred]

### (b) What overlaps
- **RetSyn:** same clinical problem and same target device (very probably), diffusion synthesis, real + synthetic classifier training, and the idea of device/quality-aware generation. [abstract] [inferred]
- **FGDM / ContourDiff:** target-trained diffusion conditioned on a structure representation of the source, used without pairs; FGDM's SDEdit start is the template for ours. [full text]
- **SCR-Net / ArcNet / GFE-Net:** the signed high-pass structure representation for fundus, and degradation-model twins that share structure. [full text]
- **BMJ 2025, Residual-CycleGAN 2020:** structure-preserving or unpaired camera translation for DR with zero target labels, including mixed training. [full text]
- **Sensor Transfer, SimGAN, DRUM:** target-fitted camera/sensor effects, and simulate-then-refine. [abstract]
- **Ktena et al., DG-ADR:** generative augmentation for distribution shift, with grade-conditioned DR augmentation. [abstract]
- **Ju et al., BDL, AdaptDiff, Diffuse-UDA:** translation or diffusion combined with pseudo-labels. [abstract]
- **CFG, DDIM and v-prediction** are standard.

### (c) What a skeptical ISBI reviewer would say
1. "RetSyn (JBI 2025), from the mBRSET authors, already uses diffusion-generated device-diverse images to improve portable-camera DR screening. What is new here?"
2. "The method is FGDM/ContourDiff with a different edge map plus an SDEdit start from a colour/noise augmentation. The 'physics' is a photometric augmentation fitted to 13 image statistics, not an identified camera model."
3. "'Dual-domain' and 'world model' are overclaims. Nothing is trained jointly; the physics output is just the initial image."
4. "The images are not realistic: a linear probe separates them from real mBRSET at AUROC 1.00, only 40% fool the device classifier, and 92% of the feature gap remains. Is this translation, or a structured augmentation?"
5. "The diffusion stage changes pixels by about 3.5/255 relative to physics-only. Any stochastic texture perturbation might do the same." Physics-only being flat answers part of this; a matched-texture control would answer it fully.
6. "Label preservation is not shown. Grade-1 judge AUROC drops 0.98 → 0.91 and healthy eyes look more diseased to the judge. Also, the judge was trained with mBRSET labels and chose your sampler settings, and validation labels chose the configuration among 12. So it is not zero-label."
7. "There are no baselines beyond ERM: no CycleGAN/CUT, FDA or colour transfer, DANN or self-training, and no RetSyn-style generator. +0.027 AUROC over source-only ERM is weak evidence on its own."
8. "F1 does not move, 3 seeds, one dataset pair, a reused test split, and the gain disappears with 250 labels. What is the practical value?"
9. "The ablation was run on a different configuration (gen_tgt, 50%) from the headline one (gen_both, 25%)."

### (d) Which claims are defensible and which are not

**Defensible now** (with the caveats attached):
- "Unpaired, physics-initialised, structure-conditioned diffusion twins of labelled tabletop images raise handheld-test DR AUROC with zero target *training* labels: 0.901 → 0.928, patient-bootstrap CI of the gain [+0.010, +0.049], 3/3 seeds, configuration chosen on validation."
- "Neither stage alone gives a significant gain. Physics alone: +0.003. Diffusion from noise: +0.017. Diffusion from the plain source: +0.013, and both diffusion-only variants significantly *lower* F1. Physics-initialised diffusion beats the no-physics start and physics-only with paired CIs above 0." State that the ablations used gen_tgt 40k / 50%.
- "The gain concentrates in mild DR (grade-1 AUROC 0.759 → 0.815) and at moderate specificity. It does not move F1 at the validation-chosen operating point." Descriptive.
- "Synthetic twins match the AUROC of about 250 real handheld labels (0.928 vs 0.922) and add nothing once those labels exist (±0.001)." Descriptive; no equivalence test.
- "Longer generator training (40k/60k/100k) and the choice of generator variant do not change the downstream result."

**Not defensible:**
- "First diffusion / synthetic-data approach for tabletop→handheld DR" (RetSyn).
- "First cross-camera translation for DR" (Residual-CycleGAN, BMJ).
- "Label-preserving" without qualification: the grade-1 loss and the healthy-eye score inflation are measured.
- "Realistic handheld images" or "crosses the device boundary".
- "Dual-domain (DuDoNet-like)" or "world model".
- "Label-free".
- "Improves F1 / the clinical operating point".
- "Better than RetSyn": no head-to-head comparison, and the numbers are not comparable.
- "Our structure map is better than edges or frequency maps": not ablated.
- "Physics model of the handheld camera": it is an empirical photometric fit.

### (e) Experiments and baselines that would make the novelty defensible
All of these use the identical swap protocol: same classifier recipe, p = 0.25, 3 seeds, validation-only choices, one test read and a patient bootstrap.

1. **Unpaired GAN translation baseline: CUT and/or CycleGAN, BRSET→mBRSET** (CycleGAN arXiv 1703.10593, CUT arXiv 2007.15651), trained on the same unlabelled images.
   - Optionally add the test-time direction (mBRSET test → BRSET style, as in Residual-CycleGAN and BMJ).
   - This answers "translation-based UDA is old". It is the single most important missing comparison.
2. **RetSyn-style class-conditioned generator.** Use the same U-Net and data, conditioned on (device, DR label) with no structure map. Sample from noise, assign the conditioning label, and use the same number of images and the same swap rate. Disclose that no DPO is possible without pairs.
   - This tests (a)3 directly ("label travels with the structure" vs "label asserted by a token") and is the fairest RetSyn proxy we can build. One more 40k-step training run is about 6 h on 4 H100s.
3. **Cheap target-informed and standard UDA baselines.**
   - FDA (arXiv 2004.05498) amplitude swap with real mBRSET images, plus Reinhard/histogram colour transfer. Physics-only already covers the photometric family.
   - DANN (arXiv 1505.07818).
   - **Because self-training is now being tested:** self-training from the *source-only* teacher's pseudo-labels on the same 3,402 mBRSET images, compared with self-training from the synthetic-trained teacher. Without this control any self-training gain cannot be attributed to the translator; BDL and Ju et al. already show that translation + pseudo-labels helps.
4. **Structure-representation ablation.** Replace our map with SCR-Net's high-pass map (I − G_σ9), FGDM's thresholded Sobel map and ContourDiff's Canny map, keeping everything else. Without this, describe the map as a design choice and claim no advantage.
5. **Label hygiene for the "zero-label" wording.**
   - Re-run the gate-2 selection with a judge trained on BRSET only, or report every configuration (already in `full_table_2026-10-14_0125.json`) to show robustness to selection.
   - State explicitly that 725 labelled validation images choose the checkpoint, cut-off and configuration.
6. **Matched-texture control.** Physics output plus random texture/noise matched in amplitude to the diffusion edit (about 3.5/255). This rules out "any perturbation helps".
7. **Repeat the ablations on the final configuration** (gen_both 40k, 25%) and add seeds 3–5 for the headline and key ablations; the CI half-width is about 0.02 at 3 seeds.
8. **Optional paired fidelity check with RetSyn's public pairs.**
   - Translate the 399-image Hugging Face subset's Visucam images, if pairing can be reconstructed from the file names, and compare with the Eyer partners (LPIPS / device-classifier score).
   - This is a source-camera shift (Visucam ≠ Canon/Nikon), and the licence and pairing structure must be checked first.
   - Never commit those images to the public repository.

### (f) Proposed novelty hypothesis / contribution statement (worded to be true today)

> We study the adaptation of a diabetic-retinopathy classifier from tabletop (BRSET) to smartphone-based handheld (mBRSET) fundus photographs without paired same-eye acquisitions and without target-domain training labels. Each labelled tabletop image is rendered as a handheld-camera twin by a two-stage translator: a 19-parameter photometric acquisition model fitted to unlabelled handheld images provides the starting point for a pixel-space diffusion model conditioned on a lesion-scale contrast map of the source eye, so the twin inherits its label through the source anatomy rather than through a class token. Replacing 25 % of training images with their twins raises handheld-test DR AUROC from 0.901 to 0.928 (three seeds; patient-bootstrap 95 % CI of the gain +0.010 to +0.049) without changing F1. In ablations, neither the physics stage alone nor diffusion without the physics start yields a significant gain, and the benefit matches that of about 250 real handheld labels and vanishes once those labels are available.

Positioning sentence for related work (true by abstract): "RetSyn [Shuai et al., 2025] instead synthesises class- and quality-conditioned images and aligns them to the handheld device with DPO on a paired smartphone–tabletop set; our setting assumes no such pairs." Add the conditional clause from (a)3 only if baseline (e)2 supports it. Add "outperforms unpaired GAN translation" only if baseline (e)1 supports it.

---

## 6. Open questions / could not verify

1. **RetSyn full text, which is not read.** Unknown so far:
   - training and test datasets, and whether they are BRSET/mBRSET;
   - whether portable-device labels or unlabelled portable images are used;
   - the size of the DPO paired set and how preference pairs are built;
   - the backbone (SD3-class is only inferred from citations), resolution and conditioning format;
   - whether device is a conditioning variable;
   - generation from noise versus image-conditioned translation;
   - the real:synthetic ratio and the label assignment;
   - the classifier, seeds, confidence intervals, baselines beyond GroupDRO, and stated limitations.

   Action: someone with browser access should open https://doi.org/10.1016/j.jbi.2025.104938 (CC BY) and fill in Section 2. A shared author may also be reachable through the mBRSET maintainers.
2. **RetSyn data release.** The card promises the whole paired set "after accepted" (the paper was accepted 11 Oct 2025), but in March 2025 the public split showed 399 device-labelled images and no visible DR or quality labels. The current file list and labels were not audited, beyond listing two folders.
3. **The earlier RetSyn title** ("…with DPO Using Paired Data"): no copy found, so it is unknown whether a workshop or preprint version exists with more detail.
4. **Residual-CycleGAN** numerical results were not extracted, only the method and the metric definitions.
5. **BMJ 2025** does not say how SCR-Net was trained on UK Biobank (which degradations). Its abstract (SSIM 0.847) and body (0.817) disagree, and the kappa type is unstated.
6. **FGDM:** the published CBCT noise schedule is not stated (the code defaults may be the MR configuration), and the direction of the MR-T1 training is ambiguous in the text.
7. **ContourDiff:** the README implies CT validation data during training, against the paper's "no input-domain access" claim. Unresolved.
8. **DG-ADR, Ktena et al., Ju et al., KI 2021, AdaptDiff, Diffuse-UDA, Uncertainty-Aware ControlNet, FD3, LED, Sensor Transfer, DRUM** were read at abstract or listing level only. The KI 2021 authors and venue details were not verified.
9. **The search was not systematic.** "No physics-initialised diffusion translator for fundus device rendering was found" is a statement about this search, not proof of absence.
10. **Whether a published paper already trains on BRSET and tests on mBRSET with synthetic data** depends on RetSyn's full text (item 1). The April 2026 foundation-model transfer study (see `DIRECTION_REVIEW.md`) covers BRSET→mBRSET transfer, but not synthesis.

---

## Sources (primary links used above)
- RetSyn: https://doi.org/10.1016/j.jbi.2025.104938 · https://pubmed.ncbi.nlm.nih.gov/41138952/ · https://api.crossref.org/works/10.1016/j.jbi.2025.104938 · https://huggingface.co/datasets/smartretina2025/paired_retina · https://datasets-server.huggingface.co/info?dataset=smartretina2025/paired_retina
- mBRSET: https://pmc.ncbi.nlm.nih.gov/articles/PMC11846882/ (DOI 10.1038/s41597-025-04627-3) · BRSET: https://pmc.ncbi.nlm.nih.gov/articles/PMC11239107/
- SCR-Net: https://arxiv.org/abs/2206.04684 · ArcNet: https://arxiv.org/abs/2203.07737 · GFE-Net: https://arxiv.org/abs/2309.00885 · code: https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement
- BMJ cross-camera: https://pmc.ncbi.nlm.nih.gov/articles/PMC12658492/ · Residual-CycleGAN: https://arxiv.org/abs/2007.15874
- ContourDiff: https://arxiv.org/abs/2403.10786 · FGDM: https://arxiv.org/abs/2304.02742
- DG-ADR: https://arxiv.org/abs/2411.02614 · Ktena et al.: https://arxiv.org/abs/2304.09218 · Ju et al. 2021: https://arxiv.org/abs/2011.13816 · Ju et al. 2020: https://arxiv.org/abs/2003.10042 · BDL: https://arxiv.org/abs/1904.10620 · CyCADA: https://arxiv.org/abs/1711.03213
- Go et al.: https://arxiv.org/abs/2305.06813 · RLAD: https://arxiv.org/abs/2503.01190 · AdaptDiff: https://arxiv.org/abs/2410.04648 · Diffuse-UDA: https://arxiv.org/abs/2408.05985 · Uncertainty-Aware ControlNet: https://arxiv.org/abs/2510.11346 · Palani et al.: https://arxiv.org/abs/2604.15723 · FD3: https://arxiv.org/abs/2409.12377 · LED: https://arxiv.org/abs/2303.04603 · source-free fundus pseudo-labelling: https://arxiv.org/abs/2109.09735
- Sensor Transfer: https://doi.org/10.1109/LRA.2019.2896470 · SimGAN: https://arxiv.org/abs/1612.07828 · SDEdit: https://arxiv.org/abs/2108.01073 · DRUM: https://doi.org/10.1109/ICRA57385.2026.11697452 · DuDoNet: https://arxiv.org/abs/1907.00273
- Baselines: CycleGAN https://arxiv.org/abs/1703.10593 · CUT https://arxiv.org/abs/2007.15651 · FDA https://arxiv.org/abs/2004.05498 · DANN https://arxiv.org/abs/1505.07818 · DDIB https://arxiv.org/abs/2203.08382 · CFG https://arxiv.org/abs/2207.12598 · GDRNet https://arxiv.org/abs/2307.04378
