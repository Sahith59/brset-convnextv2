# Method, end to end (verified against the code and checkpoints, 14 Oct 2026)

Sources: `scripts/72_physical_layer.py`, `83_train_translator.py`, `88_synth_augment.py`,
`109_gen_common.py`, `110_train_diffusion.py`, `111_translate.py`, `112_gates.py`,
`30_train_strong_baseline.py`; physics values read from
`results/xlate_v2_noartifact/translator_last.pth` (job physparams, 14 Oct).

## 0. Pipeline

```
labelled BRSET photo x ──► (1) PHYSICS camera model ──► x_phys  (global colour, light, blur, noise)
          │                                                │ re-noised to strength 0.4
          └──► (2) STRUCTURE MAP s(x) ─────────────────────┤ (vessels, disc, lesions as local contrast)
                                                           ▼
                          (3) DIFFUSION renderer, device token = mBRSET, 12 DDIM steps
                                                           ▼
                          synthetic mBRSET twin  (same eye, same DR / oedema label)
                                                           ▼
          (4) classifier trained on BRSET with each image swapped for its twin with probability p
          (5) optional: self-training on unlabelled real mBRSET photos (pseudo-labels)
```

No paired photos, no mBRSET label, no validation/test image is used by stages 1–3.

## 1. Physics stage (19 bounded parameters, DuDoNet-style "acquisition domain")

Input x ∈ [0,1]^{3×560×560}; retina mask m = 1[mean_c x > 0.04]; grey(x) = channel mean.
Each parameter θ_k = lo_k + (hi_k − lo_k)·sigmoid(raw_k), so it never leaves its physical range.
Operations, in the order a camera forms an image (fitted value in brackets):

1. **Optics (signed sharpness).** x₁ = x + k·(x − G_σ=1 ∗ x), k = −0.930 → x₁ ≈ 0.07·x + 0.93·(G₁ ∗ x): a mild 1-pixel blur.
2. **Uneven illumination.** x₂ = x₁ · (1 + L·(e^{−d²} − 0.5)), d² = ((u−c_u)² + (v−c_v)²) / (2·(0.55·560)²),
   centre (c_u, c_v) jittered uniformly ±22 % of the frame per image; L = 0.232 → smooth ±11.6 % brightness field.
3. **Halo ring.** x₃ = x₂ + h·exp(−(r − 0.82R)² / (2(0.10R)²)), r = distance from the frame centre, R = 280 px; h = 0.103.
4. **Spots / holes.** x₄ = x₃ + a·Σ_{1 spot} blob − a·Σ_{3 holes} blob, Gaussian blobs of radius 1.5–5.5 % of R at random
   positions; a = 0.020 (deliberately capped: left free, the fit painted haemorrhage-like blobs onto healthy eyes).
5. **Sensor noise (shot + read, per channel).** x₅ = x₄ + sqrt(a_c·x₄ + b_c²)·ε·m, ε ~ N(0,1);
   a = (8, 6, 2)·10⁻⁵, b = (0.00004, 0.00005, 0.0028) for (R, G, B) → effectively only faint blue read noise.
6. **Colour pipeline.** white balance / black level x₆ = x₅ ⊙ g + o, g = (0.671, 0.844, 1.342), o = (+0.111, −0.147, −0.148);
   contrast x₇ = μ + 1.547·(x₆ − μ), μ = masked mean of grey(x₆); saturation x₈ = grey(x₇) + 0.571·(x₇ − grey(x₇)).
7. **Output** x_phys = clamp(x₈ ⊙ m, 0, 1).

Every operation is photometric or a smooth global field: none can move a vessel or a lesion.

**How the 19 values were learned (script 83).** Jointly with a 68k-parameter bounded refiner (that refiner is
DISCARDED in the current method; only the physics stage is reused), 4,000 steps, unpaired and unlabelled:
(i) statistic loss — means AND spreads of 13 colour/brightness/contrast/sharpness statistics matched to real
mBRSET; (ii) style loss — Gram matrices of a frozen VGG16 on crops from inside the retina, against real mBRSET;
(iii) residual penalty (refiner only); adversarial term off. Noted: the fit chose k = −0.93 (blur) although mBRSET
carries more fine-scale energy; the diffusion stage supplies that texture.

Measured alone: device classifier calls 26 % of physics-only images mBRSET (untouched BRSET 0.2 %, our full twins
47 %); disc colour 152/55/33 vs real mBRSET 153/73/42; classifier gain at 25 % swap −0.001 AUROC (none).

## 2. Structure map (what carries the label)

Green channel g; G_σ = Gaussian blur; σ_L = 10 px:
μ = G_σL∗(g·m) / G_σL∗m,  d = (g − μ)·m,  σ_loc² = G_σL∗(d²) / G_σL∗m,
s = tanh( 0.5·d / (σ_loc + 0.03) ),  then s ← (G₁∗s)·erode(m, 4 px).
s ∈ (−1, 1): small dark structures (vessels, microaneurysms, haemorrhages) negative, bright ones (exudates, disc)
positive, flat retina ≈ 0; the 0.03 floor stops flat-region sensor noise from being amplified. At translation s is
read from the SOURCE BRSET image; in generator training it is read from a degraded copy (with prob. 0.9: Gaussian blur
σ ∈ [0, 1.8] px, Gaussian noise std ∈ [0, 0.015]) so the model learns to follow the map without copying its noise.

## 3. Diffusion renderer (gen_both)

**Network.** diffusers `UNet2DModel`, 104.7 M parameters, fully convolutional (no attention), so it trains on
256-px crops and runs on the full 560-px frame. Input 5 channels = noisy image (3) + structure map s (1) + mask m (1);
output 3 channels (velocity). 5 resolution levels with 128, 256, 256, 512, 512 channels, 2 ResNet blocks per level,
down/up-sampling with skip connections. Timestep t → sinusoidal embedding → MLP (dim 512); device token
d ∈ {BRSET 0, mBRSET 1, null 2} → learned embedding added to the timestep embedding.

**Forward (noising) process.** DDPM with T = 1000 and the cosine schedule (Nichol & Dhariwal):
x₀ ∈ [−1, 1], x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε.

**Objective (v-prediction).** v = √ᾱ_t·ε − √(1−ᾱ_t)·x₀,
L = E_{x₀, t, ε} ‖ v_θ(x_t, t; s̃, m, d̃) − v ‖², with s̃ = 0 w.p. 0.1 (structure dropout) and d̃ = null w.p. 0.1
(device dropout) — both make classifier-free guidance possible.

**Data and optimisation.** Unlabelled BRSET train (11,372) + mBRSET train (3,402); each sample: pick a camera with
probability ½, pick an image, take a random 256×256 crop that is ≥ 30 % retina, random horizontal/vertical flip.
AdamW, lr 2·10⁻⁴ after 1,000 warm-up steps, weight decay 0.01, gradient clip 1.0, bf16; batch 96 (8 A100 × 12);
40,000 steps; EMA of weights with decay min(0.9999, (1+step)/(10+step)), EMA weights used for sampling.
What it learns: "this retina structure, photographed by this camera" — one retina, two renderings.

## 4. Translation = physics + diffusion coupled (dual-domain sampling)

For each BRSET training image x (label y):
1. x_phys = Physics(x); s = s(x); m = mask(x).
2. DDIM, 30 steps, "trailing" spacing; strength 0.4 → skip the 18 noisiest steps; start at the 19th step t*:
   x_{t*} = √ᾱ_{t*}·(2·x_phys − 1) + √(1−ᾱ_{t*})·ε.
3. For the remaining 12 steps, guided velocity
   v̂ = v_θ(x_t, t; s, m, mBRSET) + (w − 1)·[ v_θ(x_t, t; s, m, mBRSET) − v_θ(x_t, t; 0, m, mBRSET) ],  w = 1.5,
   x̂₀ = √ᾱ_t·x_t − √(1−ᾱ_t)·v̂,  ε̂ = √(1−ᾱ_t)·x_t + √ᾱ_t·v̂,  x_{t'} = √ᾱ_{t'}·x̂₀ + √(1−ᾱ_{t'})·ε̂ (deterministic DDIM).
4. Twin = clamp((x₀ + 1)/2, 0, 1) ⊙ m, saved as a 560-px PNG; label of the twin = y.

Why couple: physics sets global colour/light, so diffusion starts 60 % of the way to a clean image and only adds
phone-specific fine texture; starting from pure noise loses mild lesions (grade-1 judge AUROC 0.871 vs 0.907) and,
downstream, makes F1 significantly worse. Measured downstream (gen_both 40k, 25 % swap, 3 seeds, test AUROC):
physics only 0.8999, diffusion only 0.9158, diffusion from plain BRSET 0.9167, physics → diffusion 0.9279
(each paired difference to the full method has a 95 % CI above 0).

## 5. Quality gates (before any classifier sees the twins)

546 held-out BRSET images (val + test split; 80 per grade where available + healthy): gate 1 = frozen ConvNeXt-V2-nano
features + logistic regression trained on real BRSET vs real mBRSET → mean P(mBRSET) of the twins; gate 2 = a
joint-trained judge's AUROC per ICDR grade on the twins, validated by a lesion-blur control.

## 6. Classifier and protocol

ConvNeXt-V2-Large (ImageNet-22k), 512-px random crops of 560-px images, flips/rotation/colour jitter, focal loss
(γ 2), mixup 0.2, label smoothing 0.1, AdamW lr 3·10⁻⁵, 25 epochs, batch 16 × 4 accumulation, EMA 0.999, DR and
oedema as two binary outputs. Twin swap: each time a BRSET training image is loaded it is replaced by its twin with
probability p (redrawn every load). Best epoch and the DR cut-off chosen on mBRSET validation (725, bootstrap-median
F1 cut-off); flip-4 test-time augmentation; mBRSET test (732 images, 193 patients) read once; patient-cluster
bootstrap (2,000) for all CIs.

## 7. Optional extension: self-training on unlabelled real phone photos

The twin-trained model scores the 3,402 unlabelled mBRSET training photos; photos with score < 0.10 become
"healthy", > 0.7 "DR", the rest are left out; oedema masked. A new classifier is trained on BRSET + twins (25 %) +
these pseudo-labelled photos (ST-B: 2,509 photos, test AUROC 0.9362 = 500 real labels; F1 0.7784, n.s.).
