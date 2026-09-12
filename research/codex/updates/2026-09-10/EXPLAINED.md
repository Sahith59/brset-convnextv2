**The three-page degradation report, explained — September 10, 2026**

User reports sending the PDF and message to Professor Dong in Slack. This guide explains the sent report; it does not change it or claim new diagnostic results.

**The purpose**

Our eventual goal is better DR/ME prediction on real mBRSET photographs. This report answers a smaller question: does our existing image transformation move some BRSET appearance measurements closer to mBRSET, even on additional patients? DR and ME are the two dataset disease labels. A classifier is the neural network that predicts those labels. No classifier was trained or evaluated in this appearance check.

**Page 1: the comparison**

BRSET is the source, whose images we transform. mBRSET is the target, whose appearance we use as a reference. A domain difference can involve acquisition, framing, population and disease prevalence. Appearance matching alone does not isolate a camera effect.

There were 64 source images from 64 patients, and 64 target images from 59 patients. Some patients contribute several images, which is why image counts and patient counts differ. The new sample excludes patients in reconstructed old measurement/fit samples. Reconstructed means we recovered the selection from code and seeds; a contemporaneous original file manifest was not saved. These are training-pool images, not an untouched external study.

Frozen means the transformation settings did not change during this check. Fitted means an earlier search selected them using an appearance objective. A generic setting is a manually chosen reference. Our local generic implementation is not the official FundusAug method.

Direct 512×512 resize squeezes the whole photograph into a 512-pixel square; legacy means this older preparation. The alternative first resizes to a 560-pixel square, then removes 24 pixels on every side to leave the central 512 square. Resize changes sampling/shape; crop discards an outer region. Neither square-resize route preserves the shape of a nonsquare original. Bilinear and bicubic interpolation estimate the new pixel values differently. The latter preparation matches classifier evaluation geometry before normalization: numerical scaling used at the network input. The degradation is applied after image preparation in this check.

RGB means red, green and blue channels. Divide pixel values by 255 to put them on a 0–1 scale. Grayscale here is their arithmetic mean. A mask selects which pixels to measure: grayscale >0.04 excludes most black background. It is not a learned/anatomical retinal segmentation.

Five summaries are measured. Sharpness is variance of the grayscale Laplacian, which measures local intensity changes; random noise can increase it. Brightness is average masked intensity. Contrast is how much masked intensities vary. Falloff is center brightness minus outer brightness. Saturation measures color strength, using (largest RGB channel minus smallest)/largest and averaging in the mask. Brightness/contrast/sharpness statistics cannot establish lesion visibility.

For falloff, center pixels lie at a radius less than 25% of the shorter image side; outer pixels at a radius greater than 39%; both are masked. Radius is distance from the image center. The mask avoids treating most background as dim retina. The code has minimum-pixel checks; all included quantitative measurements must be finite.

The appearance-distance formula is:

    D_r(A,T) = sum over k=1..5 of ((mu_A,r,k - mu_T,k) / max(s_T,k, 0.000001))²

Here A is a source arm, T the target, k the measurement and r the random seed. Mu is the average across images. s_T is the sample standard deviation across target images, their spread. For each measurement, subtract the target average from the source average, divide by the target spread, square, then add all five contributions. The tiny floor prevents dividing by zero. Standardizing avoids brightness or sharpness dominating only because of their units; it does not prove equal clinical importance or remove correlations among measurements.

Illustration only: source brightness 0.50, target 0.40, target spread 0.10 gives a contribution ((0.50-0.40)/0.10)²=1. These are not the measured data. A smaller overall distance means closer on five averages. Even zero would not prove identical image distributions or correct visible disease.

| Preparation | Original | Local generic | Fitted |
|---|---:|---:|---:|
| Direct 512 square | 6.742 | 10.092 | 1.261 |
| 560 square then center 512 | 8.474 | 12.011 | 4.294 |

Within each row, fitted is closer and generic farther from the target than original. These values are not accuracy percentages. Target statistics are recalculated per row, so cross-row distances are not a common-scale effectiveness comparison. The transformed columns average three transform runs on the same images. This is not three independent patient samples or a confidence interval. A seed initializes a reproducible sequence of pseudorandom choices; it is not an amount of noise.

**Page 2: the simulator and search**

The earlier procedure measured 400 training images per dataset, tried 150 setting combinations on 25 source images, and retained a minimum-distance combination. It was random search over an empirical simulator, not neural generator training. The old source-fit and target-measurement resizing differed; the report discloses that. Our new check tests the saved settings under two defined preparations without refitting them.

The ordered operations are blur, uneven illumination, bright/dark spots and halo, sensor-like noise, then brightness/contrast/saturation. A Gaussian is bell shaped: a smooth blur/spot is strongest centrally and weaker farther away. A halo is a ring-shaped brightness effect. Sensor-like noise means random perturbations, not a measured model of the actual camera sensor. Zero mean means perturbations average zero before clipping. Clipping limits computed pixels to their valid range. The sequence matters because later operations change earlier effects. There is no separate 50% coin flip for each configured operation; zero-strength terms have no effect.

| Parameter | Allowed search range | Generic | Fitted | Meaning |
|---|---|---:|---:|---|
| Blur sigma | 0–1.2 | 1.0000 | 0.9071 | Width of smoothing in pixels |
| Light strength | 0–0.6 | 0.2500 | 0.2613 | Magnitude of smooth uneven illumination |
| Bright-spot count | Integers 0–4 | 2 | 1 | Number of added soft bright patches |
| Dark-spot count | Integers 0–3 | 1 | 3 | Number of added soft dark patches |
| Halo amplitude | 0–0.20 | 0.1000 | 0.0857 | Strength of ring-shaped brightening |
| Brightness multiplier | 0.70–1.15 | 1.1500 | 1.0767 | Brightness adjustment at the color stage |
| Contrast multiplier | 0.90–2.00 | 1.1500 | 0.9160 | Contrast adjustment at that stage |
| Saturation multiplier | 0.50–1.20 | 1.1000 | 1.0028 | Color-strength adjustment |
| Noise sigma | 0–0.030 | 0.0000 | 0.0120 | Standard deviation of added noise on 0–1 RGB |

Multiplier 1 means unchanged; above 1 increases, below 1 decreases. The fitted settings jointly produce an image; multiplying brightness by 1.0767 does not describe the final brightness change after the full chain. Illumination strength 0.2613 is not “26% degradation.” Displayed fitted settings are rounded; the JSON retains full precision.

Uniform search means equal-length continuous intervals are equally likely; count values are sampled from allowed integers. Search/sample seed 0 and trial t's transformation seed t make the procedure reproducible. The saved vector matches zero-based trial 87, the 88th draw. Current image-selection seed is 20260910. Quantitative transformation seeds are 0/1/2. Figure generic seeds are 100–102; fitted seeds 200–202. Different seeds can move spots even with fixed strengths/counts.

R is half the shorter image side, or 256 pixels for a 512 square. Bright Gaussian widths are 0.015–0.055R (3.84–14.08 pixels); dark widths 0.015–0.045R (3.84–11.52 pixels). These are Gaussian widths, not hard boundaries. Centers lie within 15–85% of height/width and amplitudes vary 0.10–0.30. Halo radius is 0.82R, width 0.10R. The light-field center shifts up to ±22% of each dimension; its Gaussian width is 55% of the shorter side. The multiplier is 1 + strength×(Gaussian−0.5). Noise is applied in the pre-noise mask, independently across RGB array entries.

Variance measures squared spread; standard deviation is its square root. Pixel contrast uses population standard deviation over the included pixels. The distance denominator uses sample standard deviation across sampled target images, with the usual n−1 correction. “High-frequency energy” here is a shorthand for local rapid intensity changes, not a measure of useful clinical information.

**Page 3: examples and evidence**

Each row is original BRSET, the same eye with generic alteration, the same eye with fitted alteration, and unrelated real mBRSET. Unpaired means we do not have two-device photographs of that same eye in these rows. Thus the last column is not a desired pixel-perfect reconstruction. Original means after resize/crop, with no added degradation; it does not certify a clinically clean image.

Rows use BRSET img04157.jpg, img06199.jpg, img15850.jpg and mBRSET 1317.1.jpg, 252.3.jpg, 927.1.jpg. These are the first three randomly selected examples, not a visually chosen best case. Provenance means where data came from and what operations were applied. All six sources match raw dataset tables and train manifests. A hash is a digital fingerprint; source hashes are retained and the twelve PDF panels match regenerated pixels.

Artificial spots are not new medical annotations. A transformed image keeps the source's label in a training recipe, but the simulator might obscure visible disease or add misleading features. Preserving metadata does not guarantee preserving visual diagnostic evidence.

Ophthalmic review means review by someone qualified to interpret retinal images. Diagnostic preservation asks whether disease clues remain usable. Classification improvement asks whether trained models predict real target labels better. Novelty asks whether a specific contribution differs meaningfully from prior research. None of those three has been established by the current appearance check.

Distillation, a possible later step, trains a student to learn from a teacher's outputs/features. Consistency encourages predictions/features for different views of the same image to agree. Both ideas already exist. A fair ablation changes one component while keeping the rest controlled, so an observed difference can be attributed more narrowly.

**What we do next**

Step 1 is recorded in `../../protocol/STEP_1_PROTOCOL.md`: partition patients, pin initialization, define equal training-update budgets and fix checkpoint/threshold rules. A checkpoint is a saved model state; a threshold converts a disease probability to yes/no. Fit data teaches the model, selection data chooses settings, and assessment data scores those frozen choices. Assessment becomes development evidence if used to select future methods. This split cannot erase earlier exposure to the old test set.

Then implement and check the baseline trainer before GPU comparison runs. The contribution hypothesis and the evidence that could reject it are in `../../protocol/NOVELTY_HYPOTHESIS.md`. Our immediate achievement is a trustworthy appearance analysis and a concrete experimental contract; improved disease prediction remains to be tested.
