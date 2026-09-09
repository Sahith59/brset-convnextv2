# Research plan: BRSET to mBRSET cross-device transfer

Last updated 3 September 2026, after the advisor meeting. Priorities reordered. Target submission mid-October 2026.

This is the working plan. It records what is established, what is ruled out,
what the contribution could be, and what to do next in priority order. Every
number below is measured on our own data and traceable to a run in `results/`.

---

## 1. The research question

A diabetic retinopathy classifier trained on tabletop-camera images (BRSET)
loses half its recall on handheld phone-camera images (mBRSET), from 10 missed
of 162 to 50 missed of 159. The two datasets share no patients, so no paired
data exists.

**Question: how do you transfer a classifier to a new imaging device when the
device changes the images, the population changes the disease rate, and you
cannot pair the two datasets?**

Stated in the general form rather than the ophthalmic one: two datasets with
the same semantic content, one high quality and one low, with no correspondence
between them.

---

## 2. What is established, with evidence

| Finding | Evidence |
|---|---|
| Ranking survives, the decision collapses | AUC 0.9906 to 0.9060, recall 0.9383 to 0.6855 |
| At most 28% of the gap is the decision cutoff, at least 72% is the representation | Six target-trained endpoints spanning F1 0.8146 to 0.8355; the cutoff share never exceeds 28% |
| Calibration and label-shift methods cannot exceed F1 0.7927 here | Every cutoff from 0.005 to 0.995 swept. Monotone transformations preserve ordering, so they select a point on the same curve |
| The shift is not only prevalence | ROC is invariant to class balance (Fawcett 2006), yet AUC fell |
| Aggregating both training sets is the best available model | DR F1 0.8250, 27 of 159 missed. Statistically tied with fine-tuning (p = 0.47) but catches more patients |
| Encoder-level domain separation does not help | DSN at best setting: DR F1 0.8232, 31 missed. Difference from baseline 0.0015, p = 0.91 |
| The domain-adversarial term is actively harmful here | Sweeping its weight 0.25, 0.05, 0 raises DR F1 0.8053, 0.8227, 0.8232 and cuts missed 37, 36, 31 |

---

## 3. What is ruled out, and why

| Approach | Why it is out |
|---|---|
| Calibration, temperature scaling, label-shift correction | Provably capped at F1 0.7927 on this model. Measured, not assumed |
| Adversarial adaptation (DANN, MDD, CycleGAN) | Documented to collapse on real tabletop to portable fundus data (Lin, MICCAI 2022) |
| Test-time adaptation | Loses up to 66 points under prior shift, our exact regime (LAME, CVPR 2022) |
| Generic domain generalization | No method reliably beats plain training under fair evaluation (Gulrajani, ICLR 2021) |
| Encoder-level domain separation | Measured here. A wash against plain aggregation |
| DuDoNet's dual-domain method | Needs a sinogram, which fundus photography does not have. Framing only, not method |

---

## 4. The contribution, and the honest risk on each candidate

### Candidate A — device-fitted degradation with clean-to-degraded distillation

**The claim.** Published augmentation applies fixed operations at fixed
probabilities, tuned for no particular camera. Fitting them to the measured
statistics of the actual target device makes the synthetic pairs realistic
enough to distil from, which is what turns augmentation into a supervision
signal.

**Why it answers the unpaired objection.** Degrading a BRSET image produces a
clean and a degraded view of the SAME image. That is a perfect pair by
construction, and it unlocks every method that requires pairs.

**Papers:** cofe-Net (IEEE TMI 2021) for the physics, GDRNet (MICCAI 2023) for
the operator set, Self-Feature Distillation (ECCV 2022) for the distillation.

**The risk that could kill it.** FDA (Yang and Soatto, CVPR 2020) swaps the
low-frequency Fourier amplitude between source and target. No training, no
paired data, one FFT. If that matches a fitted physical degradation model, the
fitting adds nothing. **This must be tested as a baseline before committing.**

### Candidate B — abstention under device shift  [TESTED, RULED OUT]

**The claim.** 83 percent of mBRSET images carry an artifact. Some are
genuinely ungradable, and no model should be confident on them. Instead of
forcing a prediction, abstain and refer for re-imaging, which is normal
screening workflow rather than failure.

**Why it may matter more than a metric gain.** If a disproportionate share of
the 50 missed patients sit on the worst-quality images, the headline changes
from "misses 50" to "misses N on gradable images and flags the rest for
re-imaging". That is a better clinical claim than another 0.02 F1.

**Papers:** Chopra et al., IJCAI 2026, abstention for DR screening. Conformal
cost-aware deferral under distribution shift, Scientific Reports 2026.

**Tested 2 September 2026 and ruled out.** The premise is false here: misses do
not concentrate on low-quality images. See P1 below for the numbers.

### Candidate C — the routing gate

**The claim.** Aggregate both datasets, then let a gate decide per image how
much to draw on each source, in the encoder rather than the decision layer,
driven by conditional entropy of the labels.

**Papers:** Domain Separation Networks (NeurIPS 2016) for the architecture,
information-theoretic Mixture-of-Experts gating for the routing signal.

**The risk, now measured.** Experiment 1 showed the structure this sits on does
not help. The gate must now beat plain aggregation on its own rather than
improve something that was already working.

**Note:** this is the advisor's own proposal. Deferring or dropping it is his
call, not ours.

---

## 5. The paper this plan is aiming at

A coherent structure exists already, and it does not depend on any single
experiment succeeding:

1. **Diagnosis.** Decompose the gap. Prove the threshold family is exhausted.
   This is original and finished.
2. **Negative results.** Domain separation does not help; the adversarial term
   actively hurts. Reported rather than buried.
3. **Method.** Device-fitted degradation and distillation, compared against a
   training-free Fourier baseline and generic augmentation.
4. **Deployment.** Abstention on ungradable images, so the clinical claim is
   about what the system does with images it cannot read.

The diagnosis motivates the method, the negative results establish that the
numbers can be trusted, the method is the contribution, and abstention is the
practical payoff.

---

## 5b. Directives from the meeting of 3 September 2026

These supersede parts of the queue below.

**P0. Degraded image examples, due this weekend.** The single named deliverable.
"We need to validate this degradation procedure, so maybe it's just random. You
need to do some qualitative evaluation." He is openly sceptical that
hand-crafted degradation resembles anything real, and wants to look at images
rather than metrics. Nothing else on this list matters as much this week.

**P0b. Run far more epochs on the routing experiments.** "You may need more
epoch for this. Did you try to run this onto like a 50 or even 100 epoch?" and
"you can restart your training from the 25th checkpoint. I want you to run more
epoch." His reasoning is that the routing architecture converges more slowly.
25 was chosen for turnaround speed, not for any principled reason.

**Change of method: synthesise the degradation with a generative model.** "If
you use the generative world model, you can synthesise the degraded imagery
more clearly. More generalisable, the diffusion models. You can just use the
generative world model to generate these degraded images." The plan had
hand-crafted physical degradation fitted to measured statistics. He wants a
learned generative alternative, or at minimum a comparison against one.

**The CheXWorld position needs softening.** The claim recorded earlier was that
CheXWorld occupies the world-model direction. He disagreed, and the argument is
reasonable: "we have the same anchor, the same semantics, the object. You have
one single eye, but you have a representation from the mobile and from this.
Two are connected." Our setting has stronger structure than generic domain
variation, and degradation is precisely what manufactures the same eye in two
representations. Cite CheXWorld as related work rather than as a bound.

**A new hypothesis worth testing, and it is his.** "It's not just about the
image quality, but also it's about the label distribution. You have more
balanced data in Setup B." BRSET training is 6.6 percent DR; the aggregated set
is 10.5 percent. The Setup B gain may come from label balance rather than from
modelling the device at all. This is directly testable by subsampling BRSET to
the aggregated positive rate without adding any mBRSET image, and it points the
same way as the P1 finding that misses do not sit on degraded images.

**Venue.** Two were discussed; the transcript garbles both names. The more
premier one has a deadline in about two weeks and was agreed to be too tight.
Target remains late October. Confirm the venue in writing.

## 6. Work queue, in priority order

Priority is (impact multiplied by probability of success) divided by cost.

### P1 — Abstention analysis. DONE 2 September 2026. NEGATIVE.
Measured how the 27 missed DR patients distribute across image quality, using
the joint model's predictions plus artifact annotations and a computed
sharpness measure (Laplacian variance, saved to `results/mbrset_test_quality.csv`).

**Misses do not concentrate on low-quality images.** Miss rate is 16.4% on the
586 artifact-annotated images against 20.0% on the 146 clean ones. Across
sharpness quintiles the miss rate runs 19.4, 16.2, 16.3, 20.0, 13.0 percent
with no monotonic trend, and AUC does not degrade with blur. Abstaining on the
least sharp 30% of images lifts recall on the remainder only from 0.8302 to
0.8426, which is not a trade anyone would accept.

**Consequence, and it matters for P3.** The model fails on images that look
fine. Whatever breaks is not image quality in any measurable sense. That does
not kill the degradation work, since augmentation can aid generalisation
without degradation being the cause of failure, but the mechanism behind P3 is
weaker than assumed and the claim must be stated more carefully.

### P2 — FDA baseline. One day plus one training run.
Implement the Fourier amplitude swap and train on BRSET with it. This is the
control that decides whether Candidate A is worth a week. **Do this before
building the fitted degradation, not after.**

### P3 — Device-fitted degradation and distillation. About one week.
- 3a. Measure blur, illumination falloff and artifact statistics on mBRSET
  versus BRSET. No labels needed. One day, no GPU.
- 3b. Fit the operator parameters so degraded BRSET matches those statistics.
  One day.
- 3c. Train with the distillation loss on the resulting pairs. One day plus
  about 7 hours of GPU.
- Controls: generic FundusAug at published settings, and the FDA baseline
  from P2.

### P4 — The routing gate. About one week, pending advisor input.
Build on the zero-adversarial-weight variant rather than the published
defaults, since the sweep showed those defaults hurt here. Blocked on one
question: conditional entropy of the gate's output distribution, or of the
label given domain-specific features. Implement both if no answer arrives.

### P5 — Curriculum degradation. Ten lines, once 3b exists.
Train on progressively worse images instead of random severity. An ablation,
not a contribution.

### Deliberately not doing
Sharpness-aware minimization. Doubles training time, and recent work shows it
converges to fake flat minima in domain generalization specifically.

---

## 7. Timeline to mid-October

| Week | Work |
|---|---|
| Sep 2 to 8 | P1 abstention analysis, P2 FDA baseline. Both are decision points |
| Sep 9 to 22 | P3 fitted degradation and distillation, with both controls |
| Sep 23 to 29 | P4 routing gate, if the advisor still wants it |
| Sep 30 to Oct 6 | Ablations, second-device check if time allows, all bootstrap intervals |
| Oct 7 to 14 | Write-up |

---

## 8. Standing rules for every experiment

- Report against the same 732-image mBRSET test set, never re-split.
- Paired bootstrap with 4,000 resamples for every comparison. State p-values.
- Report missed-patient counts alongside F1. The count is what survives when
  the F1 difference is inside the noise, which it usually is.
- Every method needs a control that could disprove it. Gulrajani (ICLR 2021)
  is why.
- Negative results go in the deck with the mechanism, not in a drawer.

---

## 9. Open questions for the advisor

1. Conditional entropy of what, precisely? Gate output distribution, or label
   given domain-specific features? They differ substantially in difficulty.
2. Is the gap decomposition a contribution in its own right, or supporting
   analysis for the method?
3. Does the routing gate stay in scope for mid-October, given that Experiment 1
   removed its foundation?
4. Confirm venue and deadline. Working assumption is ISBI, mid-October.
