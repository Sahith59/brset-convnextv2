# Bugs, wrong turns, and what caught them

A running log for this project. Every entry is something that was wrong at some
point, how it was found, and what changed. Kept because the same classes of
mistake recur, and because a wrong number that goes unrecorded eventually gets
presented to somebody.

Three categories: **silent bugs** that produced plausible-looking wrong output,
**wrong hypotheses** that were tested and disproved, and **reporting errors**
where the code was right but the write-up was not.

---

## SILENT BUGS

### 1. fp16 overflow killed a training run without an error message
**Symptom.** The selected Part-1 focal run reported normal per-batch losses
while its running average had already become NaN. Weights went non-finite from
epoch 28 to 40. Its best checkpoint is epoch 7, so it trained usefully for 7 of
40 epochs and nothing in the logs said so.

**Cause.** ConvNeXt V2's GRN takes an L2 norm over a 512x512 feature map, which
can exceed the fp16 maximum of 65504. The asymmetric-focal runs survived only
because that loss clamps before taking a log; the symmetric focal loss does not.

**Fix.** `--amp_dtype bf16` (same exponent range as fp32, cannot overflow there)
plus a guard that drops any accumulation group whose loss is non-finite and
prints a warning instead of silently poisoning the weights.

**Cost.** Would have been 9 wasted hours had a clean rerun not confirmed the
result. It reached the identical peak validation score, 0.9218.

**Lesson.** A per-batch metric looking healthy does not mean training is
healthy. Log the running average too, and guard the accumulation.

### 2. Warmup schedule tuned for a large dataset destroyed a small one
**Symptom.** The mBRSET from-scratch run reached validation AUC 0.961 at epoch
2, then crashed to 0.487, below chance, at epoch 3. It crawled back to 0.91 over
37 epochs and never recovered its peak.

**Cause.** Warmup is specified in EPOCHS, not steps. Three epochs is 4,263
optimizer steps on BRSET but only 636 on the three-times-smaller mBRSET, so the
learning rate reached its peak of 5e-5 roughly seven times faster.

**Fix.** lr 1.5e-5 with warmup 8 for that dataset. No collapse.

**Lesson.** Any schedule expressed in epochs silently changes meaning when the
dataset size changes. Convert to steps before reusing a recipe.

### 3. A temperature wrong by 30x made a mechanism look uninformative
**Symptom.** The conditional-entropy routing gate held alpha at 0.501 for all
25 epochs. The obvious reading was that the routing signal does not discriminate
and the idea does not work.

**Cause.** alpha = sigmoid(difference / T) with T defaulted to 1.0. The actual
difference has standard deviation 0.0327 and spans -0.042 to +0.047, so dividing
by 1.0 squashed every sample into alpha between 0.455 and 0.554. The signal was
present the whole time; the scaling erased it.

**How it was caught.** Instead of sweeping T blindly at 7 GPU-hours per setting,
a 44-second diagnostic measured the actual distribution of the difference on the
validation set using the already-trained checkpoint, then reported what alpha
would look like at each candidate temperature. At T=0.03, 63.6 percent of
samples route decisively.

**Lesson.** When a hyperparameter divides a quantity, measure that quantity's
scale before choosing the divisor. And when a mechanism looks dead, check
whether it was ever alive before concluding anything about the idea.

---

## WRONG HYPOTHESES, TESTED AND DISPROVED

### 4. "Checkpoint selection caused the Domain Separation deficit"
**The claim.** Experiment 1 improved AUC while losing recall. Checkpoints were
selected on the average of macro AUC and macro F1, which rewards exactly what
that architecture improves, so the selection criterion looked like the culprit.

**The test.** Reran selecting on macro F1 alone.

**The result.** It chose the same epoch 16 and produced identical numbers. The
hypothesis was wrong.

**Kept on the slide** rather than dropped, because a disproved hypothesis of
your own is evidence the numbers can be trusted.

### 5. "Asymmetric focal loss fails because it is worse on precision AND recall"
**The claim.** Since it lost on both, the deficit could not be recovered by
retuning the cutoff.

**Why it was shaky.** Asymmetric focal actually has slightly HIGHER DR AUC
(0.9916 against 0.9906), so better ranking does exist in principle and the
argument did not follow.

**The proper test.** An oracle sweep over every threshold. Asymmetric focal
tops out at 0.9292 against focal's 0.9374, so the conclusion survives, but on
completely different reasoning.

**Lesson.** A correct conclusion reached by faulty reasoning is still a hole
somebody will find.

### 6. "The missed cases will concentrate on degraded images"
**The claim.** 83 percent of mBRSET images carry an artifact, so abstaining on
the worst should remove a disproportionate share of the misses.

**The test.** Joined predictions to artifact annotations and a computed
sharpness measure.

**The result.** False. Miss rate is 16.4 percent on artifact images against 20.0
percent on clean ones, and across sharpness quintiles it runs 19.4, 16.2, 16.3,
20.0, 13.0 with no trend. Abstaining on the least sharp 30 percent lifts recall
only from 0.8302 to 0.8426.

**Consequence.** Ruled out abstention as a contribution, and weakened the stated
rationale for the degradation work, since the model fails on images that look
fine.

---

## REPORTING ERRORS

### 7. Numbers in a deck that did not match the data
- Discordant-eye rate reported as 14.5 percent; recomputation gave 11.8 percent
  of patients with both eyes imaged.
- Threshold ceiling reported as 0.765; the exhaustive sweep gives 0.7628.
- Segment deltas reported as +0.048 and +0.050; correct values +0.046 and +0.052.

**Cause.** Carried forward from an earlier draft instead of recomputed.

**Fix.** Every figure in the decks is now recomputed from `results/` at build
time. Nothing is transcribed.

### 8. Two experimental setups presented as one
**Symptom.** Slide 1 said "no mBRSET image is ever seen in training" next to
conclusions drawn from a run that trains on 3,402 mBRSET images.

**Cause.** Both statements were true, of different experiments, and the deck
never distinguished them.

**Fix.** Setup A (zero-shot) and Setup B (aggregated) are now labelled
explicitly on every slide that uses either.

**Lesson.** When the same test set appears under two training regimes, label the
regime every single time.

---

## STANDING PRACTICES THAT CAME OUT OF THIS

- Recompute every reported number from the result files at build time.
- Smoke-test architecture changes on a small model before a 7-hour run. The
  Domain Separation gather and the routing gate were both checked this way.
- Measure a quantity's scale before choosing a hyperparameter that divides it.
- Report missed-patient counts alongside F1. With 159 positives, F1 differences
  are usually inside the noise and the count is what survives.
- When a result is surprising, check for a bug before writing it up as a finding.
