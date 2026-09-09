"""Status deck: everything measured on BRSET to mBRSET as of 9 September 2026.

Written to be read by someone who has not followed the work. It reports five
negative results and two positive findings, in that order, because that is the
honest weight of the evidence.
"""
from deck_common import (prs, slide, title, bullets, table, box, label, takeaway,
                         arrow, link_cell, cell_color, Inches, Pt, PP_ALIGN, MSO_SHAPE,
                         RGBColor, INK, MUTED, ACCENT, GOOD, WARN, AMBER, WHITE, PALE, RULE)
from pathlib import Path

OUT = Path(__file__).parent / "BRSET_mBRSET_Status.pptx"
R = Path(__file__).parent.parent / "results"
SOFT = RGBColor(0xE8, 0xED, 0xF3)
GREENF = RGBColor(0xDF, 0xEA, 0xDF)
REDF = RGBColor(0xF7, 0xE2, 0xDD)
AMBERF = RGBColor(0xF8, 0xEE, 0xD8)

# ═══════════════════════════ 1. the one-page summary
s = slide()
title(s, "Where the work stands",
      "Five interventions tested against a measured noise floor. Two findings survive.")

box(s, 0.65, 1.50, 12.1, 0.72,
    "Nothing tested so far improves cross-device transfer beyond run-to-run variance. "
    "The negative results are rigorous; two separate findings stand on their own.",
    AMBERF, INK, 13, True, line=AMBER)

label(s, 0.65, 2.42, 12.1, "Tested and negative", 12.5, WARN, True, PP_ALIGN.LEFT)
table(s, [
    ["Intervention", "Result", "Against noise floor"],
    ["Domain Separation, shared plus private encoders", "inside noise", "floor 0.0199 DR F1"],
    ["Entropy-gated routing, both formulations", "best +0.0087", "floor 0.0199"],
    ["Abstention on low-quality images", "premise false", "misses are not on bad images"],
    ["Aspect-ratio correction, 3 seeds", "+0.0020", "floor 0.0199"],
], top=2.76, left=0.65, width=12.1, col_w=[5.6, 2.6, 3.9], font=10.5, height=1.5)

label(s, 0.65, 4.44, 12.1, "Findings that stand", 12.5, GOOD, True, PP_ALIGN.LEFT)
box(s, 0.65, 4.78, 5.95, 1.30,
    "The threshold family is provably exhausted.\n\nEvery cutoff from 0.005 to 0.995 was swept. "
    "The best attainable F1 is 0.7927. Calibration, temperature scaling and label-shift "
    "correction cannot exceed it.",
    GREENF, INK, 11, False, line=GOOD)
box(s, 6.80, 4.78, 5.95, 1.30,
    "Published augmentation settings are worse than none.\n\nDistance to the measured mBRSET target: "
    "BRSET 5.13, GDRNet published 7.75, fitted 1.36. Their defaults move away on four of five statistics.",
    GREENF, INK, 11, False, line=GOOD)

takeaway(s, "The defensible paper is the diagnosis plus these two findings, not a performance win.",
         top=6.24)

# ═══════════════════════════ 2. the problem
s = slide()
title(s, "The problem, and the starting line",
      "One model trained on BRSET alone, evaluated on both datasets. Same weights, same cutoff.")

box(s, 0.85, 1.90, 3.6, 0.95, "BRSET\ntabletop camera, hospital", SOFT, INK, 11.5, False, line=RULE)
box(s, 0.85, 3.25, 3.6, 0.95, "mBRSET\nhandheld phone, screening", SOFT, INK, 11.5, False, line=RULE)
arrow(s, 4.62, 2.30, 0.58); arrow(s, 4.62, 3.65, 0.58)
box(s, 5.38, 1.90, 2.6, 2.30, "ConvNeXt V2 Large\n\ntrained on\nBRSET only", ACCENT, WHITE, 12, True)
arrow(s, 8.16, 2.30, 0.58); arrow(s, 8.16, 3.65, 0.58)
box(s, 8.92, 1.90, 3.85, 0.95, "10 of 162 missed", GREENF, INK, 13, True, line=GOOD)
box(s, 8.92, 3.25, 3.85, 0.95, "50 of 159 missed", REDF, INK, 13, True, line=WARN)

bullets(s, [
    "Ranking barely suffers: AUC falls only 0.9906 to 0.9060. The decision collapses: recall drops "
    "0.938 to 0.686. The model still sees the disease and stops acting on it.",
    "Two things change together. The camera differs, and the population differs: 6.6 percent with "
    "diabetic retinopathy becomes 23.3 percent.",
    "Aggregating both training sets, which is permitted since only the test split is protected, "
    "reaches 27 of 159 missed. That is the model every later experiment has to beat.",
], top=4.44, size=12, bottom=6.35)

takeaway(s, "The gap: at most 28 percent is the decision cutoff, at least 72 percent is the representation.",
         top=6.40)

# ═══════════════════════════ 3. the noise floor
s = slide()
title(s, "The measurement that reframed everything",
      "Experiment 0 repeated three times with different random seeds and nothing else changed.")

table(s, [
    ["Run", "DR AUC", "DR F1", "DR missed", "ME F1"],
    ["seed 0", "0.9377", "0.8250", "27 / 159", "0.8649"],
    ["seed 1", "0.9519", "0.8449", "31 / 159", "0.8224"],
    ["seed 2", "0.9492", "0.8396", "36 / 159", "0.8440"],
    ["spread from seed alone", "0.0142", "0.0199", "9 patients", "0.0424"],
], top=1.62, left=0.65, width=12.1, col_w=[4.4, 1.9, 1.9, 2.0, 1.9], font=11.5,
   height=1.85, highlight=(4,), hcolor=AMBERF)

label(s, 0.65, 3.70, 12.1,
      "The largest architectural gain ever recorded in this project was +0.0202 DR F1. "
      "The seed spread is 0.0199. They are the same size.",
      13, WARN, True, PP_ALIGN.LEFT, h=0.56)

label(s, 0.65, 4.34, 12.1, "What this retracts", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "Every architectural conclusion drawn before this control is unsupported: that domain separation "
    "is a wash, that the adversarial term is harmful, that the routing gate improves F1, and every "
    "comparison of missed-patient counts.",
    "A compounding error worth stating. Seed 0, used as the baseline throughout, scored the LOWEST "
    "of the three. Every architecture was measured against an unlucky draw and looked better than it "
    "was. Against the 3-seed mean of 0.8365, the best gain shrinks from +0.0202 to +0.0087.",
    "Standing rule now: three seeds before comparing architectures, never after. This triples the "
    "cost of every experiment, which should shape what is worth running at all.",
], top=4.68, size=11.5, bottom=6.45)

takeaway(s, "Most published work in this area reports gains of exactly this size without a seed control.",
         top=6.50)

# ═══════════════════════════ 4. finding 1
s = slide()
title(s, "Finding 1: an entire method family is provably exhausted",
      "Diabetic retinopathy, diseased-class F1 on mBRSET.")

BX0, BW, BY, BH = 1.55, 10.2, 2.75, 0.62
LO, HI = 0.7842, 0.8355
def bx(v): return BX0 + BW * (v - LO) / (HI - LO)
for v, val, who in [(0.7842, "0.7842", "transferred as is"),
                    (0.7927, "0.7927", "best possible cutoff"),
                    (0.8355, "0.8355", "trained with mBRSET labels")]:
    label(s, bx(v) - 0.80, 1.62, 1.6, val, 12.5, ACCENT, True, h=0.28)
    label(s, bx(v) - 0.80, 1.90, 1.6, who, 9, MUTED, False, h=0.46)
    box(s, bx(v) - 0.006, 2.40, 0.012, BY - 2.40, "", RULE, INK, 8, False)
box(s, bx(0.7842), BY, bx(0.7927) - bx(0.7842), BH, "+0.009",
    RGBColor(0xA8, 0xC0, 0xD8), INK, 13, True, line=WHITE)
box(s, bx(0.7927), BY, bx(0.8355) - bx(0.7927), BH, "+0.043",
    RGBColor(0x9B, 0x45, 0x35), WHITE, 13, True, line=WHITE)

label(s, 0.65, 3.60, 12.1, "The argument", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "Every cutoff from 0.005 to 0.995 was swept. The best attainable F1 is 0.7927.",
    "Calibration, temperature scaling and label-shift correction are monotone transformations. "
    "They rescale scores but never reorder them, so each selects a point on that same curve. "
    "None of them can exceed 0.7927.",
    "That rules out the whole family by measurement, before implementing any of it. Checked against "
    "six different target-trained endpoints spanning F1 0.8146 to 0.8355; the cutoff share never "
    "exceeds 28 percent, so the conclusion does not depend on which endpoint is chosen.",
], top=3.94, size=12, bottom=6.30)

takeaway(s, "Most cross-device papers try methods one at a time. This says in advance which cannot win.",
         GOOD, top=6.36)

# ═══════════════════════════ 5. finding 2
s = slide()
title(s, "Finding 2: published augmentation settings are worse than no augmentation",
      "Normalised distance from degraded BRSET to the appearance statistics measured on real mBRSET. "
      "Lower is closer.")

for x, lab, val, col, fg in [(1.60, "BRSET untouched", "5.13", SOFT, INK),
                             (5.30, "GDRNet published", "7.75", REDF, INK),
                             (9.00, "Fitted to measured\nstatistics", "1.36", GREENF, INK)]:
    box(s, x, 1.72, 3.0, 1.10, val, col, fg, 26, True, line=RULE)
    label(s, x, 2.90, 3.0, lab, 12, INK, True, h=0.50)

label(s, 0.65, 3.56, 12.1,
      "GDRNet's published settings score worse than leaving the image alone.",
      13.5, WARN, True, PP_ALIGN.LEFT)

table(s, [
    ["statistic", "BRSET", "GDRNet published", "fitted", "mBRSET target", "GDRNet direction"],
    ["sharpness", "0.00033", "0.00012", "0.00091", "0.00114", "away"],
    ["brightness", "0.38506", "0.47671", "0.42544", "0.35258", "away"],
    ["contrast", "0.08417", "0.10402", "0.12905", "0.14183", "closer"],
    ["falloff", "0.10154", "0.07686", "0.12775", "0.10950", "away"],
    ["saturation", "0.78221", "0.83108", "0.68934", "0.76537", "away"],
], top=3.96, left=0.65, width=12.1, col_w=[2.2, 1.8, 2.6, 1.8, 2.2, 2.4], font=10, height=1.85)

bullets(s, [
    "The reason is a wrong assumption, not bad tuning. GDRNet and cofe-Net both treat degradation as "
    "blur. Measured here, mBRSET carries five to seven times MORE high-frequency energy than BRSET, "
    "which is what a small phone sensor and in-camera sharpening produce. Blurring moves away from "
    "the target on the largest axis of all.",
], top=5.94, size=11.5, bottom=6.60)

# ═══════════════════════════ 6. the images
s = slide()
title(s, "The qualitative check, and what it shows",
      "Requested directly: validate the degradation procedure by looking at it rather than at metrics.")
s.shapes.add_picture(str(R / "degradation_examples_3row.png"), Inches(0.9), Inches(1.44),
                     height=Inches(4.30))
label(s, 0.65, 5.88, 12.1,
      "Reading the figure honestly: column 3 is quantitatively much closer to the target than column 2, "
      "but it still does not look like column 4.",
      12.5, WARN, True, PP_ALIGN.LEFT, h=0.34)
bullets(s, [
    "Two differences dominate visually and are invisible to the five statistics being fitted: framing, "
    "since mBRSET is a circle filling the frame while BRSET is wider with black bars, and hue, since "
    "mBRSET is pink-red where BRSET is orange. Statistics computed inside the fundus mask cannot see "
    "either. Matching numbers is not the same as matching appearance.",
], top=6.26, size=11, bottom=7.10)

# ═══════════════════════════ 7. what next
s = slide()
title(s, "What is left, and the honest assessment")

label(s, 0.65, 1.50, 12.1, "Untested", 12.5, ACCENT, True, PP_ALIGN.LEFT)
table(s, [
    ["", "Work", "Why it is still open"],
    ["1", "Train with the fitted degradation (Experiment 3c)",
     "The method itself has never been run. The appearance fit is quantitatively strong"],
    ["2", "Label-balance hypothesis",
     "BRSET train is 6.6 percent positive, aggregated is 10.5. The gain may be balance, not the device"],
    ["3", "Generative or diffusion degradation",
     "The qualitative check is now the evidence for why hand-crafted statistics are insufficient"],
    ["4", "Routing at 50 to 100 epochs",
     "Asked for directly. The seed control makes it unpromising, but it would close the question"],
], top=1.84, left=0.65, width=12.1, col_w=[0.4, 4.4, 7.3], font=10, height=2.3)

label(s, 0.65, 4.34, 12.1, "The assessment", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "Five interventions have been tested and none exceeds run-to-run variance. That is a hard "
    "position, and it should be stated plainly rather than dressed up.",
    "What makes it defensible is the methodology. Three seeds, paired bootstrap with 4,000 resamples, "
    "a measured noise floor, and a proven ceiling. The claim that no intervention beats seed variance "
    "on this benchmark is only possible because the seed control was run.",
    "The remaining constraint may be the data. mBRSET has 3,402 training images and 159 test "
    "positives, and the seed alone moves missed patients by 9. Part of this gap may not be resolvable "
    "at this dataset size, and that is worth saying out loud.",
], top=4.68, size=11.5, bottom=6.50)

takeaway(s, "Better to bring a clear-eyed assessment now than a hopeful one in October.", top=6.54)

prs.save(OUT)
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
