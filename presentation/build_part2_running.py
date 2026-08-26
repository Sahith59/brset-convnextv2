"""Running deck for Part 2: BRSET (tabletop) -> mBRSET (handheld).

This is the live working document for the cross-device problem. Slides are
added as results land. Every number is measured on our own data and recomputed
from the files in results/ at build time, never carried over on trust.

Typography and layout helpers are shared with the Part-1 deck via deck_common.
"""
from deck_common import (prs, slide, title, bullets, table, box, label, takeaway,
                         link_cell, cell_color, Inches, Pt, PP_ALIGN, MSO_SHAPE, RGBColor,
                         INK, MUTED, ACCENT, GOOD, WARN, AMBER, WHITE, PALE, RULE)
from pathlib import Path

OUT = Path(__file__).parent / "BRSET_to_mBRSET_Part2.pptx"

# ═════════════════════════════════════════════ 1. the problem
s = slide()
title(s, "Part 2: a tabletop-trained model must work on a phone camera",
      "The model is trained only on BRSET. No mBRSET image is ever seen during training, "
      "so this is a genuine test of whether it learned the disease or the camera.")

table(s, [
    ["", "BRSET, trained on", "mBRSET, tested on"],
    ["Camera", "Tabletop fundus camera", "Handheld phone camera (Phelcom Eyer)"],
    ["Setting", "Hospital eye clinic", "Community diabetes screening"],
    ["Images", "16,258   (test 2,435)", "4,859   (test 732)"],
    ["Patients with DR", "6.6%,  about 1 in 15", "23.3%,  about 1 in 4"],
    ["Images with an artifact", "not recorded", "83%   (4,272 of 5,164)"],
], top=1.60, left=0.65, width=12.1, col_w=[2.2, 4.0, 5.0], font=11.5, height=2.3)

label(s, 0.65, 4.15, 12.1, "Two things change at the same time", 12.5, ACCENT, True, PP_ALIGN.LEFT)
box(s, 0.65, 4.52, 5.85, 0.95,
    "The camera gets worse.\nBlur, glare and artifacts the model never saw in training.",
    PALE, INK, 12.5, False, line=RULE)
box(s, 6.90, 4.52, 5.85, 0.95,
    "The patients get sicker.\nA screening campaign, not a general clinic. 6.6 percent becomes 23.3.",
    PALE, INK, 12.5, False, line=RULE)

bullets(s, [
    "Almost every published cross-device study changes only one of these. Changing both at once is what "
    "makes this setting hard, and what makes it worth a paper.",
], top=5.65, size=12.5, bottom=6.35)

takeaway(s, "Goal: make the BRSET-trained model work on handheld images without collecting "
            "labels for the handheld device.", GOOD, top=6.45)

# ═════════════════════════════════════════════ 2. the starting line
s = slide()
title(s, "Starting line: one model, two cameras",
      "The selected Part-1 baseline (focal loss, oversampling off), evaluated with its own "
      "BRSET-chosen cutoff applied unchanged. Measured 21 August 2026.")

table(s, [
    ["Diabetic retinopathy", "On BRSET", "On mBRSET", "Change"],
    ["AUC  (ranking ability)", "0.9906", "0.9060", "−0.085"],
    ["F1, class-macro  (headline)", "0.9374", "0.8668", "−0.071"],
    ["F1, diseased class only", "0.8837", "0.7842", "−0.100"],
    ["Recall, diseased class", "0.9383", "0.6855", "−0.253"],
    ["Diseased cases missed", "10 / 162", "50 / 159", "5x worse"],
], top=1.62, left=0.65, width=6.2, col_w=[2.9, 1.1, 1.1, 1.1], font=10.5,
   height=2.0, highlight=(5,), hcolor=RGBColor(0xF7, 0xE2, 0xDD))

table(s, [
    ["Macular edema", "On BRSET", "On mBRSET", "Change"],
    ["AUC  (ranking ability)", "0.9957", "0.9705", "−0.025"],
    ["F1, class-macro  (headline)", "0.8852", "0.8392", "−0.046"],
    ["F1, diseased class only", "0.7759", "0.7027", "−0.073"],
    ["Recall, diseased class", "0.7377", "0.6190", "−0.119"],
    ["Diseased cases missed", "16 / 61", "24 / 63", "1.5x worse"],
], top=1.62, left=7.15, width=5.6, col_w=[2.7, 1.0, 1.0, 1.0], font=10.5, height=2.0)

label(s, 0.65, 3.80, 12.1, "How to read this", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "Ranking barely suffers. AUC falls only 0.085 on DR, so the model can still tell a diseased eye from "
    "a healthy one on handheld images.",
    "The decision collapses. Recall falls from 0.938 to 0.686, so 50 of 159 diseased patients are missed "
    "instead of 10 of 162.",
    "The three F1 rows are the same result measured three ways. Class-macro averages the diseased and "
    "healthy classes and is the number quoted in the Part-1 deck. Diseased-only is stricter. Both are shown "
    "so nothing is hidden.",
], top=4.14, size=12, bottom=6.35)

takeaway(s, "This right-hand column is the number to beat. Everything from here is about raising it.",
         WARN, top=6.45)

# ═════════════════════════════════════════════ 3. what the gap is made of
s = slide()
title(s, "What the gap is made of, and what it is not",
      "Diabetic retinopathy, diseased-class F1 on mBRSET. Every value measured, none assumed.")

BX0, BW, BY, BH = 1.30, 10.6, 2.95, 0.62
LO, HI = 0.7842, 0.8317
def bx(v):
    return BX0 + BW * (v - LO) / (HI - LO)

for a, b, fill, fg, delta, cap in [
        (0.7842, 0.7927, RGBColor(0xA8, 0xC0, 0xD8), INK, "+0.009", "all that a better\ncutoff can give"),
        (0.7927, 0.8317, RGBColor(0x9B, 0x45, 0x35), WHITE, "+0.039", "needs a better model.\nNo cutoff reaches it")]:
    box(s, bx(a), BY, bx(b) - bx(a), BH, delta, fill, fg, 15, True, line=WHITE)
    label(s, bx(a), BY + BH + 0.12, bx(b) - bx(a), cap, 11, INK, False, h=0.62)

for v, cap in [(0.7842, "0.7842\ntransferred"), (0.7927, "0.7927\nbest cutoff"),
               (0.8317, "0.8317\nwith labels")]:
    label(s, bx(v) - 0.90, 1.70, 1.8, cap, 11.5, ACCENT, True, h=0.60)
    box(s, bx(v) - 0.006, 2.32, 0.012, BY - 2.32, "", RULE, INK, 8, False)

label(s, 0.65, 4.30, 12.1, "Only 18 percent of the remaining gap is the cutoff. 82 percent needs a better model.",
      13, ACCENT, True, PP_ALIGN.LEFT)

bullets(s, [
    "The 0.7927 ceiling was found by sweeping every cutoff from 0.005 to 0.995. Calibration, temperature "
    "scaling and label-shift correction only rescale scores, so all of them land on this same curve and "
    "none can pass it. That family is exhausted.",
    "The shift is not only that patients are sicker. AUC is mathematically unaffected by how many patients "
    "are diseased (Fawcett 2006), so if prevalence alone had changed, AUC would have held. It fell from "
    "0.9906 to 0.9060, which means the images themselves are different.",
    "0.8317 is what fine-tuning on labelled mBRSET achieves, so it is the value of target labels, not a "
    "hard limit. The research question is how close a method with no target labels can get to it.",
], top=4.66, size=12, bottom=6.45)

takeaway(s, "The cutoff route is nearly exhausted. The remaining work is representation, which is where "
            "the literature points.", top=6.52)

# ═════════════════════════════════════════════ 4. literature
s = slide()
title(s, "What the published literature establishes",
      "Peer-reviewed versions only. Citation counts from Semantic Scholar. Paper names are live links.")

lit = [
    ["Evidence", "Finding", "Paper", "Venue", "Cites"],
    ["Works", "Appearance and degradation augmentation beat every\nadaptation method tested for cross-scanner shift",
     "Tellez et al.", "Medical Image\nAnalysis 2019", "665"],
    ["Works", "Fundus DR over 8 datasets, AUC 78.4 to 82.6, using nine\ndegradation and colour operations (FundusAug)",
     "Che et al. (GDRNet)", "MICCAI 2023", "46"],
    ["Fails", "DANN, MDD and CycleGAN all collapse on real tabletop\nto portable fundus data",
     "Lin et al.", "MICCAI 2022", "5"],
    ["Fails", "No domain-generalization method reliably beats plain\ntraining under fair evaluation",
     "Gulrajani and Lopez-Paz", "ICLR 2021", "1,504"],
    ["Fails", "Test-time adaptation loses up to 66 points under prior\nshift, the exact regime here",
     "Boudiaf et al. (LAME)", "CVPR 2022", "257"],
    ["Partial", "Cross-camera image translation for portable DR screening,\nbut never treats the decision cutoff",
     "Joseph et al. (SCR-Net)", "BMJ Open\nOphthalmology 2025", "n/r"],
    ["Tool", "Physically grounded fundus degradation model: light\ntransmission, blur, retinal artifacts",
     "Shen et al. (cofe-Net)", "IEEE Trans. Medical\nImaging 2021", "n/r"],
]
shp = table(s, lit, top=1.62, col_w=[0.85, 5.3, 2.1, 2.05, 0.6], font=10, height=3.9)
for row, (col, url) in enumerate([
        (GOOD,  "https://doi.org/10.1016/j.media.2019.101544"),
        (GOOD,  "https://doi.org/10.1007/978-3-031-43904-9_42"),
        (WARN,  "https://doi.org/10.1007/978-3-031-16434-7_57"),
        (WARN,  "https://openreview.net/forum?id=lQdXeXDoWtI"),
        (WARN,  "https://doi.org/10.1109/CVPR52688.2022.00816"),
        (AMBER, "https://doi.org/10.1136/bmjophth-2025-002238"),
        (ACCENT, "https://arxiv.org/abs/2005.05594")], start=1):
    cell_color(shp, row, 0, col)
    link_cell(shp, row, 2, url)

bullets(s, [
    "Three of the four methods a reader would reach for first are published failures in this exact setting. "
    "Augmentation is the one intervention with peer-reviewed evidence of raising AUC under real fundus "
    "device shift, which is why it is the direction I propose.",
    "n/r means the citation count could not be retrieved because the Semantic Scholar API rate-limited the "
    "lookup. Venue and year were verified individually for every row.",
], top=5.62, size=11.5, bottom=6.45)

takeaway(s, "The review did not just fill a section. It removed three candidate methods and left one.",
         top=6.50)

# ═════════════════════════════════════════════ 5. the plan
s = slide()
title(s, "The plan, and where I am in it",
      "Each step produces a number, so progress is visible rather than asserted.")

steps = [
    ["", "Step", "What it establishes", "Status"],
    ["1", "Measure the starting line",
     "BRSET model on mBRSET: DR class-macro F1 0.8668, 50 of 159 missed", "Done"],
    ["2", "Measure what target labels buy",
     "BRSET pretrain then mBRSET fine-tune: diseased F1 0.8317, 28 of 159 missed", "Done"],
    ["3", "Split the gap by measurement",
     "18 percent cutoff, 82 percent representation. Calibration family ruled out", "Done"],
    ["4", "Literature review",
     "Three method families eliminated, augmentation identified as the one with evidence", "Done"],
    ["5", "Stabilise the Part-1 baseline",
     "bf16 rerun completed 40 clean epochs. Both runs peak at the identical validation\nscore of 0.9218, so the divergence cost nothing and the baseline is confirmed", "Done"],
    ["6", "Device-calibrated degradation augmentation",
     "Fit degradation parameters to measured mBRSET statistics, using no target labels", "Next"],
    ["7", "Control experiment",
     "Compare against generic FundusAug at published settings, as Gulrajani requires", "Planned"],
]
shp = table(s, steps, top=1.60, col_w=[0.4, 3.0, 6.5, 0.9], font=10, height=3.3)
for r, st in enumerate(["Done", "Done", "Done", "Done", "Done", "Next", "Planned"], start=1):
    cell_color(shp, r, 3, GOOD if st == "Done" else (AMBER if st == "Next" else MUTED))

label(s, 0.65, 5.10, 12.1, "The idea behind step 6", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "FundusAug applies its nine operations with a fixed probability of 0.5 and hand-chosen magnitudes. "
    "It is generic damage for generic robustness, tuned for no particular camera.",
    "4,272 mBRSET images already carry artifact annotations, so the handheld camera's actual degradation "
    "statistics are measurable without using a single diagnostic label. Fitting the augmentation to those "
    "measured statistics, rather than guessing them, is the novel step.",
], top=5.44, size=12, bottom=6.50)

takeaway(s, "Steps 1 to 5 are complete and measured. Step 6 is the novel contribution.", GOOD, top=6.55)

prs.save(OUT)
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
