"""Findings deck for the BRSET to mBRSET work.

Written for a reader seeing this cold. The two training setups are kept
explicitly separate throughout, because conflating them is the easiest way to
misread every number in the project.
"""
from deck_common import (prs, slide, title, bullets, table, box, label, takeaway,
                         arrow, link_cell, cell_color, Inches, Pt, PP_ALIGN, MSO_SHAPE,
                         RGBColor, INK, MUTED, ACCENT, GOOD, WARN, AMBER, WHITE, PALE, RULE)
from pathlib import Path

OUT = Path(__file__).parent / "BRSET_mBRSET_Findings.pptx"
SOFT = RGBColor(0xE8, 0xED, 0xF3)
GREENF = RGBColor(0xDF, 0xEA, 0xDF)
REDF = RGBColor(0xF7, 0xE2, 0xDD)
AMBERF = RGBColor(0xF8, 0xEE, 0xD8)

# ══════════════════════════ 1. the starting point (SETUP A)
s = slide()
title(s, "Setup A, the starting point: nothing from the target camera",
      "Trained on BRSET alone. This is the zero-shot transfer test, and it is the number every later "
      "experiment has to improve on.")

box(s, 0.85, 1.86, 3.6, 0.95, "BRSET  train\n11,372 tabletop images", SOFT, INK, 11.5, False, line=RULE)
box(s, 0.85, 3.16, 3.6, 0.95, "mBRSET  test\n732 handheld images", SOFT, INK, 11.5, False, line=RULE)
arrow(s, 4.62, 2.24, 0.58); arrow(s, 4.62, 3.54, 0.58)
box(s, 5.38, 1.86, 2.6, 2.25, "ConvNeXt V2 Large\n\ntrained on\nBRSET only\n\nsame weights\nsame cutoff",
    ACCENT, WHITE, 11.5, True)
arrow(s, 8.16, 2.24, 0.58); arrow(s, 8.16, 3.54, 0.58)
box(s, 8.92, 1.86, 3.85, 0.95, "on BRSET, its own data\n10 of 162 missed", GREENF, INK, 12, True, line=GOOD)
box(s, 8.92, 3.16, 3.85, 0.95, "on mBRSET, unseen camera\n50 of 159 missed", REDF, INK, 12, True, line=WARN)

label(s, 0.65, 4.34, 12.1,
      "Ranking barely suffers: AUC falls 0.9906 to 0.9060, so the model still separates diseased from "
      "healthy on phone images. The decision is what collapses, with recall dropping 0.938 to 0.686.",
      12, INK, False, PP_ALIGN.LEFT, h=0.58)

label(s, 0.65, 5.02, 12.1, "Two things change at the same time, which is what makes this hard",
      12.5, ACCENT, True, PP_ALIGN.LEFT)
box(s, 0.65, 5.36, 5.9, 0.58, "The camera degrades.\n83 percent of mBRSET images carry an artifact.",
    PALE, INK, 11.5, False, line=RULE)
box(s, 6.85, 5.36, 5.9, 0.58, "The population changes.\n6.6 percent with DR becomes 23.3 percent.",
    PALE, INK, 11.5, False, line=RULE)

box(s, 0.65, 6.16, 12.1, 0.66,
    "Setup A is the reference point, not the current design. The project has since moved to Setup B, "
    "which allows mBRSET images in training. That is on the next slide.",
    AMBERF, INK, 12, True, line=AMBER)

# ══════════════════════════ 2. the two setups + experiment 0 record
s = slide()
title(s, "Setup A and Setup B, kept separate",
      "Training may draw on any available data. Only the 732-image mBRSET test split is protected, "
      "and it is identical across every experiment.")

shp = table(s, [
    ["", "Setup A  (zero-shot)", "Setup B  (aggregated training)"],
    ["Training images", "BRSET only, 11,372", "BRSET 11,372 + mBRSET 3,402 = 14,774"],
    ["mBRSET labels used in training", "No", "Yes"],
    ["Test images", "mBRSET, 732", "mBRSET, 732  (identical images)"],
    ["DR diseased-class F1", "0.7842", "0.8250"],
    ["DR patients missed", "50 of 159", "27 of 159"],
    ["ME diseased-class F1", "0.7027", "0.8649"],
], top=1.60, left=0.65, width=12.1, col_w=[3.4, 3.6, 5.1], font=11, height=2.5,
   highlight=(5,), hcolor=GREENF)

label(s, 0.65, 4.24, 12.1, "Experiment 0: the aggregated run, in full", 12.5, ACCENT, True, PP_ALIGN.LEFT)
table(s, [
    ["Data", "Configuration", "Health and result"],
    ["Train 14,774 (BRSET 77%, mBRSET 23%)\nVal 725 and test 732 are mBRSET ONLY\nDR positives 10.5% overall",
     "ConvNeXt V2 Large, 512 px\nAdamW lr 3e-5, 25 epochs, 3 warmup\nfocal gamma 2.0, no oversampling\nEMA 0.999, 4-way flip TTA, bf16",
     "25 of 25 epochs, zero non-finite batches\nbest epoch 10, 6.64 hours on one A40\nDR AUC 0.9377, ME AUC 0.9936\nME precision 1.0000 (no false positives)"],
], top=4.58, left=0.65, width=12.1, col_w=[3.9, 4.0, 4.2], font=9.5, height=1.15)

bullets(s, [
    "Honest reading: aggregation misses the fewest DR patients of any model and reaches perfect ME "
    "precision, but paired bootstrap puts it in a statistical tie with fine-tuning on DR (p = 0.47). "
    "It is a tie that catches more patients, not a win.",
], top=5.92, size=11.5, bottom=6.60)

# ══════════════════════════ 3. where the gap lives
s = slide()
title(s, "Where the remaining gap lives",
      "Diabetic retinopathy, diseased-class F1 on the mBRSET test set. Reading left to right, "
      "each step is a different kind of fix.")

BX0, BW, BY, BH = 1.55, 10.2, 2.92, 0.66
LO, HI = 0.7842, 0.8355
def bx(v):
    return BX0 + BW * (v - LO) / (HI - LO)

for v, val, who in [(0.7842, "0.7842", "Setup A,\nas transferred"),
                    (0.7927, "0.7927", "best possible\ncutoff"),
                    (0.8355, "0.8355", "best model trained\nwith mBRSET labels")]:
    label(s, bx(v) - 0.80, 1.60, 1.6, val, 12.5, ACCENT, True, h=0.28)
    label(s, bx(v) - 0.80, 1.88, 1.6, who, 9.5, MUTED, False, h=0.46)
    box(s, bx(v) - 0.006, 2.38, 0.012, BY - 2.38, "", RULE, INK, 8, False)

box(s, bx(0.7842), BY, bx(0.7927) - bx(0.7842), BH, "+0.009",
    RGBColor(0xA8, 0xC0, 0xD8), INK, 13, True, line=WHITE)
box(s, bx(0.7927), BY, bx(0.8355) - bx(0.7927), BH, "+0.043",
    RGBColor(0x9B, 0x45, 0x35), WHITE, 13, True, line=WHITE)
label(s, bx(0.7842), BY + BH + 0.08, bx(0.7927) - bx(0.7842),
      "all a better\ncutoff can give", 10, INK, False, h=0.46)
label(s, bx(0.7927), BY + BH + 0.08, bx(0.8355) - bx(0.7927),
      "no cutoff reaches this. It needs a better model", 10.5, INK, False, h=0.30)

label(s, 0.65, 4.18, 12.1,
      "At most 28 percent of the gap is the decision cutoff. At least 72 percent is the representation.",
      13.5, ACCENT, True, PP_ALIGN.LEFT)
table(s, [
    ["Endpoint used as the target", "Its F1", "Share that is cutoff", "Share that is representation"],
    ["mBRSET from scratch, old recipe", "0.8146", "28%", "72%"],
    ["mBRSET from scratch, stable schedule", "0.8239", "22%", "78%"],
    ["BRSET and mBRSET trained jointly (Setup B)", "0.8250", "21%", "79%"],
    ["BRSET pretrain then mBRSET finetune", "0.8355", "17%", "83%"],
], top=4.54, left=0.65, width=12.1, col_w=[5.4, 1.5, 2.5, 2.7], font=10, height=1.45)

bullets(s, [
    "The 0.7927 ceiling came from sweeping every cutoff from 0.005 to 0.995. Calibration, temperature "
    "scaling and label-shift correction only rescale scores, so none of them can pass it. That whole family "
    "is ruled out by measurement rather than by trial.",
], top=6.10, size=11.5, bottom=6.70)

# ══════════════════════════ 4. the obstacle and the answer
s = slide()
title(s, "The obstacle, and the proposed way around it",
      "The camera-adaptation papers all assume paired data, the same patient on both devices. "
      "BRSET and mBRSET share no patients.")

ROWS = [
    (1.72, "What the published\nmethods need",
     "Patient A\non the tabletop camera", "Patient A\non the handheld camera", "PAIRED", GOOD, GREENF),
    (2.94, "What this project\nactually has",
     "BRSET\n8,524 patients", "mBRSET\n1,291 different patients", "NOT PAIRED", WARN, REDF),
    (4.16, "The proposed answer:\ndevice-fitted degradation",
     "One BRSET image", "the same image, degraded to\nmatch the handheld camera", "PAIRED", GOOD, GREENF),
]
for y, lab, a_txt, b_txt, verdict, col, fill in ROWS:
    label(s, 0.65, y + 0.06, 2.6, lab, 11, ACCENT, True, PP_ALIGN.LEFT, h=0.62)
    box(s, 3.40, y, 3.0, 0.82, a_txt, SOFT, INK, 10.5, False, line=RULE)
    arrow(s, 6.52, y + 0.31, 0.48)
    box(s, 7.12, y, 3.1, 0.82, b_txt, SOFT, INK, 10.5, False, line=RULE)
    box(s, 10.40, y, 2.35, 0.82, verdict, fill, INK, 12, True, line=col)

label(s, 0.65, 5.16, 12.1,
      "Degradation augmentation manufactures the pairs that do not exist naturally",
      13, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "A clean image and its degraded twin are the same image, so they pair perfectly by construction. "
    "Creating them costs nothing.",
    "The method: fit the degradation to the measured statistics of the real handheld camera, using the "
    "physically grounded fundus degradation model of Shen et al., cofe-Net, IEEE TMI 2021. That fitting is "
    "what makes the synthetic pairs realistic enough to learn from, and it is the novel step.",
    "This matters beyond answering the objection. Every method that requires pairs was closed to this "
    "project and is now open, including clean-to-degraded feature distillation.",
], top=5.50, size=11.5, bottom=6.70)

# ══════════════════════════ 5. the method chain
s = slide()
title(s, "The proposed method, and where each piece comes from",
      "Four steps. Every step is published work; the contribution is the chain and the fitting in step 2.")

STEPS = [
    (0.65, "1", "Measure", "How the handheld camera really\ndegrades images, from 4,272\nartifact-annotated mBRSET images",
     "cofe-Net\nIEEE TMI 2021", "https://arxiv.org/abs/2005.05594"),
    (3.75, "2", "Degrade", "Apply that measured degradation\nto BRSET, creating clean and\ndegraded pairs",
     "GDRNet\nMICCAI 2023", "https://arxiv.org/abs/2307.04378"),
    (6.85, "3", "Distil", "Clean features teach the degraded\nones, weighted by how hard each\nregion is to recover",
     "Self-Feature Distillation\nECCV 2022", "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840544.pdf"),
    (9.95, "4", "Separate", "Split what both cameras share\nfrom what is specific to each,\ninside the encoder",
     "Domain Separation Nets\nNeurIPS 2016", "https://arxiv.org/abs/1608.06019"),
]
for x, n, head, body, paper, _ in STEPS:
    box(s, x, 1.66, 2.7, 0.44, f"{n}.  {head}", ACCENT, WHITE, 12.5, True)
    box(s, x, 2.16, 2.7, 1.18, body, SOFT, INK, 10.5, False, line=RULE)
    box(s, x, 3.42, 2.7, 0.56, paper, PALE, INK, 10, True, line=RULE)
for x in (3.40, 6.50, 9.60):
    arrow(s, x, 2.58, 0.28)

label(s, 0.65, 4.24, 12.1,
      "Steps 1 and 2 use no target diagnosis labels, only image appearance and existing artifact annotations.",
      12.5, ACCENT, True, PP_ALIGN.LEFT)

label(s, 0.65, 4.66, 12.1, "What is genuinely new, and what would disprove it", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "Step 2 is the keystone. GDRNet fires its nine operations at a fixed 0.5 probability with hand-chosen "
    "magnitudes, tuned for no particular camera. Fitting them to a measured target device is the new part, "
    "and it is what makes the pairs realistic enough for step 3 to work.",
    "The control: generic FundusAug at its published settings. If fitting to the device does not beat "
    "guessing, the contribution fails and I will report that. Gulrajani and Lopez-Paz, ICLR 2021, is the "
    "reason that control is not optional.",
], top=5.00, size=11.5, bottom=6.40)

takeaway(s, "Deployable on a new camera without a labelling campaign, which is the practical case.",
         GOOD, top=6.46)

# ══════════════════════════ 6. the wider literature, three findings
s = slide()
title(s, "Three findings from the widened review",
      "The review was widened from the application to the semantics of the problem: two datasets with the "
      "same content, one high quality and one low.")

FIND = [
    (1.66, "A", "The unpaired problem is solved in CT, not ophthalmology",
     "DuDoNet (CVPR 2019) is dual-domain but needs PAIRED data. DCDiff (MICCAI 2024) "
     "extends it with conditional diffusion. The CycleGAN family escaped the paired requirement through "
     "cycle-consistency: translate low quality to high quality and then back again, recovering the\n"
     "original.\n"
     "The catch, worth a paragraph in the paper: CycleGAN is documented to collapse on real tabletop to "
     "portable fundus data (Lin, MICCAI 2022). Borrow the framing, not the method."),
    (3.32, "B", "The routing idea has a name and a 2016 ancestor",
     "Domain Separation Networks (NeurIPS 2016) is almost exactly what he sketched: a shared encoder for "
     "what both domains have in common, a private encoder per domain for what is specific, no paired data "
     "required. For the conditional-entropy half there is an active literature on information-theoretic "
     "gating for Mixture-of-Experts, where routing is driven by mutual information between expert "
     "assignment and label. Neither piece is ours; the combination on cross-device fundus appears open."),
    (5.14, "C", "The generative world model idea is partly taken",
     "A world model for cross-domain relationships looked unexplored. It is not. CheXWorld (CVPR 2025) "
     "is a self-supervised world model for radiographs that explicitly models domain variation across "
     "hospitals and devices. It validates the direction at a top venue, but the claim must be narrower: "
     "CheXWorld does self-supervised pretraining on chest X-ray, not cross-device transfer of a trained "
     "classifier."),
]
for y, n, head, body in FIND:
    box(s, 0.65, y, 0.42, 0.40, n, ACCENT, WHITE, 12.5, True)
    label(s, 1.20, y - 0.02, 11.5, head, 12.5, ACCENT, True, PP_ALIGN.LEFT)
    label(s, 1.20, y + 0.32, 11.5, body, 10.5, INK, False, PP_ALIGN.LEFT, h=1.30)

takeaway(s, "Finding B is the one to act on: it gives the routing proposal a published architecture.",
         top=6.74)

# ══════════════════════════ 7. evidence
s = slide()
title(s, "The evidence behind each choice",
      "Peer-reviewed venues only. Most sit outside ophthalmology, which is the point. Paper names are links.")

lit = [
    ["Use", "What it establishes", "Paper", "Venue"],
    ["Adopt", "Clean features can teach degraded ones, using synthetically\ndegraded images. Exactly the setting here once step 2 makes pairs",
     "Self-Feature Distillation", "ECCV 2022"],
    ["Adopt", "Degradation and appearance augmentation beat every adaptation\nmethod tested for cross-scanner shift",
     "Tellez et al.", "Med. Image Anal. 2019"],
    ["Adopt", "Shared encoder plus a private encoder per domain, no pairs needed.\nThe architecture behind the routing idea",
     "Domain Separation Networks", "NeurIPS 2016"],
    ["Compare", "Aligns low-quality features to high-quality ones through a fixed\ninvertible decoder",
     "QualNet", "CVPR 2021"],
    ["Frame", "The same problem in CT: two views, one clean and one corrupted.\nNeeds paired data, which is the limitation inherited here",
     "DuDoNet", "CVPR 2019"],
    ["Bound", "World model that already captures cross-device domain variation,\nso the novelty claim must be narrower",
     "CheXWorld", "CVPR 2025"],
    ["Rule out", "Adversarial adaptation collapses on real tabletop to portable data",
     "Lin et al.", "MICCAI 2022"],
    ["Rule out", "Test-time adaptation loses up to 66 points under prior shift",
     "Boudiaf et al. (LAME)", "CVPR 2022"],
]
shp = table(s, lit, top=1.68, col_w=[0.9, 6.2, 2.7, 2.3], font=9, height=4.5)
for row, (col, url) in enumerate([
        (GOOD,  "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840544.pdf"),
        (GOOD,  "https://doi.org/10.1016/j.media.2019.101544"),
        (GOOD,  "https://arxiv.org/abs/1608.06019"),
        (ACCENT, "https://openaccess.thecvf.com/content/CVPR2021/html/Kim_Quality-Agnostic_Image_Recognition_via_Invertible_Decoder_CVPR_2021_paper.html"),
        (MUTED, "https://arxiv.org/abs/1907.00273"),
        (AMBER, "https://openaccess.thecvf.com/content/CVPR2025/html/Yue_CheXWorld_Exploring_Image_World_Modeling_for_Radiograph_Representation_Learning_CVPR_2025_paper.html"),
        (WARN,  "https://doi.org/10.1007/978-3-031-16434-7_57"),
        (WARN,  "https://doi.org/10.1109/CVPR52688.2022.00816")], start=1):
    cell_color(shp, row, 0, col)
    link_cell(shp, row, 2, url)

takeaway(s, "Three of the four approaches worth trying first are published failures in this exact setting.",
         top=6.42)

# ══════════════════════════ 8. experiment 1 result
s = slide()
title(s, "Experiment 1: encoder separation, and what the adversarial term was doing",
      "Four runs on the same aggregated training set. Only the architecture and the adversarial weight "
      "change. All 25 of 25 epochs, no instability.")

shp = table(s, [
    ["Run", "Adversarial\nweight", "DR AUC", "DR F1", "DR recall", "DR missed", "ME F1", "ME missed"],
    ["Plain aggregated training (no separation)", "not used", "0.9377", "0.8250", "0.8302", "27 / 159", "0.8649", "15 / 63"],
    ["Shared plus private encoders", "0.25", "0.9448", "0.8053", "0.7673", "37 / 159", "0.8224", "19 / 63"],
    ["Shared plus private encoders", "0.05", "0.9404", "0.8227", "0.7736", "36 / 159", "0.8889", "11 / 63"],
    ["Shared plus private encoders", "0.00", "0.9441", "0.8232", "0.8050", "31 / 159", "0.8833", "10 / 63"],
], top=1.58, left=0.65, width=12.1, col_w=[4.0, 1.4, 1.1, 1.1, 1.3, 1.5, 1.1, 1.4], font=10,
   height=1.75, highlight=(1,), hcolor=GREENF)

label(s, 0.65, 3.50, 12.1, "What the sweep shows", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "The adversarial term was the problem. Turning it down from 0.25 to 0.05 to zero raises DR F1 from "
    "0.8053 to 0.8227 to 0.8232, and cuts missed patients from 37 to 31. It was scrubbing device "
    "information the decision still needed, which is why ranking improved while recall fell.",
    "Even with it switched off, encoder separation does not beat plain aggregated training. On DR the two "
    "are indistinguishable (difference 0.0015, p = 0.91) and plain training still catches four more "
    "patients. On macular edema separation is ahead (10 missed against 15), but the difference is not "
    "significant (p = 0.50).",
    "Sweeping every cutoff confirms it is not a threshold artifact: the best DR F1 reachable is 0.8361 for "
    "separation against 0.8383 for plain training. There is no hidden advantage waiting behind a better "
    "operating point.",
], top=3.84, size=11.5, bottom=5.90)

box(s, 0.65, 5.94, 12.1, 0.60,
    "A hypothesis that turned out to be wrong, recorded rather than dropped: the deficit was expected to "
    "come from selecting checkpoints on the average of AUC and F1. Reselecting on F1 alone chose the same "
    "epoch and gave identical numbers, so that explanation is ruled out.",
    AMBERF, INK, 11, False, line=AMBER)

takeaway(s, "Encoder separation alone is a wash. The routing gate now has to carry the contribution itself.",
         WARN, top=6.62)

# ══════════════════════════ 9. plan
s = slide()
title(s, "What runs next", "Every experiment reports against the same 732-image mBRSET test set.")

shp = table(s, [
    ["", "Experiment", "Outcome", "Status"],
    ["0", "Plain aggregated training", "DR F1 0.8250, 27 of 159 missed. Still the model to beat", "Done"],
    ["1", "Shared plus private encoders, adversarial weight 0.25",
     "DR F1 0.8053, 37 missed. Worse than the baseline", "Done"],
    ["1a", "Same, checkpoints selected on macro F1 alone",
     "Identical result. The selection criterion was not the cause", "Done"],
    ["1b, 1c", "Adversarial weight reduced to 0.05, then to zero",
     "Recovers to DR F1 0.8232 and 31 missed, and to 10 of 63 on\nmacular edema. Still a wash against the baseline", "Done"],
    ["2", "Entropy-gated routing, built on the weight-zero variant",
     "The routing proposal itself. Both entropy formulations tested", "Next"],
    ["3", "Device-fitted degradation and distillation",
     "Control is generic FundusAug at its published settings", "Planned"],
], top=1.58, left=0.65, width=12.1, col_w=[0.6, 4.1, 6.4, 0.95], font=9.5, height=3.4)
for r, st in enumerate(["Done", "Done", "Done", "Done", "Next", "Planned"], start=1):
    cell_color(shp, r, 3, GOOD if st == "Done" else (ACCENT if st == "Next" else MUTED))

label(s, 0.65, 5.22, 12.1, "What the negative result changes", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "Experiment 2 now builds on the weight-zero variant rather than the published defaults, because the "
    "sweep showed those defaults actively hurt on this problem.",
    "The routing gate has to beat plain aggregated training on its own. It can no longer be presented as an "
    "addition to a structure that was already helping, because that structure is not helping.",
], top=5.56, size=11.5, bottom=6.50)

takeaway(s, "Methodology settled through September, experiments through early October, target mid-October.",
         GOOD, top=6.54)

prs.save(OUT)
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
