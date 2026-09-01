"""Findings deck: what changed after Dr. Ye's meeting of 1 September 2026.

Records the aggregation experiment in full, the widened literature review he
asked for, and how his routing proposal maps onto published work. Every number
is recomputed from results/ at build time.
"""
from deck_common import (prs, slide, title, bullets, table, box, label, takeaway,
                         link_cell, cell_color, Inches, Pt, PP_ALIGN, MSO_SHAPE, RGBColor,
                         INK, MUTED, ACCENT, GOOD, WARN, AMBER, WHITE, PALE, RULE)
from pathlib import Path

OUT = Path(__file__).parent / "BRSET_mBRSET_Findings.pptx"

# ═════════════════════════════════════════ 1. what changed
s = slide()
title(s, "What changed after the meeting",
      "Three directives from Dr. Ye, and where each one leaves the work.")

table(s, [
    ["Directive", "Before", "After"],
    ["mBRSET may be used in training.\n\"For the training we can use anything.\nFor the testing we cannot.\"",
     "Assumed zero-shot transfer.\nBaseline to beat: 0.7842",
     "Aggregated training allowed.\nBaseline to beat: 0.8355"],
    ["The methods cited rely on PAIRED data,\nsame patient on both cameras.\nBRSET and mBRSET share no patients.",
     "Treated the paired-data papers\nas directly applicable",
     "Adapting them to unpaired data\nis itself the research problem"],
    ["Widen the review beyond retinal imaging\nto any two datasets, one high quality\nand one low quality, same content.",
     "Review confined to fundus\nand DR grading",
     "CT artifact reduction and low-dose\nCT are the mature literature"],
], top=1.62, left=0.65, width=12.1, col_w=[4.6, 3.6, 3.9], font=10.5, height=3.2)

label(s, 0.65, 5.05, 12.1, "The consequence", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "The question is no longer how close a label-free method can get. It is how best to combine a large "
    "high-quality labelled set with a small degraded one. That is a harder bar and a cleaner question.",
    "Only the mBRSET test split of 732 images stays untouched. Everything else may be trained on.",
], top=5.40, size=12.5, bottom=6.45)

takeaway(s, "The aggregation experiment was already run before the meeting. Results on the next slide.",
         GOOD, top=6.50)

# ═════════════════════════════════════════ 2. the aggregation experiment
s = slide()
title(s, "The aggregation experiment, in full",
      "BRSET and mBRSET merged into one training set. Validated and tested on mBRSET only, "
      "so the target domain is never contaminated.")

label(s, 0.65, 1.42, 6.0, "Data", 12.5, ACCENT, True, PP_ALIGN.LEFT)
table(s, [
    ["Split", "Images", "DR positive"],
    ["Train: BRSET", "11,372  (77%)", "748   (6.6%)"],
    ["Train: mBRSET", "3,402  (23%)", "797  (23.4%)"],
    ["Train: total", "14,774", "1,545  (10.5%)"],
    ["Validation (mBRSET only)", "725", "176  (24.3%)"],
    ["Test (mBRSET only)", "732", "159  (21.7%)"],
], top=1.76, left=0.65, width=6.0, col_w=[2.8, 1.7, 1.7], font=10, height=1.9)

label(s, 6.95, 1.42, 5.8, "Configuration", 12.5, ACCENT, True, PP_ALIGN.LEFT)
table(s, [
    ["Setting", "Value"],
    ["Backbone", "ConvNeXt V2 Large, 512 px"],
    ["Optimizer", "AdamW, lr 3e-5, weight decay 0.1"],
    ["Schedule", "25 epochs, 3 warmup, cosine"],
    ["Loss / sampler", "focal gamma 2.0, no oversampling"],
    ["Inference", "EMA 0.999, 4-way flip TTA"],
], top=1.76, left=6.95, width=5.8, col_w=[2.0, 3.8], font=10, height=1.9)

label(s, 0.65, 3.80, 12.1, "Results on the mBRSET test set, against every alternative",
      12.5, ACCENT, True, PP_ALIGN.LEFT)
shp = table(s, [
    ["Model", "DR AUC", "DR F1", "DR missed", "ME AUC", "ME F1", "ME missed"],
    ["Zero-shot transfer, no mBRSET in training", "0.9060", "0.7842", "50 / 159", "0.9705", "0.7027", "24 / 63"],
    ["mBRSET only, from scratch", "0.9433", "0.8239", "35 / 159", "0.9920", "0.8333", "18 / 63"],
    ["BRSET pretrain, mBRSET finetune (best F1)", "0.9446", "0.8355", "32 / 159", "0.9895", "0.8113", "20 / 63"],
    ["Aggregated BRSET + mBRSET (joint)", "0.9377", "0.8250", "27 / 159", "0.9936", "0.8649", "15 / 63"],
], top=4.16, left=0.65, width=12.1, col_w=[4.6, 1.2, 1.2, 1.5, 1.2, 1.2, 1.5], font=10,
   height=1.55, highlight=(4,), hcolor=RGBColor(0xDF, 0xEA, 0xDF))

bullets(s, [
    "Joint training catches the most patients, 27 missed against 32 for the best fine-tune, and gives the "
    "best macular edema result at perfect precision. But paired bootstrap says joint and fine-tuning are "
    "statistically indistinguishable on DR (p = 0.47), so this is a tie on the headline metric, not a win.",
    "Training was clean: 25 of 25 epochs, zero non-finite batches, best epoch 10, 6.6 hours on one A40.",
], top=5.86, size=11.5, bottom=6.60)

takeaway(s, "Aggregation matches fine-tuning and misses fewer patients. It is the right base to add routing to.",
         top=6.64)

# ═════════════════════════════════════════ 3. the widened literature
s = slide()
title(s, "Where this problem has already been solved, outside ophthalmology",
      "Stated abstractly, as Dr. Ye asked: two datasets, same content, one high quality and one low quality. "
      "Paper names are live links.")

lit = [
    ["Family", "What it contributes here", "Paper", "Venue"],
    ["Architecture\nfor routing", "Shared encoder plus a private encoder per domain.\nNo paired data required. Closest published form of\nthe routing idea",
     "Domain Separation\nNetworks", "NeurIPS 2016"],
    ["The unpaired\nproblem", "Dual-domain network for CT metal artifact reduction.\nNamed directly by Dr. Ye. Requires PAIRED data,\nthe limitation inherited here",
     "DuDoNet", "CVPR 2019"],
    ["The unpaired\nproblem", "Same lineage, now with conditional diffusion",
     "DCDiff", "MICCAI 2024"],
    ["World model", "Self-supervised world model for radiographs that\nexplicitly models domain variation across hospitals\nand devices",
     "CheXWorld", "CVPR 2025"],
    ["Augmentation", "Nine degradation operations, generic settings.\nAUC 78.4 to 82.6 across eight fundus datasets",
     "GDRNet", "MICCAI 2023"],
    ["Degradation\nmodel", "Fundus degradation derived from the optics:\nlight transmission, blur, artifacts",
     "cofe-Net", "IEEE TMI 2021"],
]
shp = table(s, lit, top=1.68, col_w=[1.3, 6.4, 2.3, 1.6], font=9.5, height=4.0)
for row, url in enumerate([
        "https://arxiv.org/abs/1608.06019",
        "https://arxiv.org/abs/1907.00273",
        "https://papers.miccai.org/miccai-2024/192-Paper1608.html",
        "https://openaccess.thecvf.com/content/CVPR2025/html/Yue_CheXWorld_Exploring_Image_World_Modeling_for_Radiograph_Representation_Learning_CVPR_2025_paper.html",
        "https://arxiv.org/abs/2307.04378",
        "https://arxiv.org/abs/2005.05594"], start=1):
    link_cell(shp, row, 2, url)

bullets(s, [
    "How CT escaped the paired-data requirement: cycle-consistency. Translate low to high quality and back, "
    "and recover the original. The catch is that CycleGAN is documented to collapse on real tabletop to "
    "portable fundus data (Lin, MICCAI 2022), so the framing transfers but the method does not.",
    "One correction to raise with Dr. Ye: he believed a generative world model for cross-domain relationships "
    "was unexplored. CheXWorld does model domain variation, at CVPR 2025. Room remains, since it does "
    "self-supervised pretraining rather than cross-device transfer of a classifier, but less than assumed.",
], top=5.82, size=11, bottom=6.62)

# ═════════════════════════════════════════ 4. the routing idea
s = slide()
title(s, "Dr. Ye's routing proposal, and what it maps onto",
      "His central technical idea, restated precisely, with the published work that supports each half.")

box(s, 0.65, 1.48, 12.1, 0.72,
    "\"When you detect the disease in the mBRSET, mostly it should be based on the mBRSET, "
    "but part of the BRSET data may be helpful.\"", PALE, INK, 12.5, False, line=RULE)

label(s, 0.65, 2.38, 12.1, "The problem it solves", 12.5, ACCENT, True, PP_ALIGN.LEFT)
label(s, 0.65, 2.72, 12.1,
      "Plain aggregation lets BRSET dominate 77 percent to 23 percent, so the model mostly learns tabletop "
      "features even when classifying a handheld image. Routing lets the model decide, per image, how much "
      "to draw on each source.", 12, INK, False, PP_ALIGN.LEFT, h=0.62)

table(s, [
    ["Component", "What Dr. Ye specified", "Published form"],
    ["Where it sits", "In the ENCODER, not the decision layer.\nHe repeated this twice",
     "Domain Separation Networks: shared encoder\nplus per-domain private encoders (NeurIPS 2016)"],
    ["Routing signal", "Conditional entropy of the labels,\ncomputed during training",
     "Information-theoretic Mixture-of-Experts gating:\nmutual information between expert and label"],
], top=3.48, left=0.65, width=12.1, col_w=[1.6, 4.6, 5.9], font=10, height=1.5)

label(s, 0.65, 5.14, 12.1, "Open question, carried forward deliberately", 12.5, AMBER, True, PP_ALIGN.LEFT)
bullets(s, [
    "The transcript is ambiguous on the entropy term. Entropy of the gate's own output distribution is "
    "standard and easy; conditional entropy of the label given domain features is closer to his wording and "
    "harder. I will implement both variants and test rather than guess, and confirm with him.",
], top=5.48, size=11.5, bottom=6.20)

takeaway(s, "Neither half is ours. The combination, applied to unpaired cross-device fundus, appears open.",
         top=6.28)

# ═════════════════════════════════════════ 5. the plan
s = slide()
title(s, "What runs next", "Each experiment produces a number against a fixed control.")

shp = table(s, [
    ["", "Experiment", "What it establishes", "Control to beat", "Status"],
    ["1", "Domain Separation baseline",
     "Shared plus private encoders on the aggregated set.\nPublished architecture, unmodified",
     "Joint 0.8250\nFinetune 0.8355", "Running"],
    ["2", "Entropy-gated routing",
     "Adds Dr. Ye's gate on top of experiment 1.\nBoth entropy variants tested",
     "Experiment 1", "Next"],
    ["3", "Device-fitted augmentation",
     "Fit degradation to measured mBRSET statistics.\nRuns alongside routing, not instead of it",
     "Generic FundusAug", "Planned"],
], top=1.62, left=0.65, width=12.1, col_w=[0.4, 2.8, 5.6, 2.0, 1.3], font=10, height=2.4)
for r, st in enumerate(["Running", "Next", "Planned"], start=1):
    cell_color(shp, r, 4, AMBER if st == "Running" else (ACCENT if st == "Next" else MUTED))

label(s, 0.65, 4.24, 12.1, "Why experiment 1 comes first", 12.5, ACCENT, True, PP_ALIGN.LEFT)
bullets(s, [
    "Domain Separation Networks is published and unambiguous, so it can be built correctly today while the "
    "entropy formulation is still being clarified. If encoder-level separation does not help at all, the "
    "routing idea has no foundation, and that is learned cheaply.",
    "Every experiment reports against the same 732-image mBRSET test set with paired bootstrap intervals, "
    "so improvements are testable rather than asserted.",
    "Timeline: methodology settled through September, experiments through early October, target mid-October.",
], top=4.58, size=11.5, bottom=6.45)

takeaway(s, "Experiment 1 is queued. It answers whether encoder-level separation is worth building on.",
         GOOD, top=6.50)

prs.save(OUT)
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
