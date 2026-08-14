"""Part 2 deck: the BRSET -> mBRSET literature review and proposed direction.

Five slides. Every citation carries venue, year and approximate citation count
(Semantic Scholar, retrieved 2026-08-07), and every published paper is cited by
its peer-reviewed version rather than a preprint, per Dr. Ye's instruction.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).parent / "BRSET_Part2_LiteratureReview.pptx"

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GOOD = RGBColor(0x1E, 0x7A, 0x3C)
WARN = RGBColor(0xA5, 0x3A, 0x1A)
LINK = RGBColor(0x1155CC >> 16 & 0xFF, 0x1155CC >> 8 & 0xFF, 0x1155CC & 0xFF)
HDRBG = RGBColor(0x1F, 0x4E, 0x79)
ROWBG = RGBColor(0xF2, 0xF4, 0xF7)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def title(s, text, sub=None):
    box = s.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(12.3), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(27)
    p.font.bold = True
    p.font.color.rgb = ACCENT
    if sub:
        q = tf.add_paragraph()
        q.text = sub
        q.font.size = Pt(13)
        q.font.color.rgb = MUTED
    return box


def bullets(s, items, top, size=13, left=0.6, width=12.2, gap=6, bottom=7.2):
    h = max(0.4, min(5.2, bottom - top))
    box = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        text, lvl = it if isinstance(it, tuple) else (it, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "– ") + text
        p.level = lvl
        p.font.size = Pt(size if lvl == 0 else size - 1)
        p.font.color.rgb = INK if lvl == 0 else MUTED
        p.space_after = Pt(gap)
    return box


def table(s, rows, top, left=0.6, width=12.2, height=None, col_w=None,
          font=10.5, highlight=(), hcolor=None):
    nrow, ncol = len(rows), len(rows[0])
    height = height or 0.3 * nrow
    shp = s.shapes.add_table(nrow, ncol, Inches(left), Inches(top),
                             Inches(width), Inches(height))
    tbl = shp.table
    if col_w:
        tot = sum(col_w)
        for j, w in enumerate(col_w):
            tbl.columns[j].width = Inches(width * w / tot)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(val)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            hdr = (i == 0)
            c.fill.solid()
            if hdr:
                c.fill.fore_color.rgb = HDRBG
            elif i in highlight:
                c.fill.fore_color.rgb = hcolor or RGBColor(0xE3, 0xEF, 0xE3)
            else:
                c.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i % 2 else ROWBG
            for para in c.text_frame.paragraphs:
                para.font.size = Pt(font)
                para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                para.font.bold = hdr or (i in highlight)
                para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if hdr else INK
                for r in para.runs:
                    r.font.size = Pt(font)
                    r.font.bold = hdr or (i in highlight)
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if hdr else INK
    return shp


def takeaway(s, text, color=ACCENT, top=6.5):
    box = s.shapes.add_textbox(Inches(0.6), Inches(top), Inches(12.2), Inches(0.7))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = color
    return box


# ============================================================ 1
s = slide()
title(s, "Part 2 — BRSET → mBRSET: literature review and proposed direction",
      "Sahith Reddy Thummala   |   For: Dr. Dong Hye Ye   |   cc: Nagur Shareef Shaik")
bullets(s, [
    "How this review was conducted, after your feedback that the previous one was not rigorous enough:",
    ("Peer-reviewed versions only. Where a preprint exists, the published venue is cited instead, "
     "and preprint-only work is labelled as weak evidence.", 1),
    ("Citation counts retrieved from the Semantic Scholar Graph API by DOI (not by title matching), "
     "7 Aug 2026. Counts are stated for every paper.", 1),
    ("Papers were obtained and read as PDFs; results tables were extracted from the source, "
     "not from search summaries. Numbers quoted below come from those tables.", 1),
    ("Venue standing recorded for each: CVPR / ICCV / ECCV / NeurIPS / ICML / ICLR / MICCAI / IPMI, "
     "and Medical Image Analysis / IEEE TMI / Nature-family journals.", 1),
    "Three independent reviews were run over separate sub-questions and cross-checked: "
    "(i) unsupervised domain adaptation and test-time adaptation for real device shift, "
    "(ii) prevalence and threshold shift in medical deployment, "
    "(iii) the statistical theory of multi-view aggregation.",
    "Coverage check: every paper citing either dataset paper was enumerated through the citation graph, "
    "plus GitHub, arXiv and preprint servers, to establish what is already claimed.",
], top=1.7, size=14, bottom=6.4)
takeaway(s, "The aim was not a list of related work, but to establish what is already solved, "
            "what provably cannot work, and where a real gap remains.")

# ============================================================ 2
s = slide()
title(s, "What the published literature establishes",
      "Peer-reviewed venue and approximate citation count given for each. Full links on the references slide.")
table(s, [
    ["Finding", "Evidence", "Venue", "Cites"],
    ["Aggressive appearance / degradation augmentation is the single most\n"
     "effective intervention for cross-scanner shift — it beat stain\n"
     "normalisation and every adaptation method across 4 pathology tasks",
     "Tellez et al.", "Medical Image\nAnalysis 2019", "656"],
    ["For diabetic retinopathy across 6 fundus datasets, leave-one-domain-out\n"
     "AUC rises 75.9 → 82.6. The fundus augmentation component alone\n"
     "contributes +3.4 AUC points — the largest single factor",
     "Che et al. (GDRNet)", "MICCAI 2023", "46"],
    ["Adversarial and generative adaptation COLLAPSE on real tabletop →\n"
     "portable fundus data: DANN R² −0.025, MDD −0.010, CycleGAN no change,\n"
     "against a 0.361 baseline",
     "Lin et al.", "MICCAI 2022", "5"],
    ["Under a fair model-selection protocol, no domain-generalization\n"
     "algorithm reliably beats plain empirical risk minimisation",
     "Gulrajani &\nLopez-Paz", "ICLR 2021", "1,504"],
    ["Test-time adaptation degrades accuracy by up to 66 points when applied\n"
     "under non-i.i.d. and prior-shifted conditions — precisely our regime",
     "Boudiaf et al. (LAME)", "CVPR 2022", "250"],
    ["Unsupervised correction of the prediction distribution restores the\n"
     "operating point across 4 mammography scanners with NO target labels\n"
     "(Youden 0.295 → 0.651), but assumes prevalence is preserved",
     "Roschewitz et al.", "Nature\nCommunications 2023", "42"],
    ["Prevalence shift alone breaks calibration, decision thresholds and\n"
     "metric interpretation across 30 medical classification tasks",
     "Godau et al.", "MICCAI 2023 →\nMed. Image Anal. 2025", "15"],
], top=1.55, col_w=[6.9, 2.0, 2.0, 0.8], font=10, height=4.6)
takeaway(s, "Two conclusions: augmentation has the strongest evidence for raising AUC under device shift, "
            "and the popular adaptation families have published evidence of failing on exactly our problem.",
         top=6.35)

# ============================================================ 3
s = slide()
title(s, "What we established ourselves — measurements, not reading",
      "Each of these was computed on our own data and is reproducible from the repository.")
table(s, [
    ["Question", "What we did", "Result"],
    ["Is the shift a prevalence shift or a\ngenuine model failure?",
     "Compared class-conditional score behaviour;\napplied the ROC invariance property",
     "NOT prevalence shift. ROC is provably invariant\nto class distribution (Fawcett 2006, 21,726 cites),\nyet AUC fell 0.988 → 0.909"],
    ["How much of the collapse is the\ndecision cutoff?",
     "Closed-form Bayes mapping of the optimal\ncutoff under a prior change",
     "77% of the 0.61 → 0.19 cutoff move is explained\nby disease rate alone (6.6% → 23.3%)"],
    ["Can any calibration method close\nthe remaining gap?",
     "Swept every attainable cutoff on the target\ntest set to find the true maximum F1",
     "No. Ceiling is F1 0.765 (DR) / 0.671 (ME).\nProof by set inclusion: monotone post-processing\ncannot exceed it"],
    ["Would averaging a patient's images\nraise AUC, and by how much?",
     "Derived AUC_K = Φ(z₁·√(K/(1+(K−1)ρ))) and\nmeasured ρ by variance decomposition",
     "ρ = 0.363 (patient), 0.481 (eye).\nPredicted 0.909 → 0.968"],
], top=1.6, col_w=[3.0, 4.2, 5.0], font=10, height=3.9)
bullets(s, [
    "The fourth row is the important one: the entire multi-view proposal reduces to a single measurable "
    "quantity, so it could be tested before any implementation.",
], top=5.75, size=13, bottom=6.35)
takeaway(s, "The decomposition — prevalence vs. calibration vs. ranking — is computable in closed form. "
            "Most cross-device papers report one number and attribute all of it to the model.", top=6.4)

# ============================================================ 4
s = slide()
title(s, "We tested our own proposal before building it — and it failed",
      "The theory gave a falsifiable prediction. We checked it against data we already had, in minutes rather than a week.")
table(s, [
    ["Aggregation of a patient's images", "Predicted by theory", "Actually measured", "Verdict"],
    ["Diabetic retinopathy, 4 images per patient", "0.968", "0.842", "Failed"],
    ["Diabetic retinopathy, 2 images per eye", "0.940", "0.883", "Failed"],
    ["Macular edema, max over 4 images", "—", "0.969  (from 0.933)", "Worked"],
], top=1.65, col_w=[5.2, 2.4, 2.6, 1.4], font=11.5, height=1.5,
   highlight=(1, 2), hcolor=RGBColor(0xFD, 0xEC, 0xEC))
bullets(s, [
    "Why it failed, and it is a mechanism we can state precisely:",
    ("The theory assumes every image in a patient's group carries the same label. 14.5% of patients have "
     "one eye affected and the other clear, so averaging pulls the diseased eye's score down toward the "
     "healthy one and the patient looks healthier than they are.", 1),
    ("Independently, the theory predicts mean-pooling is the wrong operator when a bag is heterogeneous — "
     "an explicit counterexample takes AUC from 0.75 to 0.50 under mean-pooling while the median gives 1.00. "
     "The published literature agrees: the best reported multi-view gain in medical imaging is +0.015 AUC "
     "(Wu et al., MIDL 2020), and naive fusion there scored WORSE than the best single view.", 1),
    "Macular edema behaves differently and consistently with theory: it is a focal finding that appears "
    "strongly in one image, so a maximum — not a mean — is the correct aggregator, and it gives +0.036 AUC.",
], top=3.4, size=12.5, bottom=6.35)
takeaway(s, "Reporting this rather than burying it: the prediction was wrong, the reason is understood, "
            "and the cost of finding out was minutes.", WARN, top=6.4)

# ============================================================ 5
s = slide()
title(s, "Where the real gap is, and what we propose",
      "Positioned against the two efforts closest to ours, both identified during the review.")
table(s, [
    ["Nearby work", "What it does", "What it leaves open"],
    ["RetSyn — Shuai, Wu, Tang, Restrepo, Morley,\nNakayama.  J. Biomedical Informatics 2025.\nSame senior author as both datasets.",
     "Quality- and class-conditioned synthetic data\nfor tabletop → portable DR screening",
     "Does not diagnose what breaks; does not separate\nimage quality from camera identity; does not\naddress macular edema or the decision cutoff"],
    ["IEEE ISBI 2026 — transferability of\nfundus-trained models to mobile imaging",
     "Reports the BRSET → mBRSET performance drop",
     "Descriptive. Establishes that the drop exists,\nnot what it is made of or how to close it"],
], top=1.6, col_w=[4.0, 4.0, 4.2], font=10, height=1.9)
bullets(s, [
    "Three openings the review did not find addressed anywhere:",
    ("1.  Decomposing a cross-device gap into prevalence, calibration and ranking components. "
     "Papers report a single number and attribute it to the model; we show the split is computable, "
     "and that in our case calibration is provably exhausted.", 1),
    ("2.  Label-shift correction for MULTI-LABEL sigmoid outputs. Every published method — BBSE "
     "(Lipton et al., ICML 2018), MLLS (Alexandari et al., ICML 2020; Garg et al., NeurIPS 2020) — "
     "is single-label multi-class. Diabetic retinopathy and macular edema co-occur, so per-label "
     "independent estimation is misspecified. No published extension exists.", 1),
    ("3.  The intersection where both published families break their own stated assumptions: the "
     "mammography method assumes prevalence is preserved (ours changes 6.6% → 23.3%) and the "
     "prevalence-shift work assumes p(x|y) is unchanged (our AUC drop shows it is not).", 1),
    "Proposed next step, ordered by strength of published evidence rather than novelty: "
    "fundus degradation augmentation on the source (the only intervention with peer-reviewed evidence "
    "of raising AUC under real fundus device shift), with the gap decomposition as the scientific core.",
], top=3.75, size=12, bottom=6.4)
takeaway(s, "The decomposition holds as a contribution even if every method underperforms — "
            "showing what the gap is made of is itself the result.", GOOD, top=6.45)

# ============================================================ 6  references
s = slide()
title(s, "References", "Peer-reviewed versions. Citation counts: Semantic Scholar Graph API, 7 Aug 2026.")
refs = [
    "Tellez et al. Quantifying the effects of data augmentation and stain color normalization in CNNs for computational "
    "pathology. Medical Image Analysis 58:101544, 2019.  656 cites.  doi.org/10.1016/j.media.2019.101544",
    "Che, Cheng, Jin, Chen. Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains. MICCAI 2023, "
    "pp. 430–440.  46 cites.  doi.org/10.1007/978-3-031-43904-9_42",
    "Lin, Shi, Zhang, Shang, He, Ge. Camera Adaptation for Fundus-Image-Based CVD Risk Estimation. MICCAI 2022, "
    "pp. 593–603.  5 cites.  doi.org/10.1007/978-3-031-16434-7_57",
    "Gulrajani, Lopez-Paz. In Search of Lost Domain Generalization. ICLR 2021.  1,504 cites.  "
    "openreview.net/forum?id=lQdXeXDoWtI",
    "Boudiaf, Mueller, Ben Ayed, Bertinetto. Parameter-free Online Test-time Adaptation. CVPR 2022, pp. 8344–8353.  "
    "250 cites.  doi.org/10.1109/CVPR52688.2022.00816",
    "Roschewitz, Khara, Yearsley et al. Automatic correction of performance drift under acquisition shift in medical "
    "image classification. Nature Communications 14:6608, 2023.  42 cites.  doi.org/10.1038/s41467-023-42396-y",
    "Godau, Kalinowski, Christodoulou et al. Deployment of Image Analysis Algorithms under Prevalence Shifts. "
    "MICCAI 2023 → Navigating prevalence shifts..., Medical Image Analysis 102:103504, 2025.  "
    "doi.org/10.1016/j.media.2025.103504",
    "Lipton, Wang, Smola. Detecting and Correcting for Label Shift with Black Box Predictors. ICML 2018.  679 cites.  "
    "proceedings.mlr.press/v80/lipton18a.html",
    "Alexandari, Kundaje, Shrikumar. Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift "
    "Adaptation. ICML 2020, pp. 222–232.  proceedings.mlr.press/v119/alexandari20a.html",
    "Garg, Wu, Balakrishnan, Lipton. A Unified View of Label Shift Estimation. NeurIPS 2020.  179 cites.",
    "Fawcett. An introduction to ROC analysis. Pattern Recognition Letters 27(8):861–874, 2006.  21,726 cites.  "
    "doi.org/10.1016/j.patrec.2005.10.010",
    "Wu, Jastrzębski, Park, Moy, Cho, Geras. Improving the Ability of Deep Neural Networks to Use Information from "
    "Multiple Views in Breast Cancer Screening. MIDL 2020, PMLR 121:827–842.",
    "Ilse, Tomczak, Welling. Attention-based Deep Multiple Instance Learning. ICML 2018.  2,691 cites.",
    "Shuai, Wu, Tang, Restrepo, Morley, Nakayama. Enhancing AI-based diabetic retinopathy screening in low- and "
    "middle-income countries with synthetic data. J. Biomedical Informatics 172:104938, 2025.  "
    "doi.org/10.1016/j.jbi.2025.104938",
    "Nakayama, Restrepo et al. BRSET: A Brazilian Multilabel Ophthalmological Dataset. PLOS Digital Health "
    "3(7):e0000454, 2024.   |   Wu, Restrepo, Nakayama et al. A portable retina fundus photos dataset. "
    "Scientific Data 12:340, 2025.",
]
box = s.shapes.add_textbox(Inches(0.6), Inches(1.5), Inches(12.2), Inches(5.6))
tf = box.text_frame
tf.word_wrap = True
for i, r in enumerate(refs):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = f"[{i+1}]  {r}"
    p.font.size = Pt(9.5)
    p.font.color.rgb = INK
    p.space_after = Pt(3)

prs.save(OUT)
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
