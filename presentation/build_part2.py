"""Part 2 deck: BRSET -> mBRSET literature review and proposed direction.

Design brief: minimal words. Numbers and diagrams carry the argument; prose is
cut to what cannot be shown visually. Every citation keeps its venue and
citation count, since that was the point of redoing the review.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT = Path(__file__).parent / "BRSET_Part2_LiteratureReview.pptx"

INK    = RGBColor(0x1A, 0x1A, 0x1A)
MUTED  = RGBColor(0x6B, 0x6B, 0x6B)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GOOD   = RGBColor(0x1E, 0x7A, 0x3C)
WARN   = RGBColor(0xB3, 0x3A, 0x1A)
AMBER  = RGBColor(0xC8, 0x86, 0x00)
HDRBG  = RGBColor(0x1F, 0x4E, 0x79)
ROWBG  = RGBColor(0xF3, 0xF5, 0xF8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def title(s, text, sub=None):
    b = s.shapes.add_textbox(Inches(0.6), Inches(0.38), Inches(12.2), Inches(0.85))
    tf = b.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = ACCENT
    if sub:
        q = tf.add_paragraph(); q.text = sub
        q.font.size = Pt(13); q.font.color.rgb = MUTED
    return b


def bullets(s, items, top, size=14, left=0.65, width=12.1, gap=9, bottom=7.1):
    b = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width),
                             Inches(max(0.4, min(5.4, bottom - top))))
    tf = b.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        t, lv = it if isinstance(it, tuple) else (it, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lv == 0 else "– ") + t
        p.level = lv
        p.font.size = Pt(size if lv == 0 else size - 1.5)
        p.font.color.rgb = INK if lv == 0 else MUTED
        p.space_after = Pt(gap)
    return b


def table(s, rows, top, left=0.65, width=12.1, height=None, col_w=None,
          font=11.5, highlight=(), hcolor=None):
    nr, nc = len(rows), len(rows[0])
    shp = s.shapes.add_table(nr, nc, Inches(left), Inches(top),
                             Inches(width), Inches(height or 0.32 * nr))
    tbl = shp.table
    if col_w:
        tot = sum(col_w)
        for j, w in enumerate(col_w):
            tbl.columns[j].width = Inches(width * w / tot)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = tbl.cell(i, j); c.text = str(v)
            c.margin_left = c.margin_right = Inches(0.07)
            c.margin_top = c.margin_bottom = Inches(0.025)
            hdr = (i == 0)
            c.fill.solid()
            c.fill.fore_color.rgb = (HDRBG if hdr else
                                     (hcolor or RGBColor(0xE4, 0xEF, 0xE4)) if i in highlight else
                                     (WHITE if i % 2 else ROWBG))
            for pa in c.text_frame.paragraphs:
                pa.font.size = Pt(font)
                pa.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                pa.font.bold = hdr or (i in highlight)
                pa.font.color.rgb = WHITE if hdr else INK
                for r in pa.runs:
                    r.font.size = Pt(font); r.font.bold = hdr or (i in highlight)
                    r.font.color.rgb = WHITE if hdr else INK
    return shp


def box(s, x, y, w, h, text, fill, fg=WHITE, size=12, bold=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = fill
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = text; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = fg
    for r in p.runs:
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = fg
    return sh


def label(s, x, y, w, text, size=11, color=INK, bold=False, align=PP_ALIGN.CENTER):
    b = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.34))
    tf = b.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text; p.alignment = align
    p.font.size = Pt(size); p.font.color.rgb = color; p.font.bold = bold
    return b


def arrow(s, x, y, w, h=0.22, color=MUTED):
    sh = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background(); sh.shadow.inherit = False
    return sh


def takeaway(s, text, color=ACCENT, top=6.55):
    b = s.shapes.add_textbox(Inches(0.65), Inches(top), Inches(12.1), Inches(0.65))
    tf = b.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = color
    return b


# ══════════════════════════════════════════════════ 1 — title + method
s = slide()
title(s, "Part 2 — BRSET → mBRSET",
      "Literature review and proposed direction   |   Sahith Reddy Thummala   |   For: Dr. Dong Hye Ye")

label(s, 0.65, 1.55, 12.1, "How this review was done differently", 15, ACCENT, True, PP_ALIGN.LEFT)
y = 2.15
for t, d in [("Published only", "Peer-reviewed version cited, not the preprint.\nPreprint-only work labelled weak evidence."),
             ("Counted", "Citation counts pulled by DOI from the\nSemantic Scholar API, not title matching."),
             ("Read, not skimmed", "PDFs obtained; results tables extracted\nfrom source, not from search summaries."),
             ("Exhaustive", "Full citation graph of both dataset papers\nenumerated, plus GitHub and preprint servers.")]:
    box(s, 0.65 + (y - 2.15) * 0, y, 2.6, 0.85, t, ACCENT, size=13)
    label(s, 3.45, y + 0.06, 9.3, d, 12, INK, False, PP_ALIGN.LEFT)
    y += 1.02

takeaway(s, "Three independent sub-reviews, cross-checked: device adaptation · prevalence shift · aggregation theory.",
         MUTED, top=6.4)

# ══════════════════════════════════════════════════ 2 — what the literature says
s = slide()
title(s, "What the literature establishes",
      "Venue and citation count for every claim.")
table(s, [
    ["", "Finding", "Source", "Venue", "Cites"],
    ["✓", "Appearance augmentation beats every adaptation method\ntested for cross-scanner shift", "Tellez et al.", "Med. Image Anal. 2019", "656"],
    ["✓", "Fundus DR across 6 datasets: AUC 75.9 → 82.6.\nAugmentation alone contributes +3.4", "Che et al. (GDRNet)", "MICCAI 2023", "46"],
    ["✗", "DANN, MDD, CycleGAN all COLLAPSE on real\ntabletop → portable fundus data", "Lin et al.", "MICCAI 2022", "5"],
    ["✗", "No domain-generalization method reliably beats\nplain training under fair evaluation", "Gulrajani & Lopez-Paz", "ICLR 2021", "1,504"],
    ["✗", "Test-time adaptation loses up to 66 points under\nprior shift — our exact regime", "Boudiaf et al. (LAME)", "CVPR 2022", "250"],
    ["~", "Threshold can be fixed with no target labels, but the\nmethod assumes prevalence is unchanged", "Roschewitz et al.", "Nature Comms 2023", "42"],
], top=1.6, col_w=[0.4, 5.6, 2.4, 2.4, 0.8], font=11, height=3.6)

label(s, 0.65, 5.45, 4.0, "✓  strongest evidence", 12, GOOD, True, PP_ALIGN.LEFT)
label(s, 4.3, 5.45, 4.0, "✗  published evidence of failure", 12, WARN, True, PP_ALIGN.LEFT)
label(s, 8.4, 5.45, 4.3, "~  works, but assumptions break here", 12, AMBER, True, PP_ALIGN.LEFT)

takeaway(s, "Augmentation is the only intervention with peer-reviewed evidence of raising AUC under real fundus device shift.",
         top=6.1)

# ══════════════════════════════════════════════════ 3 — the gap decomposition (diagram)
s = slide()
title(s, "What the gap is actually made of",
      "Diabetic retinopathy, F1 on mBRSET. Every figure measured on our own data.")

# segmented bar: 0.661 -> 0.717 -> 0.765 -> 0.815
X0, Y, TOT_W, H = 0.9, 2.5, 11.4, 0.95
lo, hi = 0.661, 0.815
def px(v):  # value -> inches offset
    return TOT_W * (v - lo) / (hi - lo)

segs = [(0.661, 0.717, GOOD,  "fixed by\nretuning the cutoff", "+0.056"),
        (0.717, 0.765, AMBER, "still reachable\nby a better cutoff", "+0.048"),
        (0.765, 0.815, WARN,  "needs better ranking\n(higher AUC)", "+0.050")]
for a, b, col, txt, delta in segs:
    box(s, X0 + px(a), Y, px(b) - px(a), H, delta, col, size=15)
    label(s, X0 + px(a), Y + H + 0.12, px(b) - px(a), txt, 11.5, INK, False)

for v, cap in [(0.661, "0.661\nsource cutoff"), (0.717, "0.717\nwhere we are"),
               (0.765, "0.765\nbest any cutoff"), (0.815, "0.815\ntrained on mBRSET")]:
    label(s, X0 + px(v) - 0.85, Y - 0.72, 1.7, cap, 11, ACCENT, True)

bullets(s, [
    "Threshold retuning alone recovered a third of the loss — no retraining.",
    "The 0.765 ceiling was measured by sweeping every attainable cutoff. "
    "Monotone post-processing cannot exceed it, by set inclusion — so calibration, temperature scaling and "
    "label-shift correction are provably ruled out from here.",
    "The shift is not label shift: ROC is invariant to class distribution (Fawcett 2006, 21,726 cites), "
    "yet our AUC fell 0.988 → 0.909.",
], top=4.35, size=13, bottom=6.4)
takeaway(s, "Most cross-device papers report one number and blame the model. The split is computable.", top=6.45)

# ══════════════════════════════════════════════════ 4 — the failed prediction (diagram)
s = slide()
title(s, "We tested our own proposal before building it",
      "Theory reduced multi-view aggregation to one measurable quantity, so it could be falsified in minutes.")

# formula strip
box(s, 0.9, 1.55, 5.3, 0.62, "AUC_K  =  Φ( z₁ · √( K / (1 + (K−1)ρ) ) )", RGBColor(0xEC, 0xEF, 0xF4), INK, 14)
label(s, 6.35, 1.62, 6.2, "ρ = within-patient score correlation.\nMeasured, not assumed: ρ = 0.363",
      12, INK, False, PP_ALIGN.LEFT)

# predicted vs actual
label(s, 0.9, 2.55, 3.0, "PREDICTED", 13, MUTED, True)
label(s, 5.6, 2.55, 3.0, "MEASURED", 13, MUTED, True)
rows = [("Diabetic retinopathy\n4 images / patient", "0.968", "0.842", WARN),
        ("Diabetic retinopathy\n2 images / eye",     "0.940", "0.883", WARN),
        ("Macular edema\nmax over 4 images",         "—",     "0.969", GOOD)]
y = 3.05
for name, pred, act, col in rows:
    label(s, 0.65, y + 0.05, 3.4, name, 11.5, INK, False, PP_ALIGN.LEFT)
    box(s, 4.15, y, 1.5, 0.6, pred, RGBColor(0xD8, 0xDD, 0xE6), INK, 14)
    arrow(s, 5.85, y + 0.19, 0.7)
    box(s, 6.75, y, 1.5, 0.6, act, col, WHITE, 14)
    label(s, 8.5, y + 0.05, 4.2,
          "prediction failed" if col is WARN else "worked — and theory says why",
          12, col, True, PP_ALIGN.LEFT)
    y += 0.78

bullets(s, [
    "Why: 14.5% of patients have one eye affected and one clear, so averaging pulls the diseased eye "
    "toward the healthy one. Mean-pooling is provably wrong for heterogeneous groups.",
    "Macular edema is focal — it shows strongly in one image — so a maximum is the correct operator, "
    "and it delivers +0.036 AUC.",
], top=5.5, size=13, bottom=6.5)
takeaway(s, "Reported, not buried. The prediction was wrong, the mechanism is understood, and it cost minutes.",
         WARN, top=6.55)

# ══════════════════════════════════════════════════ 5 — gap + proposal
s = slide()
title(s, "Where the gap is, and what we propose")

label(s, 0.65, 1.4, 12.1, "Nearest existing work — and what each leaves open", 14, ACCENT, True, PP_ALIGN.LEFT)
table(s, [
    ["Work", "Does", "Leaves open"],
    ["RetSyn — J. Biomed. Inform. 2025\n(same senior author as both datasets)",
     "Synthetic data for tabletop → portable",
     "Does not diagnose what breaks; no macular\nedema; no decision-cutoff treatment"],
    ["IEEE ISBI 2026", "Reports the BRSET → mBRSET drop",
     "Descriptive — not what the drop is made of,\nnor how to close it"],
], top=1.85, col_w=[3.6, 3.6, 4.9], font=11, height=1.5)

label(s, 0.65, 3.6, 12.1, "Three openings the review did not find addressed", 14, ACCENT, True, PP_ALIGN.LEFT)
y = 4.05
for n, t in [("1", "Decomposing a cross-device gap into prevalence, calibration and ranking — and proving "
                   "calibration is exhausted"),
             ("2", "Label-shift correction for MULTI-LABEL sigmoid outputs. BBSE (ICML'18) and MLLS (ICML'20, "
                   "NeurIPS'20) are all single-label. DR and ME co-occur, so per-label estimation is misspecified"),
             ("3", "The intersection where both published families break their own assumptions — one assumes "
                   "prevalence is fixed (ours changes 6.6% → 23.3%), the other assumes p(x|y) is fixed (our AUC fell)")]:
    box(s, 0.65, y, 0.42, 0.42, n, ACCENT, size=13)
    label(s, 1.22, y - 0.02, 11.5, t, 12.5, INK, False, PP_ALIGN.LEFT)
    y += 0.68

takeaway(s, "Next step: fundus degradation augmentation — the only intervention with published evidence of raising "
            "AUC here — with the decomposition as the scientific core.", GOOD, top=6.35)

# ══════════════════════════════════════════════════ 6 — references
s = slide()
title(s, "References", "Peer-reviewed versions. Counts: Semantic Scholar, 7 Aug 2026.")
refs = [
 "Tellez et al. Quantifying the effects of data augmentation and stain color normalization in CNNs for computational pathology. Medical Image Analysis 58:101544, 2019. 656 cites. doi.org/10.1016/j.media.2019.101544",
 "Che, Cheng, Jin, Chen. Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains. MICCAI 2023, 430–440. 46 cites. doi.org/10.1007/978-3-031-43904-9_42",
 "Lin, Shi, Zhang, Shang, He, Ge. Camera Adaptation for Fundus-Image-Based CVD Risk Estimation. MICCAI 2022, 593–603. 5 cites. doi.org/10.1007/978-3-031-16434-7_57",
 "Gulrajani, Lopez-Paz. In Search of Lost Domain Generalization. ICLR 2021. 1,504 cites. openreview.net/forum?id=lQdXeXDoWtI",
 "Boudiaf, Mueller, Ben Ayed, Bertinetto. Parameter-free Online Test-time Adaptation. CVPR 2022, 8344–8353. 250 cites. doi.org/10.1109/CVPR52688.2022.00816",
 "Roschewitz, Khara, Yearsley et al. Automatic correction of performance drift under acquisition shift in medical image classification. Nature Communications 14:6608, 2023. 42 cites. doi.org/10.1038/s41467-023-42396-y",
 "Godau, Kalinowski, Christodoulou et al. Navigating prevalence shifts in image analysis algorithm deployment. Medical Image Analysis 102:103504, 2025 (MICCAI 2023). doi.org/10.1016/j.media.2025.103504",
 "Lipton, Wang, Smola. Detecting and Correcting for Label Shift with Black Box Predictors. ICML 2018. 679 cites. proceedings.mlr.press/v80/lipton18a.html",
 "Alexandari, Kundaje, Shrikumar. Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation. ICML 2020, 222–232. proceedings.mlr.press/v119/alexandari20a.html",
 "Garg, Wu, Balakrishnan, Lipton. A Unified View of Label Shift Estimation. NeurIPS 2020. 179 cites.",
 "Fawcett. An introduction to ROC analysis. Pattern Recognition Letters 27(8):861–874, 2006. 21,726 cites. doi.org/10.1016/j.patrec.2005.10.010",
 "Wu, Jastrzębski, Park, Moy, Cho, Geras. Improving the Ability of DNNs to Use Information from Multiple Views in Breast Cancer Screening. MIDL 2020, PMLR 121:827–842.",
 "Ilse, Tomczak, Welling. Attention-based Deep Multiple Instance Learning. ICML 2018. 2,691 cites.",
 "Shuai, Wu, Tang, Restrepo, Morley, Nakayama. Enhancing AI-based diabetic retinopathy screening in LMICs with synthetic data. J. Biomedical Informatics 172:104938, 2025. doi.org/10.1016/j.jbi.2025.104938",
 "Nakayama, Restrepo et al. BRSET. PLOS Digital Health 3(7):e0000454, 2024.  |  Wu, Restrepo, Nakayama et al. A portable retina fundus photos dataset (mBRSET). Scientific Data 12:340, 2025.",
]
b = s.shapes.add_textbox(Inches(0.65), Inches(1.45), Inches(12.1), Inches(5.6))
tf = b.text_frame; tf.word_wrap = True
for i, r in enumerate(refs):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = f"[{i+1}]  {r}"
    p.font.size = Pt(9.5); p.font.color.rgb = INK; p.space_after = Pt(3)

prs.save(OUT)
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
