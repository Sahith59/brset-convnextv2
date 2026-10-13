"""Advisor deck, 13 October 2026: the BRSET -> mBRSET project end to end.

Every number on a slide is copied from a saved result file named in that slide's
source line. Run on a compute node (python-pptx + Pillow only, no model loaded).
After building, a conservative text-fit check is run: each text box and table is
re-wrapped with DejaVu Serif, which is wider than Times New Roman, so anything
that fits here fits in PowerPoint.
"""
from pathlib import Path

from PIL import ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).parent
OUT = HERE / "BRSET_mBRSET_Dong_Update_2026-10-13.pptx"

FONT = "Times New Roman"
NAVY = RGBColor(23, 42, 64)
TEAL = RGBColor(0, 111, 115)
INK = RGBColor(42, 49, 58)
GREY = RGBColor(101, 111, 121)
PALE = RGBColor(236, 242, 245)
PALE_TEAL = RGBColor(226, 241, 239)
AMBER = RGBColor(169, 103, 27)
PALE_AMBER = RGBColor(251, 242, 226)
RED = RGBColor(157, 64, 54)
PALE_RED = RGBColor(250, 234, 230)
GREEN = RGBColor(46, 125, 79)
PALE_GREEN = RGBColor(230, 243, 234)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(214, 223, 228)
ICE = RGBColor(202, 220, 235)

SW, SH = 13.333, 7.5
FIT = []          # (kind, slide_no, payload) for the post-build fit check


# --------------------------------------------------------------- primitives
def _font(run, size, bold=False, color=INK, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def text(slide, x, y, w, h, paras, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, margin=0.06, space_after=4, line=1.08, name=None):
    """paras: str | list of str | list of list of (text, dict) runs."""
    if isinstance(paras, str):
        paras = [paras]
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = anchor
    flat = []
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after if i < len(paras) - 1 else 0)
        p.line_spacing = line
        runs = [(para, {})] if isinstance(para, str) else para
        ptxt = []
        for t, o in runs:
            r = p.add_run()
            r.text = t
            _font(r, o.get("size", size), o.get("bold", bold), o.get("color", color), o.get("italic", False))
            ptxt.append((t, o.get("size", size), o.get("bold", bold)))
        flat.append(ptxt)
    FIT.append(("text", _CUR[0], dict(x=x, y=y, w=w, h=h, margin=margin, paras=flat,
                                       space_after=space_after, line=line, name=name or "")))
    return box


def rect(slide, x, y, w, h, fill=PALE, line=None, rounded=True, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.9)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def circle(slide, x, y, d, fill, label, size=16, color=WHITE):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    _font(r, size, True, color)
    return c


def arrow(slide, x1, y1, x2, y2, color=GREY, width=2.0):
    a = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    a.line.color.rgb = color
    a.line.width = Pt(width)
    ln = a.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd",
                          {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return a


def card(slide, x, y, w, h, head, body, fill=PALE, head_color=TEAL, size=15, head_size=17):
    rect(slide, x, y, w, h, fill)
    text(slide, x + 0.15, y + 0.12, w - 0.3, 0.45, head, size=head_size, bold=True, color=head_color)
    text(slide, x + 0.15, y + 0.58, w - 0.3, h - 0.68, body, size=size)


def stat(slide, x, y, w, number, label, color=TEAL, num_size=34, label_size=14, h=1.5):
    text(slide, x, y, w, 0.72, number, size=num_size, bold=True, color=color)
    text(slide, x, y + 0.72, w, h - 0.72, label, size=label_size, color=INK)


def table(slide, x, y, w, widths, header, rows, size=14, row_h=0.42, head_fill=NAVY,
          fills=None, bold_cols=(), aligns=None, name=None):
    nrow, ncol = len(rows) + 1, len(header)
    gt = slide.shapes.add_table(nrow, ncol, Inches(x), Inches(y), Inches(w), Inches(row_h * nrow))
    if name:
        gt.name = name
    t = gt.table
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(cw)
    for i in range(nrow):
        t.rows[i].height = Inches(row_h)
    cells_txt = []
    for i in range(nrow):
        vals = header if i == 0 else rows[i - 1]
        row_txt = []
        for j in range(ncol):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.07)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            v = vals[j]
            style = {}
            if isinstance(v, tuple):
                v, style = v
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (aligns[j] if aligns else PP_ALIGN.LEFT)
            r = p.add_run()
            r.text = str(v)
            if i == 0:
                _font(r, size, True, WHITE)
                c.fill.solid()
                c.fill.fore_color.rgb = head_fill
            else:
                _font(r, size, style.get("bold", j in bold_cols), style.get("color", INK))
                c.fill.solid()
                c.fill.fore_color.rgb = style.get("fill", (fills[i - 1] if fills else (WHITE if i % 2 else PALE)))
            row_txt.append((str(v), size, i == 0 or style.get("bold", j in bold_cols)))
        cells_txt.append(row_txt)
    FIT.append(("table", _CUR[0], dict(x=x, y=y, widths=widths, row_h=row_h, cells=cells_txt, name=name or "")))
    return gt


def chart_style(ch, size=13):
    ch.font.name = FONT
    ch.font.size = Pt(size)
    ch.font.color.rgb = INK
    ch.has_legend = False


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


_CUR = [0]


def base(prs, kicker, title, source=None, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _CUR[0] = len(prs.slides)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY if dark else WHITE
    if not dark:
        text(s, 0.6, 0.28, 10.5, 0.36, kicker.upper(), size=12, bold=True, color=TEAL, name="kicker")
        text(s, 0.6, 0.6, 12.1, 0.8, title, size=30, bold=True, color=NAVY, name="title")
        if source:
            text(s, 0.6, 6.98, 11.0, 0.36, "Source: " + source, size=10, color=GREY, name="source")
        text(s, 12.15, 6.98, 0.6, 0.36, str(_CUR[0]), size=10, color=GREY, align=PP_ALIGN.RIGHT, name="pageno")
    return s


# --------------------------------------------------------------- slides
def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)

    # 1 ---------------------------------------------------------------- title
    s = base(prs, "", "", dark=True)
    text(s, 0.7, 0.75, 7.4, 0.4, "BRSET → mBRSET  ·  CROSS-CAMERA DIABETIC RETINOPATHY SCREENING",
         size=13, bold=True, color=ICE)
    text(s, 0.7, 1.3, 7.2, 2.3, "What limits cross-camera DR screening, and a generative fix",
         size=40, bold=True, color=WHITE, line=1.0)
    text(s, 0.7, 3.85, 7.0, 0.5, "Research update for Dr. Dong Hye Ye  ·  13 October 2026", size=18, color=ICE)
    text(s, 0.7, 4.55, 6.9, 1.4,
         ["From architectures, to a diagnosis of the gap, to a 105-million-parameter "
          "diffusion renderer coupled to a physics camera model.",
          "Every number traces to a saved result file named at the foot of its slide."],
         size=15, color=WHITE)
    pic = s.shapes.add_picture(str(HERE / "synth_grid.png"), Inches(8.35), Inches(1.15), width=Inches(4.3))
    text(s, 8.35, 5.55, 4.3, 0.75,
         ["BRSET original  ·  ours, synthetic mBRSET  ·  real mBRSET (different eye)",
          "Rows: ICDR grade 1, 3, 4"], size=11, color=ICE)
    notes(s, "This deck covers the whole project: the architecture phase, the controlled "
             "baselines from September, your 8 October review, the diagnosis that the gap is about "
             "mild disease, every mechanism that failed, and the new generative method. The last "
             "five slides are the new method in detail. The picture on the right is real output: "
             "left column BRSET photos, middle column the same eyes rendered by our generator as the "
             "phone camera, right column real mBRSET photos of other eyes.")

    # 2 ---------------------------------------------------------------- summary
    s = base(prs, "Executive summary", "Four things to take away",
             "results/lebase_b*_s*/metrics_test.json; results/data_forensics.json; results/gates_g40000_*.json")
    cw, chh = 5.9, 2.55
    pos = [(0.6, 1.55), (6.83, 1.55), (0.6, 4.25), (6.83, 4.25)]
    items = [
        ("1  The gap", "0.9007 → 0.9463",
         "DR AUROC on the mBRSET test set: BRSET-only training vs adding all 3,402 "
         "mBRSET labels (3 seeds each).", TEAL, PALE),
        ("2  Where it comes from", "42 of 42",
         "false-negative test images are mild (ICDR 1–2); none are grade 3–4. Grade-1 "
         "AUROC 0.77 vs grade-4 1.00.", AMBER, PALE_AMBER),
        ("3  What did not work", "15 mechanisms",
         "architectures, sampling, fitted augmentation, pruning, a bounded translator, "
         "reweighting, an ordinal head, external data — no reliable gain.", RED, PALE_RED),
        ("4  New method, running now", "105 M parameters",
         "diffusion renderer + physics stage. Translated BRSET moves toward the phone camera "
         "and keeps the disease; joint training results due this morning.", GREEN, PALE_GREEN),
    ]
    for (x, y), (head, big, body, col, fill) in zip(pos, items):
        rect(s, x, y, cw, chh, fill)
        text(s, x + 0.2, y + 0.12, cw - 0.4, 0.42, head, size=16, bold=True, color=col)
        text(s, x + 0.2, y + 0.55, cw - 0.4, 0.7, big, size=30, bold=True, color=NAVY)
        text(s, x + 0.2, y + 1.3, cw - 0.4, 1.15, body, size=15)
    notes(s, "Headline: we now understand the gap, and we have the method you asked for. "
             "One, adding real mBRSET labels lifts DR AUROC from 0.9007 to about 0.946. "
             "Two, the errors are concentrated in mild disease: every one of the 42 false-negative "
             "test images is grade 1 or 2. Three, fifteen mechanisms that only rearranged BRSET "
             "data failed. Four, the new generative translator passes both of its quality checks "
             "and the joint-training experiment is running.")

    # 3 ---------------------------------------------------------------- data
    s = base(prs, "Problem and data", "Same disease, two cameras, no shared patients",
             "dataset label files; data/finetune_joint splits; results/convnextv2_large_BRSET_strong_baseline, "
             "lebase_b0_s*, brset_mbrset_joint + joint_seed1/2")
    card(s, 0.6, 1.55, 3.85, 2.45, "BRSET  ·  tabletop camera",
         ["16,266 images, hospital clinics", "Train 11,372 images", "DR-positive 6.58 % of train"],
         fill=PALE, head_color=NAVY)
    card(s, 0.6, 4.2, 3.85, 2.6, "mBRSET  ·  handheld phone",
         ["5,164 images, community screening", "Train 3,402 · val 725 · test 732",
          "Test: 193 patients, 159 DR+ images", "DR-positive 23.4 % of train"],
         fill=PALE_TEAL, head_color=TEAL)
    text(s, 4.75, 1.55, 3.0, 0.4, "Task", size=17, bold=True, color=NAVY)
    text(s, 4.75, 1.98, 3.0, 2.2,
         ["Image-level referable decision for any DR (ICDR ≥ 1) and macular oedema.",
          "ConvNeXt V2 Large, 512 px."], size=15)
    cd = CategoryChartData()
    cd.categories = ["BRSET → BRSET", "BRSET + all mBRSET", "BRSET → mBRSET"]
    cd.add_series("DR AUROC", (0.9874, 0.9463, 0.9007))
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(7.9), Inches(1.5), Inches(4.85), Inches(4.0), cd)
    ch = gf.chart
    chart_style(ch)
    ch.has_title = True
    ch.chart_title.text_frame.text = "DR AUROC, mBRSET test (BRSET test for the top bar)"
    _font(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, NAVY)
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.000"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size = Pt(13)
    for i, col in enumerate((GREY, TEAL, AMBER)):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = col
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.85, 1.0
    va.has_major_gridlines = False
    va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    va.format.line.fill.background()
    ch.category_axis.reverse_order = True
    ch.category_axis.tick_labels.font.size = Pt(13)
    text(s, 7.9, 5.6, 4.85, 1.2,
         ["The question: how close can we get to the middle bar without paying for "
          "3,402 phone-camera labels?"], size=15, color=NAVY, bold=True)
    notes(s, "BRSET is the large, high-quality tabletop dataset; mBRSET is the smaller handheld "
             "phone dataset with three and a half times the disease prevalence. The bar chart has "
             "the three reference points: 0.9874 within BRSET, 0.9463 when all mBRSET labels are "
             "added (three seeds), and 0.9007 zero-shot (three seeds). Note the test set has 732 "
             "images from 193 patients, so I always say images, not patients.")

    # 4 ---------------------------------------------------------------- timeline
    s = base(prs, "Timeline", "Five phases, each one narrowing the question",
             "Claude memory files; research/codex/PLAN.md and DECISIONS.md; report/WORKING_MEMORY.md")
    phases = [
        ("Aug – 3 Sep", "Architectures", "Setup A/B, Experiments 0–2: domain separation, routing. "
         "Result: saturated.", GREY),
        ("10 – 16 Sep", "Controlled protocol", "Steps 1–5: joint training wins; augmentation, "
         "pruning and pooling fail.", NAVY),
        ("8 Oct", "Your review", "“Preprocessing … class project.” Ask: synthesis, joint training, "
         "generative world model.", AMBER),
        ("9 – 11 Oct", "First translator + diagnosis", "Bounded translator too small to move images. "
         "Gap is mild disease.", RED),
        ("12 – 13 Oct", "Generative renderer", "105 M diffusion + physics. Passes both gates; joint "
         "training running.", GREEN),
    ]
    x0, wph = 0.6, 2.42
    KEYS = [("0.8455 vs 0.8456", "F1: pooled vs best target-only"),
            ("+0.0233", "DR F1, joint vs target-only"),
            ("3 asks", "synthesis · joint · world model"),
            ("42 of 42", "false negatives are mild"),
            ("0.968", "disease AUROC kept (0.987 max)")]
    for i, (when, what, body, col) in enumerate(phases):
        x = x0 + i * (wph + 0.11)
        circle(s, x + wph / 2 - 0.3, 1.6, 0.6, col, str(i + 1), size=18)
        if i < 4:
            arrow(s, x + wph / 2 + 0.36, 1.9, x + wph + 0.11 + wph / 2 - 0.36, 1.9, color=LINE, width=2.5)
        rect(s, x, 2.45, wph, 2.95, PALE)
        text(s, x + 0.12, 2.58, wph - 0.24, 0.4, when, size=14, bold=True, color=col)
        text(s, x + 0.12, 3.0, wph - 0.24, 0.85, what, size=18, bold=True, color=NAVY, line=1.0)
        text(s, x + 0.12, 3.9, wph - 0.24, 1.45, body, size=15)
        big, lab = KEYS[i]
        rect(s, x, 5.55, wph, 1.2, WHITE, line=LINE)
        text(s, x + 0.12, 5.6, wph - 0.24, 0.5, big, size=20, bold=True, color=col)
        text(s, x + 0.12, 6.1, wph - 0.24, 0.6, lab, size=12, color=GREY)
    notes(s, "The project went through five phases. Each phase ruled something out and narrowed "
             "the question. The key turning points were your review on 8 October and the diagnosis "
             "on 10–11 October that the remaining errors are mild disease.")

    # 5 ---------------------------------------------------------------- era 1
    s = base(prs, "Phase 1 · architectures", "Architecture changes did not separate from joint training",
             "Claude memory experiment-numbering.md and where-the-project-stands.md (legacy trainer, legacy test split, DR F1)")
    table(s, 0.6, 1.55, 7.6, [3.6, 1.8, 2.2],
          ["Setting", "DR F1", "False-negative images (of 159)"],
          [["Setup A · BRSET only (zero-shot)", "0.7842", "50"],
           ["Exp 0 · joint BRSET + mBRSET", "0.8250", "27"],
           ["Exp 1 · shared/private encoders (γ 0.05 / 0)", "0.8227 / 0.8232", "36 / 31"],
           ["Exp 2 · entropy-gated routing", "0.8452", "28 (gate collapsed, α = 1)"]],
          size=14, row_h=0.62, aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    text(s, 0.6, 4.95, 7.6, 1.8,
         ["Seed-to-seed spread of DR F1 was 0.0199 (about 9 images). Domain separation and "
          "aspect-ratio correction (+0.0020) stayed inside it; routing sat at its edge (+0.0202) "
          "with its gate collapsed to a single path."], size=15)
    rect(s, 8.6, 1.55, 4.15, 5.2, PALE_AMBER)
    text(s, 8.85, 1.75, 3.7, 0.4, "Why nothing could win", size=17, bold=True, color=AMBER)
    stat(s, 8.85, 2.3, 3.7, "0.8455 vs 0.8456", "DR F1: pooled training vs the best model trained "
         "only on mBRSET, at matched best cut-off.", color=NAVY, num_size=28, h=1.8)
    text(s, 8.85, 4.2, 3.7, 2.4,
         ["Pooling already equalled the target-trained ceiling, so no architecture had room "
          "to improve on it.", "Lesson: the lever is data, not architecture."], size=15)
    notes(s, "In the first phase we tried shared/private encoders and your routing idea. None of "
             "them separated from joint training by more than the seed spread. The measurement that "
             "explained it: pooled training already matched the best model trained only on mBRSET, "
             "0.8455 versus 0.8456 F1. These are older legacy-trainer F1 numbers, so they are not "
             "directly comparable to the AUROC numbers later in the deck.")

    # 6 ---------------------------------------------------------------- step 2
    s = base(prs, "Phase 2 · controlled baselines (Step 2)", "Joint training beats target-only in every seed",
             "research/codex/step2/full_pool/STEP_2_FINAL_REVIEW.md; step5/STEP5_DIAGNOSTIC_REVIEW.md "
             "(test assessment, 732 images)")
    table(s, 0.6, 1.55, 8.0, [2.9, 1.25, 1.3, 1.25, 1.3],
          ["Arm (3 seeds, mean ± SD)", "DR F1", "DR AUROC", "ME F1", "ME AUROC"],
          [["B0 · mBRSET only", "0.8105 ± .014", "0.9395", "0.8240 ± .030", "0.9835"],
           [("B1 · joint BRSET + mBRSET", {"bold": True}), ("0.8338 ± .005", {"bold": True}),
            ("0.9467", {"bold": True}), ("0.8619 ± .042", {"bold": True}), ("0.9945", {"bold": True})],
           ["B2 · BRSET then mBRSET", "0.8139 ± .010", "0.9416", "0.8184 ± .033", "0.9882"]],
          size=14, row_h=0.6, fills=[WHITE, PALE_TEAL, WHITE],
          aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
    text(s, 0.6, 4.25, 8.0, 0.4, "Strict source-only (BRSET only, locked assessment)", size=16, bold=True, color=NAVY)
    table(s, 0.6, 4.7, 8.0, [2.9, 1.25, 1.3, 1.25, 1.3],
          ["Label", "F1", "AUROC", "Sensitivity", "FN images"],
          [["DR", "0.7218", "0.9100", "0.60", "63 / 159"],
           ["ME", "0.6972", "0.9734", "0.60", "25 / 63"]],
          size=14, row_h=0.5, aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
    rect(s, 9.0, 1.55, 3.75, 5.2, PALE_TEAL)
    stat(s, 9.25, 1.8, 3.3, "+0.0233", "DR F1, B1 − B0, mean of 3 seeds (positive in all 3)", color=TEAL)
    stat(s, 9.25, 3.45, 3.3, "+0.0379", "ME F1, B1 − B0, mean of 3 seeds (positive in all 3)", color=TEAL)
    text(s, 9.25, 5.1, 3.3, 1.5, "Your joint-training suggestion is the strongest ordinary baseline.",
         size=15, bold=True, color=NAVY)
    notes(s, "Step 2 re-ran the three basic strategies under one frozen protocol: same model, same "
             "5,775 updates, choices on validation, one locked test assessment. Joint training, B1, "
             "won in every seed for both labels. The source-only model has high specificity but only "
             "0.60 sensitivity: it misses 63 of the 159 DR-positive test images.")

    # 7 ---------------------------------------------------------------- steps 3-5
    s = base(prs, "Phase 2 · mechanisms on top of joint training (Steps 3–5)",
             "Appearance matching shrank the distance, not the error",
             "research/codex/step3/STEP_3_FINAL_REVIEW.md, step4/SEED0_RESULTS_REVIEW.md, "
             "step5/STEP5_DIAGNOSTIC_REVIEW.md (validation, 725 images; change vs joint B1)")
    R_ = {"color": RED, "bold": True}
    table(s, 0.6, 1.55, 12.15, [3.3, 3.4, 3.65, 1.8],
          ["Mechanism", "What it tests", "Change vs joint baseline", "Verdict"],
          [["Equal-domain sampling (3 seeds)", "More phone-camera draws", "DR F1 +0.0035, DR AUROC −0.0071", ("No gain", R_)],
           ["FundusAug artifacts (A1)", "Published fundus augmentation", "DR F1 +0.0069, ME F1 −0.0028", ("Fail", R_)],
           ["Fitted target transform (A2)", "Match phone appearance", "DR F1 +0.0036, ME F1 −0.0004", ("Fail", R_)],
           ["Fitted, no overlays (A3)", "Same, cleaner", "DR F1 +0.0062, DR AUROC +0.0038", ("Fail", R_)],
           ["Channel pruning (GFP)", "Remove harmful features", "ΔF1 0.0000 for DR and ME", ("Fail", R_)],
           ["Top-1 local pooling", "Lesion-scale evidence", "DR AUROC +0.0101, DR F1 −0.0012", ("Fail", R_)]],
          size=14, row_h=0.56, aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
    rect(s, 0.6, 5.7, 12.15, 0.95, PALE_AMBER)
    text(s, 0.8, 5.78, 11.8, 0.82,
         [[("Pre-registered gate: ", {"bold": True, "color": AMBER}),
           ("+0.01 F1 on one label, no label worse than −0.01. None passed. The fitted transforms "
            "cut the appearance distance strongly but did not improve diagnosis.", {})]], size=15)
    notes(s, "On top of joint training we tested sampling balance, three appearance augmentations, "
             "feature pruning and local pooling, each against a gate written in advance. None passed. "
             "The important lesson: making BRSET look statistically like mBRSET did not make the "
             "classifier better on mBRSET.")

    # 8 ---------------------------------------------------------------- label efficiency
    s = base(prs, "What does work · real phone-camera labels", "250 real labels already help; ~500 approach the all-label level",
             "results/lebase_b{0,250,500,1000,2000}_s{0,1,2} and brset_mbrset_joint, joint_seed1/2: metrics_test.json "
             "(legacy trainer, DR AUROC, 3 seeds each)")
    cd = CategoryChartData()
    cd.categories = ["0", "250", "500", "1,000", "2,000", "3,402 (all)"]
    cd.add_series("DR AUROC", (0.9007, 0.9219, 0.9360, 0.9384, 0.9475, 0.9463))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(8.1), Inches(5.2), cd)
    ch = gf.chart
    chart_style(ch)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Real mBRSET labels added to BRSET training → DR AUROC on mBRSET test"
    _font(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, NAVY)
    pl = ch.plots[0]
    pl.gap_width = 55
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.000"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size = Pt(13)
    for i in range(6):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = AMBER if i == 0 else (NAVY if i == 5 else TEAL)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.88, 0.96
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.tick_labels.font.size = Pt(12)
    va.tick_labels.number_format = "0.00"
    va.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(13)
    stat(s, 9.1, 1.6, 3.65, "+0.0212", "DR AUROC from only 250 real labels (0.9007 → 0.9219)", color=TEAL)
    stat(s, 9.1, 3.25, 3.65, "≈ 500", "labels reach the all-label level within seed spread "
         "(0.9360 vs 0.9463)", color=TEAL)
    text(s, 9.1, 4.95, 3.65, 1.75,
         ["Seed spread at 0 labels: 0.8906 – 0.9104.",
          "2,000 labels (0.9475) matches, does not beat, all labels."], size=14, color=GREY)
    notes(s, "This is the one thing that consistently worked: real labelled phone-camera images. "
             "250 labels add 0.021 AUROC, and around 500 labels are within seed spread of using "
             "all 3,402. This curve is the baseline the new synthetic-image experiment is measured "
             "against: can synthetic images replace some of these real labels?")

    # 9 ---------------------------------------------------------------- Dong review
    s = base(prs, "Your review, 8 October", "What you asked for, and what we built",
             "your 8 October conclusion as relayed; research/codex/DECISIONS.md D065–D069")
    rect(s, 0.6, 1.55, 12.15, 1.45, PALE_AMBER)
    text(s, 0.85, 1.68, 11.7, 1.2,
         [[("“Image synthesis converting BRSET to synthesized mBRSET, and also have the joint training, "
            "and make use of the generative world model … everything we did till now sounds like "
            "preprocessing, not a novel thing. Good for a class project, not as a conference submission.”",
            {"italic": True})]], size=16, color=INK)
    asks = [("Image synthesis BRSET → mBRSET", "Built",
             "104.7 M-parameter diffusion renderer, trained on real mBRSET photos, coupled to a "
             "19-parameter physics camera model.", GREEN),
            ("Joint training with the synthetic images", "Running",
             "12 runs: 2 generators × (0 or 250 real labels) × 3 seeds, on 12 A100s. Results ≈ 06:00.", AMBER),
            ("Generative world model", "Built, in its device-conditioned form",
             "gen_both: one model that renders the same retina as either camera, switched by a device "
             "token. Question for you: is this the form you meant?", TEAL)]
    for i, (ask, status, body, col) in enumerate(asks):
        x = 0.6 + i * 4.1
        rect(s, x, 3.25, 3.95, 3.5, PALE)
        text(s, x + 0.18, 3.38, 3.6, 0.75, ask, size=17, bold=True, color=NAVY, line=1.0)
        text(s, x + 0.18, 4.15, 3.6, 0.45, status, size=16, bold=True, color=col)
        text(s, x + 0.18, 4.62, 3.6, 2.05, body, size=15)
    notes(s, "Your three asks, and where each stands. Synthesis: built. Joint training with the "
             "synthetic images: running now. World model: the gen_both variant is one generative "
             "model that renders the same retina as either camera. I would like your view on "
             "whether that is the world-model form you had in mind, or whether you meant a "
             "latent-prediction model like CheXWorld.")

    # 10 --------------------------------------------------------------- bounded translator
    s = base(prs, "First synthesis attempt, 9–11 October", "The first translator was too small to move the images",
             "scripts/72_physical_layer.py, 82_refiner.py (parameter counts measured); 90_diagnose_translator.py; "
             "results/synth_b0_s0*/metrics_test.json (single seed)")
    stat(s, 0.6, 1.5, 3.95, "68,346", "parameters in the old learned stage; each pixel clamped to ±0.15 "
         "(smooth) and ±0.02 (fine detail)", color=RED, h=1.7)
    stat(s, 0.6, 3.25, 3.95, "1.0000 → 1.0000", "domain-probe AUC before vs after translation, all three "
         "variants (two adversarial)", color=RED, num_size=30, h=1.7)
    stat(s, 0.6, 5.0, 3.95, "104.7 M", "parameters in the new diffusion stage: about 1,500× larger, "
         "not clamped", color=GREEN, h=1.7)
    text(s, 4.95, 1.55, 7.8, 0.4, "Training with those images (single seed)", size=17, bold=True, color=NAVY)
    table(s, 4.95, 2.0, 7.8, [4.4, 1.6, 1.8],
          ["Arm", "DR AUROC", "vs 0.9007"],
          [["25 % of BRSET images translated", "0.8996", "−0.0011"],
           ["50 % translated", "0.9044", "+0.0037"],
           ["100 % translated", "0.8888", "−0.0119"],
           ["Real + translated copy (2× epoch)", "0.8652", "−0.0355"]],
          size=14, row_h=0.52, aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    rect(s, 4.95, 4.85, 7.8, 1.9, PALE_AMBER)
    text(s, 5.15, 4.95, 7.4, 1.7,
         [[("Correction: ", {"bold": True, "color": AMBER}),
           ("we earlier said this showed synthesis cannot work. It showed only that a generator "
            "this constrained cannot change the camera look. That claim is withdrawn; its "
            "discriminator (166,753 parameters) was 2.4× the generator.", {})]], size=15)
    notes(s, "Our first attempt at your request used a 19-parameter physics model plus a tiny 68 "
             "thousand parameter refiner whose output was clamped. A probe could still tell the "
             "translated images from real phone images perfectly, before and after. I had "
             "concluded synthesis does not work; that was wrong — the generator simply could not "
             "move the images. That is what motivated building a real generative model.")

    # 11 --------------------------------------------------------------- per grade
    s = base(prs, "Diagnosis · inside the binary label", "Every missed image is mild disease",
             "results/data_forensics.json → 4_misses.per_grade (BRSET-only model, seed 0, overall AUROC 0.9104; "
             "mBRSET test images)")
    cd = CategoryChartData()
    cd.categories = ["Grade 1 (mild)", "Grade 2", "Grade 3", "Grade 4"]
    cd.add_series("AUROC vs healthy eyes", (0.7735, 0.9390, 0.9973, 0.9963))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(6.6), Inches(5.25), cd)
    ch = gf.chart
    chart_style(ch)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Zero-shot DR AUROC by ICDR grade (each grade vs all healthy test images)"
    _font(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, NAVY)
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.000"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size = Pt(13)
    for i, col in enumerate((AMBER, TEAL, TEAL, TEAL)):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = col
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.7, 1.0
    va.major_gridlines.format.line.color.rgb = LINE
    va.tick_labels.font.size = Pt(12)
    va.tick_labels.number_format = "0.0"
    va.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(13)
    table(s, 7.6, 1.55, 5.15, [1.6, 1.75, 1.8],
          ["ICDR grade", "Missed / total", "Recall: BRSET → joint"],
          [["1 (mild)", ("24 / 40", {"bold": True, "color": AMBER}), "0.40 → 0.575"],
           ["2", ("18 / 83", {"bold": True, "color": AMBER}), "0.78 → 0.89"],
           ["3", "0 / 9", "1.00 → 1.00"],
           ["4", "0 / 27", "1.00 → 1.00"]],
          size=14, row_h=0.5, aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    rect(s, 7.6, 4.3, 5.15, 2.45, PALE_AMBER)
    stat(s, 7.85, 4.45, 4.7, "42 of 42", "false-negative test images are grade 1–2. Adding real "
         "mBRSET labels helps mild cases most.", color=AMBER, num_size=32)
    notes(s, "Here we opened up the binary label. Both datasets carry the full 0 to 4 ICDR grade. "
             "The zero-shot model is near perfect on grades 3 and 4 and weak on grade 1: AUROC 0.77. "
             "All 42 false-negative images are grade 1 or 2. When real mBRSET labels are added, "
             "grade-1 recall rises from 0.40 to 0.575, the biggest improvement of any grade. These "
             "are image counts, not patients.")

    # 12 --------------------------------------------------------------- severity mix
    s = base(prs, "Diagnosis · why the binary label hid this", "BRSET disease skews severe; mBRSET disease skews mild",
             "results/data_forensics.json → 1_severity_composition, appearance ρ vs ICDR, "
             "mbrset_frac_eyes_discordant_on_binary_dr")
    cd = CategoryChartData()
    cd.categories = ["BRSET train", "mBRSET test"]
    for g, vals in (("Grade 1", (0.1546, 0.2516)), ("Grade 2", (0.3923, 0.5220)),
                    ("Grade 3", (0.0700, 0.0566)), ("Grade 4", (0.3831, 0.1698))):
        cd.add_series(g, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED_100, Inches(0.6), Inches(1.5), Inches(7.2), Inches(3.6), cd)
    ch = gf.chart
    chart_style(ch)
    ch.has_legend = True
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Severity of DR-positive images"
    _font(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, NAVY)
    pl = ch.plots[0]
    pl.gap_width = 45
    pl.has_data_labels = True
    pl.data_labels.number_format = "0%"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.CENTER
    pl.data_labels.font.size = Pt(12)
    pl.data_labels.font.color.rgb = WHITE
    for ser, col in zip(pl.series, (AMBER, RGBColor(196, 140, 70), RGBColor(70, 110, 140), NAVY)):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    ch.category_axis.tick_labels.font.size = Pt(13)
    ch.category_axis.reverse_order = True
    text(s, 0.6, 5.25, 7.2, 1.5,
         ["χ² = 30.2, p < 0.00001. Proliferative share of positives 38 % vs 17 %. "
          "In a yes/no label a grade-1 and a grade-4 eye look identical, so the model learns "
          "“disease” mostly from florid cases."], size=15)
    for i, (big, lab) in enumerate((("6.6 % vs 21.7 %", "DR prevalence, BRSET train vs mBRSET test"),
                                    ("|ρ| ≤ 0.11", "appearance statistics vs ICDR grade (9 statistics)"),
                                    ("5.0 %", "of mBRSET eyes get opposite yes/no DR labels on repeat photos "
                                     "(119 eyes) — a ceiling on accuracy"))):
        y = 1.55 + i * 1.75
        rect(s, 8.2, y, 4.55, 1.6, PALE)
        stat(s, 8.4, y + 0.08, 4.15, big, lab, color=NAVY, num_size=26, label_size=14)
    notes(s, "Why did a binary label hide this? Among DR-positive images, BRSET has 38 percent "
             "proliferative disease, mBRSET test only 17 percent; mBRSET is dominated by grades 1 "
             "and 2. A yes/no label gives a grade-1 and a grade-4 eye the same target, so the "
             "model's notion of disease is built mostly from florid cases. Image appearance "
             "statistics are essentially unrelated to grade, rho at most 0.11. And 5 percent of "
             "mBRSET eyes get opposite labels on repeat photographs, which caps achievable accuracy.")

    # 13 --------------------------------------------------------------- exploiting severity
    s = base(prs, "Acting on the diagnosis", "Rearranging BRSET cannot create mild cases it does not have",
             "cls results sevmatch_s*, osctl_b0_s*, ordaux_ord_s*_ns, ep50_b0_s*, extmild_*_ns, tgtlab_*: metrics_test.json "
             "(3 seeds; change in mean DR AUROC)")
    cd = CategoryChartData()
    labels = ["Ordinal 5-grade head", "50 epochs instead of 25", "Severity reweighting*",
              "+1,000 DDR severe images", "+1,000 DDR mixed images", "+1,000 DDR mild images"]
    vals = (0.0100, 0.0038, -0.0003, -0.0465, -0.1253, -0.1713)
    cd.categories = labels
    cd.add_series("Δ DR AUROC", vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.5), Inches(7.6), Inches(5.0), cd)
    ch = gf.chart
    chart_style(ch)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Change in DR AUROC vs BRSET-only (0.9007)"
    _font(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, NAVY)
    pl = ch.plots[0]
    pl.gap_width = 50
    pl.has_data_labels = True
    pl.data_labels.number_format = "+0.0000;−0.0000"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size = Pt(12)
    for i, v in enumerate(vals):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = RED if v < -0.02 else GREY
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = -0.2, 0.04
    va.has_major_gridlines = False
    va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    ch.category_axis.reverse_order = True
    ch.category_axis.tick_labels.font.size = Pt(13)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    text(s, 0.6, 6.5, 7.6, 0.4, "*vs its own oversampling control (0.9098), not vs 0.9007.", size=11, color=GREY)
    card(s, 8.6, 1.55, 4.15, 2.45, "Too few mild training images",
         ["BRSET train has 117 grade-1 images. Reweighting and an ordinal head re-use the same "
          "117 pictures: +0.0100 at best, inside seed spread."], fill=PALE, head_color=NAVY)
    card(s, 8.6, 4.2, 4.15, 2.55, "Other cameras' mild labels hurt",
         ["7,093 mild images exist in DDR/APTOS/Messidor-2. Adding 1,000 DDR mild images cost "
          "0.17 AUROC; severe images cost least — consistent with mild grading transferring "
          "worst between cameras and graders."],
         fill=PALE_RED, head_color=RED)
    notes(s, "We tried to act on the diagnosis without new phone labels. The ordinal head, longer "
             "training and severity reweighting all stayed within seed spread. Importing another "
             "dataset's mild images was the worst result in the project, and mild was worse than "
             "severe, which suggests mild-disease labels are the least transferable between cameras "
             "and graders. Conclusion: the mild-disease information has to come in phone-camera "
             "form, which is what synthesis can supply.")

    # 14 --------------------------------------------------------------- QC
    s = base(prs, "Quality control", "Five measurement problems we found and fixed",
             "report/LESSONS_LOG.md; research/codex/MEMORY.md and DECISIONS.md D067 (12–13 Oct entries)")
    qc = [("Lesion detector could not separate grades",
           "Healthy 799 candidates vs grade 4 714. Two lesion-retention claims withdrawn; replaced by a validated judge."),
          ("Sampler setting differed from the baseline",
           "60–65 % vs 6.58 % positives per batch. Nine arms re-run with matched settings."),
          ("Severity reweighting was silently disabled",
           "Results were bitwise identical to baseline. Re-tested with its proper control: −0.0003."),
          ("“Missed patients” wording",
           "Counts are images: test = 732 images from 193 patients. Corrected throughout."),
          ("Test split reused for choices",
           "From now on all choices are made on validation; test is reported once.")]
    for i, (head, body) in enumerate(qc):
        y = 1.55 + i * 1.04
        circle(s, 0.6, y + 0.12, 0.62, TEAL, str(i + 1), size=18)
        text(s, 1.45, y + 0.02, 4.6, 0.85, head, size=16, bold=True, color=NAVY, line=1.0,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, 6.2, y + 0.02, 6.55, 0.85, body, size=15, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "I want to be transparent about measurement problems we found ourselves. A lesion "
             "detector we relied on could not tell grade 4 from healthy eyes, so two claims were "
             "withdrawn. A sampler setting differed from the baseline, so nine arms were re-run. "
             "One flag was silently disabled, which we caught because three seeds matched the "
             "baseline to sixteen digits. And all choices now go through validation, with the test "
             "set reported once.")

    # 15 --------------------------------------------------------------- bridge
    s = base(prs, "What the evidence says", "From diagnosis to method",
             "slides 5–14")
    cols = [("Rearranging BRSET", "No reliable gain",
             "15 mechanisms: architectures, sampling, augmentation, pruning, pooling, bounded "
             "translator, reweighting, ordinal head, longer training, other datasets.", RED, PALE_RED),
            ("Adding phone-camera images with labels", "Consistent gain",
             "+0.0212 DR AUROC from 250 real labels; joint training +0.0233 DR F1 in every seed; "
             "mild-case recall 0.40 → 0.575.", GREEN, PALE_GREEN),
            ("Therefore", "Manufacture them",
             "Render BRSET eyes, with their verified labels, as phone-camera images — and check "
             "that the disease survives before training on them.", TEAL, PALE_TEAL)]
    for i, (head, verdict, body, col, fill) in enumerate(cols):
        x = 0.6 + i * 4.15
        rect(s, x, 1.6, 3.85, 3.85, fill)
        text(s, x + 0.2, 1.78, 3.45, 0.85, head, size=18, bold=True, color=NAVY, line=1.0)
        text(s, x + 0.2, 2.7, 3.45, 0.55, verdict, size=22, bold=True, color=col)
        text(s, x + 0.2, 3.4, 3.45, 1.95, body, size=15)
        if i < 2:
            arrow(s, x + 3.88, 3.5, x + 4.12, 3.5, color=GREY, width=2.5)
    rect(s, 0.6, 5.7, 12.15, 1.05, PALE_AMBER)
    text(s, 0.8, 5.78, 11.8, 0.9,
         [[("This is the synthesis you asked for on 8 October ", {"bold": True, "color": AMBER}),
           ("— now with a model large enough to change the camera look, and a check that the "
            "disease survives before any image is used for training.", {})]], size=15)
    notes(s, "Putting it together: everything that only rearranged BRSET failed; everything that "
             "added phone-camera images with correct labels helped. So the method follows: "
             "generate phone-camera versions of BRSET eyes, keep their labels, and verify the "
             "disease survived before using them. The next five slides are that method.")

    # 16 --------------------------------------------------------------- method overview
    s = base(prs, "New method · 1 of 5", "One retina, two camera renderings",
             "scripts/109_gen_common.py (structure_map, render), 111_translate.py; panel "
             "/data/users3/sthummala2/brset_gen/runs/gen_both/panels/step_0040000.png")
    boxes = [(0.6, "BRSET photo", "label known", PALE, NAVY),
             (3.05, "Stage 1 · physics", "19 parameters: colour, light, noise", PALE_AMBER, AMBER),
             (5.5, "Partial noise", "start at strength 0.4", PALE, GREY),
             (7.95, "Stage 2 · diffusion", "104.7 M parameters, 12 denoising steps", PALE_TEAL, TEAL),
             (10.4, "Synthetic mBRSET", "same eye, same label", PALE_GREEN, GREEN)]
    for x, head, sub, fill, col in boxes:
        rect(s, x, 1.65, 2.3, 1.35, fill)
        text(s, x + 0.08, 1.72, 2.14, 0.5, head, size=16, bold=True, color=col, align=PP_ALIGN.CENTER)
        text(s, x + 0.08, 2.22, 2.14, 0.75, sub, size=14, align=PP_ALIGN.CENTER)
    for x in (2.9, 5.35, 7.8, 10.25):
        arrow(s, x, 2.32, x + 0.15, 2.32, color=GREY, width=2.5)
    rect(s, 4.2, 3.35, 3.4, 0.95, PALE)
    text(s, 4.3, 3.4, 3.2, 0.85, ["Structure map of the source", "vessels · disc · lesions"], size=14,
         bold=False, color=NAVY, align=PP_ALIGN.CENTER, space_after=0)
    arrow(s, 1.65, 3.0, 4.2, 3.8, color=GREY, width=1.8)
    arrow(s, 7.6, 3.8, 9.1, 3.02, color=GREY, width=1.8)
    s.shapes.add_picture(str(HERE / "structure_strip.png"), Inches(0.6), Inches(4.45), height=Inches(2.1))
    text(s, 0.6, 6.57, 6.6, 0.32, "BRSET crop  ·  its structure map  ·  rendered as mBRSET (gen_both, step 40,000)",
         size=11, color=GREY)
    text(s, 7.45, 4.45, 5.3, 2.4,
         ["The label lives in the structure map, taken from the source image.",
          "The camera look is generated by a model that has only learned how real phone photos "
          "look.",
          "Physics sets the global colour and noise first, so diffusion can start from less noise "
          "and wash out fewer small lesions."], size=15)
    notes(s, "The idea in one sentence: the disease is in where things are, the camera is in how "
             "they look. We extract a structure map from the BRSET image — vessels, disc and lesions "
             "as local contrast — and a diffusion model trained on real mBRSET photos paints that "
             "structure in the phone camera's style. The physics stage from DuDoNet's dual-domain "
             "idea sets the global colour, illumination and noise first, so the diffusion stage "
             "starts from a partially noised version of that, rather than pure noise.")

    # 17 --------------------------------------------------------------- tech details
    s = base(prs, "New method · 2 of 5", "Technical specification",
             "scripts/109_gen_common.py, 110_train_diffusion.py, 111_translate.py; run args in "
             "/data/users3/sthummala2/brset_gen/runs/*/ckpt_last.pt")
    table(s, 0.6, 1.5, 12.15, [2.5, 9.65],
          ["Component", "Specification"],
          [["Structure map", "Green-channel local contrast s = tanh(0.5 (g − μ) / (σ + 0.03)), μ and σ from a "
            "10 px Gaussian window inside the retina mask; read from a blurred (≤ 1.8 px) and noised copy in training"],
           ["Denoiser", "U-Net, 104.7 M parameters, 5 levels (128–512 channels), no attention: trains on 256 px "
            "crops, runs on the full 560 px image"],
           ["Inputs", "Noisy RGB + structure map + field-of-view mask (5 channels); device token BRSET / mBRSET / none"],
           ["Diffusion", "1,000-step cosine schedule, v-prediction; DDIM sampling"],
           ["Guidance", "Structure and device token each dropped 10 % in training → classifier-free guidance, w = 1.5"],
           ["Coupling", "Sampling starts from the physics output noised to strength 0.4 (12 of 30 DDIM steps)"],
           ["Training", "Batch 96, learning rate 0.0002, 40,000 steps, EMA 0.9999; ≈ 5.5 h on 4 H100 or 8 A100"],
           ["Data", "Generator sees TRAIN images only and never an mBRSET label; checks use held-out BRSET"]],
          size=14, row_h=0.6, bold_cols=(0,))
    notes(s, "Technical summary for questions. The structure map is a signed local contrast of the "
             "green channel, so dark microaneurysms and bright exudates both survive. The denoiser "
             "is a 105-million-parameter U-Net without attention, so it trains on crops and runs on "
             "full images. Classifier-free guidance on the structure map is an explicit dial for "
             "how strictly lesions are enforced. The generator never sees an mBRSET label or any "
             "validation or test image.")

    # 18 --------------------------------------------------------------- verification
    s = base(prs, "New method · 3 of 5", "Three variants, two quality gates, a rule fixed in advance",
             "scripts/112_gates.py; results/gates_g*.json (held-out BRSET val + test, 546 images); "
             "research/codex/MEMORY.md 12 Oct 21:30 selection rule")
    table(s, 0.6, 1.5, 6.0, [1.7, 2.6, 1.7],
          ["Variant", "Learns from", "Note"],
          [["gen_tgt", "3,402 mBRSET photos", "one camera"],
           ["gen_both", "14,774 photos, both cameras", "device token"],
           ["gen_tgt_sharp", "3,402 mBRSET photos", "finer map"]],
          size=14, row_h=0.5, bold_cols=(0,))
    text(s, 0.6, 3.75, 6.0, 0.4, "Selection rule, written before the final checks", size=16, bold=True, color=NAVY)
    text(s, 0.6, 4.2, 6.0, 2.5,
         ["Highest grade-1 AUROC among settings where at least 40 % of images look like mBRSET; "
          "ties broken by less inflation of healthy-eye scores.",
          "Final choice between generators: mBRSET validation in joint training."], size=15)
    card(s, 7.0, 1.5, 5.75, 2.5, "Gate 1 · did the image change camera?",
         ["Classifier trained on real BRSET vs real mBRSET (disjoint images).",
          "Untouched BRSET 0.002 · real mBRSET 0.998."], fill=PALE_TEAL, head_color=TEAL)
    card(s, 7.0, 4.2, 5.75, 2.55, "Gate 2 · did the disease survive?",
         ["Joint-trained judge, per grade. Untouched AUROC 0.987 (grade 1: 0.981).",
          "Blurring lesions away drops it to 0.915 (0.685), so the judge is sensitive."],
         fill=PALE_AMBER, head_color=AMBER)
    notes(s, "We trained three variants. Each is checked by two gates on BRSET images that neither "
             "the generator nor the judge has seen. Gate 1 asks whether the image now looks like the "
             "phone camera; gate 2 asks whether a classifier can still find the disease, grade by "
             "grade. I validated the judge first with a control that blurs lesions away — it drops "
             "sharply — because an earlier lesion detector turned out to be blind to disease.")

    # 19 --------------------------------------------------------------- results
    s = base(prs, "New method · 4 of 5", "Synthetic images move toward mBRSET and keep the disease",
             "report/figures/look_gen_both_step40000.png; results/gates_g40000_* and gates_g20000_gen_tgt_sharp_*")
    s.shapes.add_picture(str(HERE / "synth_grid.png"), Inches(0.6), Inches(1.5), height=Inches(4.9))
    text(s, 0.6, 6.42, 4.9, 0.5, "BRSET  ·  ours  ·  real mBRSET (other eye)   rows: grade 1, 3, 4",
         size=11, color=GREY)
    table(s, 5.85, 1.5, 6.9, [2.6, 1.4, 1.45, 1.45],
          ["Images", "Look like mBRSET", "Disease AUROC", "Grade-1 AUROC"],
          [["Untouched BRSET", "0.002", "0.987", "0.981"],
           ["Lesions blurred (control)", "—", "0.915", "0.685"],
           [("gen_both, 40k", {"bold": True}), ("0.41", {"bold": True}), ("0.968", {"bold": True}),
            ("0.907", {"bold": True})],
           ["gen_tgt, 40k", "0.47", "0.963", "0.902"],
           ["gen_tgt_sharp, 20k", "0.44", "0.962", "0.902"]],
          size=14, row_h=0.5, fills=[PALE, PALE, PALE_GREEN, WHITE, WHITE],
          aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 3)
    text(s, 5.85, 4.65, 6.9, 2.2,
         [[("Works: ", {"bold": True, "color": GREEN}),
           ("40–47 % of translated images classified as mBRSET (vs 0.2 %). Physics + diffusion keeps "
            "more disease than diffusion-only in 12 of 12 matched pairs (grade 1: 10 of 12).", {})],
          [("Not yet: ", {"bold": True, "color": AMBER}),
           ("no glare or uneven lighting; grade-1 AUROC 0.07 below untouched; healthy-eye "
            "score rises 0.05 → 0.13.", {})]], size=15)
    notes(s, "Results at 40,000 training steps. Between 40 and 47 percent of translated images are now "
             "classified as phone-camera images, versus 0.2 percent before. The disease survives: "
             "overall AUROC 0.968 against 0.987 untouched, grade 1 0.907 against 0.981, far above "
             "the 0.685 of the blur control. gen_both, the device-conditioned world-model form, "
             "ranked first under the rule we wrote beforehand, though the top three are within "
             "noise of each other. What is still missing: real mBRSET glare and uneven lighting, "
             "and some loss on mild disease.")

    # 20 --------------------------------------------------------------- joint training + next
    s = base(prs, "New method · 5 of 5", "Joint training is running: two ways to win",
             "scripts/114_synth_joint.slurm (jobs 4609404–4609415); research/codex/DECISIONS.md D068–D069; "
             "literature/DIRECTION_REVIEW.md")
    table(s, 0.6, 1.5, 7.4, [3.6, 2.2, 1.6],
          ["Question (3 seeds each)", "Compared with", "Win if"],
          [["BRSET + synthetic, 0 real labels", "BRSET only 0.9007", "above"],
           ["BRSET + synthetic + 250 real", "250 real 0.9219", "above"],
           ["", "500 real 0.9360", "≈ or above"]],
          size=14, row_h=0.52, aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
    text(s, 0.6, 3.75, 7.4, 3.0,
         ["Same eyes, labels and steps as the baselines; each BRSET image is swapped for its "
          "synthetic twin half the time, so only the camera look changes.",
          "12 runs on 12 A100s, done ≈ 06:00. Generator chosen on mBRSET validation; test reported once.",
          "In parallel: gen_both trained to 60k steps with a decaying learning rate (≈ 05:50)."],
         size=15)
    card(s, 8.4, 1.5, 4.35, 2.55, "Novelty, to confirm",
         ["Closest: RetSyn (diffusion across phone/tabletop cameras; uses paired photos) and "
          "SCR-Net. Ours: unpaired, physics-coupled, label checked per grade."],
         fill=PALE, head_color=NAVY, size=14)
    card(s, 8.4, 4.25, 4.35, 2.5, "Questions for you",
         ["Is the device-conditioned model the world model you meant?",
          "If synthetic + 250 labels matches 500, is that the paper?"],
         fill=PALE_AMBER, head_color=AMBER, size=14)
    notes(s, "The joint-training experiment answers two questions. First, does adding synthetic "
             "phone-camera images help with zero real phone labels? Second, can synthetic images "
             "replace real labels — does synthetic plus 250 real labels match 500 real labels? "
             "That second question is exactly your 8 October framing. The design changes only the "
             "camera look of the BRSET images, so any difference is attributable to synthesis. "
             "Closest prior work is RetSyn; our differences are no paired data, the physics "
             "coupling and per-grade label verification, which I still need to confirm against "
             "their full methods.")

    prs.save(OUT)
    return prs


# --------------------------------------------------------------- fit check
def fit_check():
    serif = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    serif_b = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    cache = {}

    def width_in(txt, size, bold):
        k = (size, bold)
        if k not in cache:
            cache[k] = ImageFont.truetype(serif_b if bold else serif, size * 10)
        return cache[k].getlength(txt) / 10 / 72.0     # points -> inches

    def lines_for(runs, avail):
        """greedy word wrap across runs; returns line count and max font size"""
        words = []
        for t, size, bold in runs:
            for w in t.replace("\n", " ").split(" "):
                if w:
                    words.append((w, size, bold))
        if not words:
            return 1, max([r[1] for r in runs] or [12])
        n, cur = 1, 0.0
        for w, size, bold in words:
            ww = width_in(w + " ", size, bold)
            if cur + ww > avail and cur > 0:
                n += 1
                cur = ww
            else:
                cur += ww
        return n, max(r[1] for r in runs)

    problems = []
    for kind, sno, d in FIT:
        if kind == "text":
            avail_w = d["w"] - 2 * d["margin"]
            hsum = 0.0
            for runs in d["paras"]:
                n, size = lines_for(runs, avail_w)
                hsum += n * size * 1.2 * d["line"] / 72.0 + d["space_after"] / 72.0
            if hsum > d["h"] - 2 * d["margin"] + 0.02:
                problems.append(f"slide {sno}: text '{d['paras'][0][0][0][:40]}' needs {hsum:.2f} in, box {d['h'] - 2 * d['margin']:.2f} in")
            if d["x"] + d["w"] > SW + 0.01 or d["y"] + d["h"] > SH + 0.01:
                problems.append(f"slide {sno}: text box off slide")
        else:
            total = 0.0
            for row in d["cells"]:
                rh = d["row_h"]
                for (t, size, bold), cw in zip(row, d["widths"]):
                    n, _ = lines_for([(t, size, bold)], cw - 0.14)
                    rh = max(rh, n * size * 1.2 / 72.0 + 0.08)
                total += rh
            bottom = d["y"] + total
            if bottom > 6.95:
                problems.append(f"slide {sno}: table '{d['cells'][0][0][0][:30]}' grows to {bottom:.2f} in (limit 6.95)")
    return problems


if __name__ == "__main__":
    build()
    print(f"wrote {OUT}  ({len(Presentation(OUT).slides)} slides); run render_preview.py for the fit check")
