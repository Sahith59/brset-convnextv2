"""Advisor deck, 13 October 2026: the BRSET -> mBRSET project end to end.

Every number on a slide is copied from a saved result file named in that slide's
source line. Run on a compute node (python-pptx + Pillow only, no model loaded).
After building, a conservative text-fit check is run: each text box and table is
re-wrapped with DejaVu Serif, which is wider than Times New Roman, so anything
that fits here fits in PowerPoint.
"""
import json
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
OUT = HERE / "BRSET_mBRSET_Dong_Update_2026-10-14.pptx"
RES = HERE.parent.parent / "results"

# Results slides read their numbers from these files at build time, so a table can
# never disagree with the saved predictions it was computed from.
FT_FILE = sorted(RES.glob("full_table_2026-10-1*_*.json"))[-1]      # scripts/119_full_table.py
FT = json.loads(FT_FILE.read_text())
PAIRS = FT.pop("_pairs", [])
BL_FILE = RES / "baselines_per_seed_2026-10-13.json"                # baselines_per_seed.py job
BL = json.loads(BL_FILE.read_text())

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
    text(s, 0.7, 3.85, 7.0, 0.5, "Research update for Dr. Dong Hye Ye  ·  14 October 2026", size=18, color=ICE)
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
             "slides 16 to 19 are the new method, and part two is the joint-training results. The picture on the right is real output: "
             "left column BRSET photos, middle column the same eyes rendered by our generator as the "
             "phone camera, right column real mBRSET photos of other eyes.")

    # 2 ---------------------------------------------------------------- summary
    s = base(prs, "Executive summary", "Four things to take away",
             "results/lebase_b*_s*/metrics_test.json; results/data_forensics.json; results/" + FT_FILE.name)
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
        ("4  New: zero labels reach the 500-label AUROC", f"{_m(FT['B0']['auc']):.4f} → {_m(FT[WIN]['auc']):.4f}",
         "Synthetic phone twins + self-training on unlabelled phone photos; no phone label used (3 seeds, interval "
         "above 0). Beats style transfer and its own ablations; F1 not yet significant.", GREEN, PALE_GREEN),
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
             "data failed. Four, the new generative method works on ranking: with zero real "
             "phone labels, synthetic images raise DR AUROC to the level of 250 real labels, and the "
             "ablation shows the diffusion stage is what does it. F1 has not improved yet.")

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
        ("12 – 13 Oct", "Generative renderer", "105 M diffusion + physics. Zero-label AUROC up; "
         "both stages needed.", GREEN),
    ]
    x0, wph = 0.6, 2.42
    KEYS = [("0.8455 vs 0.8456", "F1: pooled vs best target-only"),
            ("+0.0233", "DR F1, joint vs target-only"),
            ("3 asks", "synthesis · joint · world model"),
            ("42 of 42", "false negatives are mild"),
            (f"+{_m(FT[BEST]['auc']) - _m(FT['B0']['auc']):.3f}", "DR AUROC with zero real labels")]
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
             "your 8 October conclusion as relayed; research/codex/DECISIONS.md D065–D071")
    rect(s, 0.6, 1.55, 12.15, 1.45, PALE_AMBER)
    text(s, 0.85, 1.68, 11.7, 1.2,
         [[("“Image synthesis converting BRSET to synthesized mBRSET, and also have the joint training, "
            "and make use of the generative world model … everything we did till now sounds like "
            "preprocessing, not a novel thing. Good for a class project, not as a conference submission.”",
            {"italic": True})]], size=16, color=INK)
    asks = [("Image synthesis BRSET → mBRSET", "Built",
             "104.7 M-parameter diffusion renderer, trained on real mBRSET photos, coupled to a "
             "19-parameter physics camera model.", GREEN),
            ("Joint training with the synthetic images", "Done — results in part 2",
             f"{3 * sum(1 for k in FT if FT[k]['base'])} runs. Zero real labels: DR AUROC "
             f"{_m(FT['B0']['auc']):.4f} → {_m(FT[BEST]['auc']):.4f}; no gain on top of 250 real labels.", GREEN),
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
             "synthetic images: done, results in part two. World model: the gen_both variant is one generative "
             "model that renders the same retina as either camera. I would like your view on "
             "whether that is the world-model form you had in mind, or whether you meant a "
             "latent-prediction model like CheXWorld.")

    # 10 --------------------------------------------------------------- bounded translator
    s = base(prs, "First synthesis attempt, 9–11 October", "The first translator was too small to move the images",
             "scripts/72_physical_layer.py, 82_refiner.py (parameter counts measured); 90_diagnose_translator.py; "
             "results/synth_b0_s0*, lebase_b0_s0, syn_tgt_b0_s0 (seed 0)")
    stat(s, 0.6, 1.5, 3.95, "68,346", "parameters in the old learned stage; each pixel clamped to ±0.15 "
         "(smooth) and ±0.02 (fine detail)", color=RED, h=1.7)
    stat(s, 0.6, 3.25, 3.95, "1.0000 → 1.0000", "domain-probe AUC before vs after translation, all three "
         "variants (two adversarial)", color=RED, num_size=30, h=1.7)
    stat(s, 0.6, 5.0, 3.95, "104.7 M", "parameters in the new diffusion stage: about 1,500× larger, "
         "not clamped", color=GREEN, h=1.7)
    text(s, 4.95, 1.55, 7.8, 0.4, "Training with those images (seed 0, zero labels)", size=17, bold=True, color=NAVY)
    table(s, 4.95, 2.0, 7.8, [4.4, 1.6, 1.8],
          ["Arm (seed 0)", "DR AUROC", "vs same seed, BRSET only (0.9104)"],
          [["Old translator, 25 % of images", "0.8996", ("−0.0108", {"color": RED})],
           ["Old translator, 50 %", "0.9044", ("−0.0060", {"color": RED})],
           ["Old translator, 100 %", "0.8888", ("−0.0216", {"color": RED})],
           ["Old translator, real + copy (2× epoch)", "0.8652", ("−0.0452", {"color": RED})],
           [("New diffusion method, 50 % (same seed)", {"bold": True}), ("0.9186", {"bold": True}),
            ("+0.0082", GOOD)]],
          size=14, row_h=0.46, aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    rect(s, 4.95, 5.5, 7.8, 1.3, PALE_AMBER)
    text(s, 5.15, 5.55, 7.4, 1.2,
         [[("Correction: ", {"bold": True, "color": AMBER}),
           ("we earlier said this showed synthesis cannot work. It showed only that a generator "
            "this constrained cannot change the camera look. That claim is withdrawn; its "
            "discriminator (166,753 parameters) was 2.4× the generator.", {})]], size=15)
    notes(s, "Our first attempt at your request used a 19-parameter physics model plus a tiny 68 "
             "thousand parameter refiner whose output was clamped. Every old-translator run scored below "
             "the same-seed baseline; the new diffusion method on the same seed scored above it. A probe could still tell the "
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

    results_section(prs)

    prs.save(OUT)
    return prs


# --------------------------------------------------------------- results (part 2)
def _m(x):
    return sum(x) / len(x)


def _rn(r):
    return r["tpl"].replace("_s*", "")


def _swap(r):
    return f"{int(r['swap'])} %" if r["swap"] else "–"


def _ci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]".replace("-", "−")


def _st(r):
    return "–" if r["steps"] in ("-", "") else r["steps"]


GOOD = {"bold": True, "color": GREEN}
BADS = {"bold": True, "color": RED}
ZERO_ORDER = ["S1", "S2", "S3", "S14", "S9", "S11", "S4", "S12", "S13", "S5", "S16", "S15"]
ZERO = [k for k in ZERO_ORDER if k in FT]
LAB250 = [k for k in ["S6", "S7", "S8", "S10"] if k in FT]
BEST = max(ZERO, key=lambda k: FT[k]["val"])          # the validation rule (D071)
FTS = "results/" + FT_FILE.name + " (scripts/119_full_table.py)"


def _delta_cells(r):
    """(Δ AUROC [CI], Δ F1 [CI]) styled green if the 95% interval is above 0, red if below."""
    b = FT[r["base"]]
    da, df = _m(r["auc"]) - _m(b["auc"]), _m(r["f1"]) - _m(b["f1"])
    sa = GOOD if r["aucCI"][0] > 0 else (BADS if r["aucCI"][1] < 0 else {})
    sf = GOOD if r["f1CI"][0] > 0 else (BADS if r["f1CI"][1] < 0 else {})
    return (f"{da:+.4f} ".replace("-", "−") + _ci(*r['aucCI']), sa), (f"{df:+.4f} ".replace("-", "−") + _ci(*r['f1CI']), sf)


def _arm_row(k, label=None):
    r = FT[k]
    if r["base"] is None:
        return [(label or _rn(r) + " (baseline)", {"bold": True}), "none", "–", "–", f"{r['val']:.4f}",
                (f"{_m(r['auc']):.4f}", {"bold": True}), "–", "–", (f"{_m(r['f1']):.4f}", {"bold": True}), "–"]
    da, df = _delta_cells(r)
    return [label or _rn(r), r["gen"], _st(r), _swap(r), f"{r['val']:.4f}", (f"{_m(r['auc']):.4f}", {"bold": True}),
            da, f"{int(r['up'])}/3", (f"{_m(r['f1']):.4f}", {"bold": True}), df]


def _gate(tag):
    f = RES / f"gates_{tag}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    rows = {str(x["grade"]): x for x in d["gate2"]["rows"]}
    g1 = d["gate1"]["256"]
    return dict(p=g1["p_target_translated"], all=rows["all"]["translated"], g1=rows["1"]["translated"],
                g2=rows["2"]["translated"], neg=d["gate2"]["mean_p_neg"]["translated"],
                u_p=g1["p_target_untouched"], u_all=rows["all"]["untouched"], u_g1=rows["1"]["untouched"],
                u_g2=rows["2"]["untouched"], u_neg=d["gate2"]["mean_p_neg"]["untouched"],
                b_all=rows["all"]["blur4_control"], b_g1=rows["1"]["blur4_control"], b_g2=rows["2"]["blur4_control"])


def _bl(prefix):
    for k, v in BL.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


# Job status is a snapshot, written just before the build (squeue + logs).
STATUS_TIME = "14 Oct, 01:25"
RUNNING = [["syn_bothld100_b0  × 3 seeds", "gen_both 100k, 50 % swap, 0 labels (11,372/11,372 images made)", "started 21:55", "~01:10"],
           ["syn_bothld100_b0_p0.25  × 3 seeds", "gen_both 100k, 25 % swap, 0 labels", "started 21:55", "~01:10"]]
QUEUED = []
PLANNED = [["~01:15", "Read the 100k runs, validation first; add them to every table in part 2"],
           ["then", "Choose the final configuration on validation only; report its test result once"],
           ["this week", "Novelty statement vs RetSyn / SCR-Net; F1 analysis; ISBI draft"]]

ARM_HEAD = ["Run", "Generator", "Steps", "Swap", "Val AUROC", "Test AUROC", "Δ AUROC [95% CI]",
            "Seeds better", "Test F1", "Δ F1 [95% CI]"]
ARM_W = [2.3, 0.95, 0.65, 0.65, 0.9, 0.9, 2.05, 0.8, 0.85, 2.1]
ARM_AL = [PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 9


def results_section(prs):
    n_arms = sum(1 for k in FT if k[0] in "SA" and FT[k]["base"])
    b0, br = FT["B0"], FT[BEST]
    jc = _bl("C.")
    c_auc, c_f1 = _m([x["auc"] for x in jc]), _m([x["f1"] for x in jc])

    # divider -------------------------------------------------------------------------------
    s = base(prs, "", "", dark=True)
    text(s, 0.7, 1.9, 11.5, 0.45, "PART 2  ·  RESULTS OF JOINT TRAINING WITH SYNTHETIC mBRSET", size=14,
         bold=True, color=ICE)
    text(s, 0.7, 2.45, 11.6, 1.6, "What the synthetic phone-camera images do for the classifier",
         size=38, bold=True, color=WHITE, line=1.0)
    text(s, 0.7, 4.25, 11.0, 1.9,
         [f"{n_arms} configurations × 3 seeds = {3 * n_arms} classifier runs, all scored on the same 732 "
          "mBRSET test images (159 with DR).",
          "Choices are made on mBRSET validation; test is read once. Every table in this part reads its "
          "values from the result file named at the foot of the slide."], size=17, color=ICE)
    notes(s, "Part two is the result of the experiment you asked for: BRSET plus synthetic mBRSET "
             "images in joint training. Every configuration was run with three seeds, and every number "
             "is read directly from the saved prediction files.")

    # which baseline -------------------------------------------------------------------------
    s = base(prs, "Results · how to read them", "Zero-label runs are compared with zero-shot BRSET",
             f"{FTS}; results/{BL_FILE.name}")
    d_ = _bl("D.")
    e_ = _bl("E.")
    cats = ["BRSET only · 0 labels (baseline)", "Ours · 0 labels (best on validation)", "BRSET + 250 real labels",
            "BRSET + 500 real labels", "BRSET + all 3,402 (joint)", "Fine-tune, best lr (1 seed)", "mBRSET only (1 seed)"]
    vals = (_m(b0["auc"]), _m(br["auc"]), _m(FT["B250"]["auc"]), _m(FT["B500"]["auc"]), c_auc,
            max(x["auc"] for x in d_), e_[0]["auc"])
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("DR AUROC", vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.5), Inches(6.5), Inches(5.3), cd)
    ch = gf.chart
    chart_style(ch)
    ch.has_title = True
    ch.chart_title.text_frame.text = "DR AUROC, mBRSET test (3-seed mean unless noted)"
    _font(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, NAVY)
    pl = ch.plots[0]
    pl.gap_width = 45
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.0000"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size = Pt(12)
    for i, col in enumerate((AMBER, GREEN, TEAL, TEAL, NAVY, GREY, GREY)):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = col
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.88, 0.96
    va.has_major_gridlines = False
    va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    ch.category_axis.reverse_order = True
    ch.category_axis.tick_labels.font.size = Pt(12)
    text(s, 7.45, 1.5, 5.3, 0.4, "Putting it together", size=17, bold=True, color=NAVY)
    table(s, 7.45, 1.95, 5.3, [2.55, 0.85, 0.95, 0.95],
          ["Setup", "mBRSET train labels", "Test AUROC", "Test F1"],
          [[("A · Zero-shot, BRSET only (our baseline)", {"bold": True}), "0", f"{_m(b0['auc']):.4f}", f"{_m(b0['f1']):.4f}"],
           [(f"Ours · {br['gen']} {br['steps']}, {_swap(br)} swap", GOOD), "0", (f"{_m(br['auc']):.4f}", GOOD),
            f"{_m(br['f1']):.4f}"],
           ["C · Joint, BRSET + all mBRSET (ceiling)", "3,402", f"{c_auc:.4f}", f"{c_f1:.4f}"]],
          size=13, row_h=0.62, fills=[PALE_AMBER, PALE_GREEN, PALE_TEAL],
          aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 3)
    ga = (_m(br["auc"]) - _m(b0["auc"])) / (c_auc - _m(b0["auc"]))
    gfi = (_m(br["f1"]) - _m(b0["f1"])) / (c_f1 - _m(b0["f1"]))
    stat(s, 7.45, 4.6, 2.5, f"{ga:.0%}", "of the AUROC gap to the ceiling closed with zero labels", color=GREEN,
         num_size=32, h=1.35)
    stat(s, 10.15, 4.6, 2.6, f"{gfi:.0%}", "of the F1 gap — not significant", color=AMBER, num_size=32, h=1.35)
    text(s, 7.45, 6.0, 5.3, 0.9,
         "All runs, the baseline included, pick their checkpoint and F1 cut-off on the 725 labelled "
         "mBRSET validation images: “zero training labels”, not label-free.", size=12, color=GREY)
    notes(s, "Which baseline do we compare against? Our synthetic runs use no real mBRSET training "
             "labels, so the fair comparison is the zero-shot model trained on BRSET only, with the "
             "same recipe and the same three seeds. Joint training with all 3,402 labels, fine-tuning "
             "and mBRSET-only training all use real labels; they are the ceiling. The best "
             "zero-label configuration on validation closes most of the AUROC gap to the ceiling but "
             "hardly any of the F1 gap. One caveat: validation labels are used for model selection "
             "everywhere, so the honest phrase is zero training labels.")

    # baselines per seed ---------------------------------------------------------------------
    s = base(prs, "Results · baselines", "Every baseline, every seed",
             f"results/{BL_FILE.name} (from each run's test_predictions.npz; settings read from the Slurm logs)")
    groups = [("A · Zero-shot", "BRSET only", "0", _bl("A.")),
              ("Joint, partial", "BRSET + 250 mBRSET", "250", _bl("B. JOINT, part of mBRSET: BRSET + 250 ")),
              ("Joint, partial", "BRSET + 500 mBRSET", "500", _bl("B. JOINT, part of mBRSET: BRSET + 500 ")),
              ("Joint, partial", "BRSET + 1,000 mBRSET", "1,000", _bl("B. JOINT, part of mBRSET: BRSET + 1000 ")),
              ("Joint, partial", "BRSET + 2,000 mBRSET", "2,000", _bl("B. JOINT, part of mBRSET: BRSET + 2000 ")),
              ("C · Joint, all", "BRSET + all 3,402 mBRSET", "3,402", jc)]
    rows = []
    for name, data, lab, g in groups:
        rows.append([(name, {"bold": True}), data, lab] + [f"{x['auc']:.4f}" for x in g] +
                    [(f"{_m([x['auc'] for x in g]):.4f}", {"bold": True})] + [f"{x['f1']:.4f}" for x in g] +
                    [(f"{_m([x['f1'] for x in g]):.4f}", {"bold": True})])
    table(s, 0.6, 1.5, 12.15, [1.7, 2.45, 0.9, 0.88, 0.88, 0.88, 0.95, 0.88, 0.88, 0.88, 0.95],
          ["Setup", "Training data", "mBRSET labels", "AUROC s0", "AUROC s1", "AUROC s2", "AUROC mean",
           "F1 s0", "F1 s1", "F1 s2", "F1 mean"],
          rows, size=13, row_h=0.4, fills=[PALE_AMBER, WHITE, PALE, WHITE, PALE, PALE_TEAL],
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 9)
    text(s, 0.6, 4.5, 12.15, 0.38, "Run once each: the columns are learning rates or recipes, not seeds  (AUROC / F1)",
         size=14, bold=True, color=NAVY)
    lr = ["lr 1e-5", "lr 2e-5", "lr 5e-6"]
    table(s, 0.6, 4.9, 12.15, [3.4, 2.05, 2.25, 2.25, 2.2],
          ["Setup", "Training data", "Variant 1", "Variant 2", "Variant 3"],
          [[("D · Fine-tune: BRSET model, then mBRSET only", {"bold": True}), "3,402 mBRSET labels"] +
           [f"{l}: {x['auc']:.4f} / {x['f1']:.4f}" for l, x in zip(lr, d_)],
           [("E · mBRSET only, no BRSET", {"bold": True}), "3,402 mBRSET labels"] +
           [f"recipe {i + 1}: {x['auc']:.4f} / {x['f1']:.4f}" for i, x in enumerate(e_)] + ["–"]],
          size=13, row_h=0.44, aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 3)
    text(s, 0.6, 6.5, 12.15, 0.4,
         "0.8355 (fine-tune, lr 2e-5) is the best of three learning rates read on test, single seed. All rows: same 732 test images.",
         size=12, color=GREY)
    notes(s, "These are the baselines, every seed, recomputed from the saved predictions. A is the "
             "zero-shot model, our comparison point. The partial joint rows are the label-efficiency "
             "curve. C is joint training with all mBRSET labels, the ceiling. Fine-tuning and "
             "mBRSET-only were run once each with different learning rates or recipes, so their "
             "columns are not seeds.")

    # main result: zero labels ---------------------------------------------------------------
    s = base(prs, "Results · zero real mBRSET labels", "Zero real labels: AUROC rises, F1 does not move",
             FTS)
    rows = [_arm_row("B0", "BRSET only (baseline)")] + [_arm_row(k) for k in ZERO]
    fills = [PALE_AMBER] + [PALE_GREEN if k == BEST else (WHITE if i % 2 == 0 else PALE) for i, k in enumerate(ZERO)]
    table(s, 0.6, 1.45, 12.15, ARM_W, ARM_HEAD, rows, size=11, row_h=0.3, fills=fills, aligns=ARM_AL)
    text(s, 0.6, 6.45, 12.15, 0.5,
         f"Mean of 3 seeds; Δ vs BRSET only, same seeds; 95% patient-bootstrap interval (193 patients). Green = interval "
         f"above 0, red = below 0. Green row = highest validation AUROC ({_rn(br)}).",
         size=11, color=GREY)
    notes(s, "This is the main table. Every row uses zero real mBRSET training labels. Each row is a "
             "generator, how long it was trained, and how often a BRSET image is swapped for its "
             "synthetic twin. Most 25 and 50 percent rows raise test AUROC by 0.02 to 0.03 with the "
             "interval above zero and all three seeds better. No row improves F1 significantly. The "
             "shaded row is the current leader on validation, which is how the final configuration "
             "will be chosen.")

    # swap-rate grid ----------------------------------------------------------------------------
    s = base(prs, "Results · how much synthetic", "25 % and 50 % swap give the same AUROC gain; 75 % is worse",
             FTS)
    lines = [("gen_tgt", "40k"), ("gen_tgt", "60k"), ("gen_both", "40k"), ("gen_both", "60k"), ("gen_both", "100k")]
    look = {(FT[k]["gen"], FT[k]["steps"], int(FT[k]["swap"])): FT[k] for k in ZERO}
    rows = []
    for g, st in lines:
        row = [(f"{g}, {st} steps", {"bold": True})]
        for sw in (25, 50, 75):
            r = look.get((g, st, sw))
            if r:
                sty = GOOD if r["aucCI"][0] > 0 else {}
                row.append((f"{_m(r['auc']):.4f} / {_m(r['f1']):.4f}", sty))
            else:
                row.append(("not run" if sw == 75 else "running", {"color": GREY}))
        rows.append(row)
    table(s, 0.6, 1.5, 7.3, [2.2, 1.7, 1.7, 1.7], ["Generator", "25 % swap", "50 % swap", "75 % swap"],
          rows, size=14, row_h=0.56, aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 3)
    text(s, 0.6, 4.95, 7.3, 1.8,
         [f"Cells: test AUROC / test F1, 0 real labels, mean of 3 seeds. Baseline {_m(b0['auc']):.4f} / {_m(b0['f1']):.4f}. "
          "Green = AUROC gain with 95% interval above 0.",
          "Replacing most BRSET images (75 %) removes too much of the real camera; a quarter to a half is enough."],
         size=14)
    cd = CategoryChartData()
    cd.categories = ["25 %", "50 %", "75 %"]
    shown = [(g, st) for g, st in lines if any((g, st, sw) in look for sw in (25, 50, 75))]
    for g, st in shown:
        cd.add_series(f"{g} {st}", [(_m(look[(g, st, sw)]["auc"]) if (g, st, sw) in look else None) for sw in (25, 50, 75)])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(8.2), Inches(1.5), Inches(4.55), Inches(5.3), cd)
    ch = gf.chart
    chart_style(ch, size=12)
    ch.has_legend = True
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Test DR AUROC by swap rate (baseline 0.9007)"
    _font(ch.chart_title.text_frame.paragraphs[0].runs[0], 12, True, NAVY)
    pl = ch.plots[0]
    pl.gap_width = 60
    for ser, col in zip(pl.series, (RGBColor(120, 170, 175), TEAL, RGBColor(196, 140, 70), AMBER, NAVY)):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.90, 0.935
    va.major_gridlines.format.line.color.rgb = LINE
    va.tick_labels.font.size = Pt(11)
    va.tick_labels.number_format = "0.000"
    va.tick_labels.number_format_is_linked = False
    notes(s, "How much synthetic data? For both generators, swapping a quarter or half of the BRSET "
             "images gives the same AUROC gain; swapping three quarters is clearly worse. A quarter gives "
             "slightly higher F1 for both generators, but that difference is within noise.")

    # per seed ---------------------------------------------------------------------------------
    s = base(prs, "Results · seed by seed", "Every seed of every zero-label run", FTS)
    rows = []
    for k in ["B0"] + ZERO:
        r = FT[k]
        nm = "BRSET only (baseline)" if k == "B0" else _rn(r)
        rows.append([(nm, {"bold": k == "B0"}), r["gen"] if k != "B0" else "none", _st(r), _swap(r)] +
                    [f"{a:.4f}" for a in r["auc"]] + [(f"{_m(r['auc']):.4f}", {"bold": True})] +
                    [f"{f:.4f}" for f in r["f1"]] + [(f"{_m(r['f1']):.4f}", {"bold": True})])
    table(s, 0.6, 1.45, 12.15, [2.45, 0.95, 0.65, 0.65, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.94],
          ["Run", "Generator", "Steps", "Swap", "AUROC s0", "AUROC s1", "AUROC s2", "AUROC mean",
           "F1 s0", "F1 s1", "F1 s2", "F1 mean"],
          rows, size=11, row_h=0.3, fills=[PALE_AMBER] + [WHITE if i % 2 == 0 else PALE for i in range(len(ZERO))],
          aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 11)
    text(s, 0.6, 6.55, 12.15, 0.4,
         f"Baseline seeds: AUROC {', '.join(f'{a:.4f}' for a in b0['auc'])}. F1 at the cut-off tuned on validation "
         "(bootstrap median), applied once to test.", size=11, color=GREY)
    notes(s, "The same runs, seed by seed, so you can see the spread. Seed 0 of the baseline was a "
             "lucky 0.9104; the three-seed mean is 0.9007.")

    s = base(prs, "Results · seed by seed", "Every seed of the ablations and the 250-label runs", FTS)
    keys = [k for k in ["B0", "AB1", "AB2", "AB3", "B250"] + LAB250 if k in FT]
    rows = []
    for k in keys:
        r = FT[k]
        lab = "0" if r["labels"] == 0 else str(int(r["labels"]))
        nm = {"B0": "BRSET only (baseline, 0 labels)", "B250": "BRSET + 250 real (baseline)"}.get(k, _rn(r))
        rows.append([(nm, {"bold": k in ("B0", "B250")}), r["gen"] if r["base"] else "none", _st(r), lab] +
                    [f"{a:.4f}" for a in r["auc"]] + [(f"{_m(r['auc']):.4f}", {"bold": True})] +
                    [f"{f:.4f}" for f in r["f1"]] + [(f"{_m(r['f1']):.4f}", {"bold": True})])
    fills = [PALE_AMBER if k in ("B0", "B250") else (WHITE if i % 2 == 0 else PALE) for i, k in enumerate(keys)]
    table(s, 0.6, 1.45, 12.15, [2.45, 1.55, 0.6, 0.65, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.88],
          ["Run", "Images", "Steps", "Real labels", "AUROC s0", "AUROC s1", "AUROC s2", "AUROC mean",
           "F1 s0", "F1 s1", "F1 s2", "F1 mean"],
          rows, size=12, row_h=0.36, fills=fills, aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 11)
    text(s, 0.6, 6.45, 12.15, 0.45,
         "syn_physonly = physics only; syn_tgtnone = diffusion only (pure noise); syn_tgtsrc = diffusion from the plain "
         "BRSET image; all three use gen_tgt 40k settings and a 50 % swap.", size=11, color=GREY)
    notes(s, "Seed-by-seed values for the three ablations and for the runs that also use 250 real "
             "labels, next to their baselines.")

    # ablation --------------------------------------------------------------------------------
    s = base(prs, "Results · is it the generative model?", "Both stages are needed: physics alone and diffusion alone fall short",
             f"{FTS}; results/gates_g40000_gen_tgt_*.json, gates_gate_physonly_*.json; D070")
    abl = [("S2", "Full method: physics → diffusion (gen_tgt 40k)", "g40000_gen_tgt_physics_s0.4_w1.5"),
           ("AB1", "Physics only, no diffusion", "gate_physonly_v2noartifact"),
           ("AB2", "Diffusion only, from pure noise", "g40000_gen_tgt_none_s1.0_w1.5"),
           ("AB3", "Diffusion from the plain BRSET image", "g40000_gen_tgt_source_s0.4_w1.5")]
    rows = []
    for k, nm, gt in abl:
        r = FT[k]
        da, df = _delta_cells(r)
        gg = _gate(gt)
        rows.append([(nm, {"bold": k == "S2"}), f"{r['val']:.4f}", (f"{_m(r['auc']):.4f}", {"bold": True}), da,
                     (f"{_m(r['f1']):.4f}", {"bold": True}), df,
                     f"{gg['p']:.3f}" if gg else ("running", {"color": GREY}),
                     f"{gg['g1']:.3f}" if gg else ("running", {"color": GREY})])
    g0 = _gate("g40000_gen_tgt_physics_s0.4_w1.5")
    rows.append([("BRSET only (baseline)", {"bold": True}), f"{b0['val']:.4f}", f"{_m(b0['auc']):.4f}", "–",
                 f"{_m(b0['f1']):.4f}", "–", f"{g0['u_p']:.3f}", f"{g0['u_g1']:.3f}"])
    table(s, 0.6, 1.45, 12.15, [3.3, 0.9, 0.9, 2.05, 0.8, 2.05, 1.05, 1.1],
          ["Images used for the 50 % swap (0 labels)", "Val AUROC", "Test AUROC", "Δ AUROC [95% CI]", "Test F1",
           "Δ F1 [95% CI]", "Looks like mBRSET", "Grade-1 lesion score"],
          rows, size=12, row_h=0.44, fills=[PALE_GREEN, WHITE, PALE, WHITE, PALE_AMBER],
          aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 7)
    text(s, 0.6, 4.35, 12.15, 0.38, "Head-to-head: full method minus each ablation (same seeds, paired bootstrap)",
         size=15, bold=True, color=NAVY)
    names = {"AB1": "physics only", "AB2": "diffusion only", "AB3": "diffusion from plain BRSET"}
    for i, pr in enumerate([p for p in PAIRS if p["x"] == "S2"]):
        lo, hi = pr["aucCI"]
        if lo > 0 and pr["dval"] > 0:
            verdict, col, fill = "Needed — pre-stated rule met", GREEN, PALE_GREEN
        elif lo > -0.001 and pr["seeds"] == 3 and pr["dval"] > 0:
            verdict, col, fill = "Borderline — interval touches 0", AMBER, PALE_AMBER
        else:
            verdict, col, fill = "Not resolved", GREY, PALE
        x = 0.6 + i * 4.1
        rect(s, x, 4.8, 3.95, 1.95, fill)
        text(s, x + 0.15, 4.88, 3.65, 0.4, f"vs {names.get(pr['y'], pr['y'])}", size=15, bold=True, color=NAVY)
        text(s, x + 0.15, 5.28, 3.65, 0.55, f"{pr['dauc']:+.4f} AUROC", size=24, bold=True, color=col)
        text(s, x + 0.15, 5.85, 3.65, 0.85,
             [f"95% CI [{lo:+.4f}, {hi:+.4f}]; {pr['seeds']}/3 seeds; validation {pr['dval']:+.4f}".replace("-", "−"), verdict],
             size=12, space_after=2)
    notes(s, "This answers whether the gain is just preprocessing. Same generator, same swap rate, "
             "zero labels, three seeds; only the way the synthetic image is made changes. The physics "
             "camera model alone does almost nothing. Diffusion alone, without the physics start, gives "
             "a smaller gain that is not significant and makes F1 significantly worse. Only the "
             "combination gives the significant gain. The rule for calling a stage necessary was written "
             "before these runs: the paired interval must be above zero and validation must agree.")

    # 250 labels ------------------------------------------------------------------------------
    s = base(prs, "Results · with 250 real labels", "With 250 real labels, synthetic images add nothing measurable", FTS)
    rows = [_arm_row("B250", "BRSET + 250 real, lebase_b250 (baseline)")] + [_arm_row(k) for k in LAB250]
    table(s, 0.6, 1.45, 12.15, ARM_W, ARM_HEAD, rows, size=12, row_h=0.4,
          fills=[PALE_AMBER] + [WHITE if i % 2 == 0 else PALE for i in range(len(LAB250))], aligns=ARM_AL)
    text(s, 0.6, 4.2, 12.15, 0.38, "Reference: BRSET + real mBRSET labels only, no synthetic images (3 seeds each)",
         size=15, bold=True, color=NAVY)
    ref = [("0", FT["B0"]), ("250", FT["B250"]), ("500", FT["B500"]), ("1,000", FT["B1000"]), ("2,000", FT["B2000"])]
    table(s, 0.6, 4.62, 12.15, [2.55] + [1.6] * 6,
          ["Real mBRSET labels"] + [a for a, _ in ref] + ["3,402 (all)"],
          [["Test AUROC"] + [f"{_m(r['auc']):.4f}" for _, r in ref] + [f"{c_auc:.4f}"],
           ["Test F1"] + [f"{_m(r['f1']):.4f}" for _, r in ref] + [f"{c_f1:.4f}"]],
          size=14, row_h=0.44, bold_cols=(0,), aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 6)
    best250 = max(LAB250, key=lambda k: _m(FT[k]["auc"]))
    text(s, 0.6, 6.2, 12.15, 0.72,
         [f"Synthetic images substitute for roughly the first 250 real labels at zero cost ({_m(br['auc']):.4f} vs "
          f"{_m(FT['B250']['auc']):.4f}), but once real labels are present they do not add: best synthetic + 250 run "
          f"{_m(FT[best250]['auc']):.4f} vs 500 real {_m(FT['B500']['auc']):.4f}."], size=14)
    notes(s, "With 250 real labels already in training, adding synthetic images does not change AUROC "
             "measurably for any generator. Synthetic images therefore replace roughly the first 250 real "
             "labels, but do not stack on top of them. Synthetic plus 250 does not reach 500 real labels.")

    # gates -----------------------------------------------------------------------------------
    s = base(prs, "Results · image checks, every variant", "Image checks on 546 held-out BRSET images",
             "scripts/112_gates.py; results/gates_*.json (gate 1 device classifier; gate 2 joint-trained judge per ICDR grade)")
    gl = [("gen_tgt 40k · physics start", "g40000_gen_tgt_physics_s0.4_w1.5"),
          ("gen_tgt 60k · physics start", "g60000_gen_tgt_ld_physics_s0.4_w1.5"),
          ("gen_both 40k · physics start", "g40000_gen_both_physics_s0.4_w1.5"),
          ("gen_both 60k · physics start", "g60000_gen_both_ld_physics_s0.4_w1.5"),
          ("gen_both 100k · physics start", "g100000_gen_both_ld100_physics_s0.4_w1.5"),
          ("Physics only (no diffusion)", "gate_physonly_v2noartifact"),
          ("Diffusion only · gen_tgt 40k", "g40000_gen_tgt_none_s1.0_w1.5"),
          ("Diffusion only · gen_tgt 60k", "g60000_gen_tgt_ld_none_s1.0_w1.5"),
          ("Diffusion only · gen_both 60k", "g60000_gen_both_ld_none_s1.0_w1.5"),
          ("Diffusion from plain BRSET · gen_tgt 40k", "g40000_gen_tgt_source_s0.4_w1.5")]
    rows = [[("Untouched BRSET", {"bold": True}), f"{g0['u_p']:.3f}", f"{g0['u_all']:.3f}", f"{g0['u_g1']:.3f}",
             f"{g0['u_g2']:.3f}", f"{g0['u_neg']:.3f}"],
            [("Lesions blurred (control)", {"bold": True}), "–", f"{g0['b_all']:.3f}", f"{g0['b_g1']:.3f}",
             f"{g0['b_g2']:.3f}", "–"]]
    for nm, tg in gl:
        gg = _gate(tg)
        if gg:
            rows.append([nm, f"{gg['p']:.3f}", f"{gg['all']:.3f}", f"{gg['g1']:.3f}", f"{gg['g2']:.3f}", f"{gg['neg']:.3f}"])
        else:
            rows.append([nm] + [("running", {"color": GREY})] * 5)
    table(s, 0.6, 1.4, 12.15, [3.75, 1.75, 1.7, 1.55, 1.55, 1.85],
          ["Images", "Called mBRSET (real mBRSET 0.998)", "Lesion AUROC, all grades", "Grade 1 (mild)", "Grade 2",
           "Healthy-eye score (higher = invented)"],
          rows, size=12, row_h=0.36, fills=[PALE, PALE] + [WHITE if i % 2 == 0 else PALE for i in range(len(gl))],
          aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 5)
    notes(s, "Image checks for every generator and every ablation, on held-out BRSET images. Physics "
             "alone keeps lesions best but barely looks like the phone camera. Diffusion alone with the "
             "gen_tgt generator loses mild lesions and raises the judge's score on healthy eyes; "
             "diffusion from the plain BRSET image barely changes the camera look. The physics-started "
             "diffusion is the balance, and it is the version that helps the classifier.")

    # what the results say ---------------------------------------------------------------------
    s = base(prs, "Results · summary", "What the results say", FTS)
    ab = {k: FT[k] for k in ("AB1", "AB2") if k in FT}
    cards = [
        ("1  Zero labels: AUROC up", f"{_m(b0['auc']):.4f} → {_m(br['auc']):.4f}",
         f"{_rn(br)}: +{_m(br['auc']) - _m(b0['auc']):.4f}, 3/3 seeds, interval above 0 — the level of 250 real "
         f"labels ({_m(FT['B250']['auc']):.4f}).", GREEN, PALE_GREEN),
        ("2  It needs the generative model", f"+{_m(ab['AB1']['auc']) - _m(b0['auc']):.3f} vs +{_m(FT['S2']['auc']) - _m(b0['auc']):.3f}",
         "physics alone vs physics + diffusion (same swap, same seeds). Diffusion alone: "
         f"+{_m(ab['AB2']['auc']) - _m(b0['auc']):.3f}, not significant, F1 worse.", TEAL, PALE_TEAL),
        ("3  F1 does not move", f"{_m(b0['f1']):.4f} → {_m(br['f1']):.4f}",
         "No configuration improves F1 with an interval above 0; diffusion-only variants make it "
         "significantly worse. Ranking improves; the operating point does not.", AMBER, PALE_AMBER),
        ("4  No gain on top of real labels", f"{len(LAB250)} of {len(LAB250)} arms",
         f"with 250 real labels sit within ±{max(abs(_m(FT[k]['auc']) - _m(FT['B250']['auc'])) for k in LAB250):.3f} AUROC "
         "of the 250-label baseline.", RED, PALE_RED)]
    for (x, y), (head, big, body, col, fill) in zip([(0.6, 1.55), (6.83, 1.55), (0.6, 4.25), (6.83, 4.25)], cards):
        rect(s, x, y, 5.9, 2.55, fill)
        text(s, x + 0.2, y + 0.12, 5.5, 0.42, head, size=16, bold=True, color=col)
        text(s, x + 0.2, y + 0.55, 5.5, 0.7, big, size=28, bold=True, color=NAVY)
        text(s, x + 0.2, y + 1.3, 5.5, 1.15, body, size=14)
    notes(s, "Four messages. One: with zero real labels, synthetic images raise AUROC to the level of "
             "250 real labels. Two: the gain needs the diffusion model; physics alone gives almost "
             "nothing. Three: F1 does not move, so the model ranks better but its operating point does "
             "not improve. Four: synthetic images do not add on top of real labels.")

    # final choice (all runs finished) ------------------------------------------------------------
    s = base(prs, "Results · synthetic images only", f"Best synthetic-only setting on validation: {br['gen']} {br['steps']}, {_swap(br)}",
             FTS + "; rule: research/codex/DECISIONS.md D071")
    order = sorted(ZERO, key=lambda k: -FT[k]["val"])
    rows = []
    for i, k in enumerate(order):
        row = _arm_row(k)
        rows.append([str(i + 1)] + row[:7] + row[8:])
    rows.append(["–"] + _arm_row("B0", "BRSET only (baseline)")[:7] + _arm_row("B0", "BRSET only (baseline)")[8:])
    table(s, 0.6, 1.45, 12.15, [0.55, 2.5, 0.95, 0.65, 0.65, 0.9, 0.9, 2.05, 0.85, 2.15],
          ["Rank", "Run (0 real labels)", "Generator", "Steps", "Swap", "Val AUROC", "Test AUROC", "Δ AUROC [95% CI]",
           "Test F1", "Δ F1 [95% CI]"],
          rows, size=11, row_h=0.3, fills=[PALE_GREEN] + [WHITE if i % 2 else PALE for i in range(len(order) - 1)] + [PALE_AMBER],
          aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 8)
    text(s, 0.6, 6.2, 12.15, 0.72,
         ["Rule (D071, written 13 Oct 17:05, before the gen_both swap-rate, 60k-at-25 % and 100k results): highest mBRSET "
          "validation DR AUROC among the zero-label configurations; ablations test the method and are not candidates.",
          f"Validation and test rank configurations differently, so choosing on test would be cherry-picking. "
          f"No jobs running at {STATUS_TIME}; every run finished."], size=11, color=GREY, space_after=2)
    notes(s, "All runs have finished. Ranked by validation AUROC, which is the rule we fixed in advance, the "
             "winner is the device-conditioned generator at 40,000 steps with a quarter of BRSET images "
             "swapped. Its test result is the one we report. Note that the test ranking is different; "
             "several configurations are within noise of each other on test.")

    part3(prs)

    # next + questions -------------------------------------------------------------------------
    s = base(prs, "Next steps", "What we will do next, and questions for you",
             "research/codex/DECISIONS.md D068–D077; literature/RETSYN_NOVELTY.md")
    card(s, 0.6, 1.5, 3.85, 3.75, "Next",
         ["Stop self-training earlier: it peaks at epochs 5–13, then slips.",
          "Matched control: standard self-training with the same cut-off as the winner.",
          "Fix in advance: also report sensitivity at a fixed specificity.",
          "Draft the ISBI paper (26 Oct)."], fill=PALE, head_color=NAVY, size=14)
    card(s, 4.65, 1.5, 3.85, 3.75, "Novelty, to confirm",
         ["Closest: RetSyn (diffusion across phone/tabletop cameras, uses paired photos) and SCR-Net.",
          "Ours: unpaired, no phone labels; beats FDA and colour transfer; physics → diffusion both needed; "
          "disease checked per grade."], fill=PALE_TEAL, head_color=TEAL, size=14)
    card(s, 8.7, 1.5, 4.05, 3.75, "Questions for you",
         ["Is the device-conditioned model (gen_both) the world model you meant?",
          f"Zero labels now match 500 labels on AUROC ({_m(FT[WIN]['auc']):.4f}); F1 not yet significant. Enough for ISBI?",
          "Lead with the generator ablations, or with the zero-label = 500-label result?"],
         fill=PALE_AMBER, head_color=AMBER, size=14)
    rect(s, 0.6, 5.5, 12.15, 1.25, PALE_GREEN)
    text(s, 0.8, 5.6, 11.75, 1.05,
         [[("Bottom line: ", {"bold": True, "color": GREEN}),
           (f"with zero real phone labels, our synthetic twins plus self-training lift DR AUROC from {_m(b0['auc']):.4f} "
            f"to {_m(FT[WIN]['auc']):.4f}, the level of 500 real labels. The generator beats classical style transfer and "
            "needs both its stages. F1 is the open problem.", {})]], size=15)
    notes(s, "Next steps and the decisions I need from you. The main open question is how to frame "
             "the paper: the generative method gives a clear AUROC gain with zero labels, equal to "
             "250 real labels, but F1 does not improve yet.")


# --------------------------------------------------------------- part 3: overnight 13-14 Oct
ST_KEYS = [k for k in ("STA", "STB", "STC", "STD", "STE") if k in FT]
WIN = max(ST_KEYS, key=lambda k: _m(FT[k]["valf1"])) if ST_KEYS else None     # D073 rule
PSEUDO_N = {"STA": "984", "STB": "2,509", "STC": "812", "STD": "1,259", "STE": "984"}
ST_WHAT = {"STA": "our model (3 copies) labels only very confident phone photos; + twins",
           "STB": "our model (3 copies), looser confidence cut-off; + twins",
           "STC": "standard self-training: BRSET-only model labels the photos; no twins",
           "STD": "our model (1 copy) labels confident photos; + twins",
           "STE": "our model (3 copies), confident photos; NO twins"}


def _pair(x, y):
    return next((p for p in PAIRS if p["x"] == x and p["y"] == y), None)


def _pcell(p, key="auc"):
    """'+0.0149 [+0.002, +0.032]' for a head-to-head pair, green when the interval is above 0."""
    if p is None:
        return ("running", {"color": GREY})
    d, (lo, hi) = (p["dauc"], p["aucCI"]) if key == "auc" else (p["df1"], p["f1CI"])
    sty = GOOD if lo > 0 else (BADS if hi < 0 else {})
    return (f"{d:+.4f} ".replace("-", "−") + _ci(lo, hi), sty)


def part3(prs):
    b0, fin = FT["B0"], FT[BEST]
    have = lambda k: k in FT

    # divider
    s = base(prs, "", "", dark=True)
    text(s, 0.7, 1.9, 11.5, 0.45, "PART 3  ·  OVERNIGHT EXPERIMENTS, 13–14 OCTOBER", size=14, bold=True, color=ICE)
    text(s, 0.7, 2.45, 11.6, 1.6, "Can we fix F1, and is the generative model really the reason?",
         size=38, bold=True, color=WHITE, line=1.0)
    text(s, 0.7, 4.25, 11.0, 1.9,
         ["Four questions, each answered with 3-seed runs on the same 732 test images, zero real phone labels.",
          "Every number in this part is read from the result file named at the foot of the slide."], size=17, color=ICE)
    notes(s, "Part three is what we ran overnight after the first results: an attempt to fix F1, and the "
             "baselines and ablations a reviewer will ask for.")

    # overview: 4 questions -> answers
    s = base(prs, "Overnight · overview", "Four questions we asked overnight, and the short answers", FTS)
    qa = [("1  Can unlabelled phone photos fix F1?", "Self-training",
           (f"AUROC up to {_m(FT[WIN]['auc']):.4f} (= 500 real labels); F1 {_m(FT[WIN]['f1']):.4f}, up on every seed "
            "but not significant." if WIN else "running"), AMBER, PALE_AMBER),
          ("2  Is the gain just colour matching?", "FDA and colour transfer",
           ("No. Our generator beats both, significantly, although they match phone colour better."
            if have("BFDA") and have("BRH") else "running"), GREEN, PALE_GREEN),
          ("3  Are both stages needed at the reported setting?", "Ablation (gen_both 40k, 25 %)",
           ("Yes. Removing either stage loses AUROC significantly, and F1 too for two of them."
            if have("AQ1") else "running"), GREEN, PALE_GREEN),
          ("4  Is re-drawing a real eye better than RetSyn-style generation?", "Class-conditioned generator",
           (("Yes" if (_pair(BEST, "BCC") or {}).get("aucCI", [0])[0] > 0 else "Not clearly") +
            f": ours {_m(fin['auc']):.4f} vs RetSyn-style {_m(FT['BCC']['auc']):.4f}." if have("BCC")
            else "Running on 4 H100s; result due ≈ 14:00."), (TEAL if have("BCC") else GREY), (PALE_TEAL if have("BCC") else PALE))]
    for (x, y), (q, how, ans, col, fill) in zip([(0.6, 1.5), (6.83, 1.5), (0.6, 4.2), (6.83, 4.2)], qa):
        rect(s, x, y, 5.9, 2.55, fill)
        text(s, x + 0.2, y + 0.12, 5.5, 0.75, q, size=16, bold=True, color=NAVY, line=1.0)
        text(s, x + 0.2, y + 0.88, 5.5, 0.4, how, size=13, bold=True, color=col)
        text(s, x + 0.2, y + 1.3, 5.5, 1.15, ans, size=15)
    notes(s, "The four overnight questions and their one-line answers. The next slides show the evidence for each.")

    # how self-training works
    s = base(prs, "Overnight · 1. self-training", "How self-training uses the unlabelled phone photos",
             "scripts/120_predict.py, 121_pseudo_manifest.py, 122_pseudo_student.slurm; validation purity: fa_val_diag.py; D073")
    steps = [("Our model", "trained on BRSET + synthetic phone twins (no phone labels)", PALE_TEAL, TEAL),
             ("Scores 3,402 photos", "real mBRSET training images, labels never read", PALE, NAVY),
             ("Keeps confident ones", "very low score → 'healthy', very high → 'DR'; the rest are left out", PALE_AMBER, AMBER),
             ("Retrain", "BRSET + twins + these phone photos with their pseudo-labels", PALE_GREEN, GREEN)]
    for i, (h, sub, fill, col) in enumerate(steps):
        x = 0.6 + i * 3.1
        rect(s, x, 1.6, 2.8, 1.9, fill)
        text(s, x + 0.12, 1.7, 2.56, 0.5, h, size=16, bold=True, color=col, align=PP_ALIGN.CENTER)
        text(s, x + 0.12, 2.25, 2.56, 1.2, sub, size=13, align=PP_ALIGN.CENTER)
        if i < 3:
            arrow(s, x + 2.82, 2.55, x + 3.08, 2.55, color=GREY, width=2.5)
    text(s, 0.6, 3.85, 12.15, 0.4, "Why this could help F1, and how clean the pseudo-labels are (checked on validation, 725 images)",
         size=15, bold=True, color=NAVY)
    table(s, 0.6, 4.3, 7.4, [3.6, 1.8, 2.0],
          ["Confidence rule (validation)", "Images kept", "Wrong labels"],
          [["Score below 0.05 → 'healthy'", "142", ("0", GOOD)], ["Score below 0.10 → 'healthy'", "457", "12 (mostly grade 1)"],
           ["Score above 0.8 → 'DR'", "80", ("0", GOOD)], ["Score above 0.7 → 'DR'", "101", "1"]],
          size=13, row_h=0.42, aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    text(s, 8.3, 4.3, 4.45, 2.75,
         ["Real labels raised F1 partly by teaching what healthy PHONE eyes look like (false alarms 25 → 14 with 500 labels).",
          "Self-training gives the model real phone photos without anyone labelling them."], size=13)
    notes(s, "Self-training in four steps. Our zero-label model scores all 3,402 unlabelled phone photos. Only the "
             "photos it is very sure about are kept, with its own answer as the label. Then a new model is trained on "
             "BRSET, the synthetic twins and those real phone photos. On validation, the confident labels are almost "
             "always right.")

    # self-training results
    if ST_KEYS:
        s = base(prs, "Overnight · 1. self-training results", "Self-training: 500-label AUROC, F1 not yet significant",
                 FTS + "; pick rule D073 (validation DR F1)")
        rows = [[("BRSET only (baseline)", {"bold": True}), "BRSET only", "–", f"{_m(b0['valf1']):.4f}",
                 (f"{_m(b0['auc']):.4f}", {"bold": True}), "–", (f"{_m(b0['f1']):.4f}", {"bold": True}), "–"],
                ["Final pick, no self-training", "BRSET + twins (gen_both 40k, 25 %)", "–", f"{_m(fin['valf1']):.4f}",
                 (f"{_m(fin['auc']):.4f}", {"bold": True}), _delta_cells(fin)[0], (f"{_m(fin['f1']):.4f}", {"bold": True}), _delta_cells(fin)[1]]]
        for k in ST_KEYS:
            r = FT[k]; da, df = _delta_cells(r)
            nm = f"ST-{k[-1]}" + ("  (winner)" if k == WIN else "")
            rows.append([(nm, {"bold": k == WIN}), ST_WHAT[k], PSEUDO_N[k], f"{_m(r['valf1']):.4f}",
                         (f"{_m(r['auc']):.4f}", {"bold": True}), da, (f"{_m(r['f1']):.4f}", {"bold": True}), df])
        r5 = FT["B500"]
        rows.append([("500 real labels (reference)", {"color": GREY}), ("BRSET + 500 labelled phone photos", {"color": GREY}), "–",
                     (f"{_m(r5['valf1']):.4f}", {"color": GREY}), (f"{_m(r5['auc']):.4f}", {"color": GREY}), "–",
                     (f"{_m(r5['f1']):.4f}", {"color": GREY}), "–"])
        fills = [PALE_AMBER, PALE] + [PALE_GREEN if k == WIN else (WHITE if i % 2 == 0 else PALE) for i, k in enumerate(ST_KEYS)] + [WHITE]
        table(s, 0.6, 1.45, 12.15, [1.95, 3.15, 0.8, 0.85, 0.85, 1.85, 0.8, 1.9],
              ["Run (0 real labels)", "What the model learns from", "Phone photos used", "Val DR F1", "Test AUROC",
               "Δ AUROC vs baseline", "Test F1", "Δ F1 vs baseline"],
              rows, size=11, row_h=0.36, fills=fills, aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 6)
        pc, p5 = _pair(WIN, "STC"), _pair(WIN, "B500")
        text(s, 0.6, 5.55, 12.15, 1.35,
             [f"Winner picked on validation DR F1, as fixed before the runs: ST-{WIN[-1]}. Its test AUROC equals 500 real labels "
              f"(difference {p5['dauc']:+.4f}); its F1 is higher on all 3 seeds but the interval still includes 0, so the "
              "pre-set 'clear win' rule (AUROC and F1 both significant) is not met." if p5 else "",
              "Honest catch: standard self-training without our images (ST-C) already gets most of the AUROC gain; "
              f"our twins add mainly F1 inside self-training ({pc['df1']:+.3f}, interval just touching 0; cut-offs not matched)."
              if pc else ""], size=12, space_after=3)
        notes(s, "Results for the five self-training variants. The winner, chosen on validation, reaches the same test AUROC "
                 "as using 500 real labels, with zero labels. F1 goes up on every seed but the gain is too small to be "
                 "significant with 159 diseased test images. Plain self-training without our synthetic images also "
                 "gives most of the AUROC gain, so the synthetic images' own contribution here is mainly F1.")

    # generator vs classical style transfer (+ RetSyn-style)
    s = base(prs, "Overnight · 2 & 4. baselines", "Our generator beats simple style matching", FTS + "; D076")
    rows = [[("BRSET only (baseline)", {"bold": True}), "–", f"{_m(b0['auc']):.4f}", f"{_m(b0['f1']):.4f}", "–", "–"]]
    for k, nm, how in (("BFDA", "FDA twins", "swap the image's low-frequency colour/light with a random phone photo"),
                       ("BRH", "Colour-transfer twins", "match each image's colour statistics to a random phone photo"),
                       ("BCC", "RetSyn-style generator", "same network, makes a NEW image from 'phone + DR class' (no real eye)")):
        if have(k):
            r = FT[k]; p = _pair(BEST, k)
            rows.append([nm, how, f"{_m(r['auc']):.4f}", f"{_m(r['f1']):.4f}", _pcell(p, "auc"), _pcell(p, "f1")])
        else:
            rows.append([nm, how, ("running", {"color": GREY}), ("running", {"color": GREY}), ("running", {"color": GREY}), ("running", {"color": GREY})])
    rows.append([("Ours: " + _rn(fin), {"bold": True}), "re-draw each real BRSET eye as a phone photo (physics → diffusion)",
                 (f"{_m(fin['auc']):.4f}", {"bold": True}), (f"{_m(fin['f1']):.4f}", {"bold": True}), "–", "–"])
    table(s, 0.6, 1.45, 12.15, [2.3, 3.95, 0.9, 0.8, 2.1, 2.1],
          ["Synthetic images (25 % swap, 0 labels)", "How they are made", "Test AUROC", "Test F1", "Ours minus this: AUROC",
           "Ours minus this: F1"],
          rows, size=12, row_h=0.5, fills=[PALE_AMBER, WHITE, PALE, WHITE, PALE_GREEN],
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
    text(s, 0.6, 4.75, 12.15, 1.9,
         ["Colour inside the retina (mean RGB): real phone 153 / 73 / 42; FDA 148 / 74 / 44; colour transfer 158 / 76 / 45; "
          "ours 152 / 56 / 33. The classical twins match phone colour BETTER than ours, yet train a worse classifier, "
          "so our gain is not colour matching.",
          "Green = 95 % interval of the paired difference above 0 (same seeds, 2,000 patient resamples)."], size=13)
    notes(s, "Two classical ways to make phone-style twins, and a RetSyn-style generator that invents new images from a "
             "class label. Our generator beats the classical ones significantly even though they match phone colour "
             "better. The RetSyn-style comparison tells us whether re-drawing a real eye beats inventing one.")

    # ablation at the reported setting
    if have("AQ1"):
        s = base(prs, "Overnight · 3. ablation", "At the reported setting, both stages are needed", FTS + "; D076")
        rows = [[("Full method (physics → diffusion)", {"bold": True}), (f"{_m(fin['auc']):.4f}", {"bold": True}),
                 (f"{_m(fin['f1']):.4f}", {"bold": True}), "–", "–"]]
        for k, nm in (("AQ1", "Physics only (no diffusion)"), ("AQ2", "Diffusion only (no physics)"),
                      ("AQ3", "Diffusion from the plain BRSET image (no physics start)")):
            r = FT[k]; p = _pair(BEST, k)
            rows.append([nm, f"{_m(r['auc']):.4f}", f"{_m(r['f1']):.4f}", _pcell(p, "auc"), _pcell(p, "f1")])
        rows.append([("BRSET only (baseline)", {"color": GREY}), (f"{_m(b0['auc']):.4f}", {"color": GREY}),
                     (f"{_m(b0['f1']):.4f}", {"color": GREY}), "–", "–"])
        table(s, 0.6, 1.5, 12.15, [4.6, 1.4, 1.4, 2.375, 2.375],
              ["gen_both 40k, 25 % swap, 0 labels", "Test AUROC", "Test F1", "Full minus this: AUROC", "Full minus this: F1"],
              rows, size=13, row_h=0.55, fills=[PALE_GREEN, WHITE, PALE, WHITE, PALE_AMBER],
              aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
        text(s, 0.6, 5.1, 12.15, 1.7,
             [f"Physics alone is the 'preprocessing' part: it gives about zero ({_m(FT['AQ1']['auc']) - _m(b0['auc']):+.3f}). Diffusion without the physics start loses "
              "AUROC and makes F1 significantly worse. Only the combination works — the answer to 'is this just preprocessing?'.",
              "The earlier ablation (gen_tgt, 50 %) gave the same picture; this one matches the configuration we report."], size=14)
        notes(s, "The ablation repeated at exactly the configuration we report. Removing the diffusion stage, or the physics "
                 "starting point, loses AUROC significantly, and for two of them F1 as well. Physics alone, which is the "
                 "part that resembles preprocessing, gives no gain at all.")

    # what it proved + new winner
    if WIN:
        w = FT[WIN]; p5 = _pair(WIN, "B500")
        s = base(prs, "Overnight · the new winner", f"New best: ST-{WIN[-1]}, self-training on top of our synthetic twins",
                 FTS + "; D073, D077")
        stat(s, 0.6, 1.55, 3.9, f"{_m(w['auc']):.4f}", f"test DR AUROC, zero real labels (baseline {_m(b0['auc']):.4f}; "
             f"500 real labels {_m(FT['B500']['auc']):.4f})", color=GREEN, num_size=34, h=1.9)
        stat(s, 4.75, 1.55, 3.9, f"{_m(w['f1']):.4f}", f"test DR F1 (baseline {_m(b0['f1']):.4f}; higher on all 3 seeds, "
             "not significant)", color=AMBER, num_size=34, h=1.9)
        stat(s, 8.9, 1.55, 3.85, f"{_m(w['valf1']):.4f}", "validation DR F1 — the highest of all zero-label runs; "
             "this is how it was picked", color=NAVY, num_size=34, h=1.9)
        table(s, 0.6, 3.75, 12.15, [3.0, 1.7, 1.7, 1.7, 1.7, 2.35],
              ["Per seed", "Seed 0", "Seed 1", "Seed 2", "Mean", "Δ vs baseline [95% CI]"],
              [["Test AUROC"] + [f"{a:.4f}" for a in w["auc"]] + [(f"{_m(w['auc']):.4f}", {"bold": True}), _delta_cells(w)[0]],
               ["Test F1"] + [f"{a:.4f}" for a in w["f1"]] + [(f"{_m(w['f1']):.4f}", {"bold": True}), _delta_cells(w)[1]],
               ["Baseline test F1"] + [f"{a:.4f}" for a in b0["f1"]] + [f"{_m(b0['f1']):.4f}", "–"]],
              size=13, row_h=0.45, bold_cols=(0,), aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 5)
        rect(s, 0.6, 5.75, 12.15, 1.0, PALE_GREEN)
        text(s, 0.8, 5.82, 11.75, 0.88,
             [[("What it is: ", {"bold": True, "color": GREEN}),
               ("BRSET + our synthetic phone twins (25 %) + 2,509 real phone photos labelled by our own zero-label model. "
                "No human-labelled phone image is used for training.", {})]], size=14)
        notes(s, f"The new best configuration. With zero labelled phone images it reaches a test AUROC of "
                 f"{_m(w['auc']):.4f}, the same as training with 500 real labels. F1 is {_m(w['f1']):.4f}: higher than the "
                 "baseline on every seed, but the improvement is not statistically significant yet.")

        s = base(prs, "Overnight · what it proved", "What last night proved, in plain words", FTS)
        cards = [("Proved", "Our generator beats the simple alternatives",
                  "It beats FDA and colour transfer, and beats physics-only and diffusion-only versions of itself.", GREEN, PALE_GREEN),
                 ("Proved", "Zero labels can match 500 labels on AUROC",
                  f"Our twins + self-training (ST-{WIN[-1]}): {_m(w['auc']):.4f} vs {_m(FT['B500']['auc']):.4f} with 500 real labels.", GREEN, PALE_GREEN),
                 ("Not yet", "A significant F1 win",
                  f"F1 {_m(w['f1']):.4f} vs {_m(b0['f1']):.4f}: up on every seed, interval still includes 0 (only 159 diseased test images).",
                  AMBER, PALE_AMBER),
                 ("Caution", "Self-training does much of the work",
                  "Standard self-training without our images also gets most of the AUROC gain; our twins add mainly F1.", RED, PALE_RED)]
        for (x, y), (tag, head, body, col, fill) in zip([(0.6, 1.5), (6.83, 1.5), (0.6, 4.2), (6.83, 4.2)], cards):
            rect(s, x, y, 5.9, 2.55, fill)
            text(s, x + 0.2, y + 0.12, 5.5, 0.4, tag.upper(), size=13, bold=True, color=col)
            text(s, x + 0.2, y + 0.55, 5.5, 0.75, head, size=19, bold=True, color=NAVY, line=1.0)
            text(s, x + 0.2, y + 1.35, 5.5, 1.1, body, size=14)
        notes(s, "Two things are now proved: the generative model is what makes the difference, and with self-training "
                 "zero labels match 500 labels on AUROC. Two things are not: F1 is not yet a significant win, and much of "
                 "the self-training gain would come even without our images.")


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
