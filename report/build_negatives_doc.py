"""Build a three-page Word record of the negative experiments.

Same formatting contract as report/build_overview_doc.py: Times New Roman,
narrow (0.5 inch) margins, no em dashes, no decorative glyphs, first person
singular, and every number traceable to results/.

    python report/build_negatives_doc.py
"""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

OUT = "/home/users/sthummala2/brset-convnextv2/report/BRSET_mBRSET_Negative_Results.docx"
FONT = "Times New Roman"


def set_font(run, size, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rf = rpr.get_or_add_rFonts()
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)


def para(doc, text, size=9.5, bold=False, italic=False, align=None,
         before=0, after=2.0, bullet=False, mono=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before); pf.space_after = Pt(after); pf.line_spacing = 1.0
    if align is not None:
        p.alignment = align
    if bullet:
        pf.left_indent = Inches(0.18); pf.first_line_indent = Inches(-0.18)
        text = "•\t" + text
    r = p.add_run(text)
    set_font(r, size, bold, italic)
    if mono:
        r.font.name = "Consolas"
        r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), "Consolas")
        r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), "Consolas")
    return p


def head(doc, t):
    para(doc, t, size=10.5, bold=True, before=7, after=2.5)


def table(doc, rows, widths, header=True):
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].width = Inches(widths[j])
            p = cells[j].paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            set_font(p.add_run(str(v)), 9, bold=(header and i == 0))
    return t


def main():
    doc = Document()
    n = doc.styles["Normal"]; n.font.name = FONT; n.font.size = Pt(9.5)
    n.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.5)
        s.left_margin = s.right_margin = Inches(0.5)

    para(doc, "Negative Results Record: BRSET to mBRSET Transfer", size=13.5,
         bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    para(doc, "Every intervention tested, its measured effect, and what it rules out",
         size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    para(doc, "Sahith Reddy Thummala, 10 October 2026", size=9,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=3)

    head(doc, "1. Measurement discipline")
    para(doc, "Nothing below is reported as an effect unless it exceeds the spread "
              "produced by changing the random seed alone. I measured that spread by "
              "retraining one identical configuration three times:", after=2)
    para(doc, "    DR AUC  0.9377 / 0.9519 / 0.9492    spread 0.0142\n"
              "    DR F1   0.8250 / 0.8449 / 0.8396    spread 0.0199\n"
              "    missed patients  27 / 31 / 36       spread 9",
         size=8.5, mono=True, after=3)
    para(doc, "All results use one fixed 732-image mBRSET test set with 159 diabetic "
              "retinopathy positives, verified element-wise identical across every "
              "model compared. F1 is quoted at the threshold tuned on validation, "
              "which is what a deployed system can do. AUC is the headline metric "
              "because it requires no threshold.", after=2)

    head(doc, "2. Architecture interventions. Five tested, none measurable.")
    table(doc, [
        ["Intervention", "DR F1", "vs baseline 0.8250", "Verdict"],
        ["Domain separation, adversarial 0.25", "0.8053", "-0.0197", "inside noise"],
        ["Domain separation, adversarial 0.05", "0.8227", "-0.0023", "inside noise"],
        ["Domain separation, adversarial 0.00", "0.8232", "-0.0018", "inside noise"],
        ["Entropy-gated routing, learned gate", "0.8452", "+0.0202", "equals the floor"],
        ["Entropy routing, conditional entropy", "0.8414", "+0.0164", "inside noise"],
        ["Aspect-ratio correction, 3 seeds", "0.8385", "+0.0020", "inside noise"],
        ["Abstention on low-quality images", "n/a", "n/a", "premise false"],
    ], [2.6, 0.9, 1.5, 1.5])
    para(doc, "The abstention premise was tested directly and is false: the model's "
              "missed patients are not concentrated on poor-quality images, with a "
              "16.4 percent miss rate on images carrying artifacts against 20.0 percent "
              "on clean ones.", before=2, after=2)

    head(doc, "3. Why all five were unmeasurable")
    para(doc, "A single measurement explains the entire set. Pooling both datasets and "
              "training normally already matches the best model obtainable by training "
              "directly on target data:", after=2)
    table(doc, [
        ["Model", "val-threshold F1", "DR AUC"],
        ["Pooling BRSET + all 3,402 target, 3-seed mean", "0.8365", "0.9463"],
        ["Best of six models trained directly on target", "0.8355", "0.9446"],
        ["Difference", "+0.0010", "+0.0017"],
    ], [3.7, 1.6, 1.2])
    para(doc, "The setting was saturated. There was no headroom for any architecture to "
              "win, which is why five successive attempts all landed inside the noise "
              "floor. This is the single most useful negative in the record.", before=2, after=2)

    head(doc, "4. Image synthesis. Four regimes tested, none positive.")
    para(doc, "A two-stage translator converts BRSET images into synthetic mBRSET-like "
              "images: a bounded differentiable acquisition model with 19 interpretable "
              "parameters, followed by a bounded learned residual refiner. It fits 13 "
              "appearance statistics to within |z| <= 0.32 on held-out images, including "
              "the per-channel chromatic fractions and per-channel sharpness. Downstream, "
              "with zero real target training labels:", after=2)
    table(doc, [
        ["Training data", "DR AUC", "change", "val F1", "change"],
        ["Real BRSET only (baseline)", "0.9104", "", "0.7774", ""],
        ["50 percent synthetic", "0.9044", "-0.0060", "0.7451", "-0.0323"],
        ["25 percent synthetic", "0.8996", "-0.0108", "0.7333", "-0.0441"],
        ["100 percent synthetic (replacement)", "0.8888", "-0.0216", "0.7278", "-0.0496"],
        ["Real plus synthetic (union)", "0.8652", "-0.0452", "0.7509", "-0.0265"],
    ], [2.6, 0.9, 0.9, 0.9, 0.9])
    para(doc, "Against noise floors of 0.0142 AUC and 0.0199 F1: the two light-synthesis "
              "arms sit inside the floor and have no effect, while replacement and union "
              "cause measurable harm. More synthetic data is monotonically worse on AUC, "
              "and that dose response is itself evidence the effect is real rather than "
              "run-to-run variation. The union arm is the configuration specifically "
              "requested, real images and synthetic images trained together, and it is "
              "the worst of the four on AUC.", before=2, after=2)

    head(doc, "5. Controls that rule out competing explanations")
    para(doc, "Class balance does not explain the transfer gap.", bullet=True, after=1)
    para(doc, "    BRSET is 6.6 percent positive, mBRSET 23.4 percent, so pooling more "
              "than doubles the absolute positive count, 748 to 1,545. Resampling BRSET "
              "alone to a 23.4 percent positive rate, two seeds, closes +0.0017 of a "
              "+0.0629 gap. Three percent, far inside the floor.", size=9, after=2)
    para(doc, "The appearance gap is real, large, and was closed.", bullet=True, after=1)
    para(doc, "    Measured on 400 images per dataset: mBRSET carries 43 percent more "
              "blue as a fraction of channel energy and 2.86 times the blue variation, "
              "while red is unchanged at 1.03. With the field-of-view rim excluded, it "
              "carries 2.1 to 2.5 times more high-frequency energy, not less, which "
              "inverts the premise of the fundus-degradation literature.", size=9, after=2)
    para(doc, "A published method is worse than doing nothing.", bullet=True, after=1)
    para(doc, "    GDRNet's published augmentation settings score a normalised appearance "
              "distance of 7.75 against untouched BRSET's 5.13, and retain only 46.5 "
              "percent of red-lesion candidates under a classifier-free detector validated "
              "at 1.000 retention on an identity transform and 0.005 under heavy blur.", size=9, after=2)

    head(doc, "6. What the negatives collectively establish")
    para(doc, "Pooling a few thousand real target images closes the cross-device gap, and "
              "no domain-adaptation architecture tested improves on it, because the "
              "setting is already saturated.", bullet=True, after=1)
    para(doc, "Neither class balance nor image appearance explains the residual zero-shot "
              "gap. Both were measured and both were ruled out.", bullet=True, after=1)
    para(doc, "Matching the measured appearance of the target camera, to within a third of "
              "a standard deviation on thirteen statistics, does not improve transfer and "
              "at high substitution rates harms it. This contradicts the operating "
              "assumption of the degradation-augmentation literature.", bullet=True, after=1)
    para(doc, "Published degradation settings are actively harmful on this dataset pair, "
              "both in appearance distance and in measured destruction of lesion evidence.",
         bullet=True, after=2)
    para(doc, "Two mechanisms remain under test: whether the translator destroys lesion "
              "evidence, and whether it moves the images toward the target in a learned "
              "feature space at all rather than only in the hand-chosen statistics.",
         size=9, italic=True, after=1)

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
