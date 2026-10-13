"""
Build a one-page Word overview of the BRSET to mBRSET project.

Written for a reader who is completely new to the work: problem statement,
why it matters, what was tried, what was measured, and where it stands.

Formatting follows the project's document standards: Times New Roman
throughout, narrow (0.5 inch) margins, no em dashes, no decorative glyphs,
first person singular, and no numbers that are not in results/.

    python report/build_overview_doc.py
"""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

OUT = "/home/users/sthummala2/brset-convnextv2/report/BRSET_mBRSET_Overview.docx"
FONT = "Times New Roman"


def set_font(run, size, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    # Word picks the East Asian / complex-script font separately; pin both so
    # nothing silently falls back to Calibri.
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)


def para(doc, text, size=10, bold=False, italic=False, align=None,
         space_before=0, space_after=2.5, bullet=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.0
    if align is not None:
        p.alignment = align
    if bullet:
        pf.left_indent = Inches(0.19)
        pf.first_line_indent = Inches(-0.19)
        text = "•\t" + text
    set_font(p.add_run(text), size, bold, italic)
    return p


def heading(doc, text):
    para(doc, text, size=10.5, bold=True, space_before=6, space_after=2.5)


SECTIONS = [
    ("The problem", [
     "Diabetic retinopathy damages blood vessels at the back of the eye and is a leading cause of "
     "blindness in working-age adults. It is largely preventable if caught early, so the task is "
     "screening people before symptoms appear.",
     "Screening means photographing the retina. Hospitals use large tabletop cameras and get clean "
     "images. Community programs, which reach most people with diabetes, use cheap handheld cameras "
     "that clip onto a phone and get worse ones.",
     "There are far too few ophthalmologists to read every photo, so an AI model should read them. But "
     "the large labeled datasets all come from hospitals, which is where the model learns.",
    ]),
    ("Why this particular gap is worth closing", [
     "Two real Brazilian datasets make it concrete: BRSET, roughly 11,000 hospital images, and mBRSET, "
     "roughly 3,400 handheld images from community screening.",
     "Tested on hospital images, the model misses about 10 diseased patients out of 162. Tested on "
     "handheld images, it misses 50 out of 159, a third of everyone who has the disease.",
     "That failure lands on exactly the population with the least access to a specialist who would "
     "catch the error.",
    ]),
    ("What I tried. All five came back negative.", [
     "Domain separation network: separate what the two cameras share from what is specific to each.",
     "Entropy-gated routing: decide image by image which representation to trust.",
     "Abstention: have the model refuse to answer on poor-quality images.",
     "Aspect-ratio fix: a real bug of my own that compressed every hospital image horizontally by 23 "
     "percent while leaving handheld images untouched.",
     "Synthetic degradation: damage hospital images until they look handheld, manufacturing training data.",
    ]),
    ("The two measurements that reframed the project", [
     "Noise floor. Retraining the identical model three times, changing only the random seed, moved the "
     "score by 0.0199 F1 and 9 patients on its own. Every improvement I had measured was smaller than "
     "that, so none of them were improvements.",
     "Ceiling. Simply pooling both datasets and training on the mixture matches the best model trained "
     "directly on handheld data, 0.8455 against 0.8456. The simple approach had already saturated, "
     "leaving no room for the clever methods to find.",
     "So this was not five separate failures. It was one wrong choice of testing ground, repeated five times.",
    ]),
    ("What survives, and is useful to other researchers", [
     "The camera gap is not blur. Measured across 400 images per dataset at four resolutions, handheld "
     "images are noisier, higher in contrast, dimmer and more unevenly lit. Published degradation "
     "methods nearly all assume blur.",
     "A published method makes things worse. GDRNet (MICCAI 2023) default settings move hospital images "
     "further from handheld ones than doing nothing, because they correct an axis that is not the "
     "problem.",
     "An entire family of cheap fixes is ruled out. Exhaustively sweeping every decision threshold "
     "proves that methods which only recalibrate model confidence cannot work here.",
    ]),
    ("Where it stands", [
     "The contribution is not a winning architecture. It is that pooling a few thousand target-camera "
     "images beats every adaptation method tested, that the gap is noise and contrast rather than "
     "blur, and that a published method moves the wrong way.",
     "Next step: move the degradation experiment into the zero-shot setting, where a genuine gap of "
     "about 20 patients remains and a real effect could actually be detected.",
    ]),
]


def main():
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    for s in doc.sections:          # Word's "Narrow" margin preset
        s.top_margin = s.bottom_margin = Inches(0.5)
        s.left_margin = s.right_margin = Inches(0.5)

    para(doc, "Cross-Camera Diabetic Retinopathy Screening", size=14, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1)
    para(doc, "Transferring a retinal disease classifier from hospital cameras to handheld ones",
         size=9.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1)
    para(doc, "Sahith Reddy Thummala", size=9.5,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=3)

    for title, bullets in SECTIONS:
        heading(doc, title)
        for b in bullets:
            para(doc, b, bullet=True)

    doc.save(OUT)

    words = sum(len(b.split()) for _, bs in SECTIONS for b in bs)
    print(f"wrote {OUT}")
    print(f"{len(SECTIONS)} sections, {sum(len(b) for _, b in SECTIONS)} bullets, ~{words} words")


if __name__ == "__main__":
    main()
