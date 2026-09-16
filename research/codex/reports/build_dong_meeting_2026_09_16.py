"""Private, evidence-linked 12-slide advisor meeting deck.

Contains licensed training images. Run only on an allocated Slurm compute node;
keep the output and image manifest in the Git-ignored reports/private directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import zipfile
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "research/codex/reports"
PRIVATE = REPORTS / "private"
OUTPUT = PRIVATE / "BRSET_mBRSET_Dong_Meeting_2026-09-16.pptx"
CHECKS = PRIVATE / "dong_meeting_2026-09-16_checks.json"
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
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(214, 223, 228)

STEP0 = ROOT / "research/codex/updates/2026-09-10/validation_summary.json"
STEP2 = ROOT / "research/codex/step2/full_pool/baseline_assessment_three_seeds.json"
STEP2_PROTOCOL = ROOT / "research/codex/step2/full_pool/protocol/v1.json"
STEP2_MANIFEST = ROOT / "research/codex/step2/full_pool/protocol/full_pool_manifest.csv"
STEP3 = ROOT / "research/codex/step3/c1_validation_three_seeds.json"
STEP4_FIT = ROOT / "research/codex/step4/degradation_refit.json"
STEP4 = ROOT / "research/codex/step4/seed0_validation_summary.json"
STEP5_GATE = ROOT / "research/codex/step5/appearance_error_gate.json"
SOURCE = ROOT / "research/codex/step5/source_only_seed0_verification.json"
GFP = ROOT / "research/codex/step5/gfp_summary.json"
SPATIAL = ROOT / "research/codex/step5/spatial_diagnostic_summary.json"
IMAGES = [PRIVATE / "step4_gallery_row_2.png", PRIVATE / "step4_gallery_row_3.png"]
IMAGE_CHECKS = PRIVATE / "advisor_visual_checks.json"
RAW_IMAGE_AUDIT = PRIVATE / "gallery_source_audit.json"

PAPERS = {
    "LAT · CVPR 2021": "https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Lesion-Aware_Transformers_for_Diabetic_Retinopathy_Grading_CVPR_2021_paper.html",
    "CCT-Net · ICCV 2021": "https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_CCT-Net_Category-Invariant_Cross-Domain_Transfer_for_Medical_Single-to-Multiple_Disease_Diagnosis_ICCV_2021_paper.html",
    "Samba · NeurIPS 2024": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aa0c4d28021c0b273480f9e2aab83a6-Abstract-Conference.html",
    "CheXWorld · CVPR 2025": "https://openaccess.thecvf.com/content/CVPR2025/html/Yue_CheXWorld_Exploring_Image_World_Modeling_for_Radiograph_Representation_Learning_CVPR_2025_paper.html",
    "GDRNet · MICCAI 2023": "https://conferences.miccai.org/2023/papers/658-Paper1591.html",
    "ISBI 2027 dates": "https://biomedicalimaging.org/2027/",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def fit_prevalence():
    groups = defaultdict(list)
    with STEP2_MANIFEST.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["role"] == "fit":
                groups[row["domain"]].append(row)
    assert {domain: len(rows) for domain, rows in groups.items()} == {"BRSET": 11372, "mBRSET": 3402}
    values = {}
    for domain, rows in groups.items():
        values[domain] = {
            label: sum(int(row[label]) for row in rows) / len(rows)
            for label in ("diabetic_retinopathy", "macular_edema")
        }
    return values


def assert_compute() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Build the image-containing deck only on an allocated compute node")


def text(slide, x, y, w, h, content, size=18, bold=False, color=INK,
         align=PP_ALIGN.LEFT, italic=False, link=None, margin=.02):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(margin)
    frame.margin_top = frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.alignment = align
    p.space_before = p.space_after = Pt(0)
    p.line_spacing = 1.0
    run = p.add_run()
    run.text = str(content)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    if link:
        run.hyperlink.address = link
    return shape


def box(slide, x, y, w, h, fill=WHITE, line=LINE, radius=False):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(.8)
    return shape


def rule(slide, x, y, w, color=LINE, thickness=.015):
    shape = box(slide, x, y, w, thickness, color, color)
    return shape


def base(prs, number, kicker, title, subtitle=None, source=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    box(slide, 0, 0, 13.333, .11, NAVY, NAVY)
    text(slide, .62, .29, 4.0, .29, kicker.upper(), 10, True, TEAL)
    text(slide, .62, .63, 12.1, .55, title, 27, True, NAVY)
    if subtitle:
        text(slide, .63, 1.21, 12.0, .45, subtitle, 14, False, GREY)
    rule(slide, .62, 6.96, 12.1)
    if source:
        text(slide, .63, 7.01, 11.7, .30, source, 9, False, GREY)
    text(slide, 12.40, 7.01, .34, .3, f"{number:02d}", 10, True, GREY, PP_ALIGN.RIGHT)
    return slide


def card(slide, x, y, w, h, title, body, fill=PALE, accent=TEAL,
         title_size=16, body_size=14):
    box(slide, x, y, w, h, fill, fill, True)
    box(slide, x, y, .055, h, accent, accent)
    text(slide, x+.20, y+.12, w-.35, .40, title, title_size, True, accent)
    text(slide, x+.20, y+.56, w-.40, h-.62, body, body_size, False, INK)


def table(slide, headers, rows, x, y, widths, row_h=.56, font=14,
          highlight=None, aligns=None):
    total = sum(widths)
    box(slide, x, y, total, row_h, NAVY, NAVY)
    cx = x
    for j, (value, width) in enumerate(zip(headers, widths)):
        text(slide, cx+.09, y+.04, width-.18, row_h-.08, value, font-1, True, WHITE,
             (aligns[j] if aligns else PP_ALIGN.LEFT))
        cx += width
    for i, row in enumerate(rows):
        ry = y + (i+1)*row_h
        fill = PALE_TEAL if highlight is not None and i == highlight else WHITE if i%2 else PALE
        box(slide, x, ry, total, row_h, fill, LINE)
        cx = x
        for j, (value, width) in enumerate(zip(row, widths)):
            color = TEAL if highlight is not None and i == highlight else INK
            text(slide, cx+.09, ry+.03, width-.18, row_h-.06, value, font,
                 highlight is not None and i == highlight, color,
                 (aligns[j] if aligns else PP_ALIGN.LEFT))
            cx += width


def linked(slide, x, y, w, label, key, size=12):
    text(slide, x, y, w, .31, label, size, True, TEAL, link=PAPERS[key])


def make_deck() -> Presentation:
    step0 = load_json(STEP0)
    p2 = load_json(STEP2)
    p3 = load_json(STEP3)
    p4 = load_json(STEP4)
    fit = load_json(STEP4_FIT)
    gate = load_json(STEP5_GATE)
    source = load_json(SOURCE)
    gfp = load_json(GFP)
    spatial = load_json(SPATIAL)
    protocol = load_json(STEP2_PROTOCOL)
    prevalence = fit_prevalence()
    assert protocol["initialization"]["model"].startswith("convnextv2_large")
    heldout = {(record["preprocessing"], record["arm"]): record["mean"]
               for record in step0["heldout_distances"]}
    assert abs(heldout["legacy_512", "fitted_frozen"] - 1.2607503005384497) < 1e-9
    assert p4["reference"]["metrics"]["diabetic_retinopathy"]["f1_positive"] > .87
    assert source["assessment_metrics"]["diabetic_retinopathy"]["n_images"] == 732
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "BRSET to mBRSET: research update for Professor Dong"
    prs.core_properties.subject = "Evidence-based Steps 0–5 and research decision"

    # 1 — one-sentence state of the project.
    s = base(prs, 1, "Research update · 16 September 2026",
             "BRSET → mBRSET: what the experiments actually show",
             "Image-level diabetic retinopathy (DR) and macular edema (ME) · ConvNeXt V2 Large")
    card(s, .70, 2.05, 3.79, 2.05, "Established", "Joint training is the strongest ordinary reference across three seeds.", PALE_TEAL, TEAL, 20, 17)
    card(s, 4.77, 2.05, 3.79, 2.05, "Negative evidence", "Matching camera appearance did not produce a meaningful DR/ME gain.", PALE_AMBER, AMBER, 20, 17)
    card(s, 8.84, 2.05, 3.79, 2.05, "Next question", "Which diagnostic evidence is lost when we move between datasets?", PALE, NAVY, 20, 17)
    text(s, 1.1, 5.12, 11.1, .8,
         "A novel method has not yet been demonstrated. Today’s aim is to choose a testable mechanism, not to claim a win.",
         20, True, NAVY, PP_ALIGN.CENTER)
    text(s, .63, 7.01, 11.4, .30, "Prepared for Professor Dong · results verified against frozen run artifacts", 9, False, GREY)

    # 2 — data and settings.
    s = base(prs, 2, "Study setup", "Two training boundaries, two different questions",
             "The main study uses labeled mBRSET training images; the separate source-only run measures the zero-target-label gap.",
             "Evidence: step2/full_pool/protocol/v1.json; step5/source_only_v1.json")
    card(s, .66, 1.88, 5.92, 1.55, "Main question · joint transfer",
         "Fit: BRSET 11,372 + mBRSET 3,402 images. "
         f"DR+ {prevalence['BRSET']['diabetic_retinopathy']:.1%} vs {prevalence['mBRSET']['diabetic_retinopathy']:.1%}; "
         f"ME+ {prevalence['BRSET']['macular_edema']:.1%} vs {prevalence['mBRSET']['macular_edema']:.1%}. "
         "Target validation 725; test 732.", PALE_TEAL, TEAL, 17, 14)
    card(s, 6.77, 1.88, 5.91, 1.55, "Separate question · strict source-only",
         "BRSET fit 11,372 → BRSET validation 2,451 → mBRSET historical test 732.", PALE, NAVY, 17, 16)
    table(s, ["What is fixed", "Value", "Why it matters"], [
        ["Model", "ConvNeXt V2 Large", "The same backbone throughout Steps 2–5"],
        ["Outputs", "Any DR; ME", "Two independent image-level labels"],
        ["Compute", "5,775 updates; batch 64", "Same total update budget in baseline arms"],
        ["Selection", "Validation checkpoint + thresholds", "Historical test reuse is disclosed"],
    ], .72, 3.65, [2.3, 3.2, 6.34], .48, 13)
    text(s, .75, 6.15, 11.8, .27,
         "Recipe: pretrained Large; 560→512 pixels; AdamW 3×10⁻⁵; focal loss + Mixup; EMA 0.999; four-flip evaluation.",
         11, False, GREY)
    text(s, .75, 6.47, 11.8, .27,
         "F1 = precision–recall balance at a chosen threshold. AUROC = ranking quality across thresholds.",
         11, False, GREY)

    # 3 — plain-language timeline.
    s = base(prs, 3, "Roadmap", "What each completed step asked, in plain language",
             "A failed mechanism still helps by removing an unsupported explanation.",
             "Evidence: PLAN.md and dated Step-2/3/4/5 reviews")
    items = [
        ("0  Image simulation", "Can we make BRSET look more mobile-like?", "Appearance improved; diagnosis untested."),
        ("1  Fair protocol", "Can every method take the same exam?", "Splits, labels, budget and thresholds frozen."),
        ("2  Ordinary baselines", "Does more labeled data help?", "Joint B1 strongest on average."),
        ("3  Sampling controls", "Is B1 only a sampling effect?", "No compelling replacement for B1."),
        ("4  Degradation training", "Does target-like appearance improve DR/ME?", "No arm passed the validation gate."),
        ("5  Error diagnostics", "What is missing from the model’s evidence?", "Transfer gap measured; simple fixes failed."),
    ]
    for i, (title, question, result) in enumerate(items):
        x = .65 + (i%2)*6.16
        y = 1.84 + (i//2)*1.57
        card(s, x, y, 5.92, 1.39, title, question + "\n" + result,
             PALE_TEAL if i in (1,2) else PALE_AMBER if i in (4,5) else PALE,
             TEAL if i in (1,2) else AMBER if i in (4,5) else NAVY, 16, 13)

    # 4 — original and degraded images.
    s = base(prs, 4, "Dong’s requested visual check",
             "Verified original and degraded fundus images",
             "Columns 1–4 share one BRSET photograph; only BRSET is transformed. The mBRSET image is unpaired.",
             "Image provenance: step4/preflight.json; reports/private/gallery_source_audit.json")
    labels = ["BRSET original", "A1 · FundusAug artifacts", "A2 · target-fitted", "A3 · no overlays", "Real mBRSET · unpaired"]
    for block, path in enumerate(IMAGES):
        ly = 1.75 + block*2.43
        for i, label in enumerate(labels):
            text(s, 1.36+i*2.12, ly, 2.12, .31, label, 11, True, NAVY, PP_ALIGN.CENTER)
        s.shapes.add_picture(str(path), Inches(1.36), Inches(ly+.34), width=Inches(10.60))
    text(s, .80, 6.64, 11.75, .25,
         "Visual similarity is not a paired camera reconstruction or proof that tiny lesions are preserved.", 10, False, RED, PP_ALIGN.CENTER)

    # 5 — original appearance result and the fair protocol.
    s = base(prs, 5, "Steps 0–1", "Appearance improved; diagnosis still had to be tested",
             "The distance is a standardized sum over five image averages: sharpness, brightness, contrast, falloff and saturation.",
             "Evidence: updates/2026-09-10/validation_summary.json; full_pool/protocol/v1.json")
    table(s, ["Step-0 preparation", "Unchanged BRSET", "Frozen fitted transform", "Reading"], [
        ["Direct 512 resize", f"{heldout['legacy_512', 'BRSET_unaltered']:.3f}",
         f"{heldout['legacy_512', 'fitted_frozen']:.3f}", "Closer on five means"],
        ["560 resize → 512 crop", f"{heldout['model_eval_560_crop512', 'BRSET_unaltered']:.3f}",
         f"{heldout['model_eval_560_crop512', 'fitted_frozen']:.3f}", "Closer on five means"],
    ], .74, 1.95, [3.55, 2.35, 2.65, 3.32], .68, 15)
    card(s, .74, 4.32, 5.74, 1.55, "What Step 0 proved",
         "A frozen transform generalizes to a new training-only sample on these appearance summaries.", PALE_TEAL, TEAL, 17, 15)
    card(s, 6.73, 4.32, 5.86, 1.55, "What Step 1 prevented",
         "Changing labels, patient splits, update budget or test thresholds to make a method look better.", PALE, NAVY, 17, 15)
    text(s, .78, 6.27, 11.88, .39,
         "These distances are not F1, not a whole-image distribution measure and not evidence of clinical preservation.",
         13, True, AMBER)

    # 6 — baseline numbers from historical test.
    s = base(prs, 6, "Step 2 · three seeds", "Ordinary joint training is the baseline to beat",
             "All methods use ConvNeXt V2 Large and the same 732-image historical mBRSET test set; values are three-seed means.",
             "Evidence: step2/full_pool/baseline_assessment_three_seeds.json; STEP_2_FINAL_REVIEW.md")
    rows = []
    for arm, label in [("B0", "Target only"), ("B1", "Joint BRSET + mBRSET"), ("B2", "BRSET → mBRSET fine-tune")]:
        v = p2["aggregate"][arm]
        rows.append([arm+" · "+label,
                     f"{v['diabetic_retinopathy']['f1_positive']['mean']:.4f}",
                     f"{v['diabetic_retinopathy']['auroc']['mean']:.4f}",
                     f"{v['macular_edema']['f1_positive']['mean']:.4f}",
                     f"{v['macular_edema']['auroc']['mean']:.4f}"])
    table(s, ["Training strategy", "DR F1", "DR AUROC", "ME F1", "ME AUROC"],
          rows, .66, 2.04, [4.00, 1.97, 1.97, 1.97, 1.72], .82, 15, 1,
          [PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*4)
    card(s, .71, 5.55, 5.83, .96, "B1 − B0 · mean F1", "DR +0.0233   ·   ME +0.0379", PALE_TEAL, TEAL, 15, 16)
    card(s, 6.80, 5.55, 5.81, .96, "Boundary", "This is supervised joint transfer, not zero-shot.", PALE, NAVY, 15, 15)

    # 7 — sampler control.
    s = base(prs, 7, "Step 3 · sampling controls", "Rebalancing the two datasets did not replace B1",
             "B1 naturally draws ~23% mBRSET; C1 forces 50%, using the same images, model and update budget.",
             "Evidence: step3/c1_validation_three_seeds.json; STEP_3_FINAL_REVIEW.md")
    table(s, ["Three-seed validation difference · C1 − B1", "DR", "ME"], [
        ["F1 at selected threshold", f"{p3['aggregate']['diabetic_retinopathy']['f1_positive']['difference_mean']:+.4f}",
         f"{p3['aggregate']['macular_edema']['f1_positive']['difference_mean']:+.4f}"],
        ["AUROC", f"{p3['aggregate']['diabetic_retinopathy']['auroc']['difference_mean']:+.4f}",
         f"{p3['aggregate']['macular_edema']['auroc']['difference_mean']:+.4f}"],
        ["Average precision", f"{p3['aggregate']['diabetic_retinopathy']['average_precision']['difference_mean']:+.4f}",
         f"{p3['aggregate']['macular_edema']['average_precision']['difference_mean']:+.4f}"],
    ], .80, 2.03, [7.05, 2.35, 2.35], .72, 16)
    card(s, .85, 5.41, 11.68, .95, "Conclusion",
         "More target-image exposure gave small F1 changes but lower AUROC; retain natural B1.", PALE_AMBER, AMBER, 17, 16)
    text(s, .88, 6.48, 11.5, .27,
         "C2 matched target-label proportions; C3 combined that with equal domains. Neither passed the seed-0 screen.",
         11, False, GREY)

    # 8 — direct classifier test of degradation.
    s = base(prs, 8, "Step 4 · decisive negative", "Target-like appearance did not improve diagnosis",
             "A1: FundusAug artifacts only. A2: fit to mBRSET training averages. A3: refit with spots/holes/halo off.",
             "Evidence: step4/degradation_refit.json; seed0_validation_summary.json")
    text(s, .74, 1.79, 11.7, .34,
         f"Refit appearance distance on held-out BRSET: unchanged {fit['identity_validation']['distance_mean']:.3f} → full {fit['arms']['full']['validation']['distance_mean']:.3f} / no overlays {fit['arms']['overlay_free']['validation']['distance_mean']:.3f}",
         15, True, TEAL)
    rows = []
    ref = p4["reference"]["metrics"]
    rows.append(["B1 natural joint", f"{ref['diabetic_retinopathy']['f1_positive']:.4f}", "—",
                 f"{ref['macular_edema']['f1_positive']:.4f}", "—"])
    for label, key in [("A1 · FundusAug component", "A1"), ("A2 · full fitted", "A2"), ("A3 · no overlays", "A3")]:
        arm = next(value for name, value in p4["arms"].items() if name.startswith(key))
        m = arm["metrics"]
        rows.append([label, f"{m['diabetic_retinopathy']['f1_positive']:.4f}",
                     f"{m['diabetic_retinopathy']['f1_positive']-ref['diabetic_retinopathy']['f1_positive']:+.4f}",
                     f"{m['macular_edema']['f1_positive']:.4f}",
                     f"{m['macular_edema']['f1_positive']-ref['macular_edema']['f1_positive']:+.4f}"])
    table(s, ["Arm", "DR F1", "Δ DR", "ME F1", "Δ ME"], rows,
          .67, 2.36, [4.08, 2.02, 1.82, 2.02, 1.72], .64, 14,
          aligns=[PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*4)
    text(s, .80, 5.76, 11.7, .37,
         "Gate: +0.0100 F1 in one label, no material loss elsewhere. All A arms failed.",
         16, True, RED)
    text(s, .80, 6.16, 11.7, .32,
         "Interpretation: appearance fit cannot establish lesion preservation; A3 DR AUROC rose only +0.0038.",
         12, False, GREY)
    linked(s, .80, 6.54, 6.0, "FundusAug source: GDRNet · MICCAI 2023  ↗", "GDRNet · MICCAI 2023", 10)

    # 9 — error premise and world-model boundary.
    s = base(prs, 9, "Step 5A · test the premise", "Five appearance measures did not reliably explain errors",
             "We asked whether camera-like image summaries predict B1's DR/ME mistakes on patient-grouped validation data.",
             "Evidence: step5/appearance_error_gate.json; APPEARANCE_ERROR_GATE_REVIEW.md")
    dr_gate = gate['labels']['diabetic_retinopathy']
    me_gate = gate['labels']['macular_edema']
    def ci(label):
        low, high = label['overall_error_model']['difference_95pct']
        return f"[{low:+.4f}, {high:+.4f}]"
    table(s, ["Error-prediction evidence", "DR", "ME"], [
        ["Added AUROC from appearance", f"{dr_gate['overall_error_model']['difference_point']:+.4f}",
         f"{me_gate['overall_error_model']['difference_point']:+.4f}"],
        ["Patient-bootstrap 95% interval", ci(dr_gate), ci(me_gate)],
        ["Positive-case miss AUROC", f"{dr_gate['positive_false_negative_model']['bootstrap']['appearance_auc_point']:.3f}",
         f"{me_gate['positive_false_negative_model']['bootstrap']['appearance_auc_point']:.3f}"],
    ], .78, 1.97, [6.20, 2.77, 2.77], .72, 15)
    card(s, .83, 5.35, 11.66, 1.04, "Decision on Dong’s world-model suggestion",
         "This measured appearance signal failed the gate; a fundus adaptation of CheXWorld is deferred.",
         PALE_AMBER, AMBER, 16, 14)
    linked(s, 1.00, 6.56, 4.7, "Open CheXWorld · CVPR 2025", "CheXWorld · CVPR 2025", 11)

    # 10 — strict source-only and frozen feature diagnostics.
    s = base(prs, 10, "Step 5B · gap and feature probes", "Measure the transfer gap; test two simple explanations",
             "BRSET-only measures transfer without target labels; GFP tests channel pruning; top-1 tests local versus global evidence.",
             "Evidence: source_only_seed0_verification.json; gfp_summary.json; spatial_diagnostic_summary.json")
    sm = source["assessment_metrics"]
    table(s, ["Strict BRSET-only → mBRSET test", "DR", "ME"], [
        ["F1", f"{sm['diabetic_retinopathy']['f1_positive']:.4f}", f"{sm['macular_edema']['f1_positive']:.4f}"],
        ["AUROC", f"{sm['diabetic_retinopathy']['auroc']:.4f}", f"{sm['macular_edema']['auroc']:.4f}"],
        ["Sensitivity · positives detected", f"{sm['diabetic_retinopathy']['sensitivity']:.4f}", f"{sm['macular_edema']['sensitivity']:.4f}"],
    ], .72, 1.94, [7.06, 2.34, 2.34], .57, 14)
    card(s, .78, 4.48, 5.73, 1.25, "GFP · remove weak channels",
         f"Retained {gfp['selected']['retained_features'][0]:,}/1,536 DR and {gfp['selected']['retained_features'][1]:,}/1,536 ME channels; ΔF1 = 0 for both.",
         PALE_AMBER, AMBER, 15, 14)
    card(s, 6.76, 4.48, 5.74, 1.25, "Top-1 · strongest location",
         f"DR ΔAUROC {spatial['aggregate']['diabetic_retinopathy']['auroc']['difference_mean']:+.4f}, but ΔF1 {spatial['aggregate']['diabetic_retinopathy']['f1_positive']['difference_mean']:+.4f}; ME ΔAUROC {spatial['aggregate']['macular_edema']['auroc']['difference_mean']:+.4f}.",
         PALE_AMBER, AMBER, 15, 14)
    text(s, .83, 6.14, 11.7, .46,
         "Neither feature probe passed the F1 gate; P1/P2/P3 were not trained. A top patch is not a verified lesion.",
         14, True, NAVY)

    # 11 — literature and collision audit.
    s = base(prs, 11, "What the literature permits", "Published methods define the novelty boundary",
             "These published papers motivate tests; none demonstrates a new method for our exact BRSET/mBRSET task.",
             "Click the linked titles for primary conference papers. The proposed direction remains untested.")
    cards = [
        ("Lesion-aware regions", "LAT learns lesion-relevant regions from image labels; local evidence is occupied prior art.", "LAT · CVPR 2021"),
        ("Cross-domain category cues", "CCT-Net refines disease heatmaps in a different transfer task; category-conditioned maps alone are prior art.", "CCT-Net · ICCV 2021"),
        ("Small-severity patterns", "Samba explicitly models small severe patches across domains; patch modeling alone is not new.", "Samba · NeurIPS 2024"),
        ("Latent domain actions", "CheXWorld predicts representations under known transforms in radiographs; its fundus benefit is unproven here.", "CheXWorld · CVPR 2025"),
    ]
    for i, (heading, body, key) in enumerate(cards):
        x = .73 + (i%2)*6.10
        y = 1.82 + (i//2)*1.93
        card(s, x, y, 5.81, 1.72, heading, body, PALE, NAVY, 16, 13)
        linked(s, x+.20, y+1.31, 5.30, key + "  ↗", key, 12)
    text(s, .85, 6.09, 11.63, .47,
         "Research question: can local DR evidence become reliable across devices without losing ME?",
         16, True, TEAL)

    # 12 — actionable, falsifiable plan and advisor decision.
    s = base(prs, 12, "Decision for today", "A disciplined ConvNeXt V2 Large path to a paper claim",
             "Internal target: complete draft by 15 October; ISBI four-page deadline is 26 October 2026.",
             "Deadline: biomedicalimaging.org/2027/ · linked source below")
    for i, (head, body) in enumerate([
        ("1  Audit misses", "Use three B1 seeds to review patient-level DR/ME misses by confidence, image quality and suspected retinal cues; ask Dong to inspect a stratified sample."),
        ("2  Freeze one mechanism", "Only if local evidence is supported, specify label-wise local/global agreement on ConvNeXt V2 Large; distinguish it from prior work before training."),
        ("3  Test controls", "Compare B1, local-only, agreement-only and combined at equal data/updates. Predeclare +0.01 F1 gate; protect both labels and AUROC."),
        ("4  Replicate or stop", "Only a passing validation seed gets seeds 1/2, patient-level intervals and a post-freeze assessment, disclosed as historically reused."),
    ]):
        x = .73 + (i%2)*6.11
        y = 1.80 + (i//2)*1.82
        card(s, x, y, 5.83, 1.64, head, body, PALE_TEAL if i==0 else PALE, TEAL if i==0 else NAVY, 16, 13)
    box(s, .80, 5.63, 11.70, .89, PALE_AMBER, PALE_AMBER, True)
    text(s, 1.04, 5.73, 11.25, .66,
         "Dong’s decision: keep the primary paper supervised joint transfer, or prioritize strict source-only transfer? Would he review the missed-positive image set?",
         15, True, AMBER)
    linked(s, .91, 6.57, 4.6, "Official ISBI 2027 dates  ↗", "ISBI 2027 dates", 11)

    assert len(prs.slides) == 12
    return prs


def verify(prs: Presentation):
    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        slide_xml = "\n".join(archive.read(name).decode("utf-8", "ignore")
                              for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
        media = [name for name in names if name.startswith("ppt/media/")]
        hyperlink_relationships = sum(archive.read(name).count(b"hyperlink") for name in names
                                      if name.startswith("ppt/slides/_rels/") and name.endswith(".rels"))
    bounds = all(shape.left >= 0 and shape.top >= 0 and
                 shape.left + shape.width <= prs.slide_width and
                 shape.top + shape.height <= prs.slide_height
                 for slide in prs.slides for shape in slide.shapes)
    all_text = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if shape.has_text_frame)
    image_audit = load_json(IMAGE_CHECKS)
    source_audit = load_json(RAW_IMAGE_AUDIT)
    row_hashes = {path.name: digest(path) for path in IMAGES}
    pictures = [shape for shape in prs.slides[3].shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    image_aspect_preserved = len(pictures) == len(IMAGES)
    for picture, path in zip(pictures, IMAGES):
        with Image.open(path) as source_image:
            actual_aspect = source_image.width / source_image.height
        shown_aspect = picture.width / picture.height
        image_aspect_preserved &= abs(actual_aspect - shown_aspect) < 0.002
    results = {
        "pass": len(prs.slides) == 12 and bounds and len(media) == 2 and image_aspect_preserved
                and hyperlink_relationships >= 5
                and FONT in slide_xml and all(image_audit["row_crop_sha256"][name] == sha for name, sha in row_hashes.items())
                and source_audit["pass"] and all(term in all_text for term in ("ConvNeXt V2 Large", "unpaired", "0.8338", "0.902", "+0.0101"))
                and "STEP 4" in all_text and "STEP 5" in all_text
                and "RETFound" not in all_text,
        "output": str(OUTPUT),
        "sha256": digest(OUTPUT),
        "slides": len(prs.slides),
        "media_files": len(media),
        "hyperlink_relationships": hyperlink_relationships,
        "times_new_roman_present": FONT in slide_xml,
        "all_shape_bounds_inside_slide": bounds,
        "fundus_image_aspect_preserved": image_aspect_preserved,
        "verified_private_row_sha256": row_hashes,
        "source_gallery_sha256": image_audit["source_gallery_sha256"],
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                         (STEP0, STEP2, STEP2_PROTOCOL, STEP2_MANIFEST, STEP3, STEP4_FIT, STEP4, STEP5_GATE,
                          SOURCE, GFP, SPATIAL, IMAGE_CHECKS, RAW_IMAGE_AUDIT)},
        "privacy": "Advisor-only: contains licensed BRSET/mBRSET training images; do not commit or distribute publicly.",
    }
    CHECKS.write_text(json.dumps(results, indent=2) + "\n")
    if not results["pass"]:
        raise RuntimeError(results)
    return results


def main():
    assert_compute()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    prs = make_deck()
    prs.save(OUTPUT)
    print(json.dumps(verify(prs), indent=2))


if __name__ == "__main__":
    main()
