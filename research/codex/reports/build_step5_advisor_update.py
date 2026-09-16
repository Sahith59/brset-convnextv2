"""Build the evidence-grounded Step-5 advisor DOCX and PPTX."""
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches as PInches

import build_research_record as style


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOCX = HERE / "BRSET_mBRSET_Step5_Diagnostic_Update_for_Dong.docx"
PPTX = HERE / "BRSET_mBRSET_Step5_Diagnostic_Update_for_Dong.pptx"
CHECKS = HERE / "step5_advisor_update_checks.json"
FONT = "Times New Roman"
UPDATED = date.today().strftime("%d %B %Y").lstrip("0")

SOURCE = ROOT / "research/codex/step5/source_only_seed0_verification.json"
FEATURES = ROOT / "research/codex/step5/feature_extraction_summary.json"
GFP = ROOT / "research/codex/step5/gfp_summary.json"
SPATIAL = ROOT / "research/codex/step5/spatial_diagnostic_summary.json"
ERRORS = ROOT / "research/codex/step5/diagnostic_error_analysis.json"
PROTOCOL = ROOT / "research/codex/step5/FROZEN_FEATURE_PROTOCOL.md"
REVIEW = ROOT / "research/codex/step5/STEP5_DIAGNOSTIC_REVIEW.md"
RETFOUND = ROOT / "research/codex/step5/retfound_compatibility_audit.json"
RETFOUND_PREFLIGHT = ROOT / "research/codex/step5/retfound_preflight.json"
RETFOUND_SMOKE = ROOT / "research/codex/step5/retfound_smoke.json"
RETFOUND_RESUME = ROOT / "research/codex/step5/retfound_resume_checks.json"
RETFOUND_LAUNCH = ROOT / "research/codex/step5/launch_retfound_seed0.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load():
    required = (SOURCE, FEATURES, GFP, SPATIAL, ERRORS, PROTOCOL, REVIEW, RETFOUND,
                RETFOUND_PREFLIGHT, RETFOUND_SMOKE, RETFOUND_RESUME, RETFOUND_LAUNCH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    return tuple(json.loads(path.read_text()) for path in (
        SOURCE, FEATURES, GFP, SPATIAL, ERRORS, RETFOUND,
        RETFOUND_PREFLIGHT, RETFOUND_SMOKE, RETFOUND_RESUME, RETFOUND_LAUNCH))


def bullet(document, text):
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.first_line_indent = Inches(-0.16)
    style.doc_text(p, "•  " + text)
    style.style_doc_paragraph(p, after=2)


def build_docx(source, features, gfp, spatial, errors, retfound, rf_preflight, rf_smoke, rf_resume, rf_launch):
    document = Document()
    section = document.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.68)
    section.left_margin = section.right_margin = Inches(0.72)
    for name, size, bold, italic in [("Normal", 10, False, False), ("Title", 18, True, False)]:
        doc_style = document.styles[name]
        doc_style.font.name = FONT
        doc_style.font.size = Pt(size)
        doc_style.font.bold = bold
        doc_style.font.italic = italic
        doc_style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    style.add_page_number(section)

    p = document.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style.doc_text(p, "BRSET to mBRSET: Step-5 Diagnostic Update", 18, True)
    style.style_doc_paragraph(p, after=2)
    p = document.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style.doc_text(p, "Source-only transfer, frozen-feature tests and evidence-based decision", 11, True)
    style.style_doc_paragraph(p, after=2)
    p = document.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style.doc_text(p, f"Prepared for Professor Dong | {UPDATED}", 9, italic=True)
    style.style_doc_paragraph(p, after=8)

    p = document.add_paragraph()
    style.doc_text(p, "Executive conclusion—", 10, True, True)
    style.doc_text(p, "Step 5 produced useful transfer evidence but did not establish a new method. Strict BRSET-only training transfers poorly at the selected operating point, mainly because it misses positive mBRSET cases. A published feature-pruning comparator and a simple label-specific spatial-pooling diagnostic both failed their frozen validation gates. We therefore stop the conditional P1/P2/P3 mechanism branch and audit a protocol-matched retinal foundation model next.")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    style.style_doc_paragraph(p, after=7)

    style.add_heading(document, "1.", "QUESTION AND EVIDENCE BOUNDARY")
    style.add_body(document, "The question was whether the remaining BRSET/mBRSET error could be explained by (i) harmful or redundant final encoder channels or (ii) clinically useful local evidence being diluted by global average pooling. We froze the strongest ordinary joint model, B1 seed 0, and tested both explanations on the 725-image mBRSET validation split. No mBRSET test feature was extracted for these mechanism screens.")
    style.add_doc_table(document, [
        ["Evidence", "Cohort", "Allowed use"],
        ["Strict source-only", "11,372 BRSET fit; 2,451 BRSET selection; 732 mBRSET assessment", "Quantify transfer gap"],
        ["GFP and spatial heads", "14,774 joint fit; 725 mBRSET validation", "Select/reject mechanism"],
        ["mBRSET assessment", "732 images", "Historical reporting only; never used to rescue a failed mechanism"],
    ])
    style.add_body(document, "B1 means supervised joint training on labeled BRSET and labeled mBRSET training images. It is not zero-shot transfer. The strict source-only run alone trains and selects without any target-domain training or validation exposure.", "Terminology:")

    style.add_heading(document, "2.", "STRICT SOURCE-ONLY TRANSFER")
    style.add_body(document, "The source-only model used the same ConvNeXt V2 Large training recipe and 5,775-update budget as the Step-2 baselines. Its checkpoint and the DR/ME thresholds were selected only on BRSET validation data. The selected thresholds were 0.399 for DR and 0.451 for ME. The locked mBRSET assessment was then run once.")
    rows = [["Label", "F1", "AUROC", "Average precision", "Precision", "Sensitivity", "Specificity", "FN / positives"]]
    for key, label in [("diabetic_retinopathy", "DR"), ("macular_edema", "ME")]:
        m = source["assessment_metrics"][key]
        positives = m["confusion_matrix"][1][0] + m["confusion_matrix"][1][1]
        rows.append([label, f"{m['f1_positive']:.4f}", f"{m['auroc']:.4f}", f"{m['average_precision']:.4f}",
                     f"{m['precision']:.4f}", f"{m['sensitivity']:.4f}", f"{m['specificity']:.4f}",
                     f"{m['confusion_matrix'][1][0]} / {positives}"])
    style.add_doc_table(document, rows)
    style.add_body(document, "Sensitivity is the fraction of positive images detected. Specificity is the fraction of negative images rejected. AUROC measures ranking over every threshold, while F1 measures the selected operating point. Here, high AUROC and specificity coexist with sensitivity near 0.60: the source model still ranks many target images sensibly, but its source-selected representation and boundary miss too many target positives.", "Plain-language reading:")
    style.add_body(document, "This is evidence of a source-to-target dataset gap. It cannot isolate camera hardware as the cause because device, site, population, prevalence, image quality and unmeasured factors change together.", "Causal limit:")

    style.add_heading(document, "3.", "DESCRIPTIVE SOURCE-ONLY VERSUS JOINT ERRORS")
    style.add_body(document, "The following comparison helps describe what labeled target training changes. It is not a fair winner comparison because B1 used labeled mBRSET fitting data and the source-only model did not.")
    rows = [["Label", "Source F1", "B1 F1", "Errors fixed", "New errors", "FN fixed", "New FN", "FP fixed", "New FP"]]
    for key, label in [("diabetic_retinopathy", "DR"), ("macular_edema", "ME")]:
        record = errors["source_only_vs_joint_b1_assessment"][key]
        transitions = record["transitions_source_to_joint_b1"]
        rows.append([label, f"{record['source_metrics']['f1_positive']:.4f}", f"{record['joint_b1_metrics']['f1_positive']:.4f}",
                     transitions["errors_fixed"], transitions["new_errors"], transitions["false_negatives_fixed"],
                     transitions["new_false_negatives"], transitions["false_positives_fixed"], transitions["new_false_positives"]])
    style.add_doc_table(document, rows)
    style.add_body(document, "For DR, B1 recovered 34 source-only false negatives but introduced 17 new false positives. For ME, it fixed 15 false negatives and introduced no new assessment errors. This supports the practical value of labeled target data; it does not reveal a single causal mechanism.")

    style.add_heading(document, "4.", "FROZEN FEATURE EXTRACTION AND INTEGRITY")
    style.add_body(document, f"The exact B1 seed-0 exponential-moving-average checkpoint was frozen. For each image, the final encoder produced a 1,536-channel 16×16 feature map. The fit cache contained {features['fit_images']:,} images and the validation cache {features['selection_images']:,} images with four deterministic evaluation views. Reconstructing the original head from the saved features reproduced every saved logit and probability with maximum absolute error {features['reproduction']['probability_max_abs_error']:.1f}. This proves that the diagnostic input matches B1 exactly.")
    style.add_body(document, "Global average pooling (G0) averages each channel over all 256 spatial positions. A top-k head (G1) first computes a label-specific score at each position and then averages only the k highest positions. The test therefore asks whether concentrated local evidence improves decisions when the backbone is unchanged.")

    style.add_heading(document, "5.", "GFP FEATURE-PRUNING COMPARATOR")
    style.add_body(document, "GFP is a published validation-selected pruning procedure. It removes groups of final features only when the frozen classifier does not need them. We reproduced it as a comparator; it is not our proposed contribution. The selected group size was 16.")
    gd = gfp["delta_from_unmasked_b1"]["diabetic_retinopathy"]
    gm = gfp["delta_from_unmasked_b1"]["macular_edema"]
    style.add_doc_table(document, [
        ["Label", "Retained channels", "ΔF1", "ΔAUROC", "ΔAverage precision", "Gate"],
        ["DR", f"{gfp['selected']['retained_features'][0]} / 1536", f"{gd['f1_positive']:+.5f}", f"{gd['auroc']:+.5f}", f"{gd['average_precision']:+.5f}", "FAIL"],
        ["ME", f"{gfp['selected']['retained_features'][1]} / 1536", f"{gm['f1_positive']:+.5f}", f"{gm['auroc']:+.5f}", f"{gm['average_precision']:+.5f}", "FAIL"],
    ])
    style.add_body(document, "GFP retained nearly everything and changed no F1 value. Under this checkpoint and search, removable final-channel redundancy is not a useful explanation for the remaining errors. The result does not prove that all pruning methods fail.", "Conclusion:")

    style.add_heading(document, "6.", "GLOBAL VERSUS TOP-1 SPATIAL EVIDENCE")
    style.add_body(document, "We trained small heads on the frozen features with three deterministic seeds. Each head used 20 epochs, AdamW, learning rate 10⁻³, weight decay 10⁻⁴, batch size 64, focal loss and cosine decay. The mean DR/ME AUROC selected k=1 among k ∈ {1, 4, 8, 16, 32}; the frozen engineering gate was then applied.")
    rows = [["Metric", "G0 mean", "Top-1 mean", "Difference", "Patient-bootstrap 95% interval"]]
    boot = errors["spatial_failure"]["patient_bootstrap"]
    labels = [("diabetic_retinopathy", "DR"), ("macular_edema", "ME")]
    metrics = [("f1_positive", "f1", "F1"), ("auroc", "auroc", "AUROC"), ("average_precision", "average_precision", "Average precision")]
    for key, short in labels:
        for metric, boot_metric, shown in metrics:
            agg = spatial["aggregate"][key][metric]
            interval = boot[key][boot_metric]["patient_bootstrap_95pct"]
            rows.append([f"{short} {shown}", f"{agg['G0_mean']:.4f}", f"{agg['G1_mean']:.4f}",
                         f"{agg['difference_mean']:+.4f}", f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"])
    style.add_doc_table(document, rows)
    style.add_body(document, "DR AUROC increased in all three head seeds and its patient-bootstrap interval excluded zero. That means the top-scoring location contains a repeatable DR ranking cue. The selected-threshold DR F1 did not improve because extra positive detections came with extra false positives. ME AUROC decreased in all three seeds, and its average precision also decreased.")
    style.add_body(document, "A high-scoring patch is not automatically a lesion. Without lesion annotations, it could represent the optic disc, illumination, camera boundary, pathology or another correlated structure. We therefore do not call this lesion localization.", "Interpretation limit:")
    style.add_body(document, "The gate required at least +0.01 F1 for one label, no worse than −0.01 F1 for the other and no worse than −0.01 AUROC for either. Mean DR/ME F1 changes were −0.0012 and +0.0048. The gate failed.", "Frozen decision:")

    style.add_heading(document, "7.", "FAILURE AUDIT: COMPUTE ERRORS VERSUS SCIENTIFIC RESULTS")
    style.add_doc_table(document, [
        ["Event", "Type", "Finding and treatment"],
        ["Job 4409940", "Implementation failure", "Missing CUBLAS_WORKSPACE_CONFIG under deterministic CUDA; environment corrected, protocol unchanged; rerun 4409983 passed."],
        ["Initial head array 4409991", "Scheduling-policy correction", "Two tasks co-located on one node; array/finalizer cancelled after 45 s and all partial outputs ignored."],
        ["Jobs 4409997/8/9 + 4410000", "Valid scientific run", "Seeds 0/1/2 ran on nodes 031/032/033; finalizer completed and gate failed."],
        ["P1/P2/P3", "Deliberately not launched", "They were conditional on a passing spatial diagnostic. Stopping is the predeclared decision, not a runtime failure."],
        ["4410101 / 4410102", "Implementation/dependency failure", "JSON serialization failed after checks; dependent smoke cancelled. Corrected preflight 4410136 passed."],
        ["4410105 / 4410107", "Scheduling/evidence correction", "Wrong-partition job cancelled before execution; first smoke did not prove backbone gradients and was excluded."],
        ["4410138", "Determinism failure", "Resume states differed slightly; deterministic algorithms enabled and exact rerun 4410171 passed."],
    ])
    style.add_body(document, "The implementation failure and scheduling correction do not affect the reported science because neither partial attempt contributed to the final summaries. The negative GFP and spatial findings come only from completed, checked runs.")

    style.add_heading(document, "8.", "WHAT HAS BEEN RULED OUT")
    style.add_doc_table(document, [
        ["Tested explanation", "Evidence", "Bounded conclusion"],
        ["Sampling balance alone", "Step-3 controls", "Did not replace natural joint B1."],
        ["Target-like global appearance alone", "Step-4 A1/A2/A3", "No arm passed the +0.01 F1 replication gate."],
        ["Five appearance variables explain errors", "Step-5 appearance-error gate", "Intervals crossed zero; degradation-conditioned world-model branch suspended."],
        ["Removable final-channel redundancy", "GFP", "Nearly all channels retained; no F1 gain."],
        ["Global pooling alone hides a two-label solution", "Top-1 diagnostic", "DR ranking cue present; decision gate failed and ME ranking worsened."],
    ])
    style.add_body(document, "These conclusions apply to the tested implementations, cohorts and metrics. They do not establish universal impossibility.")

    style.add_heading(document, "9.", "NEXT PLANNED WORK")
    style.add_body(document, "The existing RETFound companion results cannot be compared directly with B1. Although the same 4,859 mBRSET files are represented, 2,262 files have different split assignments and 271 files have different DR labels because the companion predicts referable DR (ICDR ≥2), whereas this study predicts any DR. The companion has no ME output.", "Completed compatibility audit:")
    style.add_doc_table(document, [
        ["Protocol item", "B1", "Existing RETFound companion", "Aligned?"],
        *[[row["item"].capitalize(), row["B1"], row["existing_companion"], "Yes" if row["aligned"] else "No"]
          for row in retfound["protocol_matrix"]],
    ])
    style.add_body(document, "The old RETFound scripts also use 224-pixel input, 50 epochs with batch 24, a separate seed-42 split, single-dataset runs and a different selection rule. Their published-looking numbers must not appear as a fair B1 comparison. The official checkout and pretrained ViT-Large weight are usable assets for a new matched implementation.")
    style.add_body(document, "A new matched implementation now uses the exact B1 fit/validation cohorts, any-DR and ME outputs, 560-resize/512-crop input, 5,775 updates, effective batch 64, identical selection logic and no assessment access.", "Matched implementation:")
    style.add_doc_table(document, [
        ["Gate", "Job", "Verified evidence", "Status"],
        ["Preflight", rf_preflight["job_id"], f"{rf_preflight['counts']['fit_images']:,} fit; {rf_preflight['counts']['selection_images']} validation; no assessment", "PASS"],
        ["Full-backbone feasibility", rf_smoke["job_id"], f"gradient coverage {rf_smoke['backbone_gradient_numel_coverage']:.1f}; {rf_smoke['peak_allocated_gib']:.2f} GiB; {rf_smoke['estimated_seed_hours']:.2f} h", "PASS"],
        ["Exact interruption/resume", rf_resume["job_id"], f"{rf_resume['updates']} updates; {len(rf_resume['state_mismatches'])} state mismatches", "PASS"],
        ["Full seed 0", rf_launch["job_id"], f"running; first loss {rf_launch['early_health']['loss']:.5f}; stderr empty", "PENDING RESULT"],
    ])
    bullet(document, "Wait for seed 0 and independently verify the completed validation summary; the first finite update is not a performance result.")
    bullet(document, "Replicate only a material validation gain under the frozen rule; never open assessment to rescue a failed validation screen.")
    bullet(document, "Treat RETFound as a stronger published backbone comparator. A gain would strengthen the baseline and would not by itself be the paper novelty.")
    bullet(document, "Discuss with Dong whether the paper should emphasize rigorous cross-device transfer and falsified mechanisms or reserve time for one final mechanism derived from the backbone/error audit.")
    style.add_body(document, "Current novelty status: no defensible novel-method performance claim has been demonstrated. The current contribution is a rigorous evidence chain that establishes the strongest ordinary baseline and eliminates several plausible but unsupported mechanisms.", "Current novelty status:")

    style.add_heading(document, "10.", "METRIC DEFINITIONS")
    style.add_doc_table(document, [
        ["Term", "Meaning"],
        ["F1", "Harmonic mean of precision and sensitivity at one selected threshold."],
        ["AUROC", "Probability that a random positive ranks above a random negative across thresholds."],
        ["Average precision", "Precision–recall ranking summary; useful when positives are uncommon."],
        ["Sensitivity", "Fraction of positive images predicted positive."],
        ["Specificity", "Fraction of negative images predicted negative."],
        ["Patient bootstrap", "Resampling patients, rather than individual images, to preserve within-patient dependence."],
    ])
    style.add_body(document, "Evidence files: source_only_seed0_verification.json; feature_extraction_summary.json; gfp_summary.json; spatial_diagnostic_summary.json; diagnostic_error_analysis.json; retfound_preflight.json; retfound_smoke.json; retfound_resume_checks.json; launch_retfound_seed0.json; STEP5_DIAGNOSTIC_REVIEW.md.")
    document.save(DOCX)


def build_pptx(source, features, gfp, spatial, errors, retfound, rf_preflight, rf_smoke, rf_resume, rf_launch):
    prs = Presentation(); prs.slide_width = PInches(13.333); prs.slide_height = PInches(7.5)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    style.textbox(slide, .7, 1.25, 12.0, .75, "BRSET to mBRSET: Step-5 Diagnostic Update", 29, True, style.BLUE, PP_ALIGN.CENTER)
    style.textbox(slide, 1.0, 2.2, 11.3, .5, "Source-only transfer • frozen-feature tests • evidence-based decision", 19, False, style.INK, PP_ALIGN.CENTER)
    style.ppt_box(slide, 1.5, 3.25, 10.3, 1.2, "Step 5 narrows the problem, but it does not yet establish a novel method.", style.LIGHT_AMBER, style.AMBER, 21, True)
    style.textbox(slide, 1.0, 6.45, 11.3, .4, f"Prepared for Professor Dong | {UPDATED}", 13, False, style.MUTED, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Executive status")
    style.ppt_box(slide, .65, 1.55, 3.75, .75, "Established", style.LIGHT_GREEN, style.GREEN, 20, True)
    style.bullets(slide, ["Joint B1 is the strongest ordinary baseline", "Strict source-only gap is quantified", "All evidence chains independently checked"], .78, 2.55, 3.45, 3.5, 16)
    style.ppt_box(slide, 4.78, 1.55, 3.75, .75, "Closed negative", style.LIGHT_AMBER, style.AMBER, 20, True)
    style.bullets(slide, ["Global appearance matching", "Final-channel pruning", "Simple top-k spatial pooling"], 4.91, 2.55, 3.45, 3.5, 16)
    style.ppt_box(slide, 8.91, 1.55, 3.75, .75, "Next", style.PALE, style.BLUE, 20, True)
    style.bullets(slide, ["Matched RETFound seed 0 running", "Validate before replication", "Treat backbone gain as a baseline"], 9.04, 2.55, 3.45, 3.5, 16)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Frozen evidence boundary")
    items = [("Strict source-only", "BRSET fit + selection\nOne locked mBRSET assessment", .45),
             ("Frozen B1 features", "14,774 fit\n725 mBRSET validation", 3.65),
             ("GFP + spatial heads", "Mechanism selection\nNo test features", 6.85),
             ("Frozen decision", "Pass → P1/P2/P3\nFail → backbone audit", 10.05)]
    for title, body, x in items:
        style.ppt_box(slide, x, 2.0, 2.75, .7, title, style.PALE, style.BLUE, 17, True)
        style.ppt_box(slide, x, 3.0, 2.75, 1.25, body, style.WHITE, style.BLUE, 15, False)
    for x in (3.25, 6.45, 9.65):
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, PInches(x), PInches(3.25), PInches(.32), PInches(.42))
        arrow.fill.solid(); arrow.fill.fore_color.rgb = style.BLUE; arrow.line.fill.background()
    style.textbox(slide, .8, 5.35, 11.7, .75, "Validation decides the mechanism. Historical test results cannot rescue a failed gate.", 18, True, style.AMBER, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Strict BRSET-only transfer misses target positives")
    rows = [["Label", "F1", "AUROC", "AP", "Precision", "Sensitivity", "Specificity", "FN / positive"]]
    for key, label in [("diabetic_retinopathy", "DR"), ("macular_edema", "ME")]:
        m = source["assessment_metrics"][key]; positives = m["confusion_matrix"][1][0] + m["confusion_matrix"][1][1]
        rows.append([label, f"{m['f1_positive']:.4f}", f"{m['auroc']:.4f}", f"{m['average_precision']:.4f}", f"{m['precision']:.4f}", f"{m['sensitivity']:.4f}", f"{m['specificity']:.4f}", f"{m['confusion_matrix'][1][0]} / {positives}"])
    style.ppt_table(slide, rows, .35, 1.55, 12.65, 1.75, [1.0, 1.15, 1.25, 1.15, 1.3, 1.35, 1.35, 1.5], 12)
    style.ppt_box(slide, .8, 3.85, 5.7, 1.3, "High AUROC + high specificity\nThe model retains ranking information", style.PALE, style.BLUE, 19, True)
    style.ppt_box(slide, 6.85, 3.85, 5.7, 1.3, "Sensitivity ≈ 0.60\nToo many mBRSET positives are missed", style.LIGHT_AMBER, style.AMBER, 19, True)
    style.textbox(slide, .75, 5.8, 11.8, .55, "Dataset gap ≠ camera-only causal effect", 18, True, style.MUTED, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "What labeled target training changes", "Descriptive source-only → B1 transitions on the historical assessment")
    rows = [["Label", "Errors fixed", "New errors", "FN fixed", "New FN", "FP fixed", "New FP"]]
    for key, label in [("diabetic_retinopathy", "DR"), ("macular_edema", "ME")]:
        t = errors["source_only_vs_joint_b1_assessment"][key]["transitions_source_to_joint_b1"]
        rows.append([label, t["errors_fixed"], t["new_errors"], t["false_negatives_fixed"], t["new_false_negatives"], t["false_positives_fixed"], t["new_false_positives"]])
    style.ppt_table(slide, rows, .75, 1.65, 11.8, 1.8, [1.3, 1.6, 1.6, 1.45, 1.3, 1.45, 1.3], 14)
    style.bullets(slide, ["DR: B1 recovers 34 source false negatives but adds 17 false positives.", "ME: B1 recovers 15 false negatives and adds no new errors.", "Resources differ, so this describes errors and cannot select the next method."], .9, 4.0, 11.5, 2.0, 17)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "GFP pruning screen: almost every channel is needed")
    gd = gfp["delta_from_unmasked_b1"]["diabetic_retinopathy"]; gm = gfp["delta_from_unmasked_b1"]["macular_edema"]
    style.ppt_table(slide, [["Label", "Channels retained", "ΔF1", "ΔAUROC", "ΔAP"],
                            ["DR", f"{gfp['selected']['retained_features'][0]} / 1536", f"{gd['f1_positive']:+.5f}", f"{gd['auroc']:+.5f}", f"{gd['average_precision']:+.5f}"],
                            ["ME", f"{gfp['selected']['retained_features'][1]} / 1536", f"{gm['f1_positive']:+.5f}", f"{gm['auroc']:+.5f}", f"{gm['average_precision']:+.5f}"]],
                    .9, 1.65, 11.5, 1.8, [1.4, 2.4, 1.8, 1.8, 1.8], 15)
    style.ppt_box(slide, 1.25, 4.05, 10.85, 1.15, "Frozen gate: FAIL — no F1 gain and no meaningful pruning", style.LIGHT_AMBER, style.AMBER, 21, True)
    style.textbox(slide, 1.0, 5.7, 11.3, .65, "Bounded conclusion: simple final-channel redundancy is not the missing mechanism.", 17, True, style.INK, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Top-1 spatial evidence changes DR ranking, not the decision metric")
    rows = [["Label", "G0 F1", "Top-1 F1", "ΔF1", "G0 AUROC", "Top-1 AUROC", "ΔAUROC"]]
    for key, label in [("diabetic_retinopathy", "DR"), ("macular_edema", "ME")]:
        f = spatial["aggregate"][key]["f1_positive"]; a = spatial["aggregate"][key]["auroc"]
        rows.append([label, f"{f['G0_mean']:.4f}", f"{f['G1_mean']:.4f}", f"{f['difference_mean']:+.4f}", f"{a['G0_mean']:.4f}", f"{a['G1_mean']:.4f}", f"{a['difference_mean']:+.4f}"])
    style.ppt_table(slide, rows, .55, 1.5, 12.25, 1.8, [1.0, 1.4, 1.4, 1.2, 1.6, 1.6, 1.3], 13)
    style.ppt_box(slide, .75, 3.8, 5.75, 1.2, "DR AUROC +0.0101\npositive in all 3 head seeds", style.LIGHT_GREEN, style.GREEN, 19, True)
    style.ppt_box(slide, 6.85, 3.8, 5.75, 1.2, "DR F1 −0.0012; ME AUROC −0.0036\nfrozen gate failed", style.LIGHT_AMBER, style.AMBER, 19, True)
    style.textbox(slide, .8, 5.65, 11.7, .65, "A high-scoring patch is a ranking cue; without lesion labels it is not lesion localization.", 16, True, style.MUTED, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Patient-level uncertainty supports the same decision")
    boot = errors["spatial_failure"]["patient_bootstrap"]
    rows = [["Metric difference", "Point", "Patient-bootstrap 95% interval"]]
    for key, label in [("diabetic_retinopathy", "DR"), ("macular_edema", "ME")]:
        for metric, shown in [("f1", "F1"), ("auroc", "AUROC"), ("average_precision", "AP")]:
            b = boot[key][metric]
            rows.append([f"{label} {shown}", f"{b['point_mean']:+.4f}", f"[{b['patient_bootstrap_95pct'][0]:+.4f}, {b['patient_bootstrap_95pct'][1]:+.4f}]"])
    style.ppt_table(slide, rows, 1.2, 1.45, 10.95, 3.75, [3.0, 2.1, 4.0], 14)
    style.textbox(slide, .8, 5.7, 11.7, .7, "DR ranking signal is repeatable; F1 gains are uncertain and ME ranking decreases.", 17, True, style.INK, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Failure audit: only completed checked runs support the conclusion")
    style.ppt_table(slide, [["Event", "Meaning", "Treatment"],
                            ["4409940", "CUDA deterministic environment missing", "Corrected environment; rerun passed"],
                            ["4409991 array", "Two tasks shared one node", "Cancelled; partial outputs ignored"],
                            ["4409997/8/9", "Final 3-seed spatial runs", "Distinct nodes; completed"],
                            ["P1/P2/P3", "Conditional full-model branch", "Not launched after failed gate"]],
                    .45, 1.4, 12.45, 3.4, [1.9, 4.0, 5.0], 13)
    style.ppt_box(slide, 1.0, 5.4, 11.3, .9, "Scientific negative result ≠ implementation error", style.PALE, style.BLUE, 20, True)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Frozen Step-5 decision")
    style.ppt_box(slide, .8, 1.55, 5.65, .75, "Do not launch", style.LIGHT_AMBER, style.AMBER, 21, True)
    style.bullets(slide, ["P1 patch-evidence training", "P2 device adapters in this interaction plan", "P3 combined training", "Test-set rescue of failed validation"], .95, 2.55, 5.25, 3.1, 17)
    style.ppt_box(slide, 6.85, 1.55, 5.65, .75, "Preserve as evidence", style.LIGHT_GREEN, style.GREEN, 21, True)
    style.bullets(slide, ["Source-only positive-case gap", "GFP null result", "DR top-1 AUROC observation", "ME trade-off and all limitations"], 7.0, 2.55, 5.25, 3.1, 17)
    style.textbox(slide, .8, 6.15, 11.7, .45, "Current novelty status: no novel-method performance claim", 18, True, style.AMBER, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Matched RETFound engineering gates passed")
    style.ppt_table(slide, [["Gate", "Job", "Evidence", "Status"],
                            ["Cohort / label preflight", rf_preflight["job_id"], "14,774 fit; 725 validation; no assessment", "PASS"],
                            ["Full-backbone feasibility", rf_smoke["job_id"], f"100% gradients; {rf_smoke['peak_allocated_gib']:.2f} GiB; {rf_smoke['estimated_seed_hours']:.2f} h", "PASS"],
                            ["Exact resume", rf_resume["job_id"], f"{rf_resume['updates']} updates; zero state mismatch", "PASS"],
                            ["Full seed 0", rf_launch["job_id"], f"first loss {rf_launch['early_health']['loss']:.5f}; result pending", "RUNNING"]],
                    .45, 1.35, 12.45, 3.6, [2.6, 1.5, 5.4, 1.7], 12)
    style.ppt_box(slide, .9, 5.35, 11.55, .9, "A finite first update proves execution health; it does not prove better validation performance.", style.PALE, style.BLUE, 18, True)

    slide = prs.slides.add_slide(blank); style.slide_title(slide, "Next path: validate the matched retinal backbone")
    boxes = [("1  Audit", "Old result rejected\nlabels • splits • budget", .55),
             ("2  Align", "Matched protocol\nall gates passed", 3.7),
             ("3  Screen", "Seed 0 running\nvalidation only", 6.85),
             ("4  Decide", "Replicate material gain\nor stop", 10.0)]
    for title, body, x in boxes:
        style.ppt_box(slide, x, 1.75, 2.75, .7, title, style.PALE, style.BLUE, 18, True)
        style.ppt_box(slide, x, 2.7, 2.75, 1.3, body, style.WHITE, style.BLUE, 15, False)
    style.ppt_box(slide, .85, 4.55, 11.65, 1.0, f"Old RETFound results are not comparable: {retfound['cohorts']['different_split_common_files']:,} split changes; {retfound['label_alignment']['different_dr_label_common_files']:,} DR-label changes; no ME output.", style.LIGHT_AMBER, style.AMBER, 17, True)
    style.textbox(slide, .8, 5.85, 11.7, .35, "Job 4411081 is the first fair RETFound comparison; its validation result is pending.", 16, True, style.GREEN, PP_ALIGN.CENTER)
    style.textbox(slide, .8, 6.45, 11.7, .35, "A backbone gain strengthens the baseline; it does not create a novelty claim by itself.", 16, True, style.INK, PP_ALIGN.CENTER)

    prs.save(PPTX)


def verify(inputs):
    with zipfile.ZipFile(DOCX) as archive:
        doc_xml = "".join(archive.read(name).decode("utf-8", "ignore") for name in archive.namelist() if name.endswith(".xml"))
    with zipfile.ZipFile(PPTX) as archive:
        ppt_xml = "".join(archive.read(name).decode("utf-8", "ignore") for name in archive.namelist() if name.startswith("ppt/slides/") and name.endswith(".xml"))
    document = Document(DOCX); presentation = Presentation(PPTX)
    bounds = all(shape.left >= 0 and shape.top >= 0 and shape.left + shape.width <= presentation.slide_width
                 and shape.top + shape.height <= presentation.slide_height
                 for slide in presentation.slides for shape in slide.shapes)
    required_text = ["Current novelty status", "P1/P2/P3", "Source-only", "Patient-bootstrap", "RETFound"]
    content = "\n".join(p.text for p in document.paragraphs)
    content += "\n" + "\n".join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
    result = {
        "pass": True,
        "generated": UPDATED,
        "docx": {"path": str(DOCX), "sha256": sha(DOCX), "paragraphs": len(document.paragraphs),
                 "tables": len(document.tables), "times_new_roman_present": FONT in doc_xml},
        "pptx": {"path": str(PPTX), "sha256": sha(PPTX), "slides": len(presentation.slides),
                 "times_new_roman_present": FONT in ppt_xml, "all_shape_bounds_inside_slide": bounds},
        "required_content": {item: item in content or item in ppt_xml for item in required_text},
        "input_sha256": inputs,
        "privacy": "No dataset image, filename, patient identifier, private manifest or row-level prediction embedded.",
    }
    if (len(presentation.slides) != 12 or len(document.tables) < 10 or FONT not in doc_xml or FONT not in ppt_xml
            or not bounds or not all(result["required_content"].values())):
        result["pass"] = False
    CHECKS.write_text(json.dumps(result, indent=2) + "\n")
    if not result["pass"]:
        raise RuntimeError(result)
    return result


def main():
    source, features, gfp, spatial, errors, retfound, rf_preflight, rf_smoke, rf_resume, rf_launch = load()
    build_docx(source, features, gfp, spatial, errors, retfound, rf_preflight, rf_smoke, rf_resume, rf_launch)
    build_pptx(source, features, gfp, spatial, errors, retfound, rf_preflight, rf_smoke, rf_resume, rf_launch)
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in (
        SOURCE, FEATURES, GFP, SPATIAL, ERRORS, PROTOCOL, REVIEW, RETFOUND,
        RETFOUND_PREFLIGHT, RETFOUND_SMOKE, RETFOUND_RESUME, RETFOUND_LAUNCH)}
    print(json.dumps(verify(inputs), indent=2))


if __name__ == "__main__":
    main()
