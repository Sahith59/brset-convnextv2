**Current degradation report — revision 2, September 10, 2026**

The verified three-page [PDF](Degradation_Update.pdf) is the current advisor report. [Editable DOCX](Degradation_Update.docx) contains the same material; pagination is verified for the PDF.

The report defines both resizing procedures and the standardized five-summary distance, gives the nine parameter settings and search ranges, records sampling and transform seeds, and explains the simulator details and limitations.

Evidence: [image provenance](revision_2/figure_provenance.json), [deliverable checks](revision_2/deliverable_checks.json), [extracted report text](revision_2/report_text.txt), [validation summary](validation_summary.json), [appearance statistics](appearance_statistics.csv), and [distances](appearance_distances.csv). The extracted text is for searching; mathematical layout is preserved in the PDF.

Rebuild the revision using `research/codex/tools/revise_dong_report.py`; it writes to `revision_2/`. Promote outputs to this directory only after checking them. `build_dong_update.py` is the historical revision-1 builder. The earlier report is retained in `revision_1/`.

The images were regenerated from the actual dataset files and original operator implementation. All six source identities match their dataset roots, raw label tables and training manifests; all twelve embedded PDF panels match the generated pixels. Real mBRSET images are unrelated reference eyes. Improvement in appearance summaries does not establish preservation of lesions, diagnostic improvement or novelty. No message has been sent.
