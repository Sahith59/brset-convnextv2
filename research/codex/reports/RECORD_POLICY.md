# Living research record policy

The canonical maintained artifacts are `BRSET_mBRSET_Research_Record.docx` and `BRSET_mBRSET_Research_Update.pptx`. Regenerate both with `build_research_record.py` after:

1. a seed or experimental arm completes and its metrics are verified;
2. a research step is closed or its protocol is materially changed;
3. an implementation failure changes confidence in a result;
4. a decision changes the paper direction; or
5. an advisor-ready update is requested.

The DOCX is the detailed evidence record. It uses Times New Roman, 10-point body text, numbered technical headings, restrained emphasis and compact tables in an IEEE-like technical style. The PPTX is the concise advisor view; it contains only the research question, verified findings, limitations, controls and next decision gates.

Generated artifacts must distinguish completed results, descriptive findings, hypotheses and planned experiments. Dataset images, patient identifiers, per-image predictions and private manifests are excluded. Every numerical claim must trace to an aggregate artifact in the repository. After generation, `record_checks.json` records input hashes, page-independent structure checks and font checks.

Advisor-only visual companions are generated separately with `build_private_advisor_visuals.py` on an allocated Slurm node. They embed verified BRSET/mBRSET training examples, remain under the Git-ignored `reports/private/` directory and must be shared only through the authorized private advisor channel. Their mBRSET examples are explicitly unpaired. These panels verify implementation and illustrate appearance changes; they do not establish realism, paired reconstruction or diagnostic preservation. Rebuild the companions after a verified Step-4 result before sending the next advisor update.
