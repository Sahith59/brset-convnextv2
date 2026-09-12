**Contribution hypothesis after Step 1 — September 10, 2026**

Status: candidate mechanism, not a supported novelty claim. Step 1 establishes the comparison rules and patient partitions; it cannot manufacture an empirical method result. No first/best/state-of-the-art claim is justified.

**The concrete hypothesis**

Investigate a lightweight selection rule for synthetic fundus transformations that trades target-appearance matching against independently assessed loss of diagnostic evidence, using unpaired FIT images and the same labeled target budget as its comparators.

In simple words: do not accept a transformed photograph merely because its average color and texture look more like the target. Prefer transformations that move appearance toward the target while keeping the disease clues usable. Test whether this selection rule improves predictions on real target photographs beyond merely using weaker corruption.

If supported, a bounded eventual claim could be: “We introduce [the precisely specified and validated selection rule] for target-informed fundus augmentation and demonstrate incremental DR/ME transfer benefit under matched supervision and optimization budgets, without paired-camera training images.” Fill in the mechanism and measured result only after implementation, closest-work review and experiments. Do not currently substitute “lesion-preserving” for “preservation-aware”: our proxy could be wrong.

**Why the existing fitted simulator is insufficient**

It minimizes a difference between five means. It does not identify the physical camera process, fit the entire target distribution, or penalize corruption of small diagnostic features. The report provides a descriptive appearance result on additional patients. It contains no augmentation-trained classifier evaluation and no verified preservation mechanism. Choosing blur/noise/spot settings or combining an existing teacher and router is not yet the paper contribution.

**A possible mathematical design, not yet selected**

Let theta describe the distribution of allowed transformation settings, G_theta(x) an altered source image, D_app a target appearance mismatch and C_diag a validated diagnostic-damage measure. One candidate is to minimize D_app subject to expected C_diag <= epsilon. Another implementation could reject individual transformations that exceed the constraint. Neither this generic constrained-optimization formulation nor rejection sampling is intrinsically novel. The potential contribution lies in a specific reliable diagnostic-damage signal, its coupling to fitting, and evidence that this does more than limit severity.

The current project does not have a validated C_diag. A teacher prediction difference or a saliency map alone cannot certify preserved lesions; a teacher can be confidently wrong. Any teacher must be trained on FIT patients only, checked on an inner FIT holdout, and independently evaluated against expert judgments. Do not train that constraint using outer SELECTION/ASSESSMENT images. If a trustworthy preservation signal is unavailable, study plain fitted augmentation honestly and do not advertise the constraint as implemented.

**Closest work checked and resulting boundaries**

GDRNet already combines fundus visual/artifact augmentation, a supervised/contrastive objective and domain-class rebalancing. Its formulation makes a broad “augmentation + consistency + balance” contribution untenable. The current main paper was revisited; implementing its FundusAug component does not reproduce the full framework. [GDRNet main paper](https://arxiv.org/html/2307.04378v3).

RetSyn's published abstract describes disease/quality-conditioned diffusion, group-balanced training and paired-device preference alignment. Thus neither synthetic retinal augmentation nor disease-aware cross-device synthesis is our untouched space. Our prospective lightweight, unpaired fitting setting is a resource difference that needs measurement, not automatic novelty. Published abstract checked; full implementation comparison remains open. [RetSyn published record](https://pubmed.ncbi.nlm.nih.gov/41138952/).

A RetSyn-associated dataset card describes 327 patients with paired tabletop/portable photographs, but also retains a note that the full release is forthcoming after acceptance. The card was read; actual available files, label completeness and correspondence to the publication have not been audited. Do not call the entire paired dataset verified available. [Dataset card](https://huggingface.co/datasets/smartretina2025/paired_retina/blob/main/README.md).

The search also found primary work explicitly considering distribution matching while avoiding lesion tampering: “Optimal Transport Guided Unsupervised Learning for Enhancing Low-quality Retinal Images.” Indexed primary text was accessible; direct full-text opening hit a browser check. This strengthens the need to compare conceptual equivalents, not just papers named augmentation. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10513403/). Existing SCR-Net/frequency leads and access limitations remain in `../literature/DIRECTION_REVIEW.md`.

This is a focused novelty check, not a completed systematic literature review. Complete the methods/code comparison of nearest equivalents before freezing a method-level claim.

**Experiments that isolate the proposed rule**

Use the Step-1 data/selection contract. Compare ordinary augmentation; faithful FundusAug; FIT-only refitted appearance augmentation; the identical simulator with simple lower severity; and that simulator with the proposed evidence constraint. Match source views, target labels, sampler distribution, updates and inference. Also compare a matched acceptance-rate random rejection control if selecting/rejecting images: otherwise changing effective data exposure could explain a gain. Record CPU/GPU time, rejected proportions by domain and disease label, and all resulting training distributions.

Evaluate real-target DR and ME predictions, expert-judged preservation on a separately specified sample, and appearance summaries separately. A possible smaller target-label requirement needs a separate patient-budget curve with nested patient subsets; it cannot be inferred from a full-label experiment.

**What would reject the hypothesis**

No benefit over faithful generic or severity-limited augmentation; no verified relationship between the preservation signal and actual diagnostic damage; a gain explained by target-label exposure or rejection-induced class balance; benefit only in a chosen seed; or relevant ME degradation. If the rule fails, remove the claim or revise the research question. Negative results can inform the work without automatically constituting a publishable novel solution.
