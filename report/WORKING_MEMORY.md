# WORKING MEMORY

Live state of the ISBI push. Plan of record is `report/PLAN_v2.md`.
**Last updated: 01:07 UTC, 8 Nov 2026.** Update whenever a run lands or a
decision is made; first thing to read when resuming.

**Deadline: ISBI 2027, 26 October 2026. 14 days out.**

---

## READ FIRST (01:07 UTC, 8 Nov 2026) — supersedes every block below

- **ISBI 2027 analysis paper SUBMITTED** (EDAS #1571373528). Final PDF:
  `report/reference_isbi2027/FINAL_FOR_DR_YE/ISBI2027_PhoneStyleTwins_v6_final_2026-10-28.pdf`.
- **New phase: methods paper on mild DR (grade 1) with zero phone labels.** Plan of record for this phase:
  `report/reference_methods/reports/World models for mild DR transfer.md` (merged ladder; builds on
  `Closing the mild DR phone gap.md`). Both git-ignored. Decision row D114.
- **Authorized 8 Nov (user):** (1) `--skip_test` trainer flag; (2) D114 strict rule; (3) cheap experiments
  0a (grader-threshold test on saved predictions), 1a (phone vs BRSET noise spectrum at microaneurysm
  scale), 2a+2c (sham-paired microaneurysm insertion: lesion-response gate + lesion-survival audit).
  All through Slurm. Dr. Ye not yet briefed: the user will ask for the content later.
- **Key facts:** appearance is spent (D111: 40.5% -> 4.0% called phone, same AUROC); input resolution never
  changed (560 px, microaneurysm 1.3-2.6 px); 732 test set = development data from now on (~115 reads);
  fresh evaluation = mBRSET train split (899 patients, 187 grade-1), cross-fitted, read once.
- **Assets:** 251 checkpoints + 254 saved prediction sets in `/data/users3/sthummala2/brset_gen/cls_runs`.
- **Repo:** push `isbi2027-paper` only; never `main` (holds the old per-image commit). User makes the repo
  private after the 8 Nov push.

---

## READ FIRST (14 Oct, 01:30) — supersedes the blocks below

- **All runs finished.** Final zero-label configuration (validation rule, D072): gen_both 40k, 25 % swap —
  test DR AUROC 0.9279 vs 0.9007 baseline (+0.027, CI above 0, 3/3 seeds), F1 0.7746 vs 0.7670 (n.s.).
- **Ablation:** physics only +0.003 (n.s.); diffusion only +0.017 (n.s., F1 significantly worse);
  physics → diffusion +0.026 (significant). Both stages needed.
- **Longer generator training does not help downstream** (40k/60k/100k at 25 %: 0.9279/0.9271/0.9262).
- **Deck:** report/deck_2026-10-13/BRSET_mBRSET_Dong_Update_2026-10-13_v2.pptx (32 slides, numbers read from JSON at build).

## READ FIRST (13 Oct, 16:55)

- **Every arm in one table:** `scripts/119_full_table.py` -> `results/full_table_2026-10-13.json`.
  Zero real labels: BRSET-only 0.9007 test DR AUROC; + synthetic (gen_tgt 40k, p=0.5) 0.9271
  (+0.026, CI above 0, 3/3 seeds); gen_both 40k 0.9214; gen_both 60k 0.9238. With 250 real
  labels synthetic adds nothing. No reliable F1 change anywhere (all F1 CIs span 0).
- **Open risk for the claim:** full-method images are physics image + a small diffusion edit
  (3.5/255 per pixel). Ablation D070 (physics-only / diffusion-only / no-physics start, 9 runs)
  launched 16:40; read-out ~21:30. Physics-only gate: phone-likeness 0.28 vs 0.47 full method;
  mild-lesion AUROC 0.937 vs 0.902.
- **Running:** gen_both_ld100 (4 H100, ~21:40), syn_tgtld60 x6 (~19:15), syn_physonly x3 (~20:00),
  diffusion-only + no-physics translations and their runs (~21:00-21:30).

## READ FIRST (12 Oct, 17:00)

- **Authoritative record is `research/codex/` (MEMORY.md, PLAN.md, DECISIONS.md,
  SESSION_LOG.md), per `AGENTS.md`.** This file is a working scratchpad. The
  12 Oct entry there lists five corrections to statements made 10-12 Oct.
- **Hardware now (17:22):** `gen_tgt` 4x H100 (job 4599238, 1.83 it/s), `gen_both`
  8x A100 (job 4599239, 1.77 it/s), `gen_tgt_sharp` 16x A40 (job 4599320) after
  an A100 attempt was preempted. Fast runs finish ~22:20-22:45 on 12 Oct.
- **32 gate jobs queued** on snapshots 5k/10k/20k/40k -> `results/gates_g*.json`.
- **First-look figure:** `report/figures/first_look_gen_tgt.png` (step 3,000).
- **Novelty unresolved:** RetSyn and SCR-Net are close prior work.
- **Downstream comparison needs a protocol first** (validation-selected, one
  gated test assessment, patient-cluster bootstrap).

## CURRENT DIRECTION (12 Oct): the real generative translator

Decision by the user, following Dr. Ye's 8 Oct verdict ("image synthesis
converting BRSET to synthesized mBRSET ... joint training ... make use of the
generative world model ... everything we did till now sounds like preprocessing,
good for a class project not a conference submission").

**What was wrong before.** "Stage 2" was a 68,346-parameter refiner clamped to
+-0.15 / +-0.02, with a discriminator 2.4x its size. The domain probe sitting at
1.0000 was a capacity limit, not evidence about synthesis. The claim
"appearance synthesis is refuted" is WITHDRAWN.

**What is being built.** Dual-domain translator:
- stage 1, acquisition domain: the 19-parameter physics model (kept)
- stage 2, image domain: 104.7M-parameter conditional diffusion renderer
  p(image | structure map, device), trained on real images
- coupling: diffusion sampling starts from the stage-1 output at an
  intermediate noise level

**RELAUNCHED 12 Oct 15:20 on 8 A40s each (4 nodes), at the user's request.**
Effective batch 128, lr 2e-4, 40,000 steps (~22 h instead of ~64 h), snapshot
`ema_000NNNN.pt` every 2,500 steps (~85 min). Jobs 4599139 / 4599140 / 4599141,
script `scripts/110_train_diffusion_mn.slurm` with `sbatch --nodes=4`. The table
and job numbers just below describe the superseded 2-GPU launch.

27 gate jobs queued against snapshots 2,500 / 7,500 / 15,000.

**(superseded) first launch** (2x A40 each, effective batch 32, 120k steps):

| job | tag | what |
|---|---|---|
| 4599090 | `gen_tgt` | mBRSET-only renderer |
| 4599091 | `gen_both` | both devices + device token (world-model form) |
| 4599092 | `gen_tgt_sharp` | mBRSET-only, structure-map blur 0.8 (tighter lesion pinning) |

First 10k snapshot ~4.5 h after launch; 20k ~9 h. Jobs are resumable: resubmit
`scripts/110_train_diffusion.slurm` with the same TAG.

**Gates, in order. No downstream run until 1 and 2 pass.**
1. Realism — domain probe real mBRSET vs translated BRSET, well below 1.0000.
2. Label preservation — `brset_mbrset_joint` must recover BRSET labels from the
   translated images, per ICDR grade (grade 1 above all).
3. Downstream — BRSET + synthetic mBRSET, 3 seeds, `--no_oversample`.

**Measured training rate:** 0.52 it/s at effective batch 32 -> 120k steps in ~64 h.
EMA snapshots land every ~5.3 h (10k steps). Jobs 4599099 / 4599100 / 4599101.

**Multi-GPU needs `NCCL_P2P_DISABLE=1` on these nodes** (in the slurm script).
Without it DDP hangs forever in its first collective; the first launch of these
three jobs sat at 100% CPU with zero steps until that was found.

**Gates are written and validated** (`scripts/112_gates.py`, `112_gate.slurm`):
- they run on HELD-OUT BRSET (val+test, cache `brset_heldout_560.npy`), because
  the judge was trained on BRSET train
- gate 1 primary = a device classifier trained on disjoint real images, scored
  on the translated ones (did the image cross the device boundary). The direct
  real-vs-translated probe is reported but is not the criterion: it flags any
  synthetic image
- gate 2 judge validated: per-grade AUC 0.988-1.000 on untouched images, and a
  blur control drops it by 0.21. A 300-step toy model FAILS gate 2 (0.618), so
  the gate can fail a translator -- which the old lesion detector could not

**To run a gate on a snapshot:**
`sbatch -J g --export=ALL,RUN=gen_tgt,CKPT=ema_0010000.pt,OUT_TAG=<tag>,STRENGTH=0.6,W_STRUCT=1.5 scripts/112_gate.slurm`
Sweep STRENGTH (0.3-1.0), W_STRUCT (1-3), INIT (physics/source/none).

**Still to write:** downstream data-dir builder (BRSET + synthetic PNGs, absolute
paths); a CycleGAN/CUT comparator arm (the unpaired-GAN baseline a reviewer will
expect); full-corpus translation sharded over GPUs (~26 GPU-hours at 50 steps).

**Smoke evidence (300 steps, 48 images):** BRSET sources come out in mBRSET's
palette with vessels/disc/bright patches in place. Not evidence of lesion
preservation — that is gate 2.

---

## EARLIER STATE (through 12 Oct morning) — results below remain valid

## STOP — read this first (11 Oct, 13:50)

**Every comparison made on 10-11 Oct against the 0.9007 baseline is VOID.** The
baseline passes `--no_oversample`; all new slurms omitted it, so they trained at
a 60-65% effective positive rate against the baseline's 6.58%. `extmild` peaked
at best_epoch 1-3 vs the baseline's 24. All arms relaunched with the flag,
tasks tagged `_ns`. See memory `oversample-flag-confound`.

Within-batch comparisons survive (same regime):
- `tgtlab`: mild 0.9172 > severe 0.9124 (as predicted, inside floor)
- `extmild`: severe 0.8688 > mixed 0.8256 > mild 0.8102 — **opposite** of the
  mild-helps hypothesis, and all far below baseline

## Currently running

**18 jobs, all with `--no_oversample` (tasks `*_ns`):**
`extmild_{mild,severe,mixed}` x3, `sevmatch` x3, `ordaux_{ord,both}` x3.
Landing ~19:00. The `extmild` set re-asks the live question; the others
reinstate valid numbers for the severity interventions.

Superseded (confounded, kept on disk as evidence):
Imports 1,000 real external DR positives from DDR into BRSET training, matched
count across arms so severity is the only variable. ~5h, landing ~08:00.

| arm | what it adds | role |
|---|---|---|
| `mild` | 1,000 grade 1-2 | the hypothesis |
| `severe` | 1,000 grade 3-4 | control: same new camera, wrong severity |
| `mixed` | 1,000 proportional | a random mix beat both extremes at n=250 |

Read as 3-seed means against zero-shot **0.9007**, floor **0.0142**.

---

## The finding that reframes the project

**The gap is severity, not appearance.** Grade-1 AUC 0.7735 vs grade-4 0.9963.
All 42 missed patients are grade 1-2, none above. Appearance is orthogonal to
severity, |rho| <= 0.11 across 9 statistics.

That orthogonality is **generator-independent**: no appearance method can fix a
severity problem however well built. It is the keystone of the paper.

---

## The positive backbone (3 seeds each, test DR AUC)

| target labels | DR AUC | delta |
|---|---|---|
| 0 | 0.9007 | -- |
| 250 | 0.9219 | +0.0212 |
| 500 | 0.9360 | +0.0354 |
| 1000 | 0.9384 | +0.0378 |
| 2000 | **0.9475** | +0.0469 |
| pooled (3402) | 0.9377 | |

2000 labels beats the pooled model and the target-trained ceiling.

---

## Negative results, all measured

| intervention | result | verdict |
|---|---|---|
| 5 architectures | all inside 0.0199 F1 floor | dead |
| appearance synthesis, 4 ratios | -0.0060 to -0.0452 | dead |
| translator, 3 variants incl. 2 adversarial | probe AUC 1.0000 -> 1.0000 | dead |
| ensembling, 1,022 subsets | best < best single model | dead |
| calibration / thresholds | +0.0052 F1 | noise |
| aspect-ratio fix | +0.0020 F1 | noise |
| severity reweighting | +0.0088 DR, oedema -0.0442 | inside floor |
| ordinal aux head | +0.0101 DR, oedema -0.0689 | inside floor |
| both | +0.0093 DR, oedema -0.0494 | inside floor |
| 250 mild-only target labels | 0.9172 | below random 0.9219 |
| 250 severe-only target labels | 0.9124 | below random |

The three severity arms' DR gain is **one seed**: s0 +0.004, s1 +0.027,
s2 -0.000. 7/9 beat their matched baseline seed, sign test p = 0.09.

The 250-label oracle was **underpowered, not refuted**: only 59 images (0.51% of
training data) differed between arms. It did show the predicted order
(mild > severe). `extmild` applies 17x the perturbation.

**Pattern: rearranging what we have fails; adding information works.** The
bottleneck is ~115 grade-1 training images.

---

## MUST FIX before submission

**The lesion detector cannot separate ICDR grades.** At h=0.012: grade 0 = 799
candidates, grade 1 = 1133, grade 2 = 727, grade 4 = 714. No threshold in
0.012-0.15 gives monotone separation. Root cause: `robust_sigma()` returns
exactly 0.0, so `h` always collapsed to `h_abs` and `--k_rel` never did
anything in any run.

Restate these two claims:
- "translator destroys 57% of lesion evidence" -> 57% of DARK STRUCTURE candidates
- "GDRNet retains only 46.5% of lesions" -> same correction

Unaffected: the domain probe (no detector), downstream synthesis losses, the
per-grade decomposition, orthogonality, the label curve.

---

## Open items

1. `extmild` results, ~08:00. If `mild` > `mixed` > `severe` outside the floor,
   severity-targeted acquisition is a positive result.
2. **Talk to Dong.** Appearance synthesis is blocked; reframe the generative work
   onto severity (manufacturing mild cases). Also disclose the detector problem.
   Mild-case synthesis is gated behind `extmild`, which is its upper bound.
3. Mild-case synthesis blocked on annotations: BRSET has no masks, DDR's
   segmentation subset is not on the cluster, our detector cannot find lesions.
4. A practical acquisition rule (select by model uncertainty, no grades needed)
   is what turns the oracle experiment into a method.

## Code added 10-11 Oct

- `--severity_match`, `--ordinal_aux`, `--budget_severity`, `--extra_manifest`
  in `scripts/30_train_strong_baseline.py`
- `scripts/101_measure_grade_stats.py` -- grade/candidate calibration
- `scripts/102_build_extra_manifest.py` -- external dataset manifest
- `scripts/103_extra_mild.slurm`, `99_ordinal_aux.slurm`, `100_targeted_labels.slurm`

## Queued automatically (12 Oct)

12 gate jobs `g10k_<run>_<init>_s<strength>_w<w_struct>` start at 20:55 on the
10k-step snapshots of all three generators: physics init at strength 0.4 / 0.6 /
0.8 and a diffusion-only arm (init none). Each translates the 640-image held-out
gate subset and writes `results/gates_<tag>.json`. Read those first when resuming.
