**Step 2 status — September 10, 2026**

Implementation and execution checks completed; baseline experiments are running. Step 2 is not scientifically complete until results and needed replication are assessed.

| Item | Evidence-backed state |
|---|---|
| Image integrity | 14,774/14,774 decode; no exact native-RGB or EXIF-normalized duplicates. Near/recompressed duplicates not excluded. Slurm 4372368. |
| CPU checks | 13 tests pass, zero errors/failures; last check job 4372411. |
| GPU resume checks | Both ordinary interruption and source→target boundary interruption pass with identical model/EMA/optimizer tensors. Slurm 4372389 completed. |
| Actual initialized model | Seed-0 state saved and SHA256 pinned; shared by B0/B1/B2. |
| Runtime | ~4.34 seconds/update, ~49.4 seconds/full selection evaluation; ~7.36 GPU-hours/run estimate, ~9.93 with 35% buffer. Peak allocated GPU memory ~31.4 GiB on A40. |
| Baselines | Array 4372453, tasks 0/1/2 = B0/B1/B2 seed 0, concurrency 3. All three observed RUNNING on arctrdagn031/035/036. |
| Assessment | Job 4372454 depends on successful completion of entire baseline array. No baseline assessment results yet. |
| Checkpoint storage | `/data/users3/sthummala2/brset-codex/step2-runs`, linked by `runs/`; compute-node write test 4372452 passed. |

**Research interpretation.** These checks support implementation correctness within their scope, not convergence, diagnostic improvement or novelty. Smoke scores are not research outcomes. The original test/report/results remain unchanged. See `INTERPRETING_THE_BASELINES.md` for unequal domain exposure despite matched update budgets.

**Resource use.** User requested no compute-intensive work on login nodes and capacity left for others. Image/model tests and training use Slurm compute nodes. Latest user clarification authorizes three concurrent one-GPU jobs on distinct nodes, no exclusive node reservations. Original login-node decoding scan was terminated upon that instruction and restarted through Slurm. A home-storage launch gate correctly stopped submission when headroom was insufficient; a private shared-data directory resolved it. Only newly generated disposable smoke checkpoints were removed after verification, with hashes/diagnostics retained; original experimental checkpoints were not removed.

**Continuation.** Inspect `squeue -u sthummala2` and `sacct -j 4372453,4372454`; read corresponding logs. Do not resubmit blindly. If an arm fails, preserve its failure JSON/log and last state; diagnose then explicitly resume the same frozen contract when valid. Assessment depends on all three successes and is canceled if its dependency becomes impossible. Completion artifacts go in each `runs/B*_seed0/`; consolidated assessment goes in `baseline_assessment_seed0.json` and `.md`. Do not claim a winner from seed 0. Replicate relevant controls/contenders at seeds 1 and 2 before a gain claim.

**User input.** No action blocks these baselines. Share Dong's reply when received, confirm ISBI 2027 as intended venue, and help arrange qualified original/transformed-image review before a preservation claim. Clinical operating point remains open. Official ISBI deadline October 26, 2026, 11:59 p.m. EDT; internal full draft October 15.

Evidence: `image_integrity_summary.json`, `cpu_checks.json`, `smoke_checks.json`, `phase_smoke_checks.json`, `initializations/seed0.json`, `launch_seed0.json`, `frozen/`, and `logs/`. No automatic Slack messages are sent.

**Scheduling amendment.** See `scheduling_update_20260910.json`; original `launch_seed0.json` is historical. [Detailed experiment explanation](EXPERIMENTS_EXPLAINED.md) includes verified DR prevalence and configurations.

**15:17 UTC update:** All three active, sampled losses finite and checked error logs empty. Last logged updates: B0 450; B1/B2 200, of 5,775. Throughput-based training ETA approximately 22:00–23:00 UTC today, then dependent assessment subject to queueing. See `progress_20260910_1516.json` and `DATA_USAGE_CLARIFICATION.md`. Full-pool comparison design review comes before expensive replication; current runs are unchanged.

**SUPERSEDED EXECUTION — September10 15:52UTC:** User requested full-pool replacement. Old4372453/4372454 cancelled with artifacts preserved. Current status: `full_pool/STATUS.md`, launch `full_pool/launch_seed0.json`. Do not resume old reduced-pool jobs.
