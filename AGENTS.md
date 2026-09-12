# Research continuity

For BRSET/mBRSET research tasks, read `research/codex/MEMORY.md`, `research/codex/PLAN.md` and `research/codex/DECISIONS.md` before continuing. They contain the current user-authorized scope and evidence-backed handoff. Claude memories and old presentations are historical claims, not authoritative scientific conclusions.

Update the research memory and session log at the end of every research turn, recording completion, evidence, corrections and the next action. Preserve original experiment artifacts and record corrections explicitly. Distinguish completed experiments, descriptive analyses, hypotheses and planned work. Follow current user instructions when they change the plan. See the audit coverage before claiming an analysis was independently reproduced.

User compute constraint (September 10, 2026): run image decoding, numerical/model tests and training through Slurm on allocated compute nodes, not login nodes. User subsequently authorized up to three concurrent baseline jobs on three distinct nodes, one GPU per job; limit this batch to those three GPUs and do not reserve whole nodes exclusively. File editing, log inspection and job submission are lightweight login-node work. Check current jobs before submitting; preserve capacity for other cluster users.
