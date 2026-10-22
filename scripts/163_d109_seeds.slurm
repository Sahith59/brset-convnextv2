#!/bin/bash
#SBATCH --job-name=d109
#SBATCH --account=trends517s113
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/d109_%A_%a.out
# D109: seeds 3 and 4 for every Table-1 arm. Array index i -> entry i/2, seed 3 + i%2. Each entry runs the ORIGINAL
# launcher of that arm unchanged (114 synthetic mix, 71 label-budget baseline, 122 self-training student); only SEED changes.
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
S=/data/users3/sthummala2/brset_gen/synth
ENTRIES=(
  "114 gen_both $S/full_gen_both_e40k_phys_s0.4_w1.5"
  "114 gen_fda $S/full_fda_b0.01"
  "114 gen_reinhard $S/full_reinhard"
  "114 gen_classcond $S/full_gen_classcond_e40k_noise_w1.0"
  "114 gen_physonly $S/full_physonly_v2noartifact"
  "114 gen_bothnone $S/full_gen_both_e40k_none_s1.0_w1.5"
  "114 gen_bothsrc $S/full_gen_both_e40k_source_s0.4_w1.5"
  "114 gen_both_dc $S/full_gen_both_dc_phys_s0.4_w1.5"
  "114 gen_cyclegan_plain $S/full_cyclegan_plain"
  "71 0 -"
  "71 250 -"
  "71 500 -"
  "122 F -"
  "114 gen_cyclegan_residual $S/full_cyclegan_residual"
)
k=$(( SLURM_ARRAY_TASK_ID / 2 )); export SEED=$(( 3 + SLURM_ARRAY_TASK_ID % 2 ))
read -r kind a b <<< "${ENTRIES[$k]}"
echo "D109 array ${SLURM_ARRAY_TASK_ID}: ${kind} ${a} ${b} seed ${SEED} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$kind" in
  114) export GEN="$a" SD="$b" BUDGET=0 SYNTH_P=0.25 TASK_SUFFIX=_p0.25; exec bash scripts/114_synth_joint.slurm ;;
  71)  export BUDGET="$a" BSEED=0; exec bash scripts/71_label_efficiency_baseline.slurm ;;
  122) export PSEUDO=/data/users3/sthummala2/brset_gen/pseudo/ST-F_base1_loose.csv BUDGET=0 TASK="pst_F_base1loose_notwins_s${SEED}"; unset SD; exec bash scripts/122_pseudo_student.slurm ;;
esac
