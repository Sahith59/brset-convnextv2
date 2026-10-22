#!/bin/bash
# D107 chain: after each CycleGAN training job -> translate (both directions) -> 6 classifiers + test-time eval;
# after everything -> registered read (scripts/159). Slurm dependencies, so nothing depends on this shell staying alive.
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
declare -A TRAIN=( [plain]=$1 [residual]=$2 )
FAST="-p qTRDGPUH,qTRDGPU --gres=gpu:1 --exclude=arctrdgn001,arctrdagn[001-012,014-017]"   # no V100 / RTX
ALL=()
for V in plain residual; do
  T=${TRAIN[$V]}
  AB=$(sbatch --parsable $FAST --time=02:00:00 --dependency=afterok:$T --job-name=cg_ab_$V --export=ALL,MODE=translate_ab,VARIANT=$V scripts/157_cyclegan.slurm)
  BA=$(sbatch --parsable $FAST --time=01:00:00 --dependency=afterok:$T --job-name=cg_ba_$V --export=ALL,MODE=translate_ba,VARIANT=$V scripts/157_cyclegan.slurm)
  EV=$(sbatch --parsable $FAST --time=03:00:00 --dependency=afterok:$BA --job-name=cgtt_$V --export=ALL,TAG=$V scripts/158_cg_testtime.slurm)
  ALL+=($EV); echo "$V: translate_ab $AB translate_ba $BA testtime $EV"
  for P in 0.25 1.0; do for S in 0 1 2; do
    J=$(sbatch --parsable $FAST --time=14:00:00 --dependency=afterok:$AB --job-name=CG${V:0:1}${P}_s$S \
        --export=ALL,GEN=gen_cyclegan_$V,BUDGET=0,SEED=$S,SYNTH_P=$P,TASK_SUFFIX=_p$P,SD=/data/users3/sthummala2/brset_gen/synth/full_cyclegan_$V \
        scripts/114_synth_joint.slurm)
    ALL+=($J); echo "  classifier $V p$P s$S: $J"
  done; done
done
DEP=$(IFS=:; echo "${ALL[*]}")
RD=$(sbatch --parsable -p qTRDGPU --gres=gpu:0 --cpus-per-task=4 --mem=16G --time=01:00:00 --dependency=afterok:$DEP \
     --job-name=d107_read --output=/home/users/sthummala2/brset-convnextv2/logs/d107_read_%j.out \
     --wrap="cd /home/users/sthummala2/brset-convnextv2 && /home/users/sthummala2/.conda/envs/retfound/bin/python scripts/159_d107_table.py")
echo "read: $RD (after ${#ALL[@]} jobs)"
