#!/bin/bash
#SBATCH --job-name=br_smoke
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/smoke_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/smoke_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TIMM_USE_OLD_CACHE=1
py=/home/users/sthummala2/.conda/envs/retfound/bin/python
"$py" research/codex/step2/full_pool/frozen/train_baseline.py train --arm B1 --seed 0 --smoke --output research/codex/step2/full_pool/smoke_full
"$py" research/codex/step2/full_pool/frozen/train_baseline.py train --arm B1 --seed 0 --smoke --stop-after 3 --output research/codex/step2/full_pool/smoke_resume
"$py" research/codex/step2/full_pool/frozen/train_baseline.py train --arm B1 --seed 0 --smoke --resume --output research/codex/step2/full_pool/smoke_resume
"$py" research/codex/step2/full_pool/frozen/verify_smoke.py
"$py" research/codex/step2/full_pool/frozen/train_baseline.py train --arm B2 --seed 0 --smoke --output research/codex/step2/full_pool/phase_smoke_full
"$py" research/codex/step2/full_pool/frozen/train_baseline.py train --arm B2 --seed 0 --smoke --stop-after 3 --output research/codex/step2/full_pool/phase_smoke_resume
"$py" research/codex/step2/full_pool/frozen/train_baseline.py train --arm B2 --seed 0 --smoke --resume --output research/codex/step2/full_pool/phase_smoke_resume
"$py" research/codex/step2/full_pool/frozen/verify_smoke.py --prefix phase_smoke
