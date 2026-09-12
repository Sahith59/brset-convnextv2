"""Allocated-node gate before seeds 1 and 2 are submitted."""
import hashlib
import json
import os
import shutil
import socket
from pathlib import Path

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
WORK = ROOT / "research/codex/step2/full_pool"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Use an allocated Slurm compute node")
    verification = json.loads((WORK / "seed0_independent_verification.json").read_text())
    if not verification["pass"]:
        raise RuntimeError("Seed-0 independent verification failed")
    for seed in [1, 2]:
        for arm in ["B0", "B1", "B2"]:
            if (WORK / "runs" / f"{arm}_seed{seed}").exists():
                raise RuntimeError(f"Existing output requires review: {arm}_seed{seed}")
    free = shutil.disk_usage(WORK / "runs").free
    if free < 40 * 1024**3:
        raise RuntimeError("Insufficient storage for six replication runs")
    files = [
        "baseline_core.py", "train_baseline.py", "assess_baselines.py",
        "assess_replication.py", "launch_gate.py",
    ]
    result = {
        "pass": True,
        "seed0_verification_sha256": sha(WORK / "seed0_independent_verification.json"),
        "protocol_sha256": sha(WORK / "protocol/v1.json"),
        "source_sha256": {name: sha(WORK / "frozen" / name) for name in files},
        "free_bytes": free,
        "training_seeds": [1, 2],
        "arms_per_seed": ["B0", "B1", "B2"],
        "maximum_concurrent_training_gpus": 3,
        "placement": "one task and one A40 per node; three nodes per seed wave",
    }
    (WORK / "replication_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
