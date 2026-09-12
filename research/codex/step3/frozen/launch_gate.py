"""Reject a Step-3 launch if allocated preflight evidence or frozen code changed."""
import hashlib
import json
import os
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Launch gate requires an allocated compute node")
    evidence = json.loads((ROOT / "research/codex/step3/preflight.json").read_text())
    if not evidence.get("pass"):
        raise ValueError("Step-3 allocated preflight did not pass")
    checks = [(HERE / "train_controls.py", evidence["trainer_sha256"]),
              (HERE / "sampler_core.py", evidence["sampler_core_sha256"])]
    for path, expected in checks:
        if sha(path) != expected:
            raise ValueError(f"Frozen Step-3 source changed after preflight: {path.name}")
    print("Step-3 launch gate passed", flush=True)


if __name__ == "__main__":
    main()
