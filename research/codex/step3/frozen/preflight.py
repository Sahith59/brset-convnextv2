"""Allocated CPU preflight for the frozen Step-3 sampling controls."""
import json
import itertools
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STEP2 = ROOT / "research/codex/step2/full_pool/frozen"
sys.path.insert(0, str(STEP2))
from baseline_core import sha, stream_seed, validate_manifest  # noqa: E402
from sampler_core import ARMS, ControlledBatches, sampling_ledger  # noqa: E402


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Preflight must run on an allocated compute node")
    protocol_path = ROOT / "research/codex/step2/full_pool/protocol/v1.json"
    protocol = json.loads(protocol_path.read_text())
    manifest_path = protocol_path.parent / protocol["manifest"]
    manifest = validate_manifest(manifest_path, protocol["manifest_sha256"])
    rows = manifest[manifest.role == "fit"].reset_index(drop=True)
    total = 5775 * 64
    seed = stream_seed(0, "phase_main")
    ledgers = {}
    for arm in ARMS:
        ledger = sampling_ledger(rows, arm, total, seed)
        if ledger["total_draws"] != total or not ledger["all_indices_in_range"]:
            raise AssertionError(f"Invalid ledger for {arm}")
        # Verify resumability against the second half of the same declared schedule.
        start = (5775 // 2) * 64
        full = ControlledBatches(rows, arm, 16, 0, total, seed)
        suffix = ControlledBatches(rows, arm, 16, start, total, seed)
        full_tail = list(itertools.islice(iter(full), start // 16, start // 16 + 4))
        suffix_head = list(itertools.islice(iter(suffix), 4))
        if full_tail != suffix_head:
            raise AssertionError(f"Resume suffix mismatch for {arm}")
        ledger["resume_probe_pass"] = True
        ledgers[arm] = ledger
    output = {"pass": True, "job": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "protocol_sha256": sha(protocol_path), "manifest_sha256": protocol["manifest_sha256"],
              "sampler_core_sha256": sha(HERE / "sampler_core.py"),
              "trainer_sha256": sha(HERE / "train_controls.py"),
              "test_sha256": sha(HERE / "test_sampler.py"), "ledgers": ledgers}
    destination = HERE.parent / "preflight.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
