"""Step-3 trainer: Step-2 B1 recipe with only its sampler replaced."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STEP2 = ROOT / "research/codex/step2/full_pool/frozen"
sys.path.insert(0, str(STEP2))
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("step2_train", STEP2 / "train_baseline.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from sampler_core import ARMS, ControlledBatches, sampling_ledger  # noqa: E402

ORIGINAL_LOADER = base.loader
ORIGINAL_PHASE_SPECS = base.phase_specs
ACTIVE_ARM = None


def controlled_loader(rows, training, seed, workers, start=0, stop=0):
    if not training:
        return ORIGINAL_LOADER(rows, training, seed, workers, start, stop)
    dataset = base.Images(rows, True, seed)
    sampler = ControlledBatches(rows, ACTIVE_ARM, 16, start, stop, seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers, pin_memory=True,
                      generator=base.torch.Generator().manual_seed(base.stream_seed(seed, "loader")))


def controlled_contract(args, protocol, initialization):
    return {"protocol_sha256": base.sha(base.PROTOCOL / "v1.json"),
            "manifest_sha256": protocol["manifest_sha256"],
            "trainer_sha256": base.sha(__file__),
            "core_sha256": base.sha(HERE / "sampler_core.py"),
            "step2_trainer_sha256": base.sha(STEP2 / "train_baseline.py"),
            "step2_core_sha256": base.sha(STEP2 / "baseline_core.py"),
            "initialization_sha256": base.sha(initialization), "arm": args.arm,
            "seed": args.seed, "smoke": args.smoke,
            "torch": base.torch.__version__, "timm": base.timm.__version__,
            "phases": ORIGINAL_PHASE_SPECS("B1", args.smoke),
            "single_changed_factor": "training sampler"}


def controlled_phase_specs(arm, smoke=False):
    return ORIGINAL_PHASE_SPECS("B1", smoke)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    global ACTIVE_ARM
    ACTIVE_ARM = args.arm
    base.loader = controlled_loader
    base.contract = controlled_contract
    base.phase_specs = controlled_phase_specs
    base.train(args)

    protocol, manifest = base.protocol()
    rows = manifest[manifest.role == "fit"].reset_index(drop=True)
    updates = ORIGINAL_PHASE_SPECS("B1", args.smoke)[0]["updates"]
    draw_seed = base.stream_seed(args.seed, "phase_" + ("smoke" if args.smoke else "main"))
    ledger = sampling_ledger(rows, args.arm, updates * 64, draw_seed)
    ledger.update({"seed": args.seed, "updates": updates, "effective_batch": 64,
                   "selection_only": True,
                   "warning": "This control is screened on mBRSET validation; no test assessment was performed."})
    base.write_json(Path(args.output) / "sampling_ledger.json", ledger)


if __name__ == "__main__":
    main()
