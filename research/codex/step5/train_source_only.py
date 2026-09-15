"""Strict BRSET-only fitting/selection with gated mBRSET assessment."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/codex/step2/full_pool/frozen"
WORK = ROOT / "research/codex/step5"
MANIFEST = WORK / "private/source_only_manifest.csv"
PROTOCOL = WORK / "source_only_v1.json"
INTEGRITY = WORK / "source_integrity_summary.json"
sys.path.insert(0, str(BASE))
import baseline_core as core  # noqa: E402
import train_baseline as baseline  # noqa: E402


def protocol() -> tuple[dict, pd.DataFrame]:
    p = json.loads(PROTOCOL.read_text())
    data = core.validate_manifest(MANIFEST, p["private_manifest_sha256"])
    counts = data.groupby(["domain", "role"]).size().to_dict()
    expected = {("BRSET", "fit"): 11372, ("BRSET", "selection"): 2451,
                ("mBRSET", "assessment"): 732}
    if counts != expected:
        raise ValueError(f"Source-only cohort changed: {counts}")
    return p, data


def phase(smoke: bool) -> dict:
    if smoke:
        return {"name": "source", "domain": "BRSET", "updates": 6,
                "warmup": 2, "lr": 3e-5, "eval_at": [3, 6]}
    return {"name": "source", "domain": "BRSET", "updates": 5775,
            "warmup": 693, "lr": 3e-5, "eval_at": list(range(231, 5776, 231))}


def contract(args, initialization: Path) -> dict:
    return {"source_protocol_sha256": core.sha(PROTOCOL),
            "source_manifest_sha256": core.sha(MANIFEST),
            "step2_protocol_sha256": core.sha(BASE.parent / "protocol/v1.json"),
            "trainer_sha256": core.sha(Path(__file__)), "core_sha256": core.sha(BASE / "baseline_core.py"),
            "initialization_sha256": core.sha(initialization), "arm": "S0", "seed": args.seed,
            "smoke": args.smoke, "torch": torch.__version__, "timm": timm.__version__,
            "fit_domain": "BRSET", "selection_domain": "BRSET",
            "assessment_domain": "mBRSET", "phase": phase(args.smoke)}


def train(args) -> None:
    baseline.node_check(True); baseline.setup(args.seed)
    _, data = protocol()
    integrity = json.loads(INTEGRITY.read_text())
    if integrity["status"] != "pass" or integrity["manifest_sha256"] != core.sha(MANIFEST):
        raise ValueError("Source-only decoded-image integrity gate failed")
    initialization = baseline.initialize(args.seed); baseline.setup(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg = contract(args, initialization)
    if (out / "contract.json").exists():
        if json.loads((out / "contract.json").read_text()) != cfg:
            raise ValueError("Run contract changed")
        if not args.resume:
            raise ValueError("Existing run requires explicit --resume")
    else:
        baseline.write_json(out / "contract.json", cfg)

    model = timm.create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False,
                              num_classes=2, drop_path_rate=.3).cuda()
    model.load_state_dict(torch.load(initialization, map_location="cpu", weights_only=True))
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    fit = data[(data.domain == "BRSET") & (data.role == "fit")].reset_index(drop=True)
    selection = data[(data.domain == "BRSET") & (data.role == "selection")].reset_index(drop=True)
    spec = phase(args.smoke)

    resume = None
    if args.resume:
        resume = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
        if resume["contract"] != cfg:
            raise ValueError("Checkpoint contract mismatch")
        model.load_state_dict(resume["model"]); ema.load_state_dict(resume["ema"])
        baseline.restore_rng(resume["rng"])
    best = -1.0 if resume is None else resume["best"]
    start_update = 0 if resume is None else resume["update"]
    exposure = np.zeros((2, 3), dtype=np.int64) if resume is None else np.asarray(resume["exposure"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], betas=(.9, .999),
                                  eps=1e-8, weight_decay=.1)
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"]); baseline.restore_rng(resume["rng"])
    draw_seed = core.stream_seed(args.seed, "source_only_phase")
    batches = iter(baseline.loader(fit, True, draw_seed, args.workers,
                                   start_update * 64, spec["updates"] * 64))
    log = open(out / "events.jsonl", "a", buffering=1)

    def event(kind: str, **values) -> None:
        record = {"event": kind, "time": time.time(), "global_update": update, **values}
        log.write(json.dumps(record, allow_nan=False) + "\n"); print(json.dumps(record), flush=True)

    update = start_update
    event("start", job=os.environ["SLURM_JOB_ID"], resume=args.resume,
          fit_images=len(fit), selection_images=len(selection))
    checkpoint_seconds = []
    for update in range(start_update + 1, spec["updates"] + 1):
        torch.cuda.synchronize(); started = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
        lr = core.learning_rate(update, spec["updates"], spec["warmup"], spec["lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for micro in range(4):
            images, targets, indices = next(batches)
            sampled = fit.iloc[indices.tolist()]
            exposure[0] += [len(sampled), int(sampled[core.LABELS[0]].sum()),
                            int(sampled[core.LABELS[1]].sum())]
            images = images.cuda(non_blocking=True); targets = targets.cuda(non_blocking=True)
            images, targets = core.mixup(images, targets, draw_seed, (update - 1) * 4 + micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = core.focal(model(images), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {update}/{micro}")
            (loss / 4).backward(); loss_sum += loss.item() / 4
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.state_dict().values(), model.state_dict().values()):
                if target.is_floating_point():
                    target.mul_(.999).add_(source, alpha=.001)
                else:
                    target.copy_(source)
        torch.cuda.synchronize(); seconds = time.monotonic() - started
        if args.smoke or update % 25 == 0 or update == 1:
            event("update", loss=loss_sum, lr=lr, gradient_norm=float(grad_norm), seconds=seconds)
        if update in spec["eval_at"]:
            y_true, probabilities, _, infer_seconds = baseline.infer(ema, selection, args.workers)
            scored = core.metrics(y_true, probabilities, [.5, .5])
            score = float(np.mean([scored[label]["auroc"] for label in core.LABELS]))
            if score > best:
                best = score
                baseline.save(out / "best.pth", {"model": ema.state_dict(), "contract": cfg,
                                                  "global_update": update, "selection_score": score})
            event("selection", score=score, best=best, inference_seconds=infer_seconds,
                  metrics_at_05=scored)
        if update % 231 == 0 or update == spec["updates"] or (args.stop_after and update >= args.stop_after):
            started_save = time.monotonic()
            baseline.save(out / "last.pth", {"contract": cfg, "model": model.state_dict(),
                "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "rng": baseline.rng_state(),
                "update": update, "exposure": exposure.tolist(), "best": best})
            checkpoint_seconds.append(time.monotonic() - started_save)
        if args.stop_after and update >= args.stop_after:
            event("paused", reason="explicit resume verification"); log.close(); return

    chosen = torch.load(out / "best.pth", map_location="cpu", weights_only=False)
    ema.load_state_dict(chosen["model"])
    y_true, probabilities, view_logits, infer_seconds = baseline.infer(ema, selection, args.workers)
    thresholds = core.select_thresholds(y_true, probabilities)
    metadata = {**cfg, "checkpoint_sha256": core.sha(out / "best.pth"),
                "thresholds": thresholds, "checkpoint_update": chosen["global_update"]}
    baseline.save_predictions(out / "selection_predictions.npz", selection, y_true, probabilities,
                              view_logits, metadata)
    baseline.write_json(out / "selection_summary.json", {"metadata": metadata,
        "metrics": core.metrics(y_true, probabilities, thresholds),
        "warning": "BRSET selection diagnostics; no target image or label accessed."})
    baseline.write_json(out / "complete.json", {"training_complete": True, "smoke": args.smoke,
        "updates": update, "selection_inference_seconds": infer_seconds,
        "max_gpu_bytes": torch.cuda.max_memory_allocated(), "checkpoint_write_seconds": checkpoint_seconds,
        "exposure": exposure.tolist(), "contract": cfg})
    event("complete", assessment_performed=False); log.close()


def assess(args) -> None:
    baseline.node_check(True); baseline.setup(args.seed)
    _, data = protocol(); out = Path(args.output)
    complete = json.loads((out / "complete.json").read_text())
    if not complete["training_complete"] or complete["smoke"] or complete["contract"] != contract(args, baseline.initialize(args.seed)):
        raise ValueError("Training/provenance gate failed")
    selection = json.loads((out / "selection_summary.json").read_text())
    metadata = selection["metadata"]
    if metadata["checkpoint_sha256"] != core.sha(out / "best.pth"):
        raise ValueError("Frozen checkpoint changed")
    rows = data[(data.domain == "mBRSET") & (data.role == "assessment")].reset_index(drop=True)
    model = timm.create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False,
                              num_classes=2, drop_path_rate=.3).cuda()
    model.load_state_dict(torch.load(out / "best.pth", map_location="cpu", weights_only=False)["model"])
    y_true, probabilities, view_logits, seconds = baseline.infer(model, rows, args.workers)
    baseline.save_predictions(out / "assessment_predictions.npz", rows, y_true, probabilities,
                              view_logits, metadata)
    baseline.write_json(out / "assessment_summary.json", {"metadata": metadata,
        "metrics": core.metrics(y_true, probabilities, metadata["thresholds"]),
        "metrics_at_05": core.metrics(y_true, probabilities, [.5, .5]), "seconds": seconds,
        "interpretation": "Strict source-only transfer on historically reused target test; not pristine external confirmation."})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["train", "assess"])
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    try:
        train(args) if args.command == "train" else assess(args)
    except Exception as exc:
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        baseline.write_json(out / f"failure_{int(time.time())}.json",
                            {"error": repr(exc), "job": os.environ.get("SLURM_JOB_ID")})
        raise


if __name__ == "__main__":
    main()

