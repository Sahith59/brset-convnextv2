"""Resumable validation-only trainer for the protocol-matched RETFound comparator."""
import argparse
import copy
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path("/home/users/sthummala2/brset-convnextv2")
HERE = ROOT / "research/codex/step5"
BASELINE = ROOT / "research/codex/step2/full_pool/frozen"
sys.path.insert(0, str(BASELINE))
sys.path.insert(0, str(HERE))
from baseline_core import (DrawBatches, Images, LABELS, focal, flip_probabilities,  # noqa: E402
                           learning_rate, metrics, mixup, select_thresholds, sha, stream_seed)
from retfound_smoke import load_model  # noqa: E402


CONFIG = HERE / "retfound_matched_v1.json"
PREFLIGHT = HERE / "retfound_preflight.json"
SMOKE = HERE / "retfound_smoke.json"


def write_json(path, data):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save(path, state):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary); os.replace(temporary, path)


def node_check():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Allocated GPU node required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 GPU required")


def setup(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"]); random.setstate(state["python"])


def load_inputs():
    config = json.loads(CONFIG.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    smoke = json.loads(SMOKE.read_text())
    manifest = (HERE / config["manifest"]).resolve()
    if preflight["status"] != "pass" or preflight["sha256"]["config"] != sha(CONFIG):
        raise ValueError("Preflight does not match frozen config")
    if not smoke["feasibility_gate_pass"] or smoke["sha256"]["config"] != sha(CONFIG):
        raise ValueError("GPU smoke does not match frozen config")
    if smoke["microbatch"] != config["recipe"]["microbatch"] or smoke["accumulation"] != config["recipe"]["accumulation"]:
        raise ValueError("GPU smoke batch structure mismatch")
    frame = pd.read_csv(manifest)
    fit = frame[frame.role == "fit"].reset_index(drop=True)
    selection = frame[(frame.role == "selection") & (frame.domain == "mBRSET")].reset_index(drop=True)
    if len(fit) != config["fit"]["total_images"] or len(selection) != config["selection"]["images"]:
        raise ValueError("Cohort mismatch")
    return config, fit, selection


def loader(rows, training, seed, workers, microbatch=16, start_draw=0, stop_draw=0):
    dataset = Images(rows, training, seed)
    kwargs = {"num_workers": workers, "pin_memory": True,
              "generator": torch.Generator().manual_seed(stream_seed(seed, "loader"))}
    if training:
        return DataLoader(dataset, batch_sampler=DrawBatches(len(dataset), microbatch, start_draw, stop_draw, seed), **kwargs)
    return DataLoader(dataset, batch_size=4, shuffle=False, **kwargs)


@torch.inference_mode()
def infer(model, rows, workers):
    model.eval(); truths = []; probabilities = []; logits = []; identities = []; start = time.monotonic()
    for images, targets, indices in loader(rows, False, 0, workers):
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probability, view_logits = flip_probabilities(model, images)
        if not torch.isfinite(probability).all():
            raise FloatingPointError("Nonfinite validation probability")
        truths.append(targets.numpy()); probabilities.append(probability.cpu().numpy())
        logits.append(view_logits.cpu().numpy()); identities.extend(indices.tolist())
    if identities != list(range(len(rows))):
        raise ValueError("Validation identity/order mismatch")
    torch.cuda.synchronize()
    return np.concatenate(truths), np.concatenate(probabilities), np.concatenate(logits), time.monotonic() - start


def contract(args, config):
    return {
        "config_sha256": sha(CONFIG), "manifest_sha256": config["manifest_sha256"],
        "weight_sha256": config["weight_sha256"], "trainer_sha256": sha(__file__),
        "model_loader_sha256": sha(HERE / "retfound_smoke.py"),
        "baseline_core_sha256": sha(BASELINE / "baseline_core.py"),
        "preflight_sha256": sha(PREFLIGHT), "smoke_sha256": sha(SMOKE),
        "seed": args.seed, "smoke": args.smoke, "assessment_access": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    node_check(); setup(args.seed)
    config, fit, selection = load_inputs()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    run_contract = contract(args, config)
    contract_path = output / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != run_contract:
            raise ValueError("Run contract changed")
        if not args.resume:
            raise ValueError("Existing output requires --resume")
    else:
        write_json(contract_path, run_contract)

    head_seed = stream_seed(args.seed, "retfound_head")
    model, _, missing, unexpected, _, coverage = load_model(config, head_seed=head_seed)
    if set(missing) != {"head.weight", "head.bias"} or unexpected or coverage < 0.99:
        raise ValueError("Initialization coverage mismatch")
    model = model.cuda()
    ema = copy.deepcopy(model).eval()
    if hasattr(ema, "set_grad_checkpointing"):
        ema.set_grad_checkpointing(enable=False)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["recipe"]["peak_learning_rate"],
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=config["recipe"]["weight_decay"])

    total_updates = 6 if args.smoke else config["recipe"]["updates"]
    warmup = 2 if args.smoke else config["recipe"]["warmup_updates"]
    eval_at = [3, 6] if args.smoke else list(range(231, total_updates + 1, 231))
    resume = None
    if args.resume:
        resume = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
        if resume["contract"] != run_contract:
            raise ValueError("Checkpoint contract mismatch")
        model.load_state_dict(resume["model"]); ema.load_state_dict(resume["ema"])
        optimizer.load_state_dict(resume["optimizer"]); restore_rng(resume["rng"])
    start_update = 0 if resume is None else int(resume["update"])
    best = -1.0 if resume is None else float(resume["best"])
    exposure = np.zeros((2, 3), dtype=np.int64) if resume is None else np.asarray(resume["exposure"], dtype=np.int64)
    microbatch = config["recipe"]["microbatch"]; accumulation = config["recipe"]["accumulation"]
    draw_seed = stream_seed(args.seed, "retfound_fit")
    batches = iter(loader(fit, True, draw_seed, args.workers, microbatch,
                          start_update * config["recipe"]["effective_batch"],
                          total_updates * config["recipe"]["effective_batch"]))
    event_log = open(output / "events.jsonl", "a", buffering=1)

    def event(kind, **values):
        record = {"event": kind, "time": time.time(), **values}
        event_log.write(json.dumps(record, allow_nan=False) + "\n"); print(json.dumps(record), flush=True)

    def checkpoint(update):
        save(output / "last.pth", {"contract": run_contract, "model": model.state_dict(),
             "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
             "update": update, "best": best, "exposure": exposure.tolist()})

    event("start", job=os.environ["SLURM_JOB_ID"], node=socket.gethostname(), gpu=torch.cuda.get_device_name(),
          resume=args.resume, start_update=start_update, total_updates=total_updates)
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for update in range(start_update + 1, total_updates + 1):
        torch.cuda.synchronize(); start = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(update, total_updates, warmup, config["recipe"]["peak_learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        mean_loss = 0.0
        for microstep in range(accumulation):
            images, targets, indices = next(batches)
            sampled = fit.iloc[indices.tolist()]
            for domain_index, domain in enumerate(("BRSET", "mBRSET")):
                subset = sampled[sampled.domain == domain]
                exposure[domain_index] += [len(subset), int(subset[LABELS[0]].sum()), int(subset[LABELS[1]].sum())]
            images = images.cuda(non_blocking=True); images.requires_grad_(True)
            targets = targets.cuda(non_blocking=True)
            images, targets = mixup(images, targets, draw_seed, (update - 1) * accumulation + microstep)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = focal(model(images), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {update}/{microstep}")
            (loss / accumulation).backward(); mean_loss += float(loss) / accumulation
        if update == start_update + 1:
            backbone = [(name, parameter) for name, parameter in model.named_parameters() if not name.startswith("head.")]
            gradient_coverage = sum(parameter.numel() for _, parameter in backbone if parameter.grad is not None) / sum(parameter.numel() for _, parameter in backbone)
            if gradient_coverage < 0.99:
                raise RuntimeError(f"Backbone gradient coverage too low: {gradient_coverage}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["recipe"]["gradient_norm_cap"], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for target, current in zip(ema.state_dict().values(), model.state_dict().values()):
                if target.is_floating_point():
                    target.mul_(config["recipe"]["ema_decay"]).add_(current, alpha=1 - config["recipe"]["ema_decay"])
                else:
                    target.copy_(current)
        torch.cuda.synchronize(); elapsed = time.monotonic() - start; durations.append(elapsed)
        if args.smoke or update == 1 or update % 25 == 0:
            event("update", update=update, loss=mean_loss, lr=lr, gradient_norm=float(gradient_norm), seconds=elapsed)
        if update in eval_at:
            truth, probability, _, inference_seconds = infer(ema, selection, args.workers)
            at_half = metrics(truth, probability, [0.5, 0.5])
            score = float(np.mean([at_half[label]["auroc"] for label in LABELS]))
            if score > best:
                best = score
                save(output / "best.pth", {"model": ema.state_dict(), "contract": run_contract,
                     "update": update, "selection_score": score})
            event("selection", update=update, score=score, best=best, seconds=inference_seconds,
                  metrics_at_05=at_half)
            checkpoint(update)
        if args.stop_after and update >= args.stop_after:
            checkpoint(update); event("paused", update=update); event_log.close(); return

    chosen = torch.load(output / "best.pth", map_location="cpu", weights_only=False)
    ema.load_state_dict(chosen["model"])
    truth, probability, view_logits, inference_seconds = infer(ema, selection, args.workers)
    thresholds = select_thresholds(truth, probability)
    metadata = {**run_contract, "checkpoint_sha256": sha(output / "best.pth"),
                "checkpoint_update": int(chosen["update"]), "thresholds": thresholds}
    np.savez_compressed(output / "selection_predictions.npz", y_true=truth, probabilities=probability,
                        view_logits=view_logits, metadata_json=np.array(json.dumps(metadata)))
    write_json(output / "selection_summary.json", {"metadata": metadata,
               "metrics": metrics(truth, probability, thresholds),
               "warning": "Validation-only comparator screen; no assessment result."})
    write_json(output / "complete.json", {"training_complete": True, "updates": total_updates,
               "smoke": args.smoke, "assessment_performed": False, "contract": run_contract,
               "exposure": exposure.tolist(), "median_update_seconds": float(np.median(durations)),
               "selection_inference_seconds": inference_seconds,
               "max_gpu_bytes": int(torch.cuda.max_memory_allocated())})
    checkpoint(total_updates); event("complete", update=total_updates, assessment_performed=False)
    event_log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(repr(error), flush=True)
        raise
