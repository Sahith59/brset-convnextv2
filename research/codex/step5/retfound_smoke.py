"""One-A40 real-image feasibility smoke for a B1-matched 512-pixel RETFound comparator."""
import copy
import argparse
import hashlib
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
import timm


ROOT = Path("/home/users/sthummala2/brset-convnextv2")
HERE = ROOT / "research/codex/step5"
BASELINE = ROOT / "research/codex/step2/full_pool/frozen"
RETFOUND_CODE = Path("/home/users/sthummala2/mbrset-retfound/RETFound_MAE")
sys.path.insert(0, str(BASELINE))
sys.path.insert(0, str(RETFOUND_CODE))
from baseline_core import Images, focal, flip_probabilities, mixup  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402


CONFIG = HERE / "retfound_matched_v1.json"
PREFLIGHT = HERE / "retfound_preflight.json"
OUTPUT = HERE / "retfound_smoke.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup(seed=20260915):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(config, head_seed=20260915):
    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m", pretrained=False, img_size=config["input"]["crop"],
        num_classes=2, drop_path_rate=0.3,
    )
    checkpoint = torch.load(config["weight_path"], map_location="cpu", weights_only=False)
    state = checkpoint["teacher"]
    state = {key.replace("backbone.", "").replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2."): value
             for key, value in state.items()}
    current = model.state_dict()
    removed = []
    for key in ("head.weight", "head.bias"):
        if key in state and state[key].shape != current[key].shape:
            removed.append({"key": key, "checkpoint": list(state[key].shape), "model": list(current[key].shape)})
            del state[key]
    interpolate_pos_embed(model, state)
    ignored = sorted(key for key in state if key not in current)
    state = {key: value for key, value in state.items() if key in current}
    backbone_numel = sum(value.numel() for key, value in current.items() if not key.startswith("head."))
    loaded_backbone_numel = sum(value.numel() for key, value in state.items()
                                 if not key.startswith("head.") and value.shape == current[key].shape)
    coverage = loaded_backbone_numel / backbone_numel
    missing, unexpected = model.load_state_dict(state, strict=False)
    torch.manual_seed(head_seed)
    torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)
    torch.nn.init.zeros_(model.head.bias)
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)
    return model, removed, list(missing), list(unexpected), ignored, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Allocated GPU node required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 A40-class GPU required")
    setup()
    config = json.loads(CONFIG.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight["status"] != "pass" or preflight["sha256"]["config"] != sha(CONFIG):
        raise ValueError("Preflight/config mismatch")
    manifest = (HERE / config["manifest"]).resolve()
    frame = pd.read_csv(manifest)
    fit = frame[frame.role == "fit"].reset_index(drop=True)
    selection = frame[(frame.role == "selection") & (frame.domain == "mBRSET")].reset_index(drop=True)
    if (frame.role == "assessment").sum() != 732:
        raise ValueError("Assessment rows changed")

    model, removed, missing, unexpected, ignored, coverage = load_model(config)
    expected_missing = {"head.weight", "head.bias"}
    if set(missing) != expected_missing or unexpected or coverage < 0.99:
        raise ValueError({"missing": missing, "unexpected": unexpected, "removed": removed,
                          "ignored": ignored, "backbone_numel_coverage": coverage})
    model = model.cuda()
    ema = copy.deepcopy(model).eval()
    if hasattr(ema, "set_grad_checkpointing"):
        ema.set_grad_checkpointing(enable=False)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    microbatch = args.microbatch or config["recipe"]["microbatch"]
    if microbatch < 1 or config["recipe"]["effective_batch"] % microbatch:
        raise ValueError("Microbatch must divide the effective batch")
    accumulation = config["recipe"]["effective_batch"] // microbatch
    dataset = Images(fit, True, 20260915)
    microsteps = max(config["smoke"]["real_fit_microsteps"], accumulation)
    durations = []
    losses = []
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for step in range(microsteps):
        batch = [dataset[(step * microbatch + offset) % len(dataset)] for offset in range(microbatch)]
        x = torch.stack([item[0] for item in batch]).cuda(non_blocking=True)
        # Re-entrant checkpointing requires at least one gradient-bearing input;
        # otherwise only the uncheckpointed classifier head receives gradients.
        x.requires_grad_(True)
        y = torch.stack([item[1] for item in batch]).cuda(non_blocking=True)
        x, y = mixup(x, y, 20260915, step)
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = focal(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        (loss / accumulation).backward()
        torch.cuda.synchronize(); durations.append(time.monotonic() - start); losses.append(float(loss))
    backbone_numel = sum(parameter.numel() for name, parameter in model.named_parameters()
                         if not name.startswith("head."))
    backbone_grad_numel = sum(parameter.numel() for name, parameter in model.named_parameters()
                              if not name.startswith("head.") and parameter.grad is not None)
    backbone_gradient_coverage = backbone_grad_numel / backbone_numel
    if backbone_gradient_coverage < 0.99:
        raise RuntimeError(f"Backbone gradient coverage too low: {backbone_gradient_coverage}")
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        for target, current in zip(ema.state_dict().values(), model.state_dict().values()):
            if target.is_floating_point():
                target.mul_(0.999).add_(current, alpha=0.001)
            else:
                target.copy_(current)

    validation = Images(selection.iloc[:config["smoke"]["real_validation_images"]].reset_index(drop=True), False, 0)
    probabilities = []
    inference_seconds = []
    ema.eval()
    with torch.inference_mode():
        for index in range(len(validation)):
            x, _, _ = validation[index]
            x = x.unsqueeze(0).cuda()
            torch.cuda.synchronize(); start = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                probability, _ = flip_probabilities(ema, x)
            torch.cuda.synchronize(); inference_seconds.append(time.monotonic() - start)
            if not torch.isfinite(probability).all():
                raise FloatingPointError("Nonfinite probability")
            probabilities.append(probability.cpu().tolist())

    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    median_microstep = float(np.median(durations))
    estimated_update = median_microstep * accumulation
    estimated_train_hours = estimated_update * config["recipe"]["updates"] / 3600
    estimated_selection_hours = float(np.median(inference_seconds)) * config["selection"]["images"] / 3600
    estimated_seed_hours = estimated_train_hours + 26 * estimated_selection_hours
    memory_pass = peak_gib <= config["smoke"]["max_peak_allocated_gib"]
    runtime_pass = estimated_seed_hours <= config["smoke"]["max_extrapolated_seed_hours"]
    result = {
        "status": "pass",
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "resolution": config["input"]["crop"],
        "microbatch": microbatch,
        "accumulation": accumulation,
        "fit_microsteps_measured": microsteps,
        "losses": losses,
        "gradient_norm": float(grad),
        "backbone_gradient_numel_coverage": backbone_gradient_coverage,
        "median_microstep_seconds": median_microstep,
        "estimated_update_seconds": estimated_update,
        "estimated_train_hours": estimated_train_hours,
        "estimated_selection_hours_per_pass": estimated_selection_hours,
        "estimated_seed_hours": estimated_seed_hours,
        "peak_allocated_gib": peak_gib,
        "finite_validation_probabilities": probabilities,
        "checkpoint_load": {"removed_shape_mismatches": removed, "missing": missing, "unexpected": unexpected,
                            "ignored_pretraining_keys": ignored, "backbone_numel_coverage": coverage},
        "memory_gate_pass": memory_pass,
        "runtime_gate_pass": runtime_pass,
        "feasibility_gate_pass": memory_pass and runtime_pass,
        "assessment_accessed": False,
        "sha256": {"config": sha(CONFIG), "preflight": sha(PREFLIGHT), "script": sha(__file__)},
        "interpretation": "Implementation/runtime evidence only; no model comparison or performance result."
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
