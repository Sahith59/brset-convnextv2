"""Train frozen-feature global and top-k spatial heads on allocated GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "research/codex/step5"
BASE = ROOT / "research/codex/step2/full_pool/frozen"
PRIVATE = Path("/data/users3/sthummala2/brset-codex/step5/b1_features")
KS = [1, 4, 8, 16, 32]
sys.path.insert(0, str(BASE))
import baseline_core as core  # noqa: E402
import train_baseline as baseline  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DiagnosticHeads(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.global_head = nn.Linear(channels, 2)
        self.patch_weight = nn.Parameter(torch.empty(len(KS), 2, channels))
        self.patch_bias = nn.Parameter(torch.zeros(len(KS), 2))
        nn.init.xavier_uniform_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        for index in range(len(KS)):
            nn.init.xavier_uniform_(self.patch_weight[index])

    def forward(self, features):
        pooled = features.mean(dim=(-1, -2))
        global_logits = self.global_head(pooled)
        patches = features.flatten(2)
        scores = torch.einsum("bcp,qlc->bqlp", patches, self.patch_weight)
        scores = scores + self.patch_bias[None, :, :, None]
        patch_logits = torch.stack(
            [scores[:, index].topk(k, dim=-1).values.mean(-1) for index, k in enumerate(KS)], dim=1)
        return global_logits, patch_logits


def setup(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(core.stream_seed(seed, "spatial_heads"))
    torch.cuda.manual_seed_all(core.stream_seed(seed, "spatial_heads"))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def batches(size, batch_size, seed, epoch):
    order = np.random.default_rng(core.stream_seed(seed, "spatial_head_order", epoch)).permutation(size)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]


@torch.inference_mode()
def evaluate(model, spatial, truth, batch_size=8):
    model.eval()
    probabilities = {"G0": []}
    for k in KS:
        probabilities[f"G1_k{k}"] = []
    for start in range(0, len(spatial), batch_size):
        x = torch.from_numpy(np.asarray(spatial[start:start + batch_size]).copy()).cuda().float()
        batch, views = x.shape[:2]
        global_logits, patch_logits = model(x.reshape(batch * views, *x.shape[2:]))
        probabilities["G0"].append(global_logits.sigmoid().reshape(batch, views, 2).mean(1).cpu().numpy())
        for index, k in enumerate(KS):
            probabilities[f"G1_k{k}"].append(
                patch_logits[:, index].sigmoid().reshape(batch, views, 2).mean(1).cpu().numpy())
    result = {}
    combined = {}
    for name, parts in probabilities.items():
        value = np.concatenate(parts)
        thresholds = core.select_thresholds(truth, value)
        result[name] = {"thresholds": thresholds, "metrics": core.metrics(truth, value, thresholds)}
        combined[name] = value
    return result, combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    baseline.node_check(True); setup(args.head_seed)
    extraction = json.loads((WORK / "feature_extraction_summary.json").read_text())
    if extraction["status"] != "pass" or extraction["assessment_accessed"]:
        raise ValueError("Feature extraction gate failed")
    script_hash = sha(Path(__file__))
    if not args.smoke:
        smoke = json.loads((WORK / "spatial_head_smoke.json").read_text())
        if smoke["status"] != "pass" or smoke["script_sha256"] != script_hash:
            raise ValueError("Spatial-head smoke gate changed or failed")

    fit = np.load(PRIVATE / "fit_spatial.npy", mmap_mode="r")
    selection = np.load(PRIVATE / "selection_spatial.npy", mmap_mode="r")
    rows = np.load(PRIVATE / "rows.npz", allow_pickle=False)
    fit_y, selection_y = rows["fit_y"], rows["selection_y"]
    if fit.shape != (14774, 1, 1536, 16, 16) or selection.shape != (725, 4, 1536, 16, 16):
        raise ValueError(f"Spatial feature shape changed: {fit.shape}, {selection.shape}")

    model = DiagnosticHeads(1536).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 1 if args.smoke else 20
    max_batches = 2 if args.smoke else None
    total_steps = epochs * math.ceil(len(fit) / 64)
    step = 0; events = []; started = time.monotonic()
    for epoch in range(epochs):
        model.train(); losses = []
        for batch_number, indices in enumerate(batches(len(fit), 64, args.head_seed, epoch)):
            if max_batches is not None and batch_number >= max_batches:
                break
            step += 1
            lr = 1e-3 if total_steps == 1 else 5e-4 * (1 + math.cos(math.pi * (step - 1) / (total_steps - 1)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            x = torch.from_numpy(np.asarray(fit[indices, 0]).copy()).cuda().float()
            y = torch.from_numpy(fit_y[indices]).cuda().float()
            optimizer.zero_grad(set_to_none=True)
            global_logits, patch_logits = model(x)
            loss = core.focal(global_logits, y)
            for index in range(len(KS)):
                loss = loss + core.focal(patch_logits[:, index], y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite head loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        events.append({"epoch": epoch + 1, "mean_summed_head_loss": float(np.mean(losses)),
                       "gradient_norm_last": float(grad), "lr_last": lr})

    evaluation_rows = 128 if args.smoke else len(selection)
    metrics, probabilities = evaluate(model, selection[:evaluation_rows], selection_y[:evaluation_rows])
    if args.smoke:
        result = {
            "status": "pass",
            "job_id": os.environ["SLURM_JOB_ID"],
            "updates": step,
            "validation_images": evaluation_rows,
            "finite_metrics": all(np.isfinite(record["metrics"][label]["auroc"])
                                  for record in metrics.values() for label in core.LABELS),
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "seconds": time.monotonic() - started,
            "script_sha256": script_hash,
            "feature_summary_sha256": sha(WORK / "feature_extraction_summary.json"),
            "assessment_accessed": False,
        }
        if not result["finite_metrics"]:
            result["status"] = "fail"
        (WORK / "spatial_head_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if result["status"] != "pass":
            raise SystemExit(1)
        return

    private_path = PRIVATE / f"spatial_heads_seed{args.head_seed}.npz"
    np.savez_compressed(private_path, selection_y=selection_y,
                        probabilities_json=np.asarray(json.dumps({name: value.tolist() for name, value in probabilities.items()})),
                        state_json=np.asarray(json.dumps({name: tensor.detach().cpu().tolist()
                                                         for name, tensor in model.state_dict().items()})))
    result = {
        "status": "pass",
        "job_id": os.environ["SLURM_JOB_ID"],
        "head_seed": args.head_seed,
        "fit_images": len(fit),
        "selection_images": len(selection),
        "epochs": epochs,
        "updates": step,
        "heads": metrics,
        "training_events": events,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "seconds": time.monotonic() - started,
        "private_result_sha256": sha(private_path),
        "script_sha256": script_hash,
        "feature_summary_sha256": sha(WORK / "feature_extraction_summary.json"),
        "assessment_accessed": False,
        "interpretation": "Frozen-backbone validation diagnostic; patches selected by top-k score are not lesion annotations.",
    }
    output = WORK / f"spatial_head_seed{args.head_seed}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
