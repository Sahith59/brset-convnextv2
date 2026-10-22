"""Score a list of images with trained classifiers, exactly as the trainer scores test.

Same eval transform (scripts/30_train_strong_baseline.build_transforms), same flip4
test-time augmentation, same bf16 autocast, same checkpoint-best.pth weights.

--check reproduces a run's saved val_predictions.npz from scratch before anything
else is trusted: if this script and the trainer disagree on the validation images,
nothing it says about unlabelled images can be believed.

Unlabelled use: --files_csv is read with usecols=["file", "domain"]; the label
columns are never parsed, so no target label can reach the output.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent
ROOT = HERE.parent
sp = importlib.util.spec_from_file_location("T30", str(HERE / "30_train_strong_baseline.py"))
T = importlib.util.module_from_spec(sp)
sp.loader.exec_module(T)


class Files(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), torch.zeros(2)


@torch.no_grad()
def score(run_dir, paths, batch=32, workers=8):
    ck = torch.load(Path(run_dir) / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    _, eval_tf = T.build_transforms(a)
    model = timm.create_model(a.model, pretrained=False, num_classes=a.nb_classes).cuda()
    model.load_state_dict(ck["model"])
    T._EVAL_DTYPE[0] = None if a.amp_dtype == "fp32" else T.AMP_DTYPE[a.amp_dtype]
    T._NB_OUT[0] = None
    dl = DataLoader(Files(paths, eval_tf), batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    _, prob = T.run_inference(model, dl, torch.device("cuda"), tta=a.tta)
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="classifier run dirs (one per seed)")
    ap.add_argument("--files_csv", default=str(ROOT / "data/finetune_joint/train/labels.csv"))
    ap.add_argument("--split_dir", default=str(ROOT / "data/finetune_joint/train"))
    ap.add_argument("--domain", type=int, default=1, help="1 = mBRSET rows only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="first reproduce val_predictions.npz of the first run")
    a = ap.parse_args()

    if a.check:
        vl = pd.read_csv(ROOT / "data/finetune_joint/val/labels.csv", usecols=["file"])
        p = score(a.runs[0], [ROOT / "data/finetune_joint/val" / f for f in vl.file])
        ref = np.load(Path(a.runs[0]) / "val_predictions.npz")["y_prob"]
        from sklearn.metrics import roc_auc_score
        yv = np.load(Path(a.runs[0]) / "val_predictions.npz")["y_true"][:, 0]
        dmax, dmean = np.abs(p - ref).max(), np.abs(p - ref).mean()
        dauc = abs(roc_auc_score(yv, p[:, 0]) - roc_auc_score(yv, ref[:, 0]))
        print(f"check over {len(p)} validation images: max |dp| {dmax:.2e}, mean |dp| {dmean:.2e}, "
              f"|d DR AUROC| {dauc:.2e}", flush=True)
        # bf16 inference differs slightly across GPU models; a pipeline error shows up as a
        # large MEAN difference or a changed AUROC, not as one value off by a few 1e-3.
        if dmean > 5e-4 or dmax > 1e-2 or dauc > 1e-3:
            raise SystemExit("prediction pipeline does not reproduce the trainer; refusing to continue")

    df = pd.read_csv(a.files_csv, usecols=["file", "domain"])
    assert list(df.columns) == ["file", "domain"], "label columns must never be parsed"
    df = df[df.domain == a.domain].reset_index(drop=True)
    paths = [Path(a.split_dir) / f for f in df.file]
    probs = np.stack([score(r, paths) for r in a.runs])          # (seeds, n, 2)
    np.savez(a.out, files=df.file.values, prob=probs, runs=np.array([str(r) for r in a.runs]))
    m = probs.mean(0)
    print(f"scored {len(paths)} images with {len(a.runs)} models -> {a.out}; "
          f"mean DR score {m[:, 0].mean():.3f}, ME {m[:, 1].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
