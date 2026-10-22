"""D107 test-time arm (Residual-CycleGAN's protocol): the BRSET-only classifiers lebase_b0_s{0,1,2} are applied
to mBRSET val/test photos translated to tabletop style by a CycleGAN G_ba (brset_gen/cgtt/<tag>/{val,test}).
tag 'ident' = the untranslated 560 px PNGs (control); it must reproduce the stored lebase_b0 predictions.

Inference is exactly the trainer's (scripts/30_train_strong_baseline.py imported, not copied): same eval
transform (squash 560 -> centre crop 512), same autocast dtype, flip-4 TTA, cut-off = tune_thresholds on the
(translated) validation set with bootstrap 200 and the model's seed. Writes cls_runs/cgtt_<tag>_s{s}/
{val,test}_predictions.npz in the trainer's format, so the D107 read script treats it like any run.
"""
import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/users/sthummala2/brset-convnextv2")
CLS = Path("/data/users3/sthummala2/brset_gen/cls_runs")
CGTT = Path("/data/users3/sthummala2/brset_gen/cgtt")
sp = importlib.util.spec_from_file_location("T30", str(ROOT / "scripts/30_train_strong_baseline.py"))
T = importlib.util.module_from_spec(sp); sp.loader.exec_module(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="ident | plain | residual")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    dev = torch.device("cuda")
    for s in [int(x) for x in a.seeds.split(",")]:
        src = ROOT / "results" / f"lebase_b0_s{s}"          # where the lebase runs live (scripts/154 rd())
        if not (src / "checkpoint-best.pth").exists():      # D107a: s1/s2 checkpoints deleted 10 Oct -> identical-recipe retrain
            src = CLS / f"lebase_b0_s{s}_rt"
        ck = torch.load(src / "checkpoint-best.pth", map_location="cpu", weights_only=False)
        args = argparse.Namespace(**ck["args"])
        assert args.nb_classes == 2 and not getattr(args, "ordinal_aux", False)
        T._EVAL_DTYPE[0] = None if args.amp_dtype == "fp32" else T.AMP_DTYPE[args.amp_dtype]
        _, eval_tf = T.build_transforms(args)
        model = timm.create_model(args.model, pretrained=False, num_classes=args.nb_classes)
        model.load_state_dict(ck["model"]); model.to(dev).eval()
        out = {}
        for split in ("val", "test"):
            ds = T.BRSETMultiLabel(CGTT / a.tag / split, eval_tf)
            ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            out[split] = T.run_inference(model, ld, dev, tta=args.tta)
        thr = T.tune_thresholds(out["val"][0], out["val"][1], n_boot=args.thr_bootstrap, seed=args.seed)
        od = CLS / f"cgtt_{a.tag}_s{s}"; od.mkdir(parents=True, exist_ok=True)
        np.savez(od / "test_predictions.npz", y_true=out["test"][0], y_prob=out["test"][1], thresholds=np.array(thr))
        np.savez(od / "val_predictions.npz", y_true=out["val"][0], y_prob=out["val"][1])
        info = {"source_model": str(src), "epoch": ck["epoch"], "variant": ck["variant"], "tag": a.tag, "thresholds": thr}
        if a.tag == "ident":     # control: reproduce the stored predictions of the same model on the original files
            ref = np.load(src / "test_predictions.npz"); rv = np.load(src / "val_predictions.npz")
            assert (ref["y_true"] == out["test"][0]).all() and (rv["y_true"] == out["val"][0]).all(), "row order differs"
            info["max_abs_diff_test_prob"] = float(np.abs(ref["y_prob"] - out["test"][1]).max())
            info["max_abs_diff_val_prob"] = float(np.abs(rv["y_prob"] - out["val"][1]).max())
            info["stored_thresholds"] = ref["thresholds"].tolist()
        (od / "cgtt_info.json").write_text(json.dumps(info, indent=1))
        print(f"cgtt {a.tag} s{s}: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
