"""Experiment 1: Domain-Separation training on the aggregated BRSET + mBRSET set.

Dr. Ye's proposal is to aggregate both datasets and add a routing mechanism in
the ENCODER that decides, per image, how much to draw on each source. The
closest published architecture is Domain Separation Networks (Bousmalis et al.,
NeurIPS 2016, arxiv.org/abs/1608.06019): a shared encoder capturing what the
domains have in common, plus a private encoder per domain capturing what is
specific to each.

This script implements the feature-level form of that idea. The original DSN
also reconstructs the input from shared + private; that is dropped here because
a pixel decoder on a 512 px ConvNeXt V2 Large is expensive and, unlike the
original setting, BOTH of our domains are labelled, so the task loss already
supervises the private branches. This is therefore DSN-inspired rather than a
literal reimplementation, and it is reported that way.

  L = L_task + beta * L_difference + gamma * L_similarity

  L_difference  orthogonality between shared and private features, so the two
                branches cannot encode the same thing (DSN eq. 5)
  L_similarity  domain classifier on the shared features behind a gradient
                reversal layer, so shared features carry no domain identity

beta and gamma default to the values used in the DSN paper (0.075, 0.25).

Evaluation is identical to every other run in this project: mBRSET val/test
only, 4-way flip TTA, bootstrap-smoothed per-label cutoffs chosen on validation.
"""
import argparse, importlib.util, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_spec = importlib.util.spec_from_file_location(
    "t30", str(Path(__file__).parent / "30_train_strong_baseline.py"))
t30 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(t30)

LABEL_COLS = t30.LABEL_COLS
AMP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class JointDataset(Dataset):
    """Same layout as BRSETMultiLabel, but also returns the domain id."""

    def __init__(self, split_dir, transform):
        self.split_dir = Path(split_dir)
        self.df = pd.read_csv(self.split_dir / "labels.csv")
        assert "domain" in self.df.columns, "labels.csv needs a domain column"
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = self.split_dir / row["file"]
        last = None
        for attempt in range(5):
            try:
                img = self.transform(Image.open(path).convert("RGB"))
                y = torch.tensor([row[c] for c in LABEL_COLS], dtype=torch.float32)
                return img, y, int(row["domain"])
            except (OSError, IOError) as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"failed to read {path}: {last}")


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


class DomainSeparationNet(nn.Module):
    def __init__(self, backbone, n_dom=2, proj=512, n_cls=2, drop_path=0.3):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0,
                                          drop_path_rate=drop_path)
        d = self.encoder.num_features
        self.shared = nn.Sequential(nn.Linear(d, proj), nn.GELU(), nn.LayerNorm(proj))
        self.private = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, proj), nn.GELU(), nn.LayerNorm(proj)) for _ in range(n_dom)])
        self.classifier = nn.Linear(2 * proj, n_cls)
        self.domain_head = nn.Sequential(nn.Linear(proj, 256), nn.GELU(), nn.Linear(256, n_dom))

    def forward(self, x, domain, grl_lambda=1.0):
        f = self.encoder(x)
        h_s = self.shared(f)
        # gather the private branch belonging to each sample's own domain
        h_p = torch.stack([p(f) for p in self.private], dim=1)          # B x n_dom x proj
        h_p = h_p[torch.arange(f.size(0), device=f.device), domain]      # B x proj
        logits = self.classifier(torch.cat([h_s, h_p], dim=1))
        dom_logits = self.domain_head(GradReverse.apply(h_s, grl_lambda))
        return logits, h_s, h_p, dom_logits


def difference_loss(h_s, h_p):
    """Orthogonality between shared and private features (DSN eq. 5)."""
    a = h_s - h_s.mean(0, keepdim=True)
    b = h_p - h_p.mean(0, keepdim=True)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-6)
    return (a.t() @ b).pow(2).mean()


@torch.no_grad()
def infer(model, loader, device, amp, tta="flip4"):
    model.eval()
    ys, ps = [], []
    for images, y, dom in loader:
        images = images.to(device, non_blocking=True); dom = dom.to(device)
        with torch.autocast("cuda", dtype=AMP[amp], enabled=amp != "fp32"):
            views = [images]
            if tta in ("hflip", "flip4"): views.append(torch.flip(images, dims=[3]))
            if tta == "flip4":
                views += [torch.flip(images, dims=[2]), torch.flip(images, dims=[2, 3])]
            pr = torch.stack([torch.sigmoid(model(v, dom)[0].float()) for v in views]).mean(0)
        ys.append(y.numpy()); ps.append(pr.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convnextv2_large.fcmae_ft_in22k_in1k_384")
    p.add_argument("--data_path", default="/home/users/sthummala2/brset-convnextv2/data/finetune_joint")
    p.add_argument("--input_size", default=512, type=int)
    p.add_argument("--resize_size", default=560, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--accum_iter", default=4, type=int)
    p.add_argument("--epochs", default=25, type=int)
    p.add_argument("--warmup_epochs", default=3, type=int)
    p.add_argument("--lr", default=3e-5, type=float)
    p.add_argument("--weight_decay", default=0.1, type=float)
    p.add_argument("--drop_path", default=0.3, type=float)
    p.add_argument("--proj_dim", default=512, type=int)
    p.add_argument("--beta_diff", default=0.075, type=float, help="DSN difference-loss weight")
    p.add_argument("--gamma_sim", default=0.25, type=float, help="DSN similarity-loss weight")
    p.add_argument("--focal_gamma", default=2.0, type=float)
    p.add_argument("--label_smoothing", default=0.1, type=float)
    p.add_argument("--ema_decay", default=0.999, type=float)
    p.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--tta", default="flip4")
    p.add_argument("--thr_bootstrap", default=200, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--task", default="dsn_joint")
    p.add_argument("--output_dir", default="/home/users/sthummala2/brset-convnextv2/results")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.output_dir) / args.task; out.mkdir(parents=True, exist_ok=True)
    print(f"Namespace: {vars(args)}", flush=True)

    class _A: pass
    a = _A(); a.resize_size, a.input_size = args.resize_size, args.input_size
    train_tf, eval_tf = t30.build_transforms(a)
    root = Path(args.data_path)
    ds_tr = JointDataset(root / "train", train_tf)
    ds_va = JointDataset(root / "val", eval_tf)
    ds_te = JointDataset(root / "test", eval_tf)
    print(f"train {len(ds_tr)} (BRSET {int((ds_tr.df.domain==0).sum())}, "
          f"mBRSET {int((ds_tr.df.domain==1).sum())}) | val {len(ds_va)} | test {len(ds_te)}", flush=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = DomainSeparationNet(args.model, 2, args.proj_dim, len(LABEL_COLS), args.drop_path).to(device)
    print(f"n_parameters: {sum(q.numel() for q in model.parameters())}", flush=True)
    ema = t30.ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    task_loss = t30.MultiLabelFocalLoss(gamma=args.focal_gamma)
    dom_loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=args.warmup_epochs)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs - args.warmup_epochs, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda")

    best_score, best_epoch, best_variant = -1.0, -1, "raw"
    log = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        run = {"task": 0.0, "diff": 0.0, "sim": 0.0}; nb = 0; n_bad = 0
        opt.zero_grad(set_to_none=True)
        # GRL strength ramps up, as in DANN/DSN, so the encoder is not fought early
        prog = epoch / max(args.epochs - 1, 1)
        grl = 2.0 / (1.0 + math.exp(-10 * prog)) - 1.0
        for i, (x, y, dom) in enumerate(dl_tr):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            dom = dom.to(device, non_blocking=True)
            if args.label_smoothing > 0:
                y = y * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            with torch.autocast("cuda", dtype=AMP[args.amp_dtype], enabled=args.amp_dtype != "fp32"):
                logits, h_s, h_p, dlog = model(x, dom, grl)
                l_task = task_loss(logits, y)
                l_diff = difference_loss(h_s.float(), h_p.float())
                l_sim = dom_loss(dlog, dom)
                loss = (l_task + args.beta_diff * l_diff + args.gamma_sim * l_sim) / args.accum_iter
            if not torch.isfinite(loss):
                n_bad += 1; opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward()
            if (i + 1) % args.accum_iter == 0 or (i + 1) == len(dl_tr):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema is not None: ema.update(model)
            run["task"] += l_task.item(); run["diff"] += l_diff.item(); run["sim"] += l_sim.item(); nb += 1
            if i % 40 == 0:
                print(f"Epoch: [{epoch}]  [{i}/{len(dl_tr)}]  lr: {opt.param_groups[0]['lr']:.7f}  "
                      f"task: {l_task.item():.4f}  diff: {l_diff.item():.4f}  sim: {l_sim.item():.4f}  "
                      f"grl: {grl:.2f}  {time.time()-t0:.0f}s", flush=True)
        sched.step()
        if n_bad: print(f"WARNING epoch {epoch}: skipped {n_bad} non-finite batches", flush=True)

        for variant, net in (("raw", model), ("ema", ema.module if ema else None)):
            if net is None: continue
            yv, pv = infer(net, dl_va, device, args.amp_dtype, args.tta)
            thr = t30.tune_thresholds(yv, pv, n_boot=0, seed=args.seed)
            per, mauc, mf1 = t30.compute_per_label_metrics(yv, pv, thr)
            score = 0.5 * (mauc + mf1)
            print(f"val[{variant}]: macro_auc={mauc:.4f} macro_f1={mf1:.4f} score={score:.4f}", flush=True)
            if score > best_score:
                best_score, best_epoch, best_variant = score, epoch, variant
                torch.save({"model": net.state_dict(), "epoch": epoch, "variant": variant,
                            "args": vars(args)}, out / "checkpoint-best.pth")
        log.append({"epoch": epoch, **{k: v / max(nb, 1) for k, v in run.items()},
                    "best_score": best_score})
        (out / "log.txt").write_text("\n".join(json.dumps(r) for r in log))
        print(f"Best epoch = {best_epoch} ({best_variant}), Best score = {best_score:.4f}", flush=True)

    ck = torch.load(out / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); model.to(device)
    print(f"\nTest with best model: epoch {ck['epoch']} (variant={ck['variant']}), tta={args.tta}", flush=True)
    yv, pv = infer(model, dl_va, device, args.amp_dtype, args.tta)
    thr = t30.tune_thresholds(yv, pv, n_boot=args.thr_bootstrap, seed=args.seed)
    print(f"thresholds (val, bootstrap={args.thr_bootstrap}): {dict(zip(LABEL_COLS, thr))}", flush=True)
    yt, pt = infer(model, dl_te, device, args.amp_dtype, args.tta)
    per, mauc, mf1 = t30.compute_per_label_metrics(yt, pt, thr)
    print("TEST RESULTS:", flush=True)
    for c in LABEL_COLS:
        v = per[c]
        print(f"  {c}: AUC={v['auc']:.4f} F1={v['f1']:.4f} P={v['precision']:.4f} "
              f"R={v['recall']:.4f} thr={v['threshold']:.2f} cm={v['confusion_matrix']}", flush=True)
    print(f"  macro_auc={mauc:.4f} macro_f1={mf1:.4f}", flush=True)
    np.savez(out / "test_predictions.npz", y_true=yt, y_prob=pt)
    json.dump({"per_label": per, "macro_auc": mauc, "macro_f1": mf1},
              open(out / "metrics_test.json", "w"), indent=2)
    print(f"Training time {(time.time()-t0)/3600:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
