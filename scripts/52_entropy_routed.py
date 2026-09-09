"""Experiment 2: entropy-gated routing between shared and device-specific features.

The proposal is to aggregate both datasets and let the model decide, per image,
how much to draw on each source: "when you detect the disease in the mBRSET,
mostly it should be based on the mBRSET, but part of the BRSET data may be
helpful." Two constraints were specified: the routing lives in the ENCODER, not
the decision layer, and the routing signal is conditional entropy of the labels
computed during training.

Built on the zero-adversarial-weight variant of experiment 1, not the published
defaults, because the sweep showed those defaults actively hurt here
(DR F1 0.8053 at weight 0.25 against 0.8232 at zero).

Architecture: a shared branch and a per-domain private branch as before, but
instead of always concatenating both, a gate produces a per-image weight alpha
and the classifier sees a convex combination

    h = alpha * h_shared + (1 - alpha) * h_private

Two routing signals are implemented, because the intended reading of
"conditional entropy" is ambiguous and guessing would be worse than testing:

  --route gate      alpha comes from a learned gate on the encoder features,
                    regularised by the entropy of its own output distribution.
                    This is the standard Mixture-of-Experts formulation.

  --route condent   alpha is driven by MEASURED conditional entropy. Auxiliary
                    heads on each branch give H(Y | h_shared) and
                    H(Y | h_private) per sample, estimated by their own
                    cross-entropy. The gate routes toward whichever branch
                    leaves less uncertainty about the label:

                        alpha = sigmoid( (H_private - H_shared) / T )

                    so a shared branch that explains the label better pulls
                    alpha up. This is the closer reading of the wording.

Baseline to beat: plain aggregated training, DR F1 0.8250 with 27 of 159 missed.
Experiment 1 at its best setting reached 0.8232 with 31 missed, so the gate has
to carry the contribution rather than improve a structure that already helps.
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


class EntropyRoutedNet(nn.Module):
    """Shared and private branches combined by a per-image gate."""

    def __init__(self, backbone, n_dom=2, proj=512, n_cls=2, drop_path=0.3,
                 route="gate", temp=1.0):
        super().__init__()
        self.route, self.temp = route, temp
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0,
                                          drop_path_rate=drop_path)
        d = self.encoder.num_features
        self.shared = nn.Sequential(nn.Linear(d, proj), nn.GELU(), nn.LayerNorm(proj))
        self.private = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, proj), nn.GELU(), nn.LayerNorm(proj)) for _ in range(n_dom)])
        # the gate sits on encoder features, so routing happens in the encoder
        self.gate = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))
        # auxiliary heads estimate how much each branch explains the label
        self.aux_shared = nn.Linear(proj, n_cls)
        self.aux_private = nn.Linear(proj, n_cls)
        self.classifier = nn.Linear(proj, n_cls)
        self.domain_head = nn.Sequential(nn.Linear(proj, 256), nn.GELU(), nn.Linear(256, n_dom))

    def forward(self, x, domain, targets=None, grl_lambda=1.0):
        f = self.encoder(x)
        h_s = self.shared(f)
        h_p = torch.stack([p(f) for p in self.private], dim=1)
        h_p = h_p[torch.arange(f.size(0), device=f.device), domain]

        aux_s = self.aux_shared(h_s)
        aux_p = self.aux_private(h_p)

        # the gate is always computed. In condent mode it is trained to predict
        # the entropy-derived weight, so that at inference, where labels are not
        # available, routing still works.
        alpha_gate = torch.sigmoid(self.gate(f))

        alpha_target = None
        if self.route == "condent" and targets is not None:
            # per-sample conditional entropy proxies, detached so the gate is
            # driven by the measurement rather than by gradients through it
            with torch.no_grad():
                bce = nn.functional.binary_cross_entropy_with_logits
                H_s = bce(aux_s, targets, reduction="none").mean(1)
                H_p = bce(aux_p, targets, reduction="none").mean(1)
                alpha_target = torch.sigmoid((H_p - H_s) / self.temp).unsqueeze(1)

        alpha = alpha_target if alpha_target is not None else alpha_gate

        h = alpha * h_s + (1.0 - alpha) * h_p
        logits = self.classifier(h)
        dom_logits = self.domain_head(GradReverse.apply(h_s, grl_lambda))
        return logits, h_s, h_p, dom_logits, alpha, alpha_gate, alpha_target, aux_s, aux_p


def gate_entropy(alpha, eps=1e-6):
    """Entropy of the Bernoulli routing decision, averaged over the batch."""
    a = alpha.clamp(eps, 1.0 - eps)
    return (-(a * a.log() + (1 - a) * (1 - a).log())).mean()


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
            pr = torch.stack([torch.sigmoid(model(v, dom, None)[0].float()) for v in views]).mean(0)
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
    p.add_argument("--route", default="gate", choices=["gate", "condent"],
                    help="Routing signal. 'gate' is a learned gate regularised by the entropy of its "
                         "own output. 'condent' drives the weight from measured conditional entropy of "
                         "the label given each branch, and trains the gate to imitate it for inference.")
    p.add_argument("--lambda_ent", default=0.01, type=float,
                    help="Weight on the gate-entropy term. Positive values encourage decisive routing.")
    p.add_argument("--lambda_aux", default=0.1, type=float,
                    help="Weight on the auxiliary per-branch heads that estimate conditional entropy.")
    p.add_argument("--lambda_mimic", default=1.0, type=float,
                    help="Weight on training the gate to reproduce the entropy-derived weight, so "
                         "condent routing still works at inference where labels are absent.")
    p.add_argument("--route_temp", default=1.0, type=float)
    p.add_argument("--select_metric", default="score", choices=["score", "f1"],
                    help="Which validation quantity picks the best checkpoint. 'score' is the "
                         "project default, 0.5*(macro AUC + macro F1). 'f1' uses macro F1 alone. "
                         "Experiment 1 improved AUC while losing F1 and recall, so the default may "
                         "have selected for the wrong thing on this architecture.")
    p.add_argument("--task", default="routed_joint")
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

    model = EntropyRoutedNet(args.model, 2, args.proj_dim, len(LABEL_COLS), args.drop_path,
                              route=args.route, temp=args.route_temp).to(device)
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
        run = {"task": 0.0, "diff": 0.0, "sim": 0.0, "aux": 0.0, "gate": 0.0, "alpha": 0.0}
        nb = 0; n_bad = 0
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
                logits, h_s, h_p, dlog, alpha, alpha_gate, alpha_tgt, aux_s, aux_p = model(x, dom, y, grl)
                l_task = task_loss(logits, y)
                l_diff = difference_loss(h_s.float(), h_p.float())
                l_sim = dom_loss(dlog, dom)
                # auxiliary heads must be trained, or the conditional-entropy
                # estimates they provide are meaningless
                l_aux = task_loss(aux_s, y) + task_loss(aux_p, y)
                if args.route == "condent":
                    # entropy already sets the weight; train the gate to copy it
                    l_gate = nn.functional.mse_loss(alpha_gate, alpha_tgt.detach())
                else:
                    # standard Mixture-of-Experts: encourage decisive routing
                    l_gate = gate_entropy(alpha_gate.float())
                lam_gate = args.lambda_mimic if args.route == "condent" else args.lambda_ent
                loss = (l_task + args.beta_diff * l_diff + args.gamma_sim * l_sim
                        + args.lambda_aux * l_aux + lam_gate * l_gate) / args.accum_iter
            if not torch.isfinite(loss):
                n_bad += 1; opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward()
            if (i + 1) % args.accum_iter == 0 or (i + 1) == len(dl_tr):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema is not None: ema.update(model)
            run["task"] += l_task.item(); run["diff"] += l_diff.item(); run["sim"] += l_sim.item()
            run["aux"] += l_aux.item(); run["gate"] += l_gate.item()
            run["alpha"] += alpha.float().mean().item(); nb += 1
            if i % 40 == 0:
                print(f"Epoch: [{epoch}]  [{i}/{len(dl_tr)}]  lr: {opt.param_groups[0]['lr']:.7f}  "
                      f"task: {l_task.item():.4f}  aux: {l_aux.item():.4f}  gate: {l_gate.item():.4f}  "
                      f"alpha: {alpha.float().mean().item():.3f}  {time.time()-t0:.0f}s", flush=True)
        sched.step()
        if n_bad: print(f"WARNING epoch {epoch}: skipped {n_bad} non-finite batches", flush=True)

        for variant, net in (("raw", model), ("ema", ema.module if ema else None)):
            if net is None: continue
            yv, pv = infer(net, dl_va, device, args.amp_dtype, args.tta)
            thr = t30.tune_thresholds(yv, pv, n_boot=0, seed=args.seed)
            per, mauc, mf1 = t30.compute_per_label_metrics(yv, pv, thr)
            score = mf1 if args.select_metric == "f1" else 0.5 * (mauc + mf1)
            print(f"val[{variant}]: macro_auc={mauc:.4f} macro_f1={mf1:.4f} "
                  f"score={score:.4f} (select_metric={args.select_metric})", flush=True)
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
