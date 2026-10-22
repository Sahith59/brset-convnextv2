#!/usr/bin/env python
"""CPU unit tests for scripts/94_uda_align.py. Synthetic tensors and synthetic
JPEGs only: no credentialed patient data is touched, so this can run anywhere
a CPU allocation exists.

What each check would catch, in increasing order of severity:

  LOSS SEMANTICS. An alignment loss that is not zero for two identical batches
  is not measuring a discrepancy; one that is zero for two different batches is
  measuring nothing. Both failure modes still produce a descending loss curve
  and would be invisible in a training log.

  GRADIENT PATH. The whole point is that the encoder feels the target images.
  If the alignment loss reached only the classifier head, or reached nothing
  because a hook captured a detached tensor, the arm would be a slow
  reimplementation of the zero-shot baseline and would "fail to beat the noise
  floor" for a reason that has nothing to do with domain adaptation.

  BASELINE PRESERVATION. With the flags off, one epoch must leave the weights
  bitwise where the baseline leaves them. Anything else silently invalidates
  every comparison against the established numbers.

  LABEL LEAK. The target stream must be incapable of seeing a target label.
  This is checked structurally (which CSV columns are parsed, what __getitem__
  returns, which function signatures can accept a label) rather than by
  inspection, because a leak here would invalidate the result outright.
"""
import argparse
import importlib.util
import json
import inspect
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("U", str(HERE / "94_uda_align.py"))
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)
t30 = U.t30

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)


# --------------------------------------------------------------------------
class TinyNet(nn.Module):
    """Stand-in with the two structural features the taps rely on: a .stages
    sequence of 4D-output blocks and a get_classifier() linear layer."""

    def __init__(self, d=12, n_classes=2):
        super().__init__()
        self.stages = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GELU()),
            nn.Sequential(nn.Conv2d(8, d, 3, stride=2, padding=1), nn.GELU()),
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Module()
        self.head.fc = nn.Linear(d, n_classes)

    def get_classifier(self):
        return self.head.fc

    def forward_features(self, x):
        return self.stages(x)

    def forward(self, x):
        x = self.forward_features(x).mean(dim=(2, 3))
        return self.head.fc(self.norm(x))


def tiny_args(**kw):
    a = argparse.Namespace(
        model="tiny", data_path="", amp_dtype="fp32", resize_mode="squash",
        translator_ckpt="", translate_mode="prob", translate_p=1.0,
        target_pos_rate=-1.0, target_budget=0, budget_seed=0, warm_start_ckpt="",
        nb_classes=2, input_size=64, resize_size=72, batch_size=4, accum_iter=2,
        epochs=1, warmup_epochs=0, lr=1e-3, weight_decay=0.0, drop_path=0.0,
        mixup_alpha=0.0, label_smoothing=0.0, num_workers=0, ckpt_freq=0,
        no_oversample=True, loss="focal", focal_gamma=2.0, ema_decay=0.0,
        tta="none", thr_bootstrap=0, task="unit", output_dir="", seed=0,
        uda_target_path="", uda_loss="none", uda_weight=0.0, uda_taps="prelogits",
        uda_target_batch=0, uda_target_tf="train", uda_num_workers=0,
        uda_rampup_epochs=0, uda_feat_norm="none", uda_coral_mean=0.0,
        uda_scale_mode="absolute",
        uda_mmd_kernels=5, uda_mmd_mul=2.0, st_weight=0.0, st_conf=0.9,
        st_start_epoch=0, st_rampup_epochs=0, st_teacher="raw")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def make_split(root, n, labelled, seed=0):
    """A split directory of synthetic 96x96 JPEGs plus its labels.csv."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        name = f"s{seed}_{i:04d}.jpg"
        arr = rng.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / name, quality=92)
        rows.append({"file": name, "diabetic_retinopathy": int(i % 4 == 0),
                     "macular_edema": int(i % 7 == 0)})
    df = pd.DataFrame(rows)
    if not labelled:
        # still written WITH label columns, so the test proves the loader
        # declines to read columns that are present on disk
        pass
    df.to_csv(root / "labels.csv", index=False)
    return root


# --------------------------------------------------------------------------
def test_loss_semantics():
    print("\n1. alignment losses: zero on identical batches, positive on different ones")
    g = torch.Generator().manual_seed(0)
    a = torch.randn(16, 32, generator=g)
    b = a.clone()
    c = torch.randn(16, 32, generator=g) * 1.7 + 0.8      # different mean AND covariance

    c0 = U.coral_loss(a, b).item()
    c1 = U.coral_loss(a, c).item()
    check("coral == 0 for identical batches", c0 == 0.0, f"value {c0!r}")
    check("coral > 0 for different batches", c1 > 1e-6, f"value {c1:.6e}")

    m0 = U.mmd_rbf_loss(a, b).item()
    m1 = U.mmd_rbf_loss(a, c).item()
    check("mmd ~ 0 for identical batches (<1e-9)", abs(m0) < 1e-9, f"value {m0:.3e}")
    check("mmd > 0 for different batches", m1 > 1e-6, f"value {m1:.6e}")

    # a pure mean shift: published Deep CORAL is blind to it, MMD is not.
    d = a + 0.5
    cm = U.coral_loss(a, d).item()
    cmm = U.coral_loss(a, d, mean_weight=1.0).item()
    mm = U.mmd_rbf_loss(a, d).item()
    check("coral is blind to a pure mean shift (documented)", cm < 1e-12, f"value {cm:.3e}")
    check("coral_mean term sees the mean shift", cmm > 1e-6, f"value {cmm:.6e}")
    check("mmd sees the mean shift", mm > 1e-6, f"value {mm:.6e}")

    # scale sanity: a bigger discrepancy must give a bigger loss
    c_small = U.coral_loss(a, a * 1.05).item()
    c_big = U.coral_loss(a, a * 1.5).item()
    check("coral is monotone in the discrepancy", c_big > c_small,
          f"{c_small:.3e} -> {c_big:.3e}")


def test_gradients():
    print("\n2. gradients: the alignment loss reaches the encoder, not the classifier")
    torch.manual_seed(0)
    net = TinyNet()
    taps = U.AlignmentTaps(net, "prelogits")
    xs = torch.randn(8, 3, 64, 64)
    xt = torch.randn(8, 3, 64, 64) * 1.4 + 0.3
    for name, kind in (("coral", "coral"), ("mmd", "mmd")):
        net.zero_grad(set_to_none=True)
        taps.clear()
        net(xs); fs = taps.take()
        net(xt); ft = taps.take()
        loss = U.alignment_loss(fs, ft, tiny_args(uda_loss=kind))
        loss.backward()
        g_enc = net.stages[0][0].weight.grad
        g_head = net.head.fc.weight.grad
        enc_norm = float(g_enc.norm()) if g_enc is not None else 0.0
        check(f"{name}: encoder conv1 receives a finite non-zero gradient",
              g_enc is not None and torch.isfinite(g_enc).all() and enc_norm > 0,
              f"grad norm {enc_norm:.3e}")
        check(f"{name}: classifier weight receives NO gradient from alignment",
              g_head is None, f"grad is {None if g_head is None else float(g_head.norm())}")
    taps.remove()


def test_zero_weight_is_baseline():
    print("\n3. alignment weight 0 leaves the supervised path untouched")
    torch.manual_seed(0)
    net = TinyNet()
    taps = U.AlignmentTaps(net, "prelogits")
    xs = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, 2, (8, 2)).float()
    xt = torch.randn(8, 3, 64, 64) * 1.4 + 0.3
    crit = t30.MultiLabelFocalLoss(gamma=2.0)

    def grads(weight):
        net.zero_grad(set_to_none=True)
        taps.clear()
        logits = net(xs); fs = taps.take()
        sup = crit(logits, y)
        net(xt); ft = taps.take()
        align = U.alignment_loss(fs, ft, tiny_args(uda_loss="coral"))
        total = U.uda_step_loss(sup, align, weight)
        is_same_object = total is sup
        total.backward()
        return ([p.grad.clone() for p in net.parameters() if p.grad is not None],
                is_same_object, float(sup.item()), float(total.item()))

    g0, same0, sup0, tot0 = grads(0.0)
    g1, same1, sup1, tot1 = grads(1e4)
    check("weight 0 returns the supervised loss object itself", same0, "identity check")
    check("weight 0: total loss == supervised loss exactly", tot0 == sup0,
          f"{tot0!r} == {sup0!r}")
    base = []
    net.zero_grad(set_to_none=True)
    taps.clear()
    crit(net(xs), y).backward()
    base = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    allsame = len(base) == len(g0) and all(torch.equal(a, b) for a, b in zip(base, g0))
    check("weight 0: gradients bitwise identical to supervised-only backward", allsame,
          f"{len(base)} tensors compared")
    changed = any(not torch.equal(a, b) for a, b in zip(g0, g1))
    check("weight > 0: gradients do change", changed and tot1 != sup1,
          f"total {tot1:.6f} vs sup {sup1:.6f}")
    taps.remove()


def test_optimisable():
    print("\n4. the alignment losses are actually minimisable (10 Adam steps)")
    for kind in ("coral", "mmd"):
        torch.manual_seed(0)
        src = torch.randn(32, 16)
        tgt = torch.randn(32, 16) * 2.0 + 1.0
        W = torch.eye(16, requires_grad=True)
        opt = torch.optim.Adam([W], lr=0.05)
        first = last = None
        for step in range(10):
            opt.zero_grad()
            loss = U.alignment_loss([src @ W], [tgt], tiny_args(uda_loss=kind))
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss.item())
            last = float(loss.item())
        check(f"{kind}: loss decreases under optimisation", last < first,
              f"{first:.6e} -> {last:.6e}")


def test_unlabeled_loader(tmp):
    print("\n5. the target stream cannot see a target label")
    data = Path(tmp) / "fakedata"
    make_split(data / "train", 6, labelled=True, seed=1)
    make_split(data / "val", 3, labelled=True, seed=2)
    make_split(data / "test", 3, labelled=True, seed=3)
    tgt = make_split(Path(tmp) / "faketarget" / "train", 8, labelled=True, seed=4)

    tf, _ = t30.build_transforms(tiny_args())
    ds = U.UnlabeledTargetImages(tgt, tf, forbidden_files=U.eval_split_filenames(data))
    check("only the 'file' column is parsed from labels.csv",
          ds.columns_loaded == ["file"], f"columns_loaded={ds.columns_loaded}")
    on_disk = list(pd.read_csv(tgt / "labels.csv").columns)
    check("label columns ARE present on disk and were still not read",
          all(c in on_disk for c in t30.LABEL_COLS), f"on disk: {on_disk}")
    item = ds[0]
    check("__getitem__ returns a bare tensor, not (image, target)",
          torch.is_tensor(item) and item.shape == (3, 64, 64), f"type {type(item).__name__}")

    # no label value may survive anywhere on the object
    blob = repr(ds.__dict__)
    check("no label column name appears in the dataset's state",
          not any(c in blob for c in t30.LABEL_COLS), "scanned __dict__")

    # refuses the evaluation splits
    for split in ("val", "test"):
        try:
            U.UnlabeledTargetImages(data / split, tf)
            ok = False
        except SystemExit:
            ok = True
        check(f"refuses {split}/ as the unlabeled target set", ok)

    # refuses overlap with the evaluation file names
    try:
        U.UnlabeledTargetImages(tgt, tf,
                                forbidden_files=[ds.files[0]])
        ok = False
    except SystemExit:
        ok = True
    check("refuses a target set overlapping val/test file names", ok)

    # the loss and pseudo-label functions cannot accept a label
    for fn in (U.coral_loss, U.mmd_rbf_loss, U.pseudo_labels, U.alignment_loss):
        params = list(inspect.signature(fn).parameters)
        bad = [p for p in params if any(k in p.lower()
                                        for k in ("label", "target_y", "y_true", "gt"))]
        bad = [p for p in bad if not p.startswith("f_target")]
        check(f"{fn.__name__} has no label-like parameter", not bad, f"params {params}")
    return data, tgt


def test_pseudo_labels():
    print("\n6. pseudo-labels come from predictions and respect the threshold")
    logits = torch.tensor([[3.0, -3.0], [0.1, -0.1], [-2.5, 2.5]])
    y, mask = U.pseudo_labels(logits, 0.9)
    p = torch.sigmoid(logits)
    check("confident elements kept, uncertain ones dropped",
          mask.tolist() == [[True, True], [False, False], [True, True]],
          f"p={p.flatten().tolist()}")
    check("hard labels follow the sign of the logit",
          y.tolist() == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], f"y={y.tolist()}")
    y2, mask2 = U.pseudo_labels(logits, 0.99)
    check("a stricter threshold keeps fewer elements",
          int(mask2.sum()) < int(mask.sum()), f"{int(mask2.sum())} < {int(mask.sum())}")


def test_self_training_loss_dtypes():
    print("\n6b. the self-training loss survives the bf16/fp32 mix it will see on the GPU")
    crit = t30.MultiLabelFocalLoss(gamma=2.0)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        logits = (torch.randn(8, 2) * 3).to(dtype).requires_grad_(True)
        y, keep = U.pseudo_labels(logits, 0.8)
        loss = U.self_training_loss(crit, logits, y, keep)
        if loss is None:
            check(f"{dtype}: pseudo-label mask was empty", False, "no confident element")
            continue
        loss.backward()
        ok = torch.isfinite(loss) and logits.grad is not None and torch.isfinite(logits.grad).all()
        check(f"{dtype} logits + fp32 pseudo-labels: loss and gradient finite", bool(ok),
              f"loss={float(loss):.6f} kept={int(keep.sum())}/{keep.numel()}")
    empty = torch.zeros(4, 2, dtype=torch.bool)
    check("an empty mask yields no loss rather than a NaN",
          U.self_training_loss(crit, torch.randn(4, 2), torch.zeros(4, 2), empty) is None)


def test_taps_on_real_backbone():
    print("\n7. the feature tap does not change what the real backbone computes")
    try:
        import timm
        torch.manual_seed(0)
        model = timm.create_model("convnextv2_atto", pretrained=False, num_classes=2)
    except Exception as e:                                     # offline / unavailable
        check("timm ConvNeXt V2 tap test", False, f"SKIPPED: {e}")
        return
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        before = model(x).clone()
    taps = U.AlignmentTaps(model, "prelogits,stage2,stage3")
    with torch.no_grad():
        after = model(x).clone()
    feats = taps.take()
    taps.remove()
    check("logits bitwise identical with the taps registered",
          torch.equal(before, after), f"max abs diff {float((before-after).abs().max()):.3e}")
    check("prelogits tap has the classifier's input width",
          feats[0].shape[1] == model.get_classifier().in_features,
          f"{tuple(feats[0].shape)} vs in_features={model.get_classifier().in_features}")
    check("stage taps return (B, C) pooled features",
          all(f.dim() == 2 for f in feats), f"shapes {[tuple(f.shape) for f in feats]}")
    with torch.no_grad():
        again = model(x).clone()
    check("logits still identical after the taps are removed", torch.equal(before, again))


def _one_epoch(fn, args, src_dir, seed=0, net=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyNet() if net is None else net
    tf, _ = t30.build_transforms(args)
    ds = t30.BRSETMultiLabel(src_dir, tf)
    weights, _ = t30.build_sample_weights(ds.labels_df)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                                         num_workers=0, drop_last=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    crit = t30.MultiLabelFocalLoss(gamma=2.0)
    loss = fn(model, None, loader, opt, sched, scaler, crit, torch.device("cpu"), 0, args,
              print_freq=10**9)
    return model, loss


def test_off_path_is_the_baseline(tmp, src_dir):
    print("\n8. with the flags off, the baseline epoch loop runs unchanged")
    seen = {}

    def sentinel(*a, **kw):
        seen["args"] = a
        seen["kwargs"] = kw
        return 1.234

    orig = U._ORIG_TRAIN_ONE_EPOCH
    try:
        U._ORIG_TRAIN_ONE_EPOCH = sentinel
        args = tiny_args(output_dir=str(tmp))
        out = U.train_one_epoch("M", "E", "L", "O", "S", "SC", "C", "D", 7, args,
                                print_freq=5, translator=None)
        check("flags off -> delegates to t30.train_one_epoch", out == 1.234)
        check("delegation forwards every positional argument unchanged",
              seen["args"] == ("M", "E", "L", "O", "S", "SC", "C", "D", 7, args),
              f"{len(seen['args'])} positionals")
        check("delegation forwards print_freq and translator",
              seen["kwargs"] == {"print_freq": 5, "translator": None},
              f"{seen['kwargs']}")
        check("uda_enabled() is False for default flags", not U.uda_enabled(args))
        check("uda_enabled() is True once a weight is set",
              U.uda_enabled(tiny_args(uda_loss="coral", uda_weight=1.0))
              and U.uda_enabled(tiny_args(st_weight=0.1)))
        check("uda_enabled() stays False if a loss is named but the weight is 0",
              not U.uda_enabled(tiny_args(uda_loss="coral", uda_weight=0.0)))
    finally:
        U._ORIG_TRAIN_ONE_EPOCH = orig

    # end-to-end: one real epoch through each entry point, same seed
    args = tiny_args(output_dir=str(tmp))
    m_base, l_base = _one_epoch(t30.train_one_epoch, args, src_dir, seed=0)
    U.reset_state()
    m_uda, l_uda = _one_epoch(U.train_one_epoch, args, src_dir, seed=0)
    same = all(torch.equal(a, b) for a, b in zip(m_base.state_dict().values(),
                                                 m_uda.state_dict().values()))
    check("one epoch, flags off: weights bitwise identical to the baseline's", same,
          f"train_loss {l_base!r} vs {l_uda!r}")
    check("one epoch, flags off: returned train_loss identical", l_base == l_uda)


def test_end_to_end_on(tmp, data_dir, tgt_dir):
    print("\n9. with the flags on, a real epoch trains on unlabeled target images")
    for kind, weight in (("coral", 1e4), ("mmd", 1.0)):
        U.reset_state()
        args = tiny_args(output_dir=str(tmp), task=f"unit_{kind}", data_path=str(data_dir),
                         uda_target_path=str(tgt_dir), uda_loss=kind, uda_weight=weight,
                         uda_target_batch=4, uda_num_workers=0)
        base_model, _ = _one_epoch(t30.train_one_epoch, tiny_args(output_dir=str(tmp)),
                                   data_dir / "train", seed=0)
        uda_model, loss = _one_epoch(U.train_one_epoch, args, data_dir / "train", seed=0)
        finite = np.isfinite(loss)
        diff = max(float((a - b).abs().max())
                   for a, b in zip(base_model.state_dict().values(),
                                   uda_model.state_dict().values())
                   if a.dtype.is_floating_point)
        check(f"{kind}: epoch completes with a finite loss", finite, f"train_loss {loss:.6f}")
        check(f"{kind}: weights end up elsewhere than a baseline epoch", diff > 0,
              f"max |delta| vs baseline {diff:.3e} (RNG also differs; smoke check only)")
        log = Path(tmp) / f"unit_{kind}" / "uda_log.jsonl"
        check(f"{kind}: per-epoch diagnostics written", log.exists(),
              log.read_text().strip()[:200] if log.exists() else "missing")
        if log.exists():
            d = json.loads(log.read_text().strip().split("\n")[-1])
            check(f"{kind}: a non-zero alignment loss was really computed",
                  d["mean_align"] > 0 and np.isfinite(d["mean_align"]),
                  f"mean_align={d['mean_align']:.6e} lambda*align={d['mean_align_times_lambda']:.6e} "
                  f"mean_sup={d['mean_sup']:.6f}")
    # self-training path
    U.reset_state()
    args = tiny_args(output_dir=str(tmp), task="unit_st", data_path=str(data_dir),
                     uda_target_path=str(tgt_dir), uda_loss="none", uda_weight=0.0,
                     st_weight=1.0, st_conf=0.5, st_start_epoch=0, st_rampup_epochs=0,
                     st_teacher="raw", uda_target_batch=4, uda_num_workers=0)
    _, loss = _one_epoch(U.train_one_epoch, args, data_dir / "train", seed=0)
    check("self-training path completes with a finite loss", np.isfinite(loss),
          f"train_loss {loss:.6f}")
    U.reset_state()


def test_scale_calibration(tmp, data_dir, tgt_dir):
    print("\n12. relative scale mode makes lambda a fraction of the supervised loss")
    U.reset_state()
    args = tiny_args(output_dir=str(tmp), task="unit_rel", data_path=str(data_dir),
                     uda_target_path=str(tgt_dir), uda_loss="coral", uda_weight=0.05,
                     uda_scale_mode="relative", uda_target_batch=4, uda_num_workers=0)
    _one_epoch(U.train_one_epoch, args, data_dir / "train", seed=0)
    d = json.loads((Path(tmp) / "unit_rel" / "uda_log.jsonl").read_text().strip().split("\n")[-1])
    check("calibration measured a finite positive scale",
          d["scale"] > 0 and np.isfinite(d["scale"]), f"scale={d['scale']:.6e}")
    want = 0.05 * d["mean_sup"]
    got = d["mean_align_times_lambda"]
    check("the alignment term starts at exactly --uda_weight of the supervised loss",
          abs(got - want) <= 1e-9 + 1e-6 * abs(want),
          f"lambda*align={got:.8f} vs 0.05*sup={want:.8f}")
    scale_first = d["scale"]
    U.reset_state()
    args2 = tiny_args(output_dir=str(tmp), task="unit_abs", data_path=str(data_dir),
                      uda_target_path=str(tgt_dir), uda_loss="coral", uda_weight=0.05,
                      uda_scale_mode="absolute", uda_target_batch=4, uda_num_workers=0)
    _one_epoch(U.train_one_epoch, args2, data_dir / "train", seed=0)
    d2 = json.loads((Path(tmp) / "unit_abs" / "uda_log.jsonl").read_text().strip().split("\n")[-1])
    check("absolute mode leaves the weight literal", d2["scale"] == 1.0
          and abs(d2["mean_align_times_lambda"] - 0.05 * d2["mean_align"]) < 1e-12,
          f"scale={d2['scale']} lambda*align={d2['mean_align_times_lambda']:.3e}")
    check("the two modes really differ in effective weight",
          abs(scale_first - 1.0) > 1e-6, f"relative scale was {scale_first:.6e}")
    U.reset_state()


def test_schedules():
    print("\n10. weight schedules")
    a = tiny_args(uda_weight=10.0, uda_rampup_epochs=5)
    vals = [U.align_weight_at(a, e) for e in range(7)]
    check("alignment weight ramps linearly then holds",
          vals == [2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0], f"{vals}")
    b = tiny_args(st_weight=1.0, st_start_epoch=3, st_rampup_epochs=2)
    vals = [U.st_weight_at(b, e) for e in range(7)]
    check("self-training weight is 0 before the start epoch then ramps",
          vals == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], f"{vals}")


def test_arg_passthrough():
    print("\n11. argument plumbing into the baseline parser")
    argv = sys.argv
    try:
        sys.argv = ["94_uda_align.py", "--uda_loss", "mmd", "--uda_weight", "2.5",
                    "--target_budget", "-1", "--batch_size", "8", "--lr", "3e-5",
                    "--no_oversample", "--task", "probe"]
        args = U.get_args()
        check("uda flags parsed", args.uda_loss == "mmd" and args.uda_weight == 2.5)
        check("negative baseline values survive the split",
              args.target_budget == -1, f"target_budget={args.target_budget}")
        check("baseline flags reach the baseline parser",
              args.batch_size == 8 and args.lr == 3e-5 and args.no_oversample
              and args.task == "probe")
        check("baseline defaults untouched",
              args.model == "convnextv2_large.fcmae_ft_in22k_in1k_384"
              and args.input_size == 512 and args.epochs == 40)
    finally:
        sys.argv = argv


def main():
    print("=" * 78)
    print("unit tests for scripts/94_uda_align.py  (CPU, synthetic data only)")
    print("=" * 78)
    tmp = tempfile.mkdtemp(prefix="uda_unit_")
    try:
        test_loss_semantics()
        test_gradients()
        test_zero_weight_is_baseline()
        test_optimisable()
        data_dir, tgt_dir = test_unlabeled_loader(tmp)
        test_pseudo_labels()
        test_self_training_loss_dtypes()
        test_taps_on_real_backbone()
        test_off_path_is_the_baseline(tmp, data_dir / "train")
        test_end_to_end_on(tmp, data_dir, tgt_dir)
        test_scale_calibration(tmp, data_dir, tgt_dir)
        test_schedules()
        test_arg_passthrough()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAILED: {name}  {detail}")
    print("=" * 78)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
