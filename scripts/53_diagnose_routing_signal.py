"""Diagnose why conditional-entropy routing never moved off alpha = 0.5.

Experiment 2's condent variant held alpha at 0.501 for every epoch, meaning the
routing weight never discriminated between the shared and private branches. Two
explanations are possible and they have opposite consequences:

  (a) the signal is genuinely uninformative, H(Y|shared) and H(Y|private) are
      equal, and the routing idea does not work on this problem. A finding.

  (b) the difference is real but small relative to the temperature of 1.0, so
      sigmoid squashed it to 0.5. An implementation bug.

Rather than sweeping the temperature blindly at 7 GPU-hours per setting, this
measures the actual distribution of H(Y|private) - H(Y|shared) on the
validation set from the trained checkpoint, then reports what alpha would look
like at each candidate temperature. That distinguishes (a) from (b) directly.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_spec = importlib.util.spec_from_file_location(
    "routed", str(Path(__file__).parent / "52_entropy_routed.py"))
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)

CKPT = Path("/home/users/sthummala2/brset-convnextv2/results/routed_condent/checkpoint-best.pth")


class _A:
    pass


def main():
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = R.EntropyRoutedNet(a["model"], 2, a["proj_dim"], 2, 0.0,
                             route="condent", temp=a["route_temp"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"loaded epoch {ck['epoch']} ({ck['variant']}), route_temp={a['route_temp']}", flush=True)

    cfg = _A(); cfg.resize_size, cfg.input_size = a["resize_size"], a["input_size"]
    _, ev = R.t30.build_transforms(cfg)
    ds = R.JointDataset(Path(a["data_path"]) / "val", ev)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    print(f"validation set: {len(ds)} images", flush=True)

    bce = nn.functional.binary_cross_entropy_with_logits
    Hs, Hp = [], []
    with torch.no_grad():
        for k, (x, y, dom) in enumerate(dl):
            x = x.to(dev, non_blocking=True); y = y.to(dev); dom = dom.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = net.encoder(x)
                h_s = net.shared(f)
                hp = torch.stack([p(f) for p in net.private], dim=1)
                h_p = hp[torch.arange(f.size(0), device=dev), dom]
                aux_s = net.aux_shared(h_s).float()
                aux_p = net.aux_private(h_p).float()
            Hs.append(bce(aux_s, y, reduction="none").mean(1).cpu().numpy())
            Hp.append(bce(aux_p, y, reduction="none").mean(1).cpu().numpy())
            if k % 20 == 0:
                print(f"  batch {k}/{len(dl)}", flush=True)

    Hs = np.concatenate(Hs); Hp = np.concatenate(Hp); d = Hp - Hs
    print(f"\nn = {len(d)}")
    print(f"  H(Y|shared)   mean={Hs.mean():.6f}  std={Hs.std():.6f}")
    print(f"  H(Y|private)  mean={Hp.mean():.6f}  std={Hp.std():.6f}")
    print(f"  difference    mean={d.mean():+.6f}  std={d.std():.6f}")
    print(f"                p5={np.percentile(d,5):+.6f}  p50={np.percentile(d,50):+.6f}  "
          f"p95={np.percentile(d,95):+.6f}")

    print("\nalpha = sigmoid(difference / T), what each temperature would give:")
    print(f"  {'T':>8} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'frac outside [0.4,0.6]':>24}")
    for T in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
        al = 1.0 / (1.0 + np.exp(-d / T))
        frac = float(np.mean((al < 0.4) | (al > 0.6)))
        print(f"  {T:>8} {al.mean():8.4f} {al.std():8.4f} {al.min():8.4f} {al.max():8.4f} {frac:>24.3f}")

    np.savez("/home/users/sthummala2/brset-convnextv2/results/routing_signal_diagnosis.npz",
             H_shared=Hs, H_private=Hp)
    print("\nVERDICT")
    if d.std() < 1e-4:
        print("  The two branches are equally informative to within 1e-4. The signal is")
        print("  uninformative and no temperature rescues it. This is a finding, not a bug.")
    else:
        best = min((abs(0.5 - (1/(1+np.exp(-d/T))).std()), T) for T in (0.3, 0.1, 0.03, 0.01, 0.003))
        print(f"  There is real spread (std {d.std():.6f}). Temperature 1.0 was squashing it.")
        print(f"  A rerun at a smaller temperature is warranted.")


if __name__ == "__main__":
    main()
