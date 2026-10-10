"""Apply a trained BRSET -> mBRSET translator to training batches, on the fly.

WHY ON THE FLY RATHER THAN A SAVED CORPUS. Writing 11,372 translated images to
disk and reloading them loses exactly the component the translator works
hardest to match. Roughly the whole of the per-channel Laplacian-variance match
is carried by sensor noise at amplitudes of a few grey levels, and JPEG
quantises precisely those high-frequency DCT coefficients first. The trainer
also resizes, so any pixel-pitch noise written at one scale is attenuated at
another. Translating inside the training loop, at the pitch the network
actually consumes, removes both losses.

It also fixes a second defect for free. A single fitted parameter vector maps
every source image onto one point in appearance space, so a saved corpus is
under-dispersed relative to real mBRSET, whose augmentation value comes from
its spread. The acquisition model's stochastic terms (illumination centre,
artifact placement, the noise draw) are redrawn on every call here, so each
epoch sees a different sample.

ORDER OF OPERATIONS. Geometric and colour augmentation stay in the PIL stage,
before translation, so the source image is varied and THEN passed through a
fixed camera model. Normalisation moves after translation because the
translator operates on [0,1] images. Everything else about the training
pipeline is byte-identical to the baseline arms, which is what makes the
comparison valid.
"""
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).parent


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(mod, str(_HERE / name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_translator(ckpt_path, device="cuda"):
    """Rebuild the frozen two-stage translator from a script-83 checkpoint.

    The architecture is reconstructed from the hyperparameters stored in the
    checkpoint's `args`, not from defaults, so a checkpoint always restores the
    model that produced it.
    """
    P = _load("72_physical_layer.py", "P72")
    R = _load("82_refiner.py", "R82")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]

    ma = a.get("max_artifact", -1.0)
    phys = P.PhysicalAcquisition(
        learnable=False,
        max_artifact=(ma if ma is not None and ma >= 0 else None))
    with torch.no_grad():
        phys.raw.copy_(ck["phys_raw"])

    ref = R.ResidualRefiner(
        lf_stride=a["lf_stride"], lf_width=a["lf_width"], lf_depth=a["lf_depth"],
        hf_width=a["hf_width"], eps_lf=a["eps_lf"], eps_hf=a["eps_hf"],
        hf_mode=a["hf_mode"], mask_erode=a["mask_erode"], mask_taper=a["mask_taper"])
    ref.load_state_dict(ck["refiner"])

    t = R.Translator(phys=phys, refiner=ref).to(device).eval()
    for q in t.parameters():
        q.requires_grad_(False)

    b = ref.max_adjacent_step()
    print(f"translator restored from {ckpt_path} (step {ck.get('step')}); "
          f"max_artifact={ma}, eps_lf={a['eps_lf']}, eps_hf={a['eps_hf']}, "
          f"hf_mode={a['hf_mode']}", flush=True)
    print(f"  worst-case adjacent-pixel edit: deterministic "
          f"{b['deterministic_total']:.4f}, stochastic {b['hf_stochastic']:.4f}, "
          f"lesion contrast floor {R.LESION_CONTRAST_FLOOR}", flush=True)
    if b["deterministic_total"] >= R.LESION_CONTRAST_FLOOR:
        print("  [WARN] deterministic worst case reaches the lesion floor", flush=True)
    return t


@torch.no_grad()
def translate_batch(translator, x):
    """x: (B,3,H,W) in [0,1]. Returns the translated batch, clamped to [0,1].

    Stochastic terms are redrawn per call, so the same source image becomes a
    different sample on every epoch.
    """
    return translator(x).clamp_(0.0, 1.0)
