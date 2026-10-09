"""Unit tests for the 19-parameter acquisition model. Synthetic tensors only.

Why this exists. The three defects fixed on 9 Oct were all of the kind that a
loss curve cannot reveal: a target measured at the wrong scale still descends, a
noise parameter that cannot reach the blue channel still descends, and a
redundant statistic still descends. The only way to know the model can express
what it claims is to interrogate the operator directly.

Four checks, in increasing order of what they would catch:

  GRADIENTS. Every one of the 19 parameters must receive a live finite gradient
  from the fit objective. A parameter with a zero gradient is not a parameter,
  it is a constant that the report will nevertheless quote as "fitted".

  CHANNEL ASYMMETRY. The motivating measurement is that the high-frequency
  excess is channel-asymmetric (lapvar std ratio red 1.02x, blue 2.85x). Setting
  blue noise high and red noise to zero must raise b_lapvar and leave r_lapvar
  alone. If it does not, the per-channel parameterisation is cosmetic.

  REDUNDANCY. brightness is x.mean over channels, so it must equal the mean of
  the three channel means to floating-point precision, and the three fractions
  must sum to one. Both are the identities that justify the FIT_KEYS change, so
  they are asserted rather than assumed.

  BACKWARD COMPATIBILITY. An init dict written for the 15-parameter model, with
  scalar noise_a and noise_b, must still load and must broadcast to all three
  channels.

Uses synthetic discs, never patient images, so it runs on a login node without
touching credentialed data or the GPU allocation.
"""
import importlib.util, time
import numpy as np
import torch
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("P", str(HERE / "72_physical_layer.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
spec79 = importlib.util.spec_from_file_location("F", str(HERE / "79_fit_physics_v2.py"))
F = importlib.util.module_from_spec(spec79); spec79.loader.exec_module(F)


def synth(n, size, rgb=(0.70, 0.36, 0.13), noise=0.01, seed=0):
    """Synthetic fundus-like discs. Not a fundus image, just the right support:
    a circular field of view on black, non-uniform interior so the statistics
    are not degenerate, and a plausible R:G:B ratio."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    r = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2).sqrt()
    disc = (r < 0.47 * size).float()
    base = torch.stack([disc * c for c in rgb]).unsqueeze(0).repeat(n, 1, 1, 1)
    # smooth anatomy-like structure so contrast and falloff are non-zero
    struct = torch.rand(n, 1, size // 16, size // 16, generator=g)
    struct = torch.nn.functional.interpolate(struct, size=(size, size), mode="bilinear")
    base = base * (0.75 + 0.5 * struct)
    base = base + noise * torch.randn(base.shape, generator=g) * disc
    return base.clamp(0.0, 1.0)


def fake_targets(x):
    """Build a target dict in the on-disk format from a synthetic batch, so the
    real distance() from script 79 is exercised rather than a stand-in."""
    s = P.image_stats_torch(x)
    t = {}
    for k, v in s.items():
        t[k] = {"mbrset_mean": float(v.mean()),
                "mbrset_std": max(float(v.std(unbiased=False)), 1e-6),
                "brset_mean": float(v.mean()), "brset_std": 1.0}
    t["_meta"] = {"presize": 560, "crop": 512}
    return t


def test_gradients(size=128, n=8):
    print("=" * 72)
    print("TEST 1  every parameter receives a live finite gradient")
    x = synth(n, size, seed=1)
    # a target that differs from the source on every axis: bluer, noisier,
    # dimmer, so no statistic is already satisfied at init
    tgt = fake_targets(synth(n, size, rgb=(0.55, 0.30, 0.22), noise=0.05, seed=2))
    L = P.PhysicalAcquisition()
    loss = F.distance(P.image_stats_torch(L(x)), tgt)
    loss.backward()
    gr = L.raw.grad.detach().clone()
    print(f"loss {float(loss):.4f}")
    print(f"{'param':16s} {'raw':>10s} {'value':>12s} {'d loss/d raw':>15s}  status")
    dead, nonfinite = [], []
    pv = L.params()
    for i, k in enumerate(L.KEYS):
        g = float(gr[i])
        if not np.isfinite(g):
            st = "NON-FINITE"; nonfinite.append(k)
        elif abs(g) < 1e-12:
            st = "DEAD"; dead.append(k)
        else:
            st = "live"
        print(f"{k:16s} {float(L.raw[i]):+10.4f} {float(pv[k]):+12.6f} {g:+15.6e}  {st}")
    print(f"\n{len(L.KEYS) - len(dead) - len(nonfinite)} of {len(L.KEYS)} live, "
          f"dead {dead or 'none'}, non-finite {nonfinite or 'none'}")
    assert not nonfinite and not dead
    return gr


def test_channel_asymmetry(size=256, n=6):
    print("=" * 72)
    print("TEST 2  the model can produce a BLUE-ONLY noise excess")
    x = synth(n, size, noise=0.0, seed=3)
    # isolate the sensor noise: every other operator set to identity, so any
    # change in lapvar is attributable to the noise parameters alone
    ident = {"sharpen_k": 0.0, "light_strength": 0.0, "halo": 0.0,
             "spot_amp": 0.0, "hole_amp": 0.0,
             "gain_r": 1.0, "gain_g": 1.0, "gain_b": 1.0,
             "off_r": 0.0, "off_g": 0.0, "off_b": 0.0,
             "contrast_gain": 1.0, "saturation_gain": 1.0}
    off = dict(ident, noise_a=0.0, noise_b=0.0)
    blue = dict(ident, noise_a_r=0.0, noise_b_r=0.0, noise_a_g=0.0, noise_b_g=0.0,
                noise_a_b=2e-3, noise_b_b=0.03)
    torch.manual_seed(0)
    s_off = P.image_stats_torch(P.PhysicalAcquisition(init=off)(x))
    torch.manual_seed(0)
    s_blue = P.image_stats_torch(P.PhysicalAcquisition(init=blue)(x))
    print(f"{'channel':8s} {'lapvar noise off':>18s} {'lapvar blue noise':>18s} {'ratio':>10s}")
    ratios = {}
    for c in "rgb":
        a = float(s_off[f"{c}_lapvar"].mean()); b = float(s_blue[f"{c}_lapvar"].mean())
        ratios[c] = b / max(a, 1e-12)
        print(f"{c:8s} {a:18.8f} {b:18.8f} {ratios[c]:10.3f}")
    print(f"gray sharpness {float(s_off['sharpness'].mean()):.8f} -> "
          f"{float(s_blue['sharpness'].mean()):.8f}")
    print(f"\nb_lapvar ratio {ratios['b']:.2f}x, r_lapvar ratio {ratios['r']:.4f}x, "
          f"g_lapvar ratio {ratios['g']:.4f}x")
    assert ratios["b"] > 10.0, "blue noise did not raise b_lapvar"
    assert abs(ratios["r"] - 1.0) < 0.02, "red lapvar moved although red noise is zero"
    assert abs(ratios["g"] - 1.0) < 0.02, "green lapvar moved although green noise is zero"

    # the same statement in gradient form: d(b_lapvar)/d(blue noise) must be the
    # only large entry, otherwise the asymmetry above could be an artifact of
    # the particular noise draw
    L = P.PhysicalAcquisition(init=dict(ident, noise_a=1e-4, noise_b=0.004))
    torch.manual_seed(0)
    st = P.image_stats_torch(L(x))
    st["b_lapvar"].mean().backward()
    print("\nd(b_lapvar)/d(raw) for the six noise parameters:")
    for k in L.NOISE_KEYS:
        print(f"  {k:12s} {float(L.raw.grad[L.KEYS.index(k)]):+.6e}")
    gb = abs(float(L.raw.grad[L.KEYS.index("noise_b_b")]))
    gr = abs(float(L.raw.grad[L.KEYS.index("noise_b_r")]))
    print(f"  blue/red magnitude ratio {gb / max(gr, 1e-30):.3g}")
    assert gb > 1e3 * max(gr, 1e-30), "b_lapvar is not specifically driven by blue noise"
    return ratios


def test_redundancy(size=128, n=8):
    print("=" * 72)
    print("TEST 3  the redundancy that justified the FIT_KEYS change is exact")
    s = P.image_stats_torch(synth(n, size, seed=4))
    chan_mean = (s["r_mean"] + s["g_mean"] + s["b_mean"]) / 3.0
    d1 = float((s["brightness"] - chan_mean).abs().max())
    fr = s["r_frac"] + s["g_frac"] + s["b_frac"]
    d2 = float((fr - 1.0).abs().max())
    print(f"max |brightness - mean(r,g,b means)| = {d1:.3e}  (exactly redundant)")
    print(f"max |r_frac + g_frac + b_frac - 1|   = {d2:.3e}  (sum-to-one constraint)")
    print(f"FIT_KEYS now: {F.FIT_KEYS}")
    print(f"reported but not optimised: {F.DIAG_KEYS}")
    assert d1 < 1e-6 and d2 < 1e-6


def test_legacy_init():
    print("=" * 72)
    print("TEST 4  a 15-parameter init dict still loads")
    a = P.PhysicalAcquisition(init={"noise_a": 5e-4, "noise_b": 0.012})
    b = P.PhysicalAcquisition(init={"noise_a_r": 5e-4, "noise_a_g": 5e-4,
                                    "noise_a_b": 5e-4, "noise_b_r": 0.012,
                                    "noise_b_g": 0.012, "noise_b_b": 0.012})
    d = float((a.raw.data - b.raw.data).abs().max())
    print(f"max |raw(legacy) - raw(explicit)| = {d:.3e}")
    assert d < 1e-6
    try:
        P.PhysicalAcquisition(init={"blur_sigma": 0.9})
    except KeyError as e:
        print(f"unknown parameter rejected: {e}")
    else:
        raise AssertionError("unknown parameter silently ignored")


def test_variance_term(size=128, n=8):
    print("=" * 72)
    print("TEST 5  the objective now penalises a collapsed distribution")
    x = synth(n, size, seed=5)
    tgt = fake_targets(x)
    s_ok = P.image_stats_torch(x)
    # a collapsed batch: every image identical, so every statistic has the right
    # mean and zero spread. Under the old mean-only objective this scored 0.
    s_bad = {k: v.mean().expand_as(v) for k, v in s_ok.items()}
    print(f"matched batch   : mean-only {float(F.distance(s_ok, tgt, var_weight=0.0)):.6f}"
          f"   with spread term {float(F.distance(s_ok, tgt)):.6f}")
    print(f"collapsed batch : mean-only {float(F.distance(s_bad, tgt, var_weight=0.0)):.6f}"
          f"   with spread term {float(F.distance(s_bad, tgt)):.6f}")
    assert float(F.distance(s_bad, tgt, var_weight=0.0)) < 1e-6
    assert float(F.distance(s_bad, tgt)) > 1.0


def time_step(size=512, n=64):
    print("=" * 72)
    print("TEST 6  cost of one fit step at the real size, to size the CPU job")
    x = synth(n, size, seed=6)
    tgt = fake_targets(x)
    L = P.PhysicalAcquisition()
    opt = torch.optim.Adam(L.parameters(), lr=0.05)
    t0 = time.time()
    for _ in range(3):
        opt.zero_grad()
        F.distance(P.image_stats_torch(L(x)), tgt).backward()
        opt.step()
    dt = (time.time() - t0) / 3
    print(f"{n}x3x{size}x{size}: {dt:.2f} s per step, "
          f"400 steps = {400 * dt / 60:.1f} min on this machine")


if __name__ == "__main__":
    torch.set_num_threads(4)
    test_gradients()
    test_channel_asymmetry()
    test_redundancy()
    test_legacy_init()
    test_variance_term()
    time_step()
    print("=" * 72)
    print("all unit tests passed")
