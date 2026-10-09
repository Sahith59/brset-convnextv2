"""Candidate-level red-lesion RETENTION under translation. No classifier involved.

WHY THIS EXISTS
A translated BRSET image keeps its ORIGINAL diagnostic label. If the translator
erases a microaneurysm, the result is an image that looks healthy carrying a
"diabetic retinopathy" label, which is label noise manufactured by our own
pipeline. Nothing in the fitting objective (scripts 57, 79) can notice this: the
fitted statistics are global moments of brightness, contrast, Laplacian variance
and colour, and a lesion is a few dozen pixels. A translator can match every
statistic perfectly while deleting every lesion in the dataset.

WHAT WAS REJECTED, AND WHY
The first design was: run an mBRSET-trained classifier on translated BRSET
images and measure agreement with the source label. That is unsound three times
over.
  (a) Disagreement conflates lesion erasure with ordinary distribution shift.
      A classifier can flip on an image whose lesions are perfectly intact.
  (b) It rewards the wrong thing. A classifier trained on mBRSET is more
      confident on inputs that look like mBRSET, so a translator that moves the
      appearance closer to mBRSET scores better WITHOUT preserving anything.
      The metric is maximised by the appearance match it is supposed to audit.
  (c) BRSET is 6.6 percent DR-positive here (748 of 11372), so label agreement
      is about 93 percent before any lesion is considered. A translator that
      erased every lesion in the dataset would still score about 93 percent.
      The quantity has almost no dynamic range where it matters.

WHAT THIS DOES INSTEAD
Detect red-lesion CANDIDATES on the original and on the translated image with
the same deterministic detector, and ask how many original candidates still have
a candidate within a few pixels afterwards. No classifier, no training, no
dataset preference, and the answer is localised: it says WHERE structure
vanished and at what size and contrast.

The detector is the classical Spencer-Frame / Walter candidate extractor:
  1. Green channel. Red lesions are dark there and the green channel carries
     the most haemoglobin contrast.
  2. Vessel suppression by oriented morphology. Open the inverted green channel
     with a line structuring element at 8 orientations and keep the MAXIMUM over
     orientations: a vessel is elongated, so some orientation's line fits inside
     it and the opening preserves it; a blob smaller than the line in every
     direction survives no orientation and is removed. The residual
     inv - max_theta opening is therefore the compact-only part of the image.
     Identically, since black_tophat(g, se) = closing(g, se) - g
     = (1-g) - opening(1-g, se), this residual is the MINIMUM over the 8
     oriented black top-hats of the green channel. It is a background-subtracted
     dark-blob response, so uneven illumination (which the translator adds) is
     removed by construction rather than having to be modelled.
  3. Candidates are the h-maxima of that response: peaks that stand at least h
     above their own surroundings. scikit-image is NOT installed in this
     environment, so h-maxima is computed from its definition, by geodesic
     reconstruction of (response - h) under response using OpenCV dilations.

CONTROLS ARE THE POINT
A retention number is worthless without knowing what the detector does when
nothing happened and when everything happened, so every run reports:
  NULL      identity transform. Retention must be ~1.0, spurious ~0.
  NULL+     noise at 0.002, below the 1/255 quantisation step. Identity is
            deterministic and therefore guaranteed to score 1.0, which tests
            nothing; this tests whether the detector is stable under a
            perturbation smaller than the file format can represent.
  POSITIVE  Gaussian blur at sigma 3 and 5. Retention MUST collapse.
  CHANCE    the original candidate set matched against itself displaced by
            chance_shift px. This is the retention a translator would get for
            free from candidate DENSITY alone. If chance retention is high, the
            matching tolerance is too loose for the metric to mean anything, and
            every other number has to be read against this floor. No one asked
            for this control; it is the one that decides whether the metric is
            trustworthy.

It also isolates the stage-1 artifact term. PhysicalAcquisition adds n_hole dark
Gaussian blobs at amplitude up to 0.40 with sigma 0.015 to 0.055 of the fundus
radius, placed uniformly in [0.15, 0.85] of the frame. Those are dark compact
blobs, which is exactly what a haemorrhage is, so the acquisition model may be
PAINTING lesions onto images labelled healthy. The artifact_* conditions inject
those blobs at known centres and known scale and measure how many fabricate a
candidate, which converts "would it?" into a number.

Known limits, stated because they bound the interpretation:
  - Candidates are not lesions. On a healthy image most candidates are noise
    and vessel-crossing residue. Retention is therefore a measure of whether
    LESION-SCALE STRUCTURE survives, not of lesions specifically, which is why
    everything is also reported split by the DR label and by size and contrast.
  - A lesion lying on a vessel is suppressed with the vessel and is invisible
    to this metric in both images.
  - Detection at a fixed absolute h conflates erasure with global contrast
    scaling: the translator's contrast_gain of 0.916 shrinks every candidate's
    amplitude by 8 percent and pushes borderline ones under the threshold. So
    every number is reported twice, at a fixed h (abs) and at a per-image
    noise-matched h (rel, h = k * robust sigma of the response). The two
    bracket the answer; where they disagree, the loss is contrast rescaling
    rather than structural erasure.

Reads image appearance only. No diagnostic label is used by the detector; labels
are used solely to stratify the report.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).parent

# The stage-1 translator under audit, and the older numpy chain, loaded by path
# because the scripts are numbered and therefore not importable as modules.
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(HERE / path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("physical_layer", "72_physical_layer.py")
D = _load("degradation_ops", "55_degradation_ops.py")

EPS = 1e-12


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def fundus_mask_np(rgb, thresh=0.04):
    """Circular field of view, same threshold as scripts 55 and 72 so this
    agrees with what the translator itself considers inside the image."""
    return rgb.mean(axis=2) > thresh


def disc_diameter(mask):
    """Field-of-view diameter in px. Every length in the detector is expressed
    as a fraction of this, so the structuring elements and the matching
    tolerance mean the same RETINAL distance regardless of image size."""
    cols = mask.any(axis=0)
    rows = mask.any(axis=1)
    if not cols.any() or not rows.any():
        return 0
    w = int(cols.nonzero()[0].ptp() + 1)
    h = int(rows.nonzero()[0].ptp() + 1)
    return max(w, h)


# --------------------------------------------------------------------------
# detector
# --------------------------------------------------------------------------
def line_footprint(length, angle_deg):
    """A one-pixel-wide line of the given length through the centre of an odd
    square. Oversampled along the line so no orientation has gaps, which would
    break the opening at 22.5 and 67.5 degrees."""
    L = max(3, int(length) | 1)
    half = L // 2
    fp = np.zeros((L, L), np.uint8)
    t = np.deg2rad(angle_deg)
    for s in np.linspace(-half, half, 4 * L + 1):
        y = int(round(half + s * np.sin(t)))
        x = int(round(half + s * np.cos(t)))
        fp[y, x] = 1
    return fp


def lesion_response(rgb, mask, line_len, n_orient=8):
    """Dark-blob response: min over oriented black top-hats of the green channel.

    The surround outside the field of view is replaced by the interior median
    first. Left as near-1.0 in the inverted channel it is a huge bright region
    whose dilation leaks inward across the rim; made flat it cannot contribute a
    peak at all.
    """
    inv = (1.0 - rgb[..., 1]).astype(np.float32)
    if mask.any():
        inv = np.where(mask, inv, np.float32(np.median(inv[mask])))
    best = None
    for i in range(n_orient):
        fp = line_footprint(line_len, 180.0 * i / n_orient)
        op = cv2.morphologyEx(inv, cv2.MORPH_OPEN, fp,
                              borderType=cv2.BORDER_REPLICATE)
        best = op if best is None else np.maximum(best, op)
    return np.maximum(inv - best, 0.0)


def robust_sigma(resp, mask):
    """Noise scale of the response. The response is one-sided and concentrated
    near zero, so for a half-normal the median is 0.6745 sigma; this is the
    scale a noise-matched threshold is expressed in."""
    if not mask.any():
        return 0.0
    med = float(np.median(resp[mask]))
    return 1.4826 * med


def h_maxima_mask(resp, h, max_iter=512):
    """h-maxima of resp, by definition rather than by library.

    HMAX_h(f) is the reconstruction by dilation of (f - h) under f. Inside the
    dome of any peak rising more than h above its surroundings the
    reconstruction saturates at f - h, so (f - rec) >= h marks exactly the tops
    of the qualifying domes. Geodesic dilation is iterated 3x3 dilations capped
    by f, which converge when nothing changes; propagation only has to cross
    each dome, so convergence is fast in practice.
    """
    k = np.ones((3, 3), np.uint8)
    cur = np.maximum(resp - np.float32(h), 0.0).astype(np.float32)
    iters = 0
    for _ in range(max_iter):
        nxt = np.minimum(cv2.dilate(cur, k, borderType=cv2.BORDER_REPLICATE), resp)
        iters += 1
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return (resp - cur) >= np.float32(h) - 1e-7, iters


CONN8 = np.ones((3, 3), int)


def detect(resp, mask, h, max_cand=800):
    """Candidate peaks with their contrast and footprint size.

    contrast is the response at the peak, i.e. how far the blob sits below its
    own local background in green, in [0,1] intensity units. size is the area of
    the connected component of {resp > h} the peak belongs to, which is a
    footprint estimate; merged components inflate it, so only the small end of
    the size axis is used for conclusions.
    """
    em, iters = h_maxima_mask(resp, h)
    em = em & mask
    lab, n = ndimage.label(em, structure=CONN8)
    if n == 0:
        return (np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.float32), 0, iters)
    idx = np.arange(1, n + 1)
    cen = np.asarray(ndimage.center_of_mass(em, lab, idx), np.float32).reshape(-1, 2)
    peak = np.asarray(ndimage.maximum(resp, lab, idx), np.float32)

    sup, _ = ndimage.label((resp > h) & mask, structure=CONN8)
    areas = np.bincount(sup.ravel())
    yy = np.clip(np.round(cen[:, 0]).astype(int), 0, resp.shape[0] - 1)
    xx = np.clip(np.round(cen[:, 1]).astype(int), 0, resp.shape[1] - 1)
    cid = sup[yy, xx]
    area = np.where(cid > 0, areas[cid], 1).astype(np.float32)

    raw = int(n)
    if raw > max_cand:
        # Keep the strongest. A flood of near-threshold noise peaks would
        # otherwise dominate both retention and the chance floor; the raw count
        # is reported so the truncation is never silent.
        keep = np.argsort(-peak)[:max_cand]
        cen, peak, area = cen[keep], peak[keep], area[keep]
    return cen, peak, area, raw, iters


# --------------------------------------------------------------------------
# matching and binning
# --------------------------------------------------------------------------
def match_counts(src, dst, tol):
    """How many src points have a dst point within tol px."""
    if len(src) == 0:
        return np.zeros(0, bool)
    if len(dst) == 0:
        return np.zeros(len(src), bool)
    d, _ = cKDTree(dst).query(src, distance_upper_bound=tol)
    return np.isfinite(d)


def bin_keys(peak, area, h, small_px):
    """Size x contrast cell per candidate. Small and faint is the clinically
    important corner: early microaneurysms are both."""
    diam = 2.0 * np.sqrt(np.maximum(area, EPS) / np.pi)
    size = np.where(diam < small_px, "small", np.where(diam < 2 * small_px, "mid", "large"))
    con = np.where(peak < 2 * h, "faint", np.where(peak < 4 * h, "moderate", "strong"))
    return [f"{s}_{c}" for s, c in zip(size, con)]


def tally(cen, peak, area, matched, h, small_px):
    out = {}
    for k, m in zip(bin_keys(peak, area, h, small_px), matched):
        n, mm = out.get(k, (0, 0))
        out[k] = (n + 1, mm + int(m))
    return {k: list(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# transforms under test
# --------------------------------------------------------------------------
def t_identity(img, seed, c):
    return img, {}


def t_noise(img, seed, c):
    rng = np.random.default_rng(seed)
    out = img + rng.normal(0.0, c["sigma"], img.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0), {}


def t_blur(img, seed, c):
    out = np.stack([ndimage.gaussian_filter(img[..., i], c["sigma"]) for i in range(3)], 2)
    return np.clip(out, 0.0, 1.0), {}


def t_stage1(img, seed, c):
    """The actual stage-1 translator, with parameter overrides.

    PhysicalAcquisition maps its parameters through a sigmoid with the raw value
    clipped to [1e-3, 1-1e-3], so a request exactly at a bound lands a hair
    inside it; params_used records what was really applied.
    """
    torch.set_num_threads(1)
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    L = P.PhysicalAcquisition(init=c.get("over", {}), n_spot=c.get("n_spot", 1),
                              n_hole=c.get("n_hole", 3), learnable=False)
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        y = L(x, generator=g)
    return y[0].permute(1, 2, 0).numpy(), {}


def t_numpy55(img, seed, c):
    """The older numpy chain. It works in uint8, so the round trip quantises;
    that is the chain as it is actually used, not a handicap added here."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    out = D.degrade(pil, c["p"], rng)
    return np.asarray(out, np.float32) / 255.0, {}


def t_artifact(img, seed, c):
    """ONLY the stage-1 dark-blob artifact term, at a controlled scale, with the
    centres recorded. Everything else is identity, so any new candidate is
    attributable to the blobs and nothing else. Placement follows the model:
    uniform in [0.15, 0.85] of the frame, sigma as a fraction of the fundus
    radius, amplitude subtracted from all three channels."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    R = min(h, w) / 2.0
    rad = c["sigma_frac"] * R
    n = c.get("n_blob", 3)
    cen = rng.uniform(0.15, 0.85, (n, 2)) * np.array([h, w])
    yy, xx = np.mgrid[0:h, 0:w]
    out = img.copy()
    for cy, cx in cen:
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))
        out = out - np.float32(c["amp"]) * blob[..., None].astype(np.float32)
    return np.clip(out, 0.0, 1.0), {"blobs": cen.tolist(), "blob_rad": float(rad)}


TRANSFORMS = {"identity": t_identity, "noise": t_noise, "blur": t_blur,
              "stage1": t_stage1, "numpy55": t_numpy55, "artifact": t_artifact}


def build_conditions():
    """Controls first, then the stage-1 chain, then one-axis-at-a-time strength
    curves, then the artifact isolation, then the older numpy chain."""
    C = [
        {"name": "NULL_identity", "kind": "identity", "group": "control"},
        {"name": "NULL_jitter_0.002", "kind": "noise", "sigma": 0.002, "group": "control"},
        {"name": "POS_blur_sigma1", "kind": "blur", "sigma": 1.0, "group": "control"},
        {"name": "POS_blur_sigma3", "kind": "blur", "sigma": 3.0, "group": "control"},
        {"name": "POS_blur_sigma5", "kind": "blur", "sigma": 5.0, "group": "control"},
        # the translator as it stands, then with one term switched off at a time
        {"name": "stage1_fitted_full", "kind": "stage1", "over": {}, "group": "stage1"},
        {"name": "stage1_no_artifacts", "kind": "stage1", "group": "stage1",
         "over": {"spot_amp": 0.0, "hole_amp": 0.0}},
        {"name": "stage1_no_noise", "kind": "stage1", "group": "stage1",
         "over": {"noise_a": 0.0, "noise_b": 0.0}},
        {"name": "stage1_photometric_only", "kind": "stage1", "group": "stage1",
         "over": {"spot_amp": 0.0, "hole_amp": 0.0, "noise_a": 0.0, "noise_b": 0.0,
                  "halo": 0.0, "light_strength": 0.0}},
    ]
    # sharpness axis, swept across the full learnable range with every other
    # term off, so the curve is the optics axis alone. k<0 blurs, k>0 sharpens.
    for k in (-0.95, -0.60, -0.30, 0.0, 1.00, 2.50):
        C.append({"name": f"sweep_sharpen_k_{k:+.2f}", "kind": "stage1", "group": "sweep_sharpen",
                  "over": {"sharpen_k": k, "spot_amp": 0.0, "hole_amp": 0.0,
                           "noise_a": 0.0, "noise_b": 0.0, "halo": 0.0,
                           "light_strength": 0.0, "contrast_gain": 1.0,
                           "saturation_gain": 1.0}})
    # sensor-noise axis, fitted value 0.004 up to the bound 0.03
    for nb in (0.0, 0.004, 0.010, 0.030):
        C.append({"name": f"sweep_noise_b_{nb:.3f}", "kind": "stage1", "group": "sweep_noise",
                  "over": {"noise_b": nb, "noise_a": 2e-3, "spot_amp": 0.0,
                           "hole_amp": 0.0, "halo": 0.0, "light_strength": 0.0,
                           "contrast_gain": 1.0, "saturation_gain": 1.0}})
    # artifact amplitude axis inside the real chain, fitted 0.20 up to bound 0.40
    for ha in (0.0, 0.10, 0.20, 0.40):
        C.append({"name": f"sweep_hole_amp_{ha:.2f}", "kind": "stage1", "group": "sweep_hole",
                  "over": {"hole_amp": ha, "spot_amp": 0.0, "noise_a": 0.0,
                           "noise_b": 0.0, "halo": 0.0, "light_strength": 0.0,
                           "contrast_gain": 1.0, "saturation_gain": 1.0}})
    # artifact term in isolation, scale by scale, with known centres
    for sf in (0.015, 0.025, 0.040, 0.055):
        for amp in (0.20, 0.40):
            C.append({"name": f"artifact_sigma{sf:.3f}_amp{amp:.2f}", "kind": "artifact",
                      "group": "artifact", "sigma_frac": sf, "amp": amp, "n_blob": 3})
    C.append({"name": "numpy55_fitted", "kind": "numpy55", "group": "numpy55",
              "p": dict(D.FIT_INIT)})
    C.append({"name": "numpy55_gdrnet_generic", "kind": "numpy55", "group": "numpy55",
              "p": dict(D.GDRNET_GENERIC)})
    return C


# --------------------------------------------------------------------------
# per-image work
# --------------------------------------------------------------------------
CFG = {}


def _init(cfg):
    CFG.update(cfg)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def process_image(task):
    from PIL import Image
    fn, label = task
    cfg = CFG
    try:
        im = Image.open(Path(cfg["root"]) / fn).convert("RGB")
        # Measure at the pitch the TRAINER actually feeds the translator:
        # Resize((560,560)) then a 512 centre crop (scripts 79, 81). Lesion
        # survival is resolution dependent, so measuring at native resolution
        # would answer a question the pipeline never asks. --presize 0 --size 0
        # measures natively instead, which is the more generous reference.
        if cfg["presize"]:
            im = im.resize((cfg["presize"], cfg["presize"]), Image.BILINEAR)
            o = (cfg["presize"] - cfg["size"]) // 2
            im = im.crop((o, o, o + cfg["size"], o + cfg["size"]))
        elif cfg["size"]:
            im = im.resize((cfg["size"], cfg["size"]), Image.BILINEAR)
        img = np.asarray(im, np.float32) / 255.0
    except Exception as e:  # unreadable file, reported not hidden
        return {"file": fn, "error": repr(e)}

    mask0 = fundus_mask_np(img)
    disc = disc_diameter(mask0)
    if disc < 100 or mask0.sum() < 1000:
        return {"file": fn, "error": f"degenerate fov disc={disc}"}

    # Geometry is fixed by the ORIGINAL for both images: the translator cannot
    # move anatomy, so a mask change could only move candidates in or out of
    # scope and would be read as retention or spurious.
    er = max(1, int(round(cfg["erode_frac"] * disc)))
    mask = ndimage.binary_erosion(mask0, iterations=er)
    line_len = max(5, int(round(cfg["line_len_frac"] * disc)))
    tol = max(2.0, cfg["tol_frac"] * disc)
    small_px = max(2.0, cfg["small_frac"] * disc)

    r0 = lesion_response(img, mask, line_len, cfg["n_orient"])
    sig0 = robust_sigma(r0, mask)
    h0 = {"abs": cfg["h_abs"], "rel": max(cfg["h_floor"], cfg["k_rel"] * sig0)}
    base, max_iters = {}, 0
    for v in ("abs", "rel"):
        cen, peak, area, raw, it = detect(r0, mask, h0[v], cfg["max_cand"])
        base[v] = (cen, peak, area, raw)
        max_iters = max(max_iters, it)

    rec = {"file": fn, "dr": int(label), "disc": int(disc), "line_len": line_len,
           "tol": float(tol), "sigma0": float(sig0),
           "h": {v: float(h0[v]) for v in h0}, "cond": {}}

    # chance floor: the original candidates against themselves, displaced. This
    # is the retention that candidate DENSITY alone buys at this tolerance.
    for v in ("abs", "rel"):
        cen, peak, area, raw = base[v]
        sh = cen + np.float32(cfg["chance_shift_frac"] * disc)
        m = match_counts(cen, sh, tol)
        rec.setdefault("chance", {})[v] = [int(len(cen)), int(m.sum())]

    for c in cfg["conditions"]:
        # zlib.crc32, not hash(): Python salts string hashing per process, so
        # hash() would silently give a different noise and blob realisation on
        # every run and nothing here would reproduce.
        seed = zlib.crc32(f"{fn}|{c['name']}".encode()) + 1
        out, meta = TRANSFORMS[c["kind"]](img, seed, c)
        r1 = lesion_response(out, mask, line_len, cfg["n_orient"])
        sig1 = robust_sigma(r1, mask)
        entry = {"sigma1": float(sig1)}
        for v in ("abs", "rel"):
            hv = cfg["h_abs"] if v == "abs" else max(cfg["h_floor"], cfg["k_rel"] * sig1)
            c0, p0, a0, raw0 = base[v]
            c1, p1, a1, raw1, it = detect(r1, mask, hv, cfg["max_cand"])
            max_iters = max(max_iters, it)
            keep = match_counts(c0, c1, tol)           # original survived?
            fresh = ~match_counts(c1, c0, tol)         # translated candidate is new?
            e = {"n_orig": int(len(c0)), "n_trans": int(len(c1)),
                 "raw_orig": int(raw0), "raw_trans": int(raw1),
                 "kept": int(keep.sum()), "new": int(fresh.sum()),
                 "h_trans": float(hv),
                 "bins": tally(c0, p0, a0, keep, h0[v], small_px)}
            if "blobs" in meta:
                # Does the injected artifact blob FABRICATE a lesion candidate?
                # Two directions, because they answer different questions:
                # per blob, did a NEW candidate appear on it (how often the
                # model paints something the detector calls a lesion); and per
                # new candidate, is it sitting on a blob (whether the blobs
                # explain the spurious rate).
                bl = np.asarray(meta["blobs"], np.float32).reshape(-1, 2)
                rad = float(meta["blob_rad"])
                near = max(2.0, 2.0 * rad)
                newc = c1[fresh] if len(c1) else c1
                e["blob"] = {"n_blob": int(len(bl)), "rad": rad,
                             "blob_made_new_cand": int(match_counts(bl, newc, near).sum()),
                             "blob_has_any_cand": int(match_counts(bl, c1, near).sum()),
                             "new_near_blob": int(match_counts(newc, bl, near).sum()),
                             "new_total": int(fresh.sum())}
            entry[v] = e
        rec["cond"][c["name"]] = entry
    # If the reconstruction ever hits its iteration cap the h-maxima are only
    # partial, so this must be visible rather than assumed fine.
    rec["recon_iters_max"] = int(max_iters)
    return rec


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def agg(recs, name, v, sel=None):
    rows = [r for r in recs if "cond" in r and (sel is None or sel(r))]
    no = sum(r["cond"][name][v]["n_orig"] for r in rows)
    nt = sum(r["cond"][name][v]["n_trans"] for r in rows)
    kept = sum(r["cond"][name][v]["kept"] for r in rows)
    new = sum(r["cond"][name][v]["new"] for r in rows)
    per = [r["cond"][name][v]["kept"] / r["cond"][name][v]["n_orig"]
           for r in rows if r["cond"][name][v]["n_orig"] > 0]
    out = {"n_images": len(rows),
           "retention_pooled": kept / no if no else float("nan"),
           "retention_macro": float(np.mean(per)) if per else float("nan"),
           "retention_macro_sd": float(np.std(per)) if per else float("nan"),
           "spurious_frac_of_trans": new / nt if nt else float("nan"),
           "new_per_orig": new / no if no else float("nan"),
           "cand_orig_mean": no / len(rows) if rows else float("nan"),
           "cand_trans_mean": nt / len(rows) if rows else float("nan"),
           "raw_orig_mean": float(np.mean([r["cond"][name][v]["raw_orig"] for r in rows])) if rows else float("nan"),
           "raw_trans_mean": float(np.mean([r["cond"][name][v]["raw_trans"] for r in rows])) if rows else float("nan")}
    bins = {}
    for r in rows:
        for k, (n, m) in r["cond"][name][v]["bins"].items():
            a, b = bins.get(k, (0, 0))
            bins[k] = (a + n, b + m)
    out["bins"] = {k: {"n": n, "kept": m, "retention": m / n if n else float("nan")}
                   for k, (n, m) in sorted(bins.items())}
    bl = [r["cond"][name][v]["blob"] for r in rows if "blob" in r["cond"][name][v]]
    if bl:
        nb = sum(b["n_blob"] for b in bl)
        out["blob"] = {"n_blob": nb,
                       "frac_blobs_fabricating_candidate": sum(b["blob_made_new_cand"] for b in bl) / nb,
                       "frac_blobs_with_any_candidate": sum(b["blob_has_any_cand"] for b in bl) / nb,
                       "new_cand_per_image": float(np.mean([b["new_total"] for b in bl])),
                       "frac_new_cand_at_blob": (sum(b["new_near_blob"] for b in bl)
                                                 / max(sum(b["new_total"] for b in bl), 1)),
                       "blob_radius_px_mean": float(np.mean([b["rad"] for b in bl]))}
    return out


def agg_chance(recs, v):
    rows = [r for r in recs if "chance" in r]
    no = sum(r["chance"][v][0] for r in rows)
    k = sum(r["chance"][v][1] for r in rows)
    per = [r["chance"][v][1] / r["chance"][v][0] for r in rows if r["chance"][v][0] > 0]
    return {"n_images": len(rows), "retention_pooled": k / no if no else float("nan"),
            "retention_macro": float(np.mean(per)) if per else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/finetune_multilabel/train")
    ap.add_argument("--labels", default=None, help="defaults to <root>/labels.csv")
    ap.add_argument("--n_pos", type=int, default=80)
    ap.add_argument("--n_neg", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    # detector geometry, all as fractions of the fundus disc diameter
    ap.add_argument("--line_len_frac", type=float, default=0.017,
                    help="vessel-suppression line length; 0.017*870px = 15px, "
                         "longer than any lesion, shorter than a vessel run")
    ap.add_argument("--n_orient", type=int, default=8)
    ap.add_argument("--tol_frac", type=float, default=0.0057,
                    help="match tolerance; 0.0057*870px = 5px")
    ap.add_argument("--small_frac", type=float, default=0.0046,
                    help="small-candidate diameter; 0.0046*870px = 4px ~ 65um")
    ap.add_argument("--erode_frac", type=float, default=0.03)
    ap.add_argument("--chance_shift_frac", type=float, default=0.046,
                    help="chance-floor displacement; 0.046*870px = 40px")
    ap.add_argument("--h_abs", type=float, default=0.025,
                    help="fixed detection threshold in [0,1] green-contrast units")
    ap.add_argument("--k_rel", type=float, default=4.0,
                    help="noise-matched threshold, k * robust sigma of the response")
    ap.add_argument("--h_floor", type=float, default=0.004)
    ap.add_argument("--max_cand", type=int, default=800)
    ap.add_argument("--presize", type=int, default=560,
                    help="trainer's Resize((S,S)) before the crop; 0 disables")
    ap.add_argument("--size", type=int, default=512,
                    help="trainer's crop; with --presize 0 --size 0, measure natively")
    ap.add_argument("--only", default=None, help="comma-separated condition name filter")
    ap.add_argument("--out", default="results/lesion_retention.json")
    ap.add_argument("--per_image", action="store_true", help="also dump per-image rows")
    ap.add_argument("--print_bins", action="store_true",
                    help="print the size x contrast retention breakdown")
    a = ap.parse_args()

    lab = Path(a.labels) if a.labels else Path(a.root) / "labels.csv"
    df = pd.read_csv(lab)
    rng = np.random.default_rng(a.seed)
    pos = df[df.diabetic_retinopathy == 1]
    neg = df[df.diabetic_retinopathy == 0]
    take = lambda d, n: d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)]
    # Stratified on purpose: retention has to be reported on the DR-positive
    # images separately, because those are the ones whose label is at risk.
    sub = pd.concat([take(pos, a.n_pos), take(neg, a.n_neg)])
    tasks = list(zip(sub.file.values, sub.diabetic_retinopathy.values))

    conds = build_conditions()
    if a.only:
        want = set(a.only.split(","))
        conds = [c for c in conds if c["name"] in want]
    cfg = {k: getattr(a, k) for k in
           ("root", "line_len_frac", "n_orient", "tol_frac", "small_frac",
            "erode_frac", "chance_shift_frac", "h_abs", "k_rel", "h_floor",
            "max_cand", "presize", "size")}
    cfg["conditions"] = conds

    print(f"images={len(tasks)} (pos={int(sub.diabetic_retinopathy.sum())}) "
          f"conditions={len(conds)} workers={a.workers}", flush=True)
    t0 = time.time()
    if a.workers > 1:
        with mp.Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
            recs = []
            for i, r in enumerate(pool.imap_unordered(process_image, tasks, chunksize=1), 1):
                recs.append(r)
                if i % 10 == 0:
                    print(f"  {i}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    else:
        _init(cfg)
        recs = [process_image(t) for t in tasks]
    bad = [r for r in recs if "error" in r]
    ok = [r for r in recs if "cond" in r]
    print(f"done in {time.time()-t0:.0f}s  ok={len(ok)} failed={len(bad)}", flush=True)
    for r in bad[:5]:
        print("  FAILED", r["file"], r["error"])

    res = {"config": {k: v for k, v in vars(a).items()},
           "n_images": len(ok), "n_failed": len(bad),
           "n_dr_pos": sum(r["dr"] == 1 for r in ok),
           "disc_px_mean": float(np.mean([r["disc"] for r in ok])) if ok else None,
           "line_len_px_mean": float(np.mean([r["line_len"] for r in ok])) if ok else None,
           "tol_px_mean": float(np.mean([r["tol"] for r in ok])) if ok else None,
           "recon_iters_max": int(max([r.get("recon_iters_max", 0) for r in ok], default=0)),
           "chance": {v: agg_chance(ok, v) for v in ("abs", "rel")},
           "conditions": {}}
    for c in conds:
        res["conditions"][c["name"]] = {
            "group": c["group"],
            "abs": agg(ok, c["name"], "abs"),
            "rel": agg(ok, c["name"], "rel"),
            "abs_dr_pos": agg(ok, c["name"], "abs", lambda r: r["dr"] == 1),
            "abs_dr_neg": agg(ok, c["name"], "abs", lambda r: r["dr"] == 0)}
    if a.per_image:
        res["per_image"] = ok

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    # report
    print("\n" + "=" * 100)
    print("CANDIDATE-LEVEL RED-LESION RETENTION   (no classifier anywhere)")
    print("=" * 100)
    print(f"images={len(ok)}  DR+={res['n_dr_pos']}  disc={res['disc_px_mean']:.0f}px  "
          f"line SE={res['line_len_px_mean']:.0f}px  tol={res['tol_px_mean']:.1f}px")
    for v in ("abs", "rel"):
        ch = res["chance"][v]
        print(f"CHANCE FLOOR ({v}): retention of the original candidates against themselves "
              f"displaced {a.chance_shift_frac*res['disc_px_mean']:.0f}px = "
              f"{ch['retention_pooled']:.4f} pooled / {ch['retention_macro']:.4f} macro")
    hdr = (f"{'condition':34s} {'ret_abs':>8} {'spur_abs':>9} {'ret_rel':>8} {'spur_rel':>9} "
           f"{'ret_DR+':>8} {'cand_o':>7} {'cand_t':>7} {'sm&faint':>9}")
    last = None
    for c in conds:
        if c["group"] != last:
            print("\n" + hdr)
            print("-" * len(hdr))
            last = c["group"]
        A, B = res["conditions"][c["name"]]["abs"], res["conditions"][c["name"]]["rel"]
        Pp = res["conditions"][c["name"]]["abs_dr_pos"]
        sf = A["bins"].get("small_faint", {}).get("retention", float("nan"))
        print(f"{c['name']:34s} {A['retention_pooled']:8.4f} {A['spurious_frac_of_trans']:9.4f} "
              f"{B['retention_pooled']:8.4f} {B['spurious_frac_of_trans']:9.4f} "
              f"{Pp['retention_pooled']:8.4f} {A['cand_orig_mean']:7.1f} "
              f"{A['cand_trans_mean']:7.1f} {sf:9.4f}")
    if a.print_bins:
        # The breakdown is where "is this metric trustworthy" is answered: a
        # threshold detector is unstable exactly at its threshold, so faint
        # candidates must be expected to churn while strong ones must not.
        print("\nSIZE x CONTRAST BREAKDOWN (retention, pooled over images; "
              "small<" f"{a.small_frac:.4f}*disc diameter, faint<2h)")
        names = [c["name"] for c in conds if c["group"] in ("control", "stage1")]
        cells = sorted({k for n in names for k in res["conditions"][n]["abs"]["bins"]})
        print(f"{'condition':34s} " + " ".join(f"{c:>16s}" for c in cells))
        for n in names:
            b = res["conditions"][n]["abs"]["bins"]
            row = " ".join(
                (f"{b[c]['retention']:.3f}({b[c]['n']:5d})".rjust(16) if c in b
                 else "-".rjust(16)) for c in cells)
            print(f"{n:34s} " + row)

    print("\nARTIFACT BLOBS: do they FABRICATE lesion candidates?")
    print(f"{'condition':34s} {'rad_px':>7} {'frac blobs making a cand':>26} "
          f"{'new cands/img':>14} {'frac new at blob':>17}")
    for c in conds:
        b = res["conditions"][c["name"]]["abs"].get("blob")
        if b:
            print(f"{c['name']:34s} {b['blob_radius_px_mean']:7.1f} "
                  f"{b['frac_blobs_fabricating_candidate']:26.3f} "
                  f"{b['new_cand_per_image']:14.1f} {b['frac_new_cand_at_blob']:17.3f}")
    print(f"\nmax reconstruction iterations used: {res['recon_iters_max']} "
          f"(cap 512; at the cap the h-maxima would be partial)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
