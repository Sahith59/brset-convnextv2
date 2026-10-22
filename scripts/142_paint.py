"""C6: paint mild-DR lesions onto HEALTHY BRSET eyes with the lesion painter (research/codex D097).

For each healthy BRSET train eye (ICDR 0): physics camera image → noised to `strength` → rendered by the painter
(scripts/141) with camera = mBRSET, the requested ICDR grade and oedema, following the eye's own (spot-erased)
structure map. Guidance: v = v_c + (w_s − 1)(v_c − v_no-map) + (w_g − 1)(v_c − v_no-grade).
Each source eye is used once (distinct anatomy); --control also renders the SAME eyes with grade 0 so the gate
can compare painted vs unpainted versions of identical eyes. Output: synth/<out_tag>/<stem>_g<grade>.png and
labels_shard<k>.csv (file, diabetic_retinopathy, macular_edema, grade, source).
"""
import argparse, importlib.util, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent
GEN = Path("/data/users3/sthummala2/brset_gen")


def _load(name, mod):
    sp = importlib.util.spec_from_file_location(mod, str(HERE / name))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


P = _load("141_train_painter.py", "P141")
C, A = P.C, P.A
from diffusers import DDIMScheduler  # noqa: E402


@torch.no_grad()
def paint(model, x_init, s, m, labels, strength, steps, w_s, w_g, gen):
    dev = x_init.device; B = x_init.shape[0]
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction",
                          clip_sample=False, timestep_spacing="trailing")
    sched.set_timesteps(steps, device=dev)
    ts = sched.timesteps[int(round(steps * (1.0 - strength))):]
    x = sched.add_noise(x_init * 2 - 1, torch.randn(x_init.shape, generator=gen, device=dev), ts[:1].expand(B))
    nog = labels.clone(); nog[:, 1] = A.GRADE_NULL; nog[:, 2] = A.ME_NULL
    zero = torch.zeros_like(s)
    for t in ts:
        inp = torch.cat([torch.cat([x, s, m], 1), torch.cat([x, zero, m], 1), torch.cat([x, s, m], 1)])
        lab = torch.cat([labels, labels, nog])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(inp, t.expand(3 * B), lab).float()
        vc, vns, vng = v.chunk(3)
        x = sched.step(vc + (w_s - 1) * (vc - vns) + (w_g - 1) * (vc - vng), t, x).prev_sample
    return ((x + 1) / 2).clamp(0, 1) * m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--painter_run", required=True)
    ap.add_argument("--painter_ckpt", required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--n_g1", type=int, default=2000)
    ap.add_argument("--n_g2", type=int, default=2000)
    ap.add_argument("--control", type=int, default=0, help="also render the grade-1 eyes with grade 0")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--w_struct", type=float, default=1.5)
    ap.add_argument("--w_grade", type=float, default=2.0)
    ap.add_argument("--phys_ckpt", default=str(ROOT / "results/xlate_v2_noartifact/translator_last.pth"))
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(GEN / "runs" / a.painter_run / a.painter_ckpt, map_location="cpu", weights_only=False)
    model = P.build_painter("").to(dev).eval(); model.load_state_dict(ck["ema"])
    print(f"painter {a.painter_run}/{a.painter_ckpt} step {ck['step']}; strength {a.strength}, w_grade {a.w_grade}", flush=True)
    phys = _load("88_synth_augment.py", "S88").load_translator(a.phys_ckpt, device=dev).phys

    meta = pd.read_csv(GEN / "cache/brset_train_meta.csv")
    healthy = meta[(meta.dr == 0) & (meta.grade == 0)].idx.values
    order = np.random.RandomState(7).permutation(healthy)                       # fixed, distinct eyes per grade
    rs = np.random.RandomState(11)
    plan = [(int(i), 1, int(rs.rand() < 10 / 187)) for i in order[:a.n_g1]]   # oedema rates of mBRSET train grades 1 / 2
    plan += [(int(i), 2, int(rs.rand() < 158 / 405)) for i in order[a.n_g1:a.n_g1 + a.n_g2]]
    if a.control:
        plan += [(int(i), 0, 0) for i in order[:a.n_g1]]
    plan = plan[a.shard::a.n_shards]
    arr = np.load(GEN / "cache/brset_train_560.npy", mmap_mode="r")
    fname = dict(zip(meta.idx.values, meta.file.values))
    out = GEN / "synth" / a.out_tag; out.mkdir(parents=True, exist_ok=True)
    name = lambda i, g: f"{Path(fname[i]).stem}_g{g}"
    todo = [p for p in plan if not (out / f"{name(p[0], p[1])}.png").exists()]
    print(f"shard {a.shard}/{a.n_shards}: {len(todo)} of {len(plan)} to do -> {out}", flush=True)
    t0 = time.time()
    for b0 in range(0, len(todo), a.batch):
        bt = todo[b0:b0 + a.batch]
        x = torch.from_numpy(np.stack([np.array(arr[i]) for i, _, _ in bt])).permute(0, 3, 1, 2).float().div(255).to(dev)
        g = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + bt[0][0] * 7 + bt[0][1])
        m = C.fundus_mask(x)
        xs = x.clone()
        xs[:, 1] = torch.from_numpy(np.stack([P.suppress_dark_spots((x[k, 1] * 255).round().byte().cpu().numpy())
                                               for k in range(x.shape[0])])).float().div(255).to(dev)
        s = C.structure_map(xs, m)
        xp = phys(x, generator=g).clamp(0, 1)
        lab = torch.tensor([[1, gr, e] for _, gr, e in bt], device=dev)
        y = paint(model, xp, s, m, lab, a.strength, a.steps, a.w_struct, a.w_grade, g)
        y = (y * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for k, (i, gr, _) in enumerate(bt):
            tmp = out / f".{name(i, gr)}.tmp.png"; Image.fromarray(y[k]).save(tmp); tmp.replace(out / f"{name(i, gr)}.png")
        if (b0 // a.batch) % 20 == 0:
            r = (b0 + len(bt)) / max(time.time() - t0, 1e-6)
            print(f"  {b0 + len(bt)}/{len(todo)} {r:.2f} img/s", flush=True)
    pd.DataFrame(dict(file=[str(out / f"{name(i, gr)}.png") for i, gr, _ in plan],
                      diabetic_retinopathy=[int(gr >= 1) for _, gr, _ in plan], macular_edema=[e for _, _, e in plan],
                      grade=[gr for _, gr, _ in plan], source=[fname[i] for i, _, _ in plan])
                 ).to_csv(out / f"labels_shard{a.shard:03d}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
