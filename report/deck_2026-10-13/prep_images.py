"""Crop slide figures from generator outputs. Every crop records its source file."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
R = Path("/home/users/sthummala2/brset-convnextv2")
OUT = R / "report/deck_2026-10-13"
G = Path("/data/users3/sthummala2/brset_gen/runs")
prov = {}
# 1) synthetic grid: rows = grades 1,3,4 ; cols = BRSET original | ours physics+diffusion | real mBRSET
src = R / "report/figures/look_gen_both_step40000.png"
im = Image.open(src).convert("RGB")
cell = lambda r, c: im.crop((150 + c * 428, 70 + r * 428, 150 + c * 428 + 420, 70 + r * 428 + 420))
rows, cols, S, gap = [1, 3, 4], [0, 2, 3], 420, 10
grid = Image.new("RGB", (3 * S + 2 * gap, 3 * S + 2 * gap), (255, 255, 255))
for i, r in enumerate(rows):
    for j, c in enumerate(cols):
        grid.paste(cell(r, c), (j * (S + gap), i * (S + gap)))
grid.save(OUT / "synth_grid.png")
prov["synth_grid.png"] = dict(source=str(src), rows_grades=rows, cols=["BRSET original", "physics+diffusion gen_both 40k", "real mBRSET (different eye)"])
# 2) structure-map strip from the training panel: [BRSET source | structure map | rendered as mBRSET]
psrc = G / "gen_both/panels/step_0040000.png"
p = Image.open(psrc).convert("RGB")
W, H = p.size; rh = H // 6
strip = p.crop((0, 3 * rh, W, 4 * rh))
strip.save(OUT / "structure_strip.png")
prov["structure_strip.png"] = dict(source=str(psrc), row="first BRSET monitoring image (row 4 of 6)", size=strip.size)
(OUT / "image_provenance.json").write_text(json.dumps(prov, indent=2))
print(json.dumps(prov, indent=1))
