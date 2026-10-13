"""Approximate slide previews for visual QA where LibreOffice is unavailable.

Draws backgrounds, shapes, connectors, pictures, tables and wrapped text with
Liberation Serif, which has the same advance widths as Times New Roman, so line
breaks and overflow match PowerPoint closely. Charts are drawn as labelled boxes
(PowerPoint renders them natively). Any text that runs past its box, any table
that grows past the footer, and any shape off the slide is listed and outlined red.
"""
import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu

FD = "/data/users3/sthummala2/fonts/ex/liberation-fonts-ttf-2.1.5/"
DPI = 110
EMU_IN = 914400


def px(emu):
    return int(round(emu / EMU_IN * DPI))


_fc = {}


def font(size_pt, bold=False, italic=False):
    k = (round(size_pt * 2) / 2, bold, italic)
    if k not in _fc:
        name = "LiberationSerif-" + ({(0, 0): "Regular", (1, 0): "Bold", (0, 1): "Italic", (1, 1): "BoldItalic"}[(int(bold), int(italic))])
        _fc[k] = ImageFont.truetype(FD + name + ".ttf", max(1, int(round(size_pt * DPI / 72))))
    return _fc[k]


def rgb(fmt_color, default=(42, 49, 58)):
    try:
        c = fmt_color.rgb
        return (c[0], c[1], c[2]) if c is not None else default
    except Exception:
        return default


def layout_frame(tf, width_px):
    """Return list of lines: (segments[(text, font, color)], height_px, align)."""
    out = []
    for p in tf.paragraphs:
        runs = [r for r in p.runs]
        size = None
        words = []
        for r in runs:
            sz = r.font.size.pt if r.font.size else 18
            size = max(size or 0, sz)
            f = font(sz, bool(r.font.bold), bool(r.font.italic))
            col = rgb(r.font.color) if r.font.color and r.font.color.type is not None else (42, 49, 58)
            parts = r.text.split(" ")
            for i, w in enumerate(parts):
                words.append((w + (" " if i < len(parts) - 1 else ""), f, col))
        size = size or 18
        ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
        lh = size * 1.17 * ls * DPI / 72
        sa = (p.space_after.pt if p.space_after is not None else 0) * DPI / 72
        align = p.alignment
        line, cur = [], 0.0
        lines = []
        for w, f, col in words:
            ww = f.getlength(w)
            if cur + f.getlength(w.rstrip()) > width_px and line:
                lines.append(line)
                line, cur = [], 0.0
            line.append((w, f, col))
            cur += ww
        lines.append(line)
        for k, ln in enumerate(lines):
            out.append((ln, lh + (sa if (k == len(lines) - 1 and p is not tf.paragraphs[-1]) else 0), align))
    return out


def draw_text(d, tf, box, issues, tag):
    x0, y0, x1, y1 = box
    ml, mr = px(tf.margin_left), px(tf.margin_right)
    mt, mb = px(tf.margin_top), px(tf.margin_bottom)
    lines = layout_frame(tf, (x1 - x0) - ml - mr)
    total = sum(h for _, h, _ in lines)
    anchor = str(tf.vertical_anchor) if tf.vertical_anchor is not None else "TOP"
    if "MIDDLE" in anchor:
        y = y0 + ((y1 - y0) - total) / 2
    elif "BOTTOM" in anchor:
        y = y1 - mb - total
    else:
        y = y0 + mt
    for ln, h, align in lines:
        wsum = sum(f.getlength(w) for w, f, _ in ln)
        if align is not None and "CENTER" in str(align):
            x = x0 + ((x1 - x0) - wsum) / 2
        elif align is not None and "RIGHT" in str(align):
            x = x1 - mr - wsum
        else:
            x = x0 + ml
        for w, f, col in ln:
            d.text((x, y), w, font=f, fill=col)
            x += f.getlength(w)
        y += h
    if y - (y1 - mb) > 3:
        issues.append(f"{tag}: text overflows its box by {(y - (y1 - mb)) / DPI:.2f} in")
        d.rectangle(box, outline=(220, 0, 0), width=3)


def render(path, outdir):
    prs = Presentation(path)
    W, H = px(prs.slide_width), px(prs.slide_height)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = []
    for si, slide in enumerate(prs.slides, 1):
        issues = []
        bgc = (255, 255, 255)
        try:
            bgc = rgb(slide.background.fill.fore_color, (255, 255, 255))
        except Exception:
            pass
        im = Image.new("RGB", (W, H), bgc)
        d = ImageDraw.Draw(im)
        for shp in slide.shapes:
            x0, y0 = px(shp.left), px(shp.top)
            x1, y1 = x0 + px(shp.width), y0 + px(shp.height)
            tag = f"slide {si} '{shp.name}'"
            if x1 > W + 2 or y1 > H + 2 or x0 < -2 or y0 < -2:
                issues.append(f"{tag}: off slide")
            if shp.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pic = Image.open(io.BytesIO(shp.image.blob)).convert("RGB").resize((max(1, x1 - x0), max(1, y1 - y0)))
                im.paste(pic, (x0, y0))
                continue
            if shp.shape_type == MSO_SHAPE_TYPE.LINE or shp.has_text_frame is False and shp.shape_type is None:
                pass
            if getattr(shp, "has_chart", False) and shp.has_chart:
                d.rectangle((x0, y0, x1, y1), outline=(160, 170, 180), width=2, fill=(248, 250, 252))
                ttl = ""
                try:
                    ttl = shp.chart.chart_title.text_frame.text
                except Exception:
                    pass
                d.text((x0 + 8, y0 + 6), "[native chart] " + ttl, font=font(12, True), fill=(23, 42, 64))
                try:
                    pl = shp.chart.plots[0]
                    cats = list(pl.categories)
                    vals = [list(sr.values) for sr in pl.series]
                    d.text((x0 + 8, y0 + 30), "; ".join(f"{c}: {', '.join(f'{v[i]:.4g}' for v in vals)}" for i, c in enumerate(cats))[:150],
                           font=font(10), fill=(90, 90, 90))
                except Exception:
                    pass
                continue
            if shp.shape_type == MSO_SHAPE_TYPE.TABLE:
                tbl = shp.table
                y = y0
                for r in tbl.rows:
                    rh = px(r.height)
                    need = rh
                    for ci, c in enumerate(r.cells):
                        cw = px(tbl.columns[ci].width)
                        lines = layout_frame(c.text_frame, cw - px(c.margin_left) - px(c.margin_right))
                        need = max(need, int(sum(h for _, h, _ in lines)) + px(c.margin_top) + px(c.margin_bottom))
                    x = x0
                    for ci, c in enumerate(r.cells):
                        cw = px(tbl.columns[ci].width)
                        fill = (255, 255, 255)
                        try:
                            fill = rgb(c.fill.fore_color, (255, 255, 255))
                        except Exception:
                            pass
                        d.rectangle((x, y, x + cw, y + need), fill=fill, outline=(214, 223, 228))
                        cbox = (x, y, x + cw, y + need)

                        class _TF:  # cell text frame with cell margins
                            pass
                        tf = c.text_frame
                        tf_m = _TF()
                        tf_m.paragraphs = tf.paragraphs
                        tf_m.margin_left, tf_m.margin_right = c.margin_left, c.margin_right
                        tf_m.margin_top, tf_m.margin_bottom = c.margin_top, c.margin_bottom
                        tf_m.vertical_anchor = c.vertical_anchor
                        draw_text(d, tf_m, cbox, issues, tag + " cell")
                        x += cw
                    y += need
                if y > px(Emu(int(6.95 * EMU_IN))):
                    issues.append(f"{tag}: table bottom at {y / DPI:.2f} in (footer at 6.95)")
                    d.rectangle((x0, y0, x1, y), outline=(220, 0, 0), width=3)
                continue
            if shp.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE or shp.shape_type == MSO_SHAPE_TYPE.TEXT_BOX:
                try:
                    if shp.fill.type is not None and shp.fill.type == 1:
                        col = rgb(shp.fill.fore_color, (236, 242, 245))
                        if "OVAL" in str(getattr(shp, "auto_shape_type", "")):
                            d.ellipse((x0, y0, x1, y1), fill=col)
                        else:
                            d.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=col)
                except Exception:
                    pass
                if shp.has_text_frame and shp.text_frame.text.strip():
                    draw_text(d, shp.text_frame, (x0, y0, x1, y1), issues, tag)
                continue
            if shp.shape_type == MSO_SHAPE_TYPE.LINE or "Connector" in type(shp).__name__:
                d.line((px(shp.begin_x), px(shp.begin_y), px(shp.end_x), px(shp.end_y)), fill=(120, 120, 120), width=3)
                ex, ey = px(shp.end_x), px(shp.end_y)
                d.ellipse((ex - 5, ey - 5, ex + 5, ey + 5), fill=(120, 120, 120))
        im.save(outdir / f"slide-{si:02d}.png")
        report += issues
    return report


if __name__ == "__main__":
    rep = render(sys.argv[1], sys.argv[2])
    print("preview issues:", "none" if not rep else len(rep))
    for r in rep:
        print("  ", r)
