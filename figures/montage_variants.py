"""把图5-2 的候选版本拼成一张对比大图。

布局：2×2 图片网格，每张图正下方以灰色小字标注。
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import output_dir

SRC = output_dir() / "fig5_2_variants"

ITEMS = [
    ("fig5_2_v1_current.png",
     "(a) von Mises 应力云图 + 主应力方向场（真实 ANSYS 数据）"),
    ("fig5_2_v4_topview.png", "(b) XY 平面应力分布 + 方向箭头"),
    ("fig5_2_v5_template.png", "(c) 模板库 test1 的应力场"),
    ("fig5_2_v6_mesh_surface.png", "(d) 零件1 网格表面映射应力（FEM 风格）"),
]

FONT_TITLE = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_NOTE = r"C:\Windows\Fonts\msyh.ttc"


def main():
    images = []
    for fname, _caption in ITEMS:
        path = SRC / fname
        if not path.exists():
            raise FileNotFoundError(f"缺少候选图：{path}")
        images.append(Image.open(path).convert("RGB"))

    cell_w, max_h = 760, 620
    label_h = 72
    margin, gap, title_h = 48, 36, 100

    scaled = []
    for im in images:
        r = min(cell_w / im.width, max_h / im.height)
        scaled.append(im.resize(
            (max(1, int(im.width * r)), max(1, int(im.height * r))),
            Image.LANCZOS,
        ))
    cell_w_real = max(s.width for s in scaled)
    cell_h_real = max(s.height for s in scaled) + label_h

    cols = rows = 2
    W = margin + cols * cell_w_real + (cols - 1) * gap + margin
    H = title_h + margin + rows * cell_h_real + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = ImageFont.truetype(FONT_TITLE, 40)
    font_note = ImageFont.truetype(FONT_NOTE, 22)

    draw.text((W / 2, 46), "图5-2 应力场可视化 · 候选版本对比",
              font=font_title, fill="black", anchor="mm")

    for k, im in enumerate(scaled):
        r, c = divmod(k, cols)
        x = margin + c * (cell_w_real + gap)
        y = title_h + margin + r * (cell_h_real + gap)
        ox = x + (cell_w_real - im.width) // 2
        oy = y + (cell_h_real - label_h - im.height) // 2
        canvas.paste(im, (ox, oy))

        # 正下方灰色小字标注
        caption = ITEMS[k][1]
        cx = x + cell_w_real / 2
        draw.text((cx, y + cell_h_real - label_h + 34), caption,
                  font=font_note, fill="#555555", anchor="mm")

    out = SRC / "fig5_2_all_variants.png"
    canvas.save(out, dpi=(300, 300))
    print("对比大图已生成:", out, f"({W}x{H})")


if __name__ == "__main__":
    main()
