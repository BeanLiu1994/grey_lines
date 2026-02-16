from grey_lines import canvas
import svgwrite
import math
import numpy as np

def save_svg(filename:str, cvs:canvas.canvas, solved, scale: float=1,
             gamma: float=1.0, line_width: float=1.0, cutoff: float=0.005):
    """
    保存 SVG 线条画。

    参数:
        gamma:  对比度控制。< 1 增强弱线（整体更暗/更清晰），> 1 抑制弱线。
                推荐范围 0.3 ~ 2.0，默认 1.0（线性）。
        line_width: SVG 线条宽度（像素），默认 1.0。密点时可设为 0.5 让线更细。
        cutoff: 低于此强度的线条不渲染，默认 0.005。
    """
    # 计算缩放后的 SVG 实际尺寸
    # 图像区域由 img_corner_lt / img_corner_rb 决定
    lt = cvs.img_corner_lt
    rb = cvs.img_corner_rb
    svg_w = (rb.x - lt.x) * scale
    svg_h = (rb.y - lt.y) * scale
    # 偏移量：将坐标原点移到图像左上角
    ox = lt.x * scale
    oy = lt.y * scale

    dwg = svgwrite.Drawing(
        filename,
        size=(f"{svg_w:.2f}", f"{svg_h:.2f}"),
        viewBox=f"0 0 {svg_w:.2f} {svg_h:.2f}",
    )
    # 黑色背景
    dwg.add(dwg.rect(insert=(0, 0), size=(svg_w, svg_h), fill='black'))

    # 归一化到 [0, 1]
    max_val = solved.max()
    if max_val > 0:
        norm = solved / max_val
    else:
        norm = solved.copy()

    # 应用 gamma 校正: intensity = norm ^ gamma
    # gamma < 1 → 弱线被增强（暗部拉亮），整体更深
    # gamma > 1 → 弱线被抑制，只保留最亮的线
    if gamma != 1.0:
        norm = np.clip(norm, 0, None)
        norm = np.power(norm, gamma)

    lines = cvs.lines()
    for idx, line in enumerate(lines):
        intensity = float(norm[idx])
        if intensity < cutoff:
            continue
        color = svgwrite.rgb(255, 255, 255, '%')
        x1 = line.dot1.x * scale - ox
        y1 = line.dot1.y * scale - oy
        x2 = line.dot2.x * scale - ox
        y2 = line.dot2.y * scale - oy
        dwg.add(dwg.line(
            (x1, y1), (x2, y2),
            stroke=color,
            opacity=min(intensity, 1.0),
            stroke_width=line_width,
        ))

    dwg.save()