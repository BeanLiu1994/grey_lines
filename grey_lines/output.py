from grey_lines import canvas
import svgwrite

def save_svg(filename:str, cvs:canvas.canvas, solved):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    # Set the background color
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='black'))

    lines = cvs.lines()
    for idx, line in enumerate(lines):
        intensity = solved[idx]
        dwg.add(dwg.line((line.dot1.x, line.dot1.y), (line.dot2.x, line.dot2.y), stroke=svgwrite.rgb(intensity,intensity,intensity, '%')))

    dwg.save()