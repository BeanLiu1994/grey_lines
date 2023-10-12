from grey_lines import canvas
import svgwrite
import math

def save_svg(filename:str, cvs:canvas.canvas, solved, scale: float=1):
    dwg = svgwrite.Drawing(filename)
    # Set the background color
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='black'))

    # # Create the filter element with custom blend mode
    # filter_element = dwg.filter(id='blend', x='0', y='0', width='100%', height='100%')
    # filter_element.feComposite(operator='arithmetic', k1='0', k2='1', k3='1', k4='0', in_='SourceGraphic', in2='BackgroundImage')

    # Add the filter element to the SVG drawing
    # dwg.defs.add(filter_element)

    solved = solved/solved.max()

    lines = cvs.lines()
    for idx, line in enumerate(lines):
        intensity = solved[idx]
        color = svgwrite.rgb(255,255,255, '%')
        # color = svgwrite.rgb(intensity,intensity,intensity, '%')
        dwg.add(dwg.line((line.dot1.x*scale, line.dot1.y*scale), (line.dot2.x*scale, line.dot2.y*scale), stroke=color, opacity=intensity))

    dwg.save()