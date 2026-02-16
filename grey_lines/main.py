from grey_lines import input, canvas, solve, output
from grey_lines.gui import recommend_gamma_linewidth
import os
import numpy as np
import click

@click.command()
@click.option("-c", "--canvas_type", default="rect", show_default=True, help="canvas type, circle or rect")
@click.option("-d", "--density", type=float, default=4.0, show_default=True, help="may be extremely slow if too large")
@click.option("-f", "--fixed", type=int, default=0, help="fixed size (width)")
@click.option("-s", "--solver_method", default="direct", show_default=True,
              type=click.Choice(["direct", "lsmr"]), help="solver method: direct (precise) or lsmr (faster)")
@click.option("-g", "--gamma", type=float, default=0, show_default=True,
              help="gamma correction for contrast. <1 boosts weak lines, >1 suppresses. 0=auto")
@click.option("-w", "--line_width", type=float, default=0, show_default=True,
              help="SVG stroke width in pixels. 0=auto")
@click.argument("path")
@click.argument("output_path")
def cli(canvas_type, path, density, fixed, solver_method, gamma, line_width, output_path):
    max_w = 300
    sz, data, scale = input.load_image(os.path.expanduser(path), max_w)
    if canvas_type == "circle":
        cvs = canvas.canvas.circle_canvas(sz[0], sz[1], int(density * 30) + 1)
    else:
        cvs = canvas.canvas.rectangle_canvas(sz[0], sz[1], int(density * 7) + 1, int(density * 7) + 1)
    
    assert(cvs.canvas_pixel_cnt() == sz[0] * sz[1])

    if cvs.lines_cnt() > 10000:
        click.echo("warning: calculation may be extremely slow", err=True)

    click.echo(f"solving problem (method={solver_method})...")
    if solver_method == "lsmr":
        sv = solve.solver_lsmr(cvs)
    else:
        sv = solve.solver(cvs)
    result = sv.solve(data)
    
    output_path = os.path.expanduser(output_path)
    output_path = process_path(output_path)
    inv_scale = 1 / scale
    if fixed:
        inv_scale = fixed / max_w
    if gamma <= 0 or line_width <= 0:
        n_dots = cvs.edge_dots_cnt()
        auto_gamma, auto_lw = recommend_gamma_linewidth(n_dots, sz[0], sz[1])
        if gamma <= 0:
            gamma = auto_gamma
        if line_width <= 0:
            line_width = auto_lw
        click.echo(f"auto params: gamma={gamma}, line_width={line_width}")

    output.save_svg(output_path, cvs, result, inv_scale, gamma=gamma, line_width=line_width)
    click.echo(f"saved to {output_path}")


def process_path(path):
    if os.path.isdir(path):
        output_path = os.path.join(path, "output.svg")
    else:
        base_path, ext = os.path.splitext(path)
        if ext.lower() != ".svg":
            output_path = base_path + ".svg"
        else:
            output_path = path
    return output_path


if __name__ == "__main__":
    cli()
