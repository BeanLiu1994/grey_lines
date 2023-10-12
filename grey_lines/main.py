from grey_lines import input, canvas, solve, output
import os
import numpy as np
import click

@click.command()
@click.option("-c", "--canvas_type", help="canvas type, circle or rect")
@click.option("-d", "--density", type=float, default=4.0, help="may be extremely slow if too large")
@click.option("-f", "--fixed", type=int, default=0, help="fixed size (width)")
@click.argument("path")
@click.argument("output_path")
def cli(canvas_type, path, density, fixed, output_path):
    max_w = 300
    sz, data, scale = input.load_image(os.path.expanduser(path), max_w)
    if canvas_type == "circle":
        cvs = canvas.canvas.circle_canvas(sz[0], sz[1], int(density * 30) + 1)
    else:
        cvs = canvas.canvas.rectangle_canvas(sz[0], sz[1], int(density * 7) + 1, int(density * 7) + 1)
    
    assert(cvs.canvas_pixel_cnt() == sz[0] * sz[1])

    if cvs.lines_cnt() > 10000:
        click.echo("warning: calculation may be extremely slow", err=True)

    click.echo(f"solving problem...")
    sv = solve.solver(cvs)
    result = sv.solve(data)
    
    output_path = os.path.expanduser(output_path)
    output_path = process_path(output_path)
    if fixed:
        inv_scale = fixed / max_w
    output.save_svg(output_path, cvs, result, inv_scale)
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
