from grey_lines import input, canvas, solve, output
import os
import numpy as np

if __name__ == "__main__":
    sz, data, scale = input.load_image(os.path.expanduser("~/avatar.png"))

    print(sz)
    # cvs = canvas.canvas.rectangle_canvas(sz[0], sz[1], 17, 11)
    cvs = canvas.canvas.circle_canvas(sz[0], sz[1], 79)
    print(cvs.edge_dots_cnt())
    print(cvs.lines_cnt())
    print(cvs.canvas_pixel_cnt())

    assert(cvs.canvas_pixel_cnt() == sz[0] * sz[1])

    sv = solve.solver(cvs)
    result = sv.solve(data)
    print(result)
    
    output.save_svg("out.svg", cvs, result, 1/scale)
