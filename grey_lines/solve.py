from grey_lines.canvas import canvas
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

@dataclass
class solver_config:
    dispersion: float = 3.0
    normalizer: float = 0.001


default_solver_config = solver_config()


class solver:
    def __init__(self, _canvas: canvas):
        self._canvas = _canvas
        self.prepare_canvas(config=default_solver_config)

    def prepare_canvas(self, config:solver_config):
        if self._canvas.edge_dots_cnt() < 3:
            raise RuntimeError("should have more than 3 edge dots")
        lines = self._canvas.lines()
        rows = []
        cols = []
        values = []
        for ln_index, ln in enumerate(tqdm(lines)):
            dots = ln.band_dots(config.dispersion, self._canvas.img_corner_lt, self._canvas.img_corner_rb)
            for dot in dots:
                px_index = self._canvas.canvas_pixel_dot_index(dot[0])
                rows.append(px_index)
                cols.append(ln_index)
                values.append(1/(1+dot[1]))
        self.relation_matrix = coo_matrix((values, (rows, cols)), shape=(self._canvas.canvas_pixel_cnt(), self._canvas.lines_cnt()))
        self.relation_matrix_T = self.relation_matrix.transpose()
        self.normal_matrix = self.relation_matrix_T @ self.relation_matrix + np.eye(self._canvas.lines_cnt()) * config.normalizer

    def solve(self, input):
        if self._canvas.canvas_pixel_cnt() != len(input):
            raise RuntimeError("input size might mismatch")
        b = self.relation_matrix_T @ np.array(input)
        coeff = spsolve(self.normal_matrix, b)
        return coeff