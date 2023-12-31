from grey_lines.canvas import canvas
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz, eye
from scipy.sparse.linalg import spsolve
import cvxpy as cp
import os
import shutil
from tqdm import tqdm

@dataclass
class solver_config:
    dispersion: float = 3.0
    normalizer: float = 0.1
    cache_dir: str = "./cache/"


default_solver_config = solver_config()


class solver:
    def __init__(self, _canvas: canvas):
        self._canvas = _canvas
        self.prepare_canvas(config=default_solver_config)

    def get_relation_matrix_from_cache(self, config:solver_config):
        file = os.path.expanduser(os.path.join(config.cache_dir, f"mat_{self._canvas.hash()}.npz"))
        if os.path.isfile(file):
            print(f"use cache file {file}")
            return load_npz(file)
        mat = self.get_relation_matrix(config)
        if os.path.isdir(config.cache_dir):
            shutil.rmtree(config.cache_dir)
        os.mkdir(config.cache_dir)
        save_npz(file, mat)
        return mat

    def get_relation_matrix(self, config:solver_config):
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
        return coo_matrix((values, (rows, cols)), shape=(self._canvas.canvas_pixel_cnt(), self._canvas.lines_cnt()))

    def prepare_canvas(self, config:solver_config):
        self.relation_matrix = self.get_relation_matrix_from_cache(config)
        self.relation_matrix_T = self.relation_matrix.transpose()
        self.normal_matrix = self.relation_matrix_T @ self.relation_matrix + eye(self._canvas.lines_cnt()) * config.normalizer

    def solve(self, input):
        if self._canvas.canvas_pixel_cnt() != len(input):
            raise RuntimeError("input size might mismatch")
        b = self.relation_matrix_T @ np.array(input)
        coeff = spsolve(self.normal_matrix, b)
        return coeff
    

class solver2(solver):
    def __init__(self, _canvas: canvas):
        super().__init__(_canvas)

    def solve(self, input):
        if self._canvas.canvas_pixel_cnt() != len(input):
            raise RuntimeError("input size might mismatch")
        A = self.normal_matrix
        b = self.relation_matrix_T @ np.array(input)
        # Define the variable x
        x = cp.Variable(A.shape[1], nonneg=True)
        # Define the objective function and the problem
        objective = cp.Minimize(cp.norm(A @ x - b, 2))
        constraints = [x >= 0]
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve(solver=cp.SCS)
        return x.value
