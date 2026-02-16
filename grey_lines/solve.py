from grey_lines.canvas import canvas
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, save_npz, load_npz, eye
from scipy.sparse.linalg import spsolve, lsmr
import cvxpy as cp
import os
import shutil
from tqdm import tqdm
import math

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
        # 缓存 key 包含 'v2' 标记以区分归一化后的矩阵
        file = os.path.expanduser(os.path.join(config.cache_dir, f"mat_v2_{self._canvas.hash()}.npz"))
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
        base_x = math.ceil(self._canvas.img_corner_lt.x)
        base_y = math.ceil(self._canvas.img_corner_lt.y)
        xcnt = (math.floor(self._canvas.img_corner_rb.x) - base_x + 1)
        rows = []
        cols = []
        values = []
        for ln_index, ln in enumerate(tqdm(lines)):
            px_indices, weights = ln.band_dots_fast(
                config.dispersion, self._canvas.img_corner_lt, self._canvas.img_corner_rb,
                base_x, base_y, xcnt
            )
            if len(px_indices) > 0:
                rows.append(px_indices)
                cols.append(np.full(len(px_indices), ln_index, dtype=np.int64))
                values.append(weights)
        if rows:
            all_rows = np.concatenate(rows)
            all_cols = np.concatenate(cols)
            all_values = np.concatenate(values)
        else:
            all_rows = np.array([], dtype=np.int64)
            all_cols = np.array([], dtype=np.int64)
            all_values = np.array([], dtype=np.float64)
        mat = coo_matrix((all_values, (all_rows, all_cols)), shape=(self._canvas.canvas_pixel_cnt(), self._canvas.lines_cnt()))
        # 行归一化：每行除以该行权重和，使得每个像素的模型变成加权平均
        # 这样不管有多少线穿过一个像素，对目标亮度的尺度是一致的
        mat_csr = mat.tocsr()
        row_sums = np.array(mat_csr.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # 避免除零
        inv_row_sums = 1.0 / row_sums
        D_inv = diags(inv_row_sums)
        mat_normalized = D_inv @ mat_csr
        return mat_normalized

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


class solver_lsmr:
    """使用 LSMR 迭代求解器，跳过 A^T A 计算，适合大规模问题。"""
    def __init__(self, _canvas: canvas, config: solver_config = None):
        self._canvas = _canvas
        self._config = config or default_solver_config
        self.relation_matrix = self._get_relation_matrix_from_cache()

    def _get_relation_matrix_from_cache(self):
        file = os.path.expanduser(
            os.path.join(self._config.cache_dir, f"mat_v2_{self._canvas.hash()}.npz"))
        if os.path.isfile(file):
            print(f"use cache file {file}")
            return load_npz(file)
        mat = solver.get_relation_matrix(
            type('_', (), {'_canvas': self._canvas})(), self._config)
        if os.path.isdir(self._config.cache_dir):
            shutil.rmtree(self._config.cache_dir)
        os.mkdir(self._config.cache_dir)
        save_npz(file, mat)
        return mat

    def solve(self, input_data):
        if self._canvas.canvas_pixel_cnt() != len(input_data):
            raise RuntimeError("input size might mismatch")
        A = self.relation_matrix.tocsc()
        b = np.array(input_data)
        result = lsmr(A, b, damp=math.sqrt(self._config.normalizer),
                       atol=1e-5, btol=1e-5, maxiter=500)
        return result[0]
