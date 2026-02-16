"""
简单的 PyQt5 GUI，用于 grey_lines 线条画生成器。
支持交互式编辑画布边缘固定点（edge_dots）。
"""

import os
import sys
import math
import multiprocessing as mp
import pickle

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QRadioButton,
    QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox, QProgressBar,
    QButtonGroup, QSizePolicy, QComboBox, QShortcut, QToolButton,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer

from PIL import Image
import numpy as np

from grey_lines import input as img_input, canvas, solve, output
from grey_lines.canvas import dot, canvas as Canvas


# ------------------------------------------------------------------ 校验工具
def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain 算法求凸包，返回逆时针排列的顶点。"""
    pts = sorted(points)
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_convex_hull(hull: list[tuple[float, float]], px: float, py: float) -> bool:
    """判断点 (px, py) 是否在凸包内部（含边界）。"""
    n = len(hull)
    if n < 3:
        return False
    for i in range(n):
        j = (i + 1) % n
        # 逆时针排列，叉积 >= 0 表示在左侧或边上
        if _cross(hull[i], hull[j], (px, py)) < -1e-9:
            return False
    return True


def _polygon_area(hull: list[tuple[float, float]]) -> float:
    """Shoelace 公式计算多边形面积。"""
    n = len(hull)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _clip_polygon_to_rect(hull, x_min, y_min, x_max, y_max):
    """Sutherland-Hodgman 算法裁剪多边形到矩形，返回交集多边形顶点列表。"""
    output_list = list(hull)
    edges = [
        (lambda p, b=x_min: p[0] >= b, lambda p1, p2, b=x_min: _intersect_x(p1, p2, b)),
        (lambda p, b=x_max: p[0] <= b, lambda p1, p2, b=x_max: _intersect_x(p1, p2, b)),
        (lambda p, b=y_min: p[1] >= b, lambda p1, p2, b=y_min: _intersect_y(p1, p2, b)),
        (lambda p, b=y_max: p[1] <= b, lambda p1, p2, b=y_max: _intersect_y(p1, p2, b)),
    ]
    for inside_fn, intersect_fn in edges:
        if not output_list:
            break
        input_list = output_list
        output_list = []
        for i in range(len(input_list)):
            current = input_list[i]
            prev = input_list[i - 1]
            if inside_fn(current):
                if not inside_fn(prev):
                    output_list.append(intersect_fn(prev, current))
                output_list.append(current)
            elif inside_fn(prev):
                output_list.append(intersect_fn(prev, current))
    return output_list


def _intersect_x(p1, p2, x):
    if abs(p2[0] - p1[0]) < 1e-12:
        return (x, p1[1])
    t = (x - p1[0]) / (p2[0] - p1[0])
    return (x, p1[1] + t * (p2[1] - p1[1]))


def _intersect_y(p1, p2, y):
    if abs(p2[1] - p1[1]) < 1e-12:
        return (p1[0], y)
    t = (y - p1[1]) / (p2[1] - p1[1])
    return (p1[0] + t * (p2[0] - p1[0]), y)


def validate_dots(edge_dots: list[dot], img_lt: dot, img_rb: dot) -> list[str]:
    """
    校验 edge_dots 的质量，返回警告消息列表（空列表 = 全部通过）。
    检查项目：
      1. 点数过少（< 3 无法形成有效面积）
      2. 点的凸包与图片区域无重叠
      3. 凸包对图片区域的覆盖率过低
      4. 所有点几乎共线
    """
    warnings = []
    n = len(edge_dots)

    # --- 1. 点数 ---
    if n < 2:
        warnings.append(f"点数过少（{n}），至少需要 2 个点才能生成线条。")
        return warnings  # 后续检查无意义
    if n < 3:
        warnings.append(f"只有 {n} 个点，无法形成面积，生成效果会很差。")

    # --- 2. 共线检测 ---
    pts = [(d.x, d.y) for d in edge_dots]
    hull = _convex_hull(pts)
    hull_area = _polygon_area(hull)
    if len(hull) < 3 or hull_area < 1.0:
        warnings.append("所有点几乎共线，线条将无法覆盖图片区域。")
        return warnings

    # --- 3. 凸包与图片区域重叠检测 ---
    img_x_min, img_y_min = img_lt.x, img_lt.y
    img_x_max, img_y_max = img_rb.x, img_rb.y
    img_area = (img_x_max - img_x_min) * (img_y_max - img_y_min)
    if img_area <= 0:
        return warnings

    clipped = _clip_polygon_to_rect(hull, img_x_min, img_y_min, img_x_max, img_y_max)
    overlap_area = _polygon_area(clipped) if len(clipped) >= 3 else 0.0
    coverage = overlap_area / img_area if img_area > 0 else 0.0

    if coverage < 0.01:
        warnings.append(
            "边缘点的凸包几乎不覆盖图片区域，线条将无法穿过图片。\n"
            "请调整点的位置使其围绕或跨越图片区域。"
        )
    elif coverage < 0.3:
        warnings.append(
            f"边缘点对图片区域的覆盖率仅 {coverage:.0%}，\n"
            f"大部分图片区域不会有线条经过，建议添加更多点或调整位置。"
        )
    elif coverage < 0.6:
        warnings.append(
            f"边缘点对图片区域的覆盖率为 {coverage:.0%}，部分区域可能缺少线条。"
        )

    # --- 4. 图片四角是否都在凸包内 ---
    corners = [
        (img_x_min, img_y_min), (img_x_max, img_y_min),
        (img_x_min, img_y_max), (img_x_max, img_y_max),
    ]
    corners_inside = sum(1 for c in corners if _point_in_convex_hull(hull, c[0], c[1]))
    if corners_inside == 0 and coverage > 0.01:
        warnings.append("图片的四个角均不在边缘点围成的区域内，效果可能不理想。")
    elif corners_inside < 4 and coverage >= 0.3:
        warnings.append(
            f"图片的 {4 - corners_inside} 个角不在边缘点围成的区域内，"
            f"这些角落的线条覆盖会较弱。"
        )

    return warnings


# ------------------------------------------------------------------ 自动参数推荐
def recommend_gamma_linewidth(n_dots: int, img_w: int, img_h: int) -> tuple[float, float]:
    """
    根据点数和图片尺寸自动推荐 gamma 和 linewidth。

    原理：
      - 线条数 = n*(n-1)/2
      - 像素数 = w*h
      - 线条密度 = lines / pixels
      - 密度越高，弱线噪声越多，需要更高的 gamma 来抑制弱线、
        突出主要线条，同时需要更细的线宽避免过度重叠。
    """
    if n_dots < 2 or img_w <= 0 or img_h <= 0:
        return 1.0, 1.0

    lines_cnt = n_dots * (n_dots - 1) / 2
    pixel_cnt = img_w * img_h
    density = lines_cnt / pixel_cnt  # 线条/像素

    # 基准密度：约 30 个点在 300x200 图上 ≈ 0.007
    ref_density = 0.007
    ratio = density / ref_density  # > 1 说明比基准更密

    # gamma: 密度越高越大，抑制弱线噪声，让主要线条更突出
    # ratio=1 → gamma=0.5;  ratio=10 → gamma=1.0;  ratio=100 → gamma=1.5
    if ratio > 0:
        gamma = 0.5 + 0.5 * math.log10(max(ratio, 1.0))
    else:
        gamma = 0.5
    gamma = max(0.3, min(gamma, 2.0))

    # linewidth: 密度越高越细
    if ratio > 1:
        lw = 1.0 / math.sqrt(ratio)
    else:
        lw = 1.0
    lw = max(0.3, min(lw, 2.0))

    # 四舍五入到一位小数，方便界面显示
    gamma = round(gamma, 1)
    lw = round(lw, 1)

    return gamma, lw


# ------------------------------------------------------------------ 子进程求解函数
def _solve_in_process(path, edge_dots, img_corner_lt, img_corner_rb, fixed_width, solver_method, result_queue):
    """在独立进程中执行求解，通过 queue 返回结果，完全规避 GIL。"""
    try:
        max_w = 300
        sz, data, scale = img_input.load_image(os.path.expanduser(path), max_w)

        cvs = Canvas(
            edge_dots=list(edge_dots),
            img_corner_lt=img_corner_lt,
            img_corner_rb=img_corner_rb,
        )

        assert cvs.canvas_pixel_cnt() == sz[0] * sz[1], \
            f"像素数不匹配: canvas={cvs.canvas_pixel_cnt()}, image={sz[0]*sz[1]}"

        if solver_method == "lsmr":
            sv = solve.solver_lsmr(cvs)
        else:
            sv = solve.solver(cvs)
        result = sv.solve(data)

        inv_scale = 1 / scale
        if fixed_width:
            inv_scale = fixed_width / max_w

        result_queue.put(("ok", cvs, result, inv_scale))
    except Exception as e:
        result_queue.put(("error", str(e)))


# ------------------------------------------------------------------ 后台求解轮询器
class SolveWorker(QThread):
    """
    在独立 **进程** 中运行耗时求解，QThread 仅用来轮询结果队列，
    从而完全避免 GIL 导致的 UI 卡顿。
    """
    finished = pyqtSignal(object, object, float)  # (cvs, result, inv_scale)
    error = pyqtSignal(str)

    def __init__(self, path, edge_dots, img_corner_lt, img_corner_rb, fixed_width, solver_method="direct"):
        super().__init__()
        self.path = path
        self.edge_dots = edge_dots
        self.img_corner_lt = img_corner_lt
        self.img_corner_rb = img_corner_rb
        self.fixed_width = fixed_width
        self.solver_method = solver_method

    def run(self):
        result_queue = mp.Queue()
        proc = mp.Process(
            target=_solve_in_process,
            args=(self.path, self.edge_dots, self.img_corner_lt,
                  self.img_corner_rb, self.fixed_width, self.solver_method, result_queue),
            daemon=True,
        )
        proc.start()

        # 轮询队列，间隔 100ms，不占 GIL
        while True:
            proc.join(timeout=0.1)
            if not result_queue.empty():
                break
            if not proc.is_alive():
                break

        try:
            msg = result_queue.get_nowait()
        except Exception:
            self.error.emit("求解进程异常退出")
            return

        if msg[0] == "ok":
            self.finished.emit(msg[1], msg[2], msg[3])
        else:
            self.error.emit(msg[1])


# ------------------------------------------------------------------ 点编辑器 Widget
class DotEditorWidget(QWidget):
    """
    可交互的点编辑器：
    - 显示画布区域（矩形/圆形边界）和所有 edge_dots
    - 左键点击空白处 → 添加新点
    - 右键点击已有点 → 删除该点
    """
    dots_changed = pyqtSignal()  # 点列表发生变化时发出

    DOT_RADIUS = 5          # 屏幕像素半径
    HIT_RADIUS = 8          # 点击命中半径
    PADDING = 20            # 画布四周留白

    # 工具模式常量
    TOOL_POINT = "point"   # 默认：左键添加点，右键删除点
    TOOL_LINE = "line"     # 画线模式：画一条线段，沿线分布点

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # 接收键盘事件

        # 画布信息（真实坐标系）
        self._edge_dots: list[dot] = []
        self._img_corner_lt: dot = None
        self._img_corner_rb: dot = None
        # 画布外边界（用于绘制模板轮廓）
        self._boundary_pts: list[tuple[float, float]] = []  # 顺序连接
        self._boundary_type = "rect"  # "rect" | "circle"
        self._canvas_w = 0
        self._canvas_h = 0

        # 显示变换参数（延迟计算）
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        # 背景图片（原图预览）
        self._bg_image: QImage = None

        # 悬停高亮
        self._hover_idx = -1

        # 校验缓存（避免 paintEvent 中重复计算）
        self._cached_warnings: list[str] = []
        self._warnings_dirty = True

        # ---- Undo / Redo 栈 ----
        self._undo_stack: list[list[dot]] = []   # 之前的快照
        self._redo_stack: list[list[dot]] = []   # 被 undo 的快照
        self._MAX_UNDO = 200

        # ---- 工具模式 ----
        self._tool_mode = self.TOOL_POINT
        self._line_point_count = 5  # 画线模式：沿线段生成的点数

        # 画线模式的临时状态
        self._line_start: tuple[float, float] | None = None   # 画布坐标
        self._line_end_screen: tuple[float, float] | None = None  # 屏幕坐标（拖拽中）
        self._line_dragging = False

    # ----- 公共接口 -----

    def set_canvas_rect(self, w, h):
        """设置矩形画布尺寸（图片像素大小）。"""
        self._canvas_w = w
        self._canvas_h = h

    def set_background_image(self, qimg: QImage):
        """设置背景图片（原图预览），将绘制在画布图像区域内。"""
        self._bg_image = qimg
        self.update()

    def set_dots(self, dots_list: list[dot], *, record_undo=True):
        if record_undo:
            self._push_undo()
        self._edge_dots = list(dots_list)
        self._warnings_dirty = True
        self.dots_changed.emit()
        self.update()

    def get_dots(self) -> list[dot]:
        return list(self._edge_dots)

    # ----- 工具模式 -----

    def set_tool_mode(self, mode: str):
        """切换工具模式 (TOOL_POINT / TOOL_LINE)"""
        self._tool_mode = mode
        # 取消画线中间状态
        self._line_start = None
        self._line_end_screen = None
        self._line_dragging = False
        self.update()

    def set_line_point_count(self, count: int):
        """设置画线模式下沿线段生成的点数。"""
        self._line_point_count = max(2, count)

    # ----- Undo / Redo -----

    def _push_undo(self):
        """将当前 _edge_dots 快照压入 undo 栈，清空 redo 栈。"""
        snapshot = [dot(d.x, d.y) for d in self._edge_dots]
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        # 当前状态压入 redo
        self._redo_stack.append([dot(d.x, d.y) for d in self._edge_dots])
        # 恢复上一个快照
        prev = self._undo_stack.pop()
        self._edge_dots = prev
        self._warnings_dirty = True
        self.dots_changed.emit()
        self.update()

    def redo(self):
        if not self._redo_stack:
            return
        # 当前状态压入 undo
        self._undo_stack.append([dot(d.x, d.y) for d in self._edge_dots])
        # 恢复 redo 快照
        nxt = self._redo_stack.pop()
        self._edge_dots = nxt
        self._warnings_dirty = True
        self.dots_changed.emit()
        self.update()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def get_img_corners(self):
        return self._img_corner_lt, self._img_corner_rb

    def load_rect_template(self, w, h, density, jitter=0.0):
        """加载矩形模板点。jitter: 随机抖动幅度 (0~1)"""
        self._push_undo()
        split_w = int(density * 7) + 1
        split_h = int(density * 7) + 1
        cvs = Canvas.rectangle_canvas(w, h, split_w, split_h, jitter=jitter)
        self._edge_dots = list(cvs.edge_dots)
        self._img_corner_lt = cvs.img_corner_lt
        self._img_corner_rb = cvs.img_corner_rb
        self._boundary_type = "rect"
        self._canvas_w = w
        self._canvas_h = h
        self._warnings_dirty = True
        self.dots_changed.emit()
        self.update()

    def load_circle_template(self, w, h, density, jitter=0.0):
        """加载圆形模板点。jitter: 随机抖动幅度 (0~1)"""
        self._push_undo()
        split_cnt = int(density * 30) + 1
        cvs = Canvas.circle_canvas(w, h, split_cnt, jitter=jitter)
        self._edge_dots = list(cvs.edge_dots)
        self._img_corner_lt = cvs.img_corner_lt
        self._img_corner_rb = cvs.img_corner_rb
        self._boundary_type = "circle"
        self._canvas_w = w
        self._canvas_h = h
        self._warnings_dirty = True
        self.dots_changed.emit()
        self.update()

    # ----- 坐标变换 -----

    def _update_transform(self):
        """根据当前控件大小和画布范围计算缩放/偏移。"""
        if not self._edge_dots and self._canvas_w == 0:
            self._scale = 1.0
            self._offset_x = self.PADDING
            self._offset_y = self.PADDING
            return

        # 确定画布坐标范围
        if self._edge_dots:
            xs = [d.x for d in self._edge_dots]
            ys = [d.y for d in self._edge_dots]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            # 额外包含 img_corner
            if self._img_corner_lt:
                min_x = min(min_x, self._img_corner_lt.x)
                min_y = min(min_y, self._img_corner_lt.y)
            if self._img_corner_rb:
                max_x = max(max_x, self._img_corner_rb.x)
                max_y = max(max_y, self._img_corner_rb.y)
        else:
            min_x, min_y = 0, 0
            max_x = self._canvas_w or 100
            max_y = self._canvas_h or 100

        span_x = max_x - min_x or 1
        span_y = max_y - min_y or 1

        avail_w = self.width() - 2 * self.PADDING
        avail_h = self.height() - 2 * self.PADDING
        if avail_w <= 0 or avail_h <= 0:
            self._scale = 1.0
            self._offset_x = self.PADDING
            self._offset_y = self.PADDING
            return

        self._scale = min(avail_w / span_x, avail_h / span_y)
        # 居中
        self._offset_x = self.PADDING + (avail_w - span_x * self._scale) / 2 - min_x * self._scale
        self._offset_y = self.PADDING + (avail_h - span_y * self._scale) / 2 - min_y * self._scale

    def _to_screen(self, cx, cy):
        """画布坐标 → 屏幕坐标"""
        return cx * self._scale + self._offset_x, cy * self._scale + self._offset_y

    def _to_canvas(self, sx, sy):
        """屏幕坐标 → 画布坐标"""
        cx = (sx - self._offset_x) / self._scale
        cy = (sy - self._offset_y) / self._scale
        return cx, cy

    # ----- 命中检测 -----

    def _hit_test(self, sx, sy):
        """返回距离屏幕坐标 (sx,sy) 最近的点索引，若超出 HIT_RADIUS 返回 -1。"""
        best_idx = -1
        best_dist = self.HIT_RADIUS + 1
        for i, d in enumerate(self._edge_dots):
            dx_s, dy_s = self._to_screen(d.x, d.y)
            dist = math.hypot(sx - dx_s, sy - dy_s)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    # ----- Qt 事件 -----

    # ----- 键盘快捷键 -----

    def keyPressEvent(self, event):
        # Ctrl+Z → undo,  Ctrl+Shift+Z / Ctrl+Y → redo
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Z:
                if event.modifiers() & Qt.ShiftModifier:
                    self.redo()
                else:
                    self.undo()
                return
            if event.key() == Qt.Key_Y:
                self.redo()
                return
        # Esc → 取消画线中间状态
        if event.key() == Qt.Key_Escape and self._tool_mode == self.TOOL_LINE:
            self._line_start = None
            self._line_end_screen = None
            self._line_dragging = False
            self.update()
            return
        super().keyPressEvent(event)

    # ----- Qt 鼠标事件 -----

    def mousePressEvent(self, event):
        sx, sy = event.x(), event.y()

        if self._tool_mode == self.TOOL_LINE:
            self._mousePressLine(event, sx, sy)
        else:
            self._mousePressPoint(event, sx, sy)

    def _mousePressPoint(self, event, sx, sy):
        """点模式：左键添加，右键删除"""
        if event.button() == Qt.LeftButton:
            idx = self._hit_test(sx, sy)
            if idx >= 0:
                return
            self._push_undo()
            cx, cy = self._to_canvas(sx, sy)
            self._edge_dots.append(dot(cx, cy))
            self._warnings_dirty = True
            self.dots_changed.emit()
            self.update()
        elif event.button() == Qt.RightButton:
            idx = self._hit_test(sx, sy)
            if idx >= 0:
                self._push_undo()
                del self._edge_dots[idx]
                self._hover_idx = -1
                self._warnings_dirty = True
                self.dots_changed.emit()
                self.update()

    def _mousePressLine(self, event, sx, sy):
        """画线模式：左键第一次点击设置起点，拖拽到终点释放生成点；右键取消"""
        if event.button() == Qt.RightButton:
            # 取消画线
            self._line_start = None
            self._line_end_screen = None
            self._line_dragging = False
            self.update()
            return
        if event.button() == Qt.LeftButton:
            cx, cy = self._to_canvas(sx, sy)
            self._line_start = (cx, cy)
            self._line_end_screen = (sx, sy)
            self._line_dragging = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._tool_mode == self.TOOL_LINE and self._line_dragging:
            self._line_end_screen = (event.x(), event.y())
            self.update()
            return
        old = self._hover_idx
        self._hover_idx = self._hit_test(event.x(), event.y())
        if old != self._hover_idx:
            self.update()

    def mouseReleaseEvent(self, event):
        if self._tool_mode == self.TOOL_LINE and self._line_dragging and event.button() == Qt.LeftButton:
            sx, sy = event.x(), event.y()
            cx_end, cy_end = self._to_canvas(sx, sy)
            cx_start, cy_start = self._line_start

            # 线段太短则忽略
            dist_screen = math.hypot(sx - self._to_screen(cx_start, cy_start)[0],
                                     sy - self._to_screen(cx_start, cy_start)[1])
            self._line_dragging = False
            self._line_start = None
            self._line_end_screen = None

            if dist_screen < 5:
                self.update()
                return

            # 沿线段均匀分布点
            n = max(2, self._line_point_count)
            self._push_undo()
            for i in range(n):
                t = i / (n - 1)
                px = cx_start + t * (cx_end - cx_start)
                py = cy_start + t * (cy_end - cy_start)
                self._edge_dots.append(dot(px, py))
            self._warnings_dirty = True
            self.dots_changed.emit()
            self.update()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self._hover_idx >= 0:
            self._hover_idx = -1
            self.update()

    def paintEvent(self, event):
        self._update_transform()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # 背景
        p.fillRect(self.rect(), QColor(30, 30, 30))

        # 绘制画布图像区域
        if self._img_corner_lt and self._img_corner_rb:
            lt_sx, lt_sy = self._to_screen(self._img_corner_lt.x, self._img_corner_lt.y)
            rb_sx, rb_sy = self._to_screen(self._img_corner_rb.x, self._img_corner_rb.y)
            img_rect = QRectF(lt_sx, lt_sy, rb_sx - lt_sx, rb_sy - lt_sy)

            # 绘制背景图片（原图预览）
            if self._bg_image is not None:
                p.setOpacity(0.45)
                p.drawImage(img_rect, self._bg_image)
                p.setOpacity(1.0)
            else:
                p.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
                p.setBrush(QBrush(QColor(50, 50, 50, 80)))
                p.drawRect(img_rect)

            # 图像区域边框
            p.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(img_rect)

        # 绘制圆形轮廓（如果是圆形模板）
        if self._boundary_type == "circle" and self._edge_dots:
            p.setPen(QPen(QColor(100, 100, 100), 1, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            # 通过边缘点推算圆心和半径
            xs = [d.x for d in self._edge_dots]
            ys = [d.y for d in self._edge_dots]
            cx_center = (min(xs) + max(xs)) / 2
            cy_center = (min(ys) + max(ys)) / 2
            radius = max(max(xs) - min(xs), max(ys) - min(ys)) / 2
            sx_c, sy_c = self._to_screen(cx_center, cy_center)
            r_screen = radius * self._scale
            p.drawEllipse(QPointF(sx_c, sy_c), r_screen, r_screen)

        # 绘制线条预览（淡线） — 点数多时只画相邻点连线，避免 O(n²) 卡顿
        n_dots = len(self._edge_dots)
        if n_dots > 1:
            p.setPen(QPen(QColor(60, 60, 60, 40), 1))
            if n_dots <= 40:
                # 点数少时画所有线对
                for i in range(n_dots):
                    for j in range(i + 1, n_dots):
                        d1, d2 = self._edge_dots[i], self._edge_dots[j]
                        x1, y1 = self._to_screen(d1.x, d1.y)
                        x2, y2 = self._to_screen(d2.x, d2.y)
                        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            else:
                # 点数多时只画相邻点连线（边界轮廓），保持流畅
                for i in range(n_dots):
                    d1 = self._edge_dots[i]
                    d2 = self._edge_dots[(i + 1) % n_dots]
                    x1, y1 = self._to_screen(d1.x, d1.y)
                    x2, y2 = self._to_screen(d2.x, d2.y)
                    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # 绘制点
        for i, d in enumerate(self._edge_dots):
            sx, sy = self._to_screen(d.x, d.y)
            if i == self._hover_idx:
                # 高亮：红色，表示可右键删除
                p.setPen(QPen(QColor(255, 80, 80), 2))
                p.setBrush(QBrush(QColor(255, 80, 80, 180)))
                p.drawEllipse(QPointF(sx, sy), self.DOT_RADIUS + 2, self.DOT_RADIUS + 2)
            else:
                p.setPen(QPen(QColor(0, 200, 255), 1))
                p.setBrush(QBrush(QColor(0, 200, 255, 160)))
                p.drawEllipse(QPointF(sx, sy), self.DOT_RADIUS, self.DOT_RADIUS)

        # 绘制画线模式的临时线段预览
        if self._tool_mode == self.TOOL_LINE and self._line_dragging and self._line_start and self._line_end_screen:
            sx_start, sy_start = self._to_screen(self._line_start[0], self._line_start[1])
            sx_end, sy_end = self._line_end_screen
            # 绘制线段
            p.setPen(QPen(QColor(255, 200, 0, 180), 2, Qt.DashLine))
            p.drawLine(QPointF(sx_start, sy_start), QPointF(sx_end, sy_end))
            # 绘制线段上将要生成的点的预览
            n = max(2, self._line_point_count)
            cx_start, cy_start = self._line_start
            cx_end, cy_end = self._to_canvas(sx_end, sy_end)
            p.setPen(QPen(QColor(255, 200, 0), 1))
            p.setBrush(QBrush(QColor(255, 200, 0, 200)))
            for i in range(n):
                t = i / (n - 1)
                preview_cx = cx_start + t * (cx_end - cx_start)
                preview_cy = cy_start + t * (cy_end - cy_start)
                psx, psy = self._to_screen(preview_cx, preview_cy)
                p.drawEllipse(QPointF(psx, psy), 3, 3)

        # 绘制校验警告（使用缓存，仅在点变化时重新计算）
        if self._warnings_dirty:
            self._cached_warnings = []
            if self._edge_dots and self._img_corner_lt and self._img_corner_rb:
                self._cached_warnings = validate_dots(
                    self._edge_dots, self._img_corner_lt, self._img_corner_rb
                )
            self._warnings_dirty = False
        validation_warnings = self._cached_warnings

        # 如果有覆盖率警告，用半透明红色渲染未覆盖区域的提示
        if validation_warnings:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 50, 50, 25)))
            p.drawRect(self.rect())

        # 提示文字
        p.setFont(QFont("sans-serif", 9))
        if validation_warnings:
            # 显示第一条警告（截取单行）
            warn_text = validation_warnings[0].split('\n')[0]
            p.setPen(QColor(255, 120, 80))
            p.drawText(8, self.height() - 24, f"⚠ {warn_text}")
        p.setPen(QColor(150, 150, 150))
        if self._tool_mode == self.TOOL_LINE:
            hint = f"点数: {len(self._edge_dots)}  |  画线模式: 拖拽画线(沿线生成{self._line_point_count}个点)  |  右键/Esc取消  |  Ctrl+Z撤销"
        else:
            hint = f"点数: {len(self._edge_dots)}  |  左键添加  |  右键删除  |  Ctrl+Z撤销  Ctrl+Shift+Z重做"
        p.drawText(8, self.height() - 8, hint)

        p.end()


# ------------------------------------------------------------------ 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grey Lines – 线条画生成器")
        self.setMinimumSize(960, 640)

        self._result = None
        self._cvs = None
        self._inv_scale = 1.0
        self._worker = None
        self._img_size = None  # (w, h) 加载图片后的像素尺寸

        self._build_ui()

    # ----------------------------------------------------------- 构建 UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # 顶部：参数 + 输入/输出路径
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)

        # 第一行：输入图片
        row0 = QHBoxLayout()
        row0.addWidget(QLabel("输入图片:"))
        self.image_path_edit = QLineEdit()
        row0.addWidget(self.image_path_edit, 1)
        browse_img_btn = QPushButton("浏览…")
        browse_img_btn.clicked.connect(self._browse_image)
        row0.addWidget(browse_img_btn)
        param_layout.addLayout(row0)

        # 第二行：画布类型 / 密度 / 固定宽度
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("画布模板:"))
        self.radio_rect = QRadioButton("矩形")
        self.radio_rect.setChecked(True)
        self.radio_circle = QRadioButton("圆形")
        self._canvas_group = QButtonGroup()
        self._canvas_group.addButton(self.radio_rect)
        self._canvas_group.addButton(self.radio_circle)
        row1.addWidget(self.radio_rect)
        row1.addWidget(self.radio_circle)

        row1.addSpacing(16)
        row1.addWidget(QLabel("密度:"))
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(1.0, 10.0)
        self.density_spin.setSingleStep(0.5)
        self.density_spin.setValue(4.0)
        row1.addWidget(self.density_spin)

        row1.addSpacing(16)
        row1.addWidget(QLabel("随机抖动:"))
        self.jitter_spin = QDoubleSpinBox()
        self.jitter_spin.setRange(0.0, 1.0)
        self.jitter_spin.setSingleStep(0.1)
        self.jitter_spin.setValue(0.0)
        self.jitter_spin.setToolTip(
            "对模板点施加随机位置抖动 (0=无抖动, 1=最大抖动)。\n"
            "抖动可打破规则排列导致的奇异点（多条线精确交汇于一点）。\n"
            "建议值: 0.3~0.5"
        )
        row1.addWidget(self.jitter_spin)

        row1.addSpacing(16)
        self.load_template_btn = QPushButton("加载模板点")
        self.load_template_btn.clicked.connect(self._load_template)
        row1.addWidget(self.load_template_btn)

        row1.addSpacing(16)
        self.clear_dots_btn = QPushButton("清空所有点")
        self.clear_dots_btn.clicked.connect(self._clear_dots)
        row1.addWidget(self.clear_dots_btn)

        row1.addSpacing(16)
        row1.addWidget(QLabel("输出固定宽度 (0=自动):"))
        self.fixed_spin = QSpinBox()
        self.fixed_spin.setRange(0, 4000)
        self.fixed_spin.setSingleStep(100)
        self.fixed_spin.setValue(0)
        row1.addWidget(self.fixed_spin)

        row1.addSpacing(16)
        row1.addWidget(QLabel("求解方法:"))
        self.solver_combo = QComboBox()
        self.solver_combo.addItem("直接法 (精确)", "direct")
        self.solver_combo.addItem("LSMR 迭代法 (更快)", "lsmr")
        self.solver_combo.setCurrentIndex(0)
        self.solver_combo.setToolTip(
            "直接法：使用 spsolve 精确求解，大密度下较慢\n"
            "LSMR：迭代法，跳过 AᵀA 矩阵构建，速度更快，精度略低"
        )
        row1.addWidget(self.solver_combo)

        row1.addStretch()
        param_layout.addLayout(row1)

        # 第 1.5 行：输出增强参数
        row1b = QHBoxLayout()
        row1b.addWidget(QLabel("Gamma 对比度:"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 3.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setToolTip(
            "Gamma 校正：< 1 增强弱线条（整体更暗更清晰），> 1 抑制弱线条。\n"
            "点密度高时建议设为 0.4 ~ 0.7 以获得更好的对比度。"
        )
        row1b.addWidget(self.gamma_spin)

        row1b.addSpacing(16)
        row1b.addWidget(QLabel("线宽:"))
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.1, 5.0)
        self.linewidth_spin.setSingleStep(0.1)
        self.linewidth_spin.setValue(1.0)
        self.linewidth_spin.setToolTip("SVG 线条宽度（像素）。密集点时可设为 0.5 让线更细更清晰。")
        row1b.addWidget(self.linewidth_spin)

        row1b.addSpacing(16)
        self.auto_param_btn = QPushButton("🔄 自动推荐")
        self.auto_param_btn.setToolTip(
            "根据当前点数和图片尺寸自动计算最佳 Gamma 和线宽。\n"
            "加载模板或增删点时会自动触发。"
        )
        self.auto_param_btn.clicked.connect(self._auto_recommend_params)
        row1b.addWidget(self.auto_param_btn)

        row1b.addStretch()
        param_layout.addLayout(row1b)

        # 第三行：输出路径
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("输出路径:"))
        self.output_path_edit = QLineEdit()
        row2.addWidget(self.output_path_edit, 1)
        browse_out_btn = QPushButton("浏览…")
        browse_out_btn.clicked.connect(self._browse_output)
        row2.addWidget(browse_out_btn)
        param_layout.addLayout(row2)

        root_layout.addWidget(param_group)

        # ===== 按钮区 =====
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶ 开始生成")
        self.run_btn.clicked.connect(self._run)
        btn_layout.addWidget(self.run_btn)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: gray;")
        btn_layout.addWidget(self.status_label)
        btn_layout.addStretch()
        root_layout.addLayout(btn_layout)

        # ===== 进度条 =====
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root_layout.addWidget(self.progress)

        # ===== 中间区域：工具栏 + 点编辑器 =====
        editor_group = QGroupBox("边缘点编辑（背景为原图预览）")
        editor_layout = QVBoxLayout(editor_group)

        # ---- 工具栏 ----
        tool_bar = QHBoxLayout()

        tool_bar.addWidget(QLabel("工具:"))
        self.tool_point_btn = QToolButton()
        self.tool_point_btn.setText("✏ 点")
        self.tool_point_btn.setCheckable(True)
        self.tool_point_btn.setChecked(True)
        self.tool_point_btn.setToolTip("点模式：左键添加点，右键删除点")
        tool_bar.addWidget(self.tool_point_btn)

        self.tool_line_btn = QToolButton()
        self.tool_line_btn.setText("📏 画线")
        self.tool_line_btn.setCheckable(True)
        self.tool_line_btn.setToolTip("画线模式：拖拽画一条线段，沿线均匀分布指定数量的点")
        tool_bar.addWidget(self.tool_line_btn)

        self._tool_btn_group = QButtonGroup()
        self._tool_btn_group.setExclusive(True)
        self._tool_btn_group.addButton(self.tool_point_btn, 0)
        self._tool_btn_group.addButton(self.tool_line_btn, 1)
        self._tool_btn_group.buttonClicked.connect(self._on_tool_changed)

        tool_bar.addSpacing(16)
        tool_bar.addWidget(QLabel("画线点数:"))
        self.line_pts_spin = QSpinBox()
        self.line_pts_spin.setRange(2, 200)
        self.line_pts_spin.setValue(5)
        self.line_pts_spin.setToolTip("画线模式下，沿线段均匀分布的点数")
        self.line_pts_spin.valueChanged.connect(self._on_line_pts_changed)
        tool_bar.addWidget(self.line_pts_spin)

        tool_bar.addSpacing(16)
        self.undo_btn = QPushButton("↩ 撤销")
        self.undo_btn.setToolTip("Ctrl+Z")
        self.undo_btn.clicked.connect(self._do_undo)
        tool_bar.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("↪ 重做")
        self.redo_btn.setToolTip("Ctrl+Shift+Z / Ctrl+Y")
        self.redo_btn.clicked.connect(self._do_redo)
        tool_bar.addWidget(self.redo_btn)

        tool_bar.addStretch()
        editor_layout.addLayout(tool_bar)

        # ---- 点编辑画布 ----
        self.dot_editor = DotEditorWidget()
        editor_layout.addWidget(self.dot_editor, 1)

        # 监听点变化，自动更新推荐参数
        self.dot_editor.dots_changed.connect(self._auto_recommend_params)
        self.dot_editor.dots_changed.connect(self._update_undo_redo_btns)

        root_layout.addWidget(editor_group, 1)

    # ----------------------------------------------------------- 模板加载
    def _load_template(self):
        """根据当前参数加载矩形/圆形模板点到编辑器。"""
        if self._img_size is None:
            QMessageBox.warning(self, "提示", "请先选择输入图片，以确定画布尺寸。")
            return
        w, h = self._img_size
        density = self.density_spin.value()
        jitter = self.jitter_spin.value()
        if self.radio_circle.isChecked():
            self.dot_editor.load_circle_template(w, h, density, jitter=jitter)
        else:
            self.dot_editor.load_rect_template(w, h, density, jitter=jitter)

    def _clear_dots(self):
        self.dot_editor.set_dots([])

    # ----------------------------------------------------------- 工具栏回调
    def _on_tool_changed(self, btn):
        if btn == self.tool_point_btn:
            self.dot_editor.set_tool_mode(DotEditorWidget.TOOL_POINT)
        else:
            self.dot_editor.set_tool_mode(DotEditorWidget.TOOL_LINE)

    def _on_line_pts_changed(self, value):
        self.dot_editor.set_line_point_count(value)

    def _do_undo(self):
        self.dot_editor.undo()

    def _do_redo(self):
        self.dot_editor.redo()

    def _update_undo_redo_btns(self):
        self.undo_btn.setEnabled(self.dot_editor.can_undo())
        self.redo_btn.setEnabled(self.dot_editor.can_redo())

    def _auto_recommend_params(self):
        """根据当前点数和图片尺寸自动设置 gamma、linewidth 和求解方法。"""
        if self._img_size is None:
            return
        n_dots = len(self.dot_editor.get_dots())
        w, h = self._img_size
        gamma, lw = recommend_gamma_linewidth(n_dots, w, h)
        self.gamma_spin.setValue(gamma)
        self.linewidth_spin.setValue(lw)

        # 线条数 = n*(n-1)/2，超过阈值时自动切换到 LSMR 迭代法
        line_cnt = n_dots * (n_dots - 1) // 2
        if line_cnt > 3000:
            self.solver_combo.setCurrentIndex(1)  # LSMR 迭代法
        else:
            self.solver_combo.setCurrentIndex(0)  # 直接法

    # ----------------------------------------------------------- 浏览文件
    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择输入图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;所有文件 (*)")
        if path:
            self.image_path_edit.setText(path)
            self._show_input_preview(path)
            # 每次加载新图片都自动更新输出路径
            base, _ = os.path.splitext(path)
            self.output_path_edit.setText(base + "_out.svg")
            # 加载图片尺寸，用于初始化画布
            self._load_image_size(path)

    def _load_image_size(self, path):
        """加载图片后确定内部工作尺寸，并自动加载模板。"""
        try:
            max_w = 300
            sz, _, _ = img_input.load_image(os.path.expanduser(path), max_w)
            self._img_size = (sz[0], sz[1])
            # 自动加载一次模板
            self._load_template()
        except Exception as e:
            QMessageBox.critical(self, "图片加载错误", str(e))

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 SVG 文件", "",
            "SVG 文件 (*.svg);;所有文件 (*)")
        if path:
            self.output_path_edit.setText(path)

    # ----------------------------------------------------------- 预览
    def _show_input_preview(self, path):
        """将原图加载为 QImage 并设置为点编辑器的背景。"""
        try:
            img = Image.open(path).convert("L")
            qimg = QImage(img.tobytes(), img.width, img.height,
                          img.width, QImage.Format_Grayscale8)
            # 保留一份副本，避免底层数据被回收
            self.dot_editor.set_background_image(qimg.copy())
        except Exception as e:
            QMessageBox.critical(self, "预览错误", str(e))

    # ----------------------------------------------------------- 状态
    def _set_status(self, msg, color="gray"):
        self.status_label.setText(msg)
        self.status_label.setStyleSheet(f"color: {color};")

    # ----------------------------------------------------------- 运行
    def _run(self):
        path = self.image_path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "提示", "请先选择有效的输入图片。")
            return
        if self._worker and self._worker.isRunning():
            return

        edge_dots = self.dot_editor.get_dots()
        if len(edge_dots) < 2:
            QMessageBox.warning(self, "提示", "至少需要 2 个边缘点才能生成线条。")
            return

        img_corner_lt, img_corner_rb = self.dot_editor.get_img_corners()
        if img_corner_lt is None or img_corner_rb is None:
            QMessageBox.warning(self, "提示", "请先加载模板以初始化画布区域。")
            return

        # ---- 校验点的质量 ----
        warnings = validate_dots(edge_dots, img_corner_lt, img_corner_rb)
        if warnings:
            detail = "\n\n".join(f"• {w}" for w in warnings)
            reply = QMessageBox.warning(
                self, "边缘点校验警告",
                f"检测到以下问题：\n\n{detail}\n\n是否仍要继续生成？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self._set_status("计算中…", "blue")

        solver_method = self.solver_combo.currentData()
        self._worker = SolveWorker(
            path, edge_dots, img_corner_lt, img_corner_rb, self.fixed_spin.value(),
            solver_method=solver_method,
        )
        self._worker.finished.connect(self._on_solve_done)
        self._worker.error.connect(self._on_solve_error)
        self._worker.start()

    def _on_solve_done(self, cvs, result, inv_scale):
        self._cvs = cvs
        self._result = result
        self._inv_scale = inv_scale

        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self._set_status("完成，正在保存…", "blue")
        self._save()

    def _on_solve_error(self, msg):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self._set_status("出错 ✗", "red")
        QMessageBox.critical(self, "运行错误", msg)

    # ----------------------------------------------------------- 保存
    def _save(self):
        if self._result is None or self._cvs is None:
            return
        out_path = self.output_path_edit.text().strip()
        if not out_path:
            self._browse_output()
            out_path = self.output_path_edit.text().strip()
        if not out_path:
            self._set_status("未指定输出路径，跳过保存", "red")
            return

        base, ext = os.path.splitext(out_path)
        if ext.lower() != ".svg":
            out_path = base + ".svg"

        try:
            output.save_svg(
                out_path, self._cvs, self._result, self._inv_scale,
                gamma=self.gamma_spin.value(),
                line_width=self.linewidth_spin.value(),
            )
            self._set_status(f"已保存: {out_path}", "green")
        except Exception as e:
            self._set_status("保存失败", "red")
            QMessageBox.critical(self, "保存错误", str(e))


def main():
    mp.set_start_method("spawn", force=True)  # macOS 需要 spawn
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
