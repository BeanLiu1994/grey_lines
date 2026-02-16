from dataclasses import dataclass
import math
import numpy as np
import hashlib
import random

@dataclass
class dot:
    x: float
    y: float


@dataclass
class line:
    a: float
    b: float
    c: float
    denominator: float = 0.0
    dot1: dot = None
    dot2: dot = None

    def __post_init__(self):
        self.denominator = math.sqrt(self.a**2 + self.b**2)

    def eval(self, d: dot):
        return self.a * d.x + self.b * d.y + self.c

    def distance(self, d: dot):
        return abs(self.eval(d)) / self.denominator
    
    def band_dots(self, distance:float, top_left_dot:dot, right_bottom_dot:dot):
        dots_in_band = []
        if self.b != 0:
            slope = -self.a / self.b # y/x
            intercept = -self.c / self.b
            cos = math.sqrt(1 + slope**2)
            y_range = distance * cos
            for x in range(math.ceil(top_left_dot.x), math.floor(right_bottom_dot.x) + 1):
                y_line = slope * x + intercept
                start = max(math.ceil(y_line - y_range), math.ceil(top_left_dot.y))
                end = min(math.floor(y_line + y_range) + 1, math.floor(right_bottom_dot.y) + 1)
                for y in range(start, end):
                    if top_left_dot.y <= y <= right_bottom_dot.y:
                        distance_to_line = abs(y-y_line) / cos
                        dots_in_band.append((dot(x, y), distance_to_line))
        elif self.a != 0:
            x_line = -self.c / self.a
            for x in range(math.ceil(x_line - distance), math.floor(x_line + distance) + 1):
                if top_left_dot.x <= x <= right_bottom_dot.x:
                    for y in range(math.ceil(top_left_dot.y), math.floor(right_bottom_dot.y) + 1):
                        dots_in_band.append((dot(x, y), abs(x-x_line)))
        else:
            raise RuntimeError(f"abnormal line {self}")
        return dots_in_band

    def band_dots_fast(self, distance: float, top_left_dot: dot, right_bottom_dot: dot, base_x: int, base_y: int, xcnt: int):
        """向量化版本的 band_dots，直接返回像素索引和权重数组，避免创建大量 dot 对象"""
        if self.b != 0:
            slope = -self.a / self.b
            intercept = -self.c / self.b
            cos_val = math.sqrt(1 + slope ** 2)
            y_range = distance * cos_val
            x_start = math.ceil(top_left_dot.x)
            x_end = math.floor(right_bottom_dot.x) + 1
            y_ceil_tl = math.ceil(top_left_dot.y)
            y_floor_rb = math.floor(right_bottom_dot.y) + 1

            xs = np.arange(x_start, x_end)
            if len(xs) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
            y_lines = slope * xs + intercept
            starts = np.maximum(np.ceil(y_lines - y_range).astype(np.int64), y_ceil_tl)
            ends = np.minimum(np.floor(y_lines + y_range).astype(np.int64) + 1, y_floor_rb)
            counts = np.maximum(ends - starts, 0)
            total = counts.sum()
            if total == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

            # 全向量化：用 np.repeat 展开每列的像素，消除 Python for 循环
            mask = counts > 0
            xs_m = xs[mask]
            starts_m = starts[mask]
            counts_m = counts[mask]
            y_lines_m = y_lines[mask]

            # 为每列生成连续 y 坐标
            offsets = np.repeat(starts_m, counts_m)
            # 每列内的递增序号 0,1,2,...
            group_ids = np.arange(total) - np.repeat(np.concatenate(([0], np.cumsum(counts_m[:-1]))), counts_m)
            all_ys = offsets + group_ids

            all_xs = np.repeat(xs_m, counts_m)
            all_y_lines = np.repeat(y_lines_m, counts_m)

            dist_to_line = np.abs(all_ys - all_y_lines) / cos_val
            px_indices = (all_xs - base_x) + (all_ys - base_y) * xcnt
            weights = 1.0 / (1.0 + dist_to_line)
            return px_indices.astype(np.int64), weights

        elif self.a != 0:
            x_line = -self.c / self.a
            x_start = max(math.ceil(x_line - distance), math.ceil(top_left_dot.x))
            x_end = min(math.floor(x_line + distance) + 1, math.floor(right_bottom_dot.x) + 1)
            y_start = math.ceil(top_left_dot.y)
            y_end = math.floor(right_bottom_dot.y) + 1

            xs = np.arange(x_start, x_end)
            ys = np.arange(y_start, y_end)
            if len(xs) == 0 or len(ys) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
            xx, yy = np.meshgrid(xs, ys)
            xx_flat = xx.ravel()
            yy_flat = yy.ravel()
            dist = np.abs(xx_flat - x_line)
            idx = (xx_flat - base_x) + (yy_flat - base_y) * xcnt
            w = 1.0 / (1.0 + dist)
            return idx.astype(np.int64), w
        else:
            raise RuntimeError(f"abnormal line {self}")

    @staticmethod
    def from_dots(dot1: dot, dot2: dot):
        a = dot2.y - dot1.y
        b = dot1.x - dot2.x
        c = dot2.x * dot1.y - dot1.x * dot2.y
        return line(a, b, c, dot1=dot1, dot2=dot2)


@dataclass
class canvas:
    edge_dots: list
    img_corner_lt: dot
    img_corner_rb: dot

    def hash(self):
        data = ""
        data += f"len:{len(self.edge_dots)}"
        for idx, d in enumerate(self.edge_dots):
            data += f"{idx}:({d.x},{d.y})"
        data += f"lt:({self.img_corner_lt.x}, {self.img_corner_lt.y})"
        data += f"rb:({self.img_corner_rb.x}, {self.img_corner_rb.y})"
        return hashlib.sha256(data.encode()).hexdigest()

    def edge_dots_cnt(self):
        return len(self.edge_dots)

    def lines_cnt(self):
        m = self.edge_dots_cnt()
        return int(m * (m -1) / 2)
        
    def line_between_dots(self, dot_ind1: int, dot_ind2: int):
        d1 = self.edge_dots[dot_ind1]
        d2 = self.edge_dots[dot_ind2]
        return line.from_dots(d1, d2)

    def lines(self):
        ret = []
        m = self.edge_dots_cnt()
        for i in range(m):
            for j in range(m):
                if i >=j:
                    continue
                # must have i < j
                ret.append(self.line_between_dots(i, j))
        return ret
    
    def canvas_pixel_cnt(self):
        xcnt = (math.floor(self.img_corner_rb.x)-math.ceil(self.img_corner_lt.x)+1)
        ycnt = (math.floor(self.img_corner_rb.y)-math.ceil(self.img_corner_lt.y)+1)
        return xcnt * ycnt
    
    def canvas_pixel_dot_index(self, d: dot):
        if d.x > self.img_corner_rb.x or  d.y > self.img_corner_rb.y:
            raise RuntimeError("out of area")
        if d.x < self.img_corner_lt.x or  d.y < self.img_corner_lt.y:
            raise RuntimeError("out of area")
        base_x = math.ceil(self.img_corner_lt.x)
        base_y = math.ceil(self.img_corner_lt.y)
        xcnt = (math.floor(self.img_corner_rb.x)-base_x+1)
        return math.floor(d.x) - base_x + (math.floor(d.y) - base_y) * xcnt

    def canvas_pixel_dots(self):
        ret = []
        base_x = math.ceil(self.img_corner_lt.x)
        base_y = math.ceil(self.img_corner_lt.y)
        xcnt = (math.floor(self.img_corner_rb.x)-base_x+1)
        ycnt = (math.floor(self.img_corner_rb.y)-base_y+1)
        for i in range(xcnt):
            for j in range(ycnt):
                ret.append(dot(i+base_x, j+base_y))
        return ret

    @staticmethod
    def rectangle_canvas(w:int, h:int, split_w: int = None, split_h:int = None, jitter:float = 0.0):
        """创建矩形画布。jitter: 随机抖动幅度 (0~1)，按相邻点间距的比例施加随机偏移以消除奇异点。"""
        split_w = split_w or 10
        split_h = split_h or 10
        dots = []
        i = 0.5
        j = 0.5
        xlist = np.linspace(0.5, w+0.5, num=split_w, endpoint=True)
        ylist = np.linspace(0.5, h+0.5, num=split_h, endpoint=True)
        # 计算相邻点间距，用于确定抖动幅度
        x_step = (xlist[1] - xlist[0]) if len(xlist) > 1 else 1.0
        y_step = (ylist[1] - ylist[0]) if len(ylist) > 1 else 1.0
        for idx, i in enumerate(xlist):
            if idx == len(xlist)-1:
                continue
            pi = i + (random.random() - 0.5) * x_step * jitter if jitter > 0 else i
            dots.append(dot(pi, j))
        for idx, j in enumerate(ylist):
            if idx == len(ylist)-1:
                continue
            pj = j + (random.random() - 0.5) * y_step * jitter if jitter > 0 else j
            dots.append(dot(i, pj))
        for idx, i in enumerate(reversed(xlist)):
            if idx == len(xlist)-1:
                continue
            pi = i + (random.random() - 0.5) * x_step * jitter if jitter > 0 else i
            dots.append(dot(pi, j))
        for idx, j in enumerate(reversed(ylist)):
            if idx == len(ylist)-1:
                continue
            pj = j + (random.random() - 0.5) * y_step * jitter if jitter > 0 else j
            dots.append(dot(i, pj))
        return canvas(edge_dots=dots, img_corner_lt=dot(1,1), img_corner_rb=dot(w,h))

    @staticmethod
    def circle_canvas(w:int, h:int, split_cnt:int = None, jitter:float = 0.0):
        """创建圆形画布。jitter: 随机抖动幅度 (0~1)，按相邻点角度间距的比例施加随机偏移以消除奇异点。"""
        split_cnt = split_cnt or 50
        diameter = math.sqrt(w**2 + h**2)
        radius = diameter / 2
        corner_lt = dot(radius - w / 2 + 0.5, radius - h / 2 + 0.5)
        corner_rb = dot(radius + w / 2 + 0.5, radius + h / 2 + 0.5)
        dots = []
        theta_list = np.linspace(0, math.pi*2, num=split_cnt, endpoint=True)
        theta_step = (theta_list[1] - theta_list[0]) if len(theta_list) > 1 else 0.1
        for idx, i in enumerate(theta_list):
            if idx == len(theta_list)-1:
                continue
            theta = i + (random.random() - 0.5) * theta_step * jitter if jitter > 0 else i
            dot_x = radius + radius * math.cos(theta)
            dot_y = radius + radius * math.sin(theta)
            dots.append(dot(dot_x, dot_y))
        return canvas(edge_dots=dots, img_corner_lt=corner_lt, img_corner_rb=corner_rb)
