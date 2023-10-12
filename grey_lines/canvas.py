from dataclasses import dataclass
import math
import numpy as np

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
            for x in range(int(top_left_dot.x), int(right_bottom_dot.x) + 1):
                y_line = slope * x + intercept
                for y in range(int(y_line - y_range), int(y_line + y_range) + 1):
                    if top_left_dot.y <= y <= right_bottom_dot.y:
                        distance_to_line = abs(y-y_line) / cos
                        dots_in_band.append((dot(x, y), distance_to_line))
        elif self.a != 0:
            x_line = -self.c / self.a
            for x in range(int(x_line - distance), int(x_line + distance) + 1):
                if top_left_dot.x <= x <= right_bottom_dot.x:
                    for y in range(int(top_left_dot.y), int(right_bottom_dot.y) + 1):
                        dots_in_band.append((dot(x, y), abs(x-x_line)))
        else:
            raise RuntimeError(f"abnormal line {self}")
        return dots_in_band

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
    def rectangle_canvas(w:int, h:int, split_w: int = None, split_h:int = None):
        split_w = split_w or 20
        split_h = split_h or 20
        dots = []
        i = 0.5
        j = 0.5
        xlist = np.linspace(0.5, w+0.5, num=split_w, endpoint=True)
        ylist = np.linspace(0.5, h+0.5, num=split_h, endpoint=True)
        for idx, i in enumerate(xlist):
            if idx == len(xlist)-1:
                continue
            dots.append(dot(i, j))
        for idx, j in enumerate(ylist):
            if idx == len(ylist)-1:
                continue
            dots.append(dot(i, j))
        for idx, i in enumerate(reversed(xlist)):
            if idx == len(xlist)-1:
                continue
            dots.append(dot(i, j))
        for idx, j in enumerate(reversed(ylist)):
            if idx == len(ylist)-1:
                continue
            dots.append(dot(i, j))
        return canvas(edge_dots=dots, img_corner_lt=dot(1,1), img_corner_rb=dot(w,h))
