import numpy as np
from dolfinx import fem, default_scalar_type

AXIS_DICT = {"x" : 0, "y" : 1}

def on_line(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy, _ = c
    xprod = (cy - ay) * (bx - ax) - (cx - ax) * (by - ay)
    dprod = (cx - ax) * (bx - ax) + (cy - ay) * (by - ay)
    sqlen = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
    return np.logical_and(
        np.logical_and(
            np.isclose(np.abs(xprod), 0),
            dprod >= -1e-8
        ),
        dprod <= sqlen + 1e-8
    )
    

def on_polygon(points):
    n = len(points)
    def _on_polygon(x):
        return np.logical_or.reduce(tuple(on_line(points[i], points[(i + 1) % n], x) for i in range(n)))
    return _on_polygon

def x_equals(value):
    return lambda x: np.isclose(x[0], value)

def y_equals(value):
    return lambda x: np.isclose(x[1], value)

def process_pair(x, pair):
    ax, val = pair
    return np.isclose(x[AXIS_DICT[ax]], val)

def any_of(pairs):
    def _any_of(x):
        return np.logical_or.reduce(tuple(process_pair(x, p) for p in pairs))
    return _any_of

def constant(value, mesh, space):
    return fem.Expression(fem.Constant(mesh, default_scalar_type(value)), space.element.interpolation_points())

def axes_aligned_boundary(points: tuple[tuple]):
    p1, p2 = points
    axes, val = (0, p1[0]) if p1[0] == p2[0] else (1, p1[1])
    x1, x2 = (points[0][1^axes], points[1][1^axes]) if points[0][1^axes] <= points[1][1^axes] else (points[1][1^axes], points[0][1^axes])
    def on_boundary(x):
        return np.logical_and(
            np.isclose(x[axes], val + 1e-8),
            np.logical_or(x[1^axes] >= x1, x[1^axes] <= x2) 
        )
    return on_boundary


