from tqdm import tqdm
import math
import dolfinx
from dolfinx import mesh, geometry
import numpy as np
from mpi4py import MPI

def step_until(T, solver, method):
    for _ in tqdm(range(math.ceil(T / solver.dt))):
        method(solver)

def constant(value, mesh, space):
    return dolfinx.fem.Expression(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value)), space.element.interpolation_points())



def compute_convergence(find_error, points):
    hs = []
    es = []
    if type(points) == int:
        points = [2 ** i for i in range(1, points + 1)]
    for n in points:
        h  = 1 / n
        error = find_error(h)
        es.append(error)
        hs.append(h)
    h = np.array(hs)
    e = np.array(es)
    logh = np.log2(h)
    loge = np.log2(e)
    a, _ = np.polyfit(logh, loge, deg=1)
    print("Convergence: ", a)
    print(hs)
    print(es)

def x_equals(value):
    return lambda x: np.isclose(x[0], value)

def y_equals(value):
    return lambda x: np.isclose(x[1], value)


def unit_square(dx, show=False):
    n = math.ceil(1 / dx)
    square = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.quadrilateral)
    return square, dolfinx.geometry.bb_tree(square, 2)

def rectangular_domain(dx, p1, p2, show=False):
    x1, y1 = p1
    x2, y2 = p2
    nx = math.ceil(abs(x2 - x1) / dx)
    ny = math.ceil(abs(y2 - y1) / dx)
    print(f"x cells: {nx}, y cells: {ny}")
    rect = mesh.create_rectangle(MPI.COMM_WORLD, [np.array(p1), np.array(p2)], (nx, ny), cell_type=mesh.CellType.quadrilateral)
    return rect, dolfinx.geometry.bb_tree(rect, 2)

def eval_sol(u, points, mesh, tree, forse_eval=False):
    cells = []
    points_on_proc = []
    cell_candidates = geometry.compute_collisions_points(tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    out_of_bounds_points = np.full(len(points.T), False)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
        elif forse_eval:
            points_on_proc.append(point)
            cells.append(0)
            out_of_bounds_points[i] = True
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    res = u.eval(points_on_proc, cells)
    res[out_of_bounds_points] = np.zeros(res[out_of_bounds_points].shape)
    return res

def fem_scalar_func_at_points(u_func, mesh, tree, xbounds, ybounds, npoints, forse_eval=False):
    x = np.linspace(*xbounds, npoints)
    y = np.linspace(*ybounds, npoints)
    x, y = np.meshgrid(x, y, indexing="ij")
    points = np.zeros((3, npoints ** 2))
    points[0] = x.flatten()
    points[1] = y.flatten()
    u = eval_sol(u_func, points, mesh, tree, forse_eval)
    return x.reshape((npoints, npoints)), y.reshape((npoints, npoints)), u.reshape((npoints, npoints))

def fem_scalar_func_at_given_points(u_func, mesh, tree, x, y, forse_eval=False):
    points = np.zeros((3, len(x)))
    points[0] = np.array(x)
    points[1] = np.array(y)
    u = eval_sol(u_func, points, mesh, tree, forse_eval=forse_eval)
    return x, y, u

def fem_vector_func_at_points(u_func, mesh, tree, xbounds, ybounds, npoints, forse_eval=False):
    x = np.linspace(*xbounds, npoints)
    y = np.linspace(*ybounds, npoints)
    x, y = np.meshgrid(x, y, indexing="ij")
    points = np.zeros((3, npoints ** 2))
    points[0] = x.flatten()
    points[1] = y.flatten()
    uvs = eval_sol(u_func, points, mesh, tree, forse_eval=forse_eval)
    u = np.array(list(uv[0] for uv in uvs))
    v = np.array(list(uv[1] for uv in uvs))
    return x.reshape((npoints, npoints)), y.reshape((npoints, npoints)), u.reshape((npoints, npoints)), v.reshape((npoints, npoints))

def fem_vector_func_at_given_points(u_func, mesh, tree, x, y, forse_eval=False):
    points = np.zeros((3, len(x)))
    points[0] = np.array(x)
    points[1] = np.array(y)
    uvs = eval_sol(u_func, points, mesh, tree, forse_eval=forse_eval)
    u = np.array(list(uv[0] for uv in uvs))
    v = np.array(list(uv[1] for uv in uvs))
    return x, y, u, v