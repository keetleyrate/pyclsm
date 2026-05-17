from common.visualize import *
from levelset import *
import matplotlib
import time
import os
import scipy
import copy
import dolfinx
import ufl
import sdf
from tqdm import tqdm


class FixedDomain:

    def __init__(self, h, left, top):
        n = math.ceil(1 / h)
        square = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [left, top], [n, n], cell_type=dolfinx.mesh.CellType.quadrilateral)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)



def sdf_circle(x):
    x, y, _ = x
    r = 0.5
    return np.sqrt((x)**2 + (y)**2) - r

def f_ellipce(x):
    return x[0]**2 + (1 + eps)**2*x[1]**2 - 1 - eps

def g_ellipce(x):
    return np.array([2 * x[0], 2*(1 + eps)**2*x[1]])

def inital_condtion(X):
    d = np.array([sdf.signed_distance_to_zero_level_set(f_ellipce, g_ellipce, (xi, yi)) for xi, yi in tqdm(zip(X[0], X[1]))])
    return 1 / (1 + np.exp(d / (1.5 * h)))

def inital_condtion(X):
    d = np.array([sdf.signed_distance_to_zero_level_set(f_ellipce, g_ellipce, (xi, yi)) for xi, yi in tqdm(zip(X[0], X[1]))])
    return 1 / (1 + np.exp(d / (1.5 * h)))

h = 1 / 128
dt = 0.005
eps = 1
max_reinit_iters = 100

fig, ax = plt.subplots()

domain = FixedDomain(h, [-2, -2], [2, 2])
solver = ConservativeLevelSet(
    domain,
    h,
    dt,
    solver_options={"normal_method": "std", "fix_inteface": True}
)
solver.ϕ.interpolate(lambda x: inital_condtion(x))
solver.compute_gradient()
solver.compute_curvature()
x, y, kappa_full = fem_scalar_func_at_given_points(solver.κ, domain, np.linspace(0, 1, 250), np.zeros(250))

domain = FixedDomain(2*h, [0, 0], [2, 2])
solver = ConservativeLevelSet(
    domain,
    h,
    dt,
    sym_bcs=True,
    solver_options={"normal_method": "std", "fix_inteface": True}
)
solver.ϕ.interpolate(lambda x: inital_condtion(x))
solver.compute_gradient()
solver.compute_curvature()
x, y, kappa_quart = fem_scalar_func_at_given_points(solver.κ, domain, np.linspace(0, 1, 250), np.zeros(250))

plt.plot(x, kappa_full, label="Full x/k")
plt.plot(x, kappa_quart, label="Quat x/k")
plt.show()