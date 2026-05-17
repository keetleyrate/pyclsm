from common.visualize import *
from levelset import *
import matplotlib
import time
import os
import scipy
import copy
import dolfinx
import ufl
from tqdm import tqdm


class FixedDomain:

    def __init__(self, h):
        n = math.ceil(1 / h)
        square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)



def sdf_circle(x):
    x, y, _ = x
    r = 0.5
    return np.sqrt((x)**2 + (y)**2) - r


h = 1 / 256
dt = 0.005
max_reinit_iters = 100

fig, ax = plt.subplots()


domain = FixedDomain(h)
solver = ConservativeLevelSet(
    domain,
    h,
    dt,
    solver_options={"normal_method": "std", "fix_inteface": True}
)
solver.ϕ.interpolate(lambda x: 1/(1 + np.exp(sdf_circle(x) / (1.5 * h))))
solver.compute_gradient()
solver.compute_curvature()
print(solver.κ.x.array.max())
g = solver.grad_ϕ
interpolate_expression(ufl.sqrt(ufl.inner(g, g)) * solver.κ, solver.κ)
fem_plot_contor_filled(fig, ax, solver.κ, domain, colorbar=True)
plt.show()