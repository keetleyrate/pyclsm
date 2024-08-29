from normal import NormalProjector
from ellipticproject import EllipticProjector

import dolfinx
import dolfinx.fem.petsc
import ufl
import mpi4py
import numpy as np
import matplotlib.pyplot as plt
from visualise import *


hs = []
es = []
for n in [2, 4, 8, 16, 32]:
    h = 1 / n
    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
    tree = dolfinx.geometry.bb_tree(mesh, 2)
    f_space = dolfinx.fem.functionspace(mesh, ("P", 2))
    Vh = dolfinx.fem.VectorFunctionSpace(mesh, ("P", 2))
    normal_solver = NormalProjector(Vh, h, delta=1e-6)
    f = dolfinx.fem.Function(f_space)
    f.interpolate(lambda x: 0.5 * (x[0]**2 + x[1]**2))

    normal_solver.set_function(f)
    nh = normal_solver.compute_normals()
    projector = EllipticProjector(f_space, h, 0)
    projector.set_projected_function(-ufl.div(nh))
    kappa_h = projector.project()

    x, y, kh = fem_scalar_func_at_points(kappa_h, mesh, tree, (0.1, 1), (0.1, 1), 100)
    k = -1 / np.sqrt(x**2 + y**2)

    error = np.sqrt(np.sum((np.square(k - kh))))
    hs.append(h)
    es.append(error)
h = np.array(hs)
e = np.array(es)

logh = np.log2(h)
loge = np.log2(e)

a, _ = np.polyfit(logh, loge, deg=1)
print("Convergence: ", a)

