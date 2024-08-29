from ellipticproject import EllipticProjector
import dolfinx
import dolfinx.fem.petsc
import ufl


import mpi4py
import matplotlib.pyplot as plt
import numpy as np
from visualise import *

class NormalProjector:

    def __init__(self, Vh, h, c_e=2, delta=1e-6) -> None:
        self.Vh = Vh
        self.projector = EllipticProjector(Vh, h, c_e)
        self.delta = delta

    def set_function(self, f):
        self.projector.set_projected_function(ufl.grad(f))
        

    def compute_normals(self):
        nabla_f = self.projector.project()
        n_h = dolfinx.fem.Function(self.Vh)
        n_h.interpolate(
            dolfinx.fem.Expression(
                nabla_f / (ufl.sqrt(ufl.inner(nabla_f, nabla_f)) + self.delta),
                self.Vh.element.interpolation_points()
            )
        )
        return n_h
    
def test():
    hs = []
    es = []

    for n in [2, 4, 8, 16, 32, 64]:
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
        n = dolfinx.fem.Function(Vh)
        n.interpolate(lambda x: (x[0] / np.sqrt(x[0]**2 + x[1]**2), x[1] / np.sqrt(x[0]**2 + x[1]**2)))
        error = ufl.sqrt(ufl.inner(n - nh, n - nh)) * ufl.dx
        es.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(error)))
        hs.append(h)

    h = np.array(hs)
    e = np.array(es)

    logh = np.log2(h)
    loge = np.log2(e)

    a, _ = np.polyfit(logh, loge, deg=1)
    print("Convergence: ", a)
