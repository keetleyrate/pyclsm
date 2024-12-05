import dolfinx
import dolfinx.fem.petsc
import ufl

import mpi4py
import matplotlib.pyplot as plt

from common.visualize import *

class EllipticProjector:

    def __init__(self, Vh, h, c_e) -> None:
        self.f = dolfinx.fem.Function(Vh)
        self.u_e = dolfinx.fem.Function(Vh)
        self.u_trail = ufl.TrialFunction(Vh)
        self.v = ufl.TestFunction(Vh)
        self.c_e = c_e
        self.h = h

    def set_projected_function(self, f):
        self.form = (ufl.inner(self.u_trail, self.v) + self.c_e * self.h * self.h * ufl.inner(ufl.grad(self.u_trail), ufl.grad(self.v)) - ufl.inner(f, self.v)) * ufl.dx
        self.solver = dolfinx.fem.petsc.LinearProblem(
            dolfinx.fem.form(ufl.lhs(self.form)),
            dolfinx.fem.form(ufl.rhs(self.form)),
            [],
            self.u_e,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"}
        )
        
    def project(self):
        self.solver.solve()
        return self.u_e
    

def test():
    hs = []
    es = []

    for n in [2, 4, 8, 16, 32, 64]:
        h = 1 / n
        mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
        tree = dolfinx.geometry.bb_tree(mesh, 2)
        f_space = dolfinx.fem.functionspace(mesh, ("P", 2))
        Vh = dolfinx.fem.VectorFunctionSpace(mesh, ("P", 2))
        projector = EllipticProjector(Vh, h, 2)
        f = dolfinx.fem.Function(f_space)
        f.interpolate(lambda x: 0.5 * (x[0]**2 + x[1]**2))
        projector.set_projected_function(ufl.grad(f))
        grad_fh = projector.project()
        dfdx = dolfinx.fem.Function(Vh)
        dfdx.interpolate(lambda x: (x[0], x[1]))
        error = ufl.sqrt(ufl.inner(dfdx - grad_fh, dfdx - grad_fh)) * ufl.dx
        es.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(error)))
        hs.append(h)

    h = np.array(hs)
    e = np.array(es)

    logh = np.log2(h)
    loge = np.log2(e)

    a, _ = np.polyfit(logh, loge, deg=1)
    print("Convergence: ", a)


