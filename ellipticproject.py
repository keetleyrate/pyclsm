import dolfinx
import dolfinx.fem.petsc
import ufl

import mpi4py
import matplotlib.pyplot as plt

from common.visualize import *

def create_test_and_trail_functions(solution_space: dolfinx.fem.FunctionSpace):
    trail = ufl.TrialFunction(solution_space)
    test = ufl.TestFunction(solution_space)
    solution = dolfinx.fem.Function(solution_space)
    return solution, trail, test


class EllipticProjector:

    def __init__(self, C) -> None:
        self.C = C
        self.bcs = []

    def build_problem(self, solution_space, to_project, h):
        solution, trail, test = create_test_and_trail_functions(solution_space)
        self.solution = solution
        form = (ufl.inner(trail, test) + self.C * h * h * ufl.inner(ufl.grad(trail), ufl.grad(test)) - ufl.inner(to_project, test)) * ufl.dx
        self.problem = dolfinx.fem.petsc.LinearProblem(
            dolfinx.fem.form(ufl.lhs(form)),
            dolfinx.fem.form(ufl.rhs(form)),
            bcs=self.bcs,
            u=self.solution,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"},
            petsc_options_prefix="elliptic_proj_"
        )
        
    def project(self):
        self.problem.solve()
    

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


