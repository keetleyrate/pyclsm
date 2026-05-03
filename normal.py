from ellipticproject import EllipticProjector
import basix
import dolfinx
import ufl
import mpi4py
import numpy as np
from common.visualize import *
from common.common import *

def create_solution_space(funciton: dolfinx.fem.Function, p: int):
    mesh = funciton.function_space.mesh
    vector_element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), p, shape=(mesh.geometry.dim,))
    vector_space = dolfinx.fem.functionspace(mesh, vector_element)
    n = dolfinx.fem.Function(vector_space)
    return vector_space, n

class NormalProjector:

    def __init__(self, function: dolfinx.fem.Function, h_min, p=1, c_e=2, delta=1e-8) -> None:
        self.gradient_projector = EllipticProjector(c_e)
        self.function = function
        self.delta = delta
        self.h_min = h_min
        self.solution_space, self.n = create_solution_space(function, p)

    def build_problem(self):
        self.gradient_projector.build_problem(self.solution_space, ufl.grad(self.function), self.h_min)

    def set_boundary_condtion(self, value, geometry_fn):
        dofs = dolfinx.fem.locate_dofs_geometrical(self.solution_space, geometry_fn)
        self.gradient_projector.bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(value), dofs, self.solution_space))
 
    def compute_normals(self):
        self.gradient_projector.project()
        nabla_f = self.gradient_projector.solution
        interpolate_expression(nabla_f / (ufl.sqrt(ufl.inner(nabla_f, nabla_f)) + self.delta), self.n)
    
def test():
    n = 256
    p = 2
    mesh = dolfinx.mesh.create_rectangle(mpi4py.MPI.COMM_WORLD, [np.array([1, 1]), np.array([2, 2])], [n, n], cell_type=dolfinx.mesh.CellType.quadrilateral)
    f_space = dolfinx.fem.functionspace(mesh, ("P", p))
    n_space = dolfinx.fem.functionspace(mesh, ("P", p, (mesh.geometry.dim,)))
    n = dolfinx.fem.Function(n_space)
    f = dolfinx.fem.Function(f_space)
    f.interpolate(lambda x: x[0] + 2 * x[1])
    normal_solver = NormalProjector(f, p, 0, 1e-16)
    normal_solver.build_problem()
    normal_solver.compute_normals()
    nh = normal_solver.n
    n = dolfinx.fem.Constant(n_space.mesh, dolfinx.default_scalar_type((1/np.sqrt(5), 2/np.sqrt(5))))
    error = ufl.sqrt(ufl.inner(n - nh, n - nh)) * ufl.dx
    print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(error)))
