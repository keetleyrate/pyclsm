import ufl
from ufl import inner, dot, grad, dx, nabla_grad, div, ds
import numpy as np
import dolfinx
import basix
from dolfinx import fem, default_scalar_type
from dolfinx.mesh import locate_entities_boundary
import dolfinx.fem.petsc as petsc
import dolfinx
import mpi4py
from common.visualize import *

def u_dot_grad(u):
    return dot(u, ufl.nabla_grad(u))

def γ(u):
    return grad(u) + grad(u).T

def create_spaces_and_functions(mesh, p):
    # Taylor-Hood Elements
    scalar_element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), p - 1)
    vector_element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), p, shape=(mesh.geometry.dim,))
    scalar_space = dolfinx.fem.functionspace(mesh, scalar_element)
    vector_space = dolfinx.fem.functionspace(mesh, vector_element)
    u = dolfinx.fem.Function(vector_space)
    u_last = dolfinx.fem.Function(vector_space)
    u_star = dolfinx.fem.Function(vector_space)
    p = dolfinx.fem.Function(scalar_space)
    p_last = dolfinx.fem.Function(scalar_space)
    rho = dolfinx.fem.Function(scalar_space)
    rho_last = dolfinx.fem.Function(scalar_space)
    mu = dolfinx.fem.Function(scalar_space)
    return scalar_space, vector_space, u_last, u, u_star, p_last, p, rho_last, rho, mu


class IncompressibleNavierStokesSolver:
    """Class for solving the incompressable Navier-Stokes equations using the fenicsx framework."""

    def __init__(self, domain, dt, p=2, kinematic_bc=False, density_viscosity_order=1) -> None:
        """Creates a new solver object. Initalises variational forms used as well as solution
        fields for velocity, pressure, dencity and viscosity. Must be given the approprite
        function space objects the the soltuion will be defined over."""
        self.domain = domain
        self.kinematic = kinematic_bc
        (
            self.scalar_space,
            self.vector_space,
            self.u_last,
            self.u,
            self.u_star,
            self.p_last,
            self.p,
            self.rho_last,
            self.rho,
            self.mu
        ) = create_spaces_and_functions(domain.mesh, p)
        self.u_bcs = []  
        self.p_bcs = []
        self.t = 0
        self.dt = dt
        self.set_density_as_const(1)
        self.set_viscosity_as_const(1)


    def build_predictor_problem(self, f):
        def σ(u, p):
            return -p * ufl.Identity(2) + self.mu * γ(u)
        trail, test = self.u_trail, self.u_test
        n = ufl.FacetNormal(self.domain.mesh)
        dt = self.dt
        form = (
            dot((self.rho * trail - self.rho_last * self.u_last) / dt, test) * dx
            + inner(σ((self.u_last + trail)/2, self.p_last), nabla_grad(test)) * dx
            + dot(self.p_last * n, test) * ds 
            - dot(self.mu * nabla_grad((self.u_last + trail)/2) * n, test) * ds
            - dot(f, test) * dx
        )
        if self.kinematic:
            form += - inner(dot(self.u_last, grad(test)), self.rho * trail) * dx
        else:
            form += self.rho * dot(dot(trail, nabla_grad(self.u_last)), test) * dx
        self.predictor_problem = petsc.LinearProblem(
            dolfinx.fem.form(ufl.lhs(form)),
            dolfinx.fem.form(ufl.rhs(form)),
            bcs=self.u_bcs,
            u=self.u_star,
            petsc_options={"ksp_type": "bcgs", "pc_type": "jacobi"},
            petsc_options_prefix="u_predictor_"
        )

    def build_pressure_problem(self):
        trail, test = self.p_trail, self.p_test
        dt = self.dt
        form = (
            (1 / dt) * test * div(self.u_star) * dx
            + inner((1 / self.rho)*grad(test), grad(trail - self.p_last)) * dx
        )
        self.pressure_problem = petsc.LinearProblem(
            dolfinx.fem.form(ufl.lhs(form)),
            dolfinx.fem.form(ufl.rhs(form)),
            bcs=self.p_bcs,
            u=self.p,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"},
            petsc_options_prefix="pressure_"
        )
       

    def build_corrector_problem(self):
        dt = self.dt
        trail, test = self.u_trail, self.u_test
        form = (
            (1 / dt) * inner(trail - self.u_star, test) * dx
            + inner(grad(self.p - self.p_last), (1 / self.rho) * test) * dx
        )
        self.correction_problem = petsc.LinearProblem(
            dolfinx.fem.form(ufl.lhs(form)),
            dolfinx.fem.form(ufl.rhs(form)),
            bcs=self.u_bcs,
            u=self.u,
            petsc_options={"ksp_type": "cg", "pc_type": "sor"},
            petsc_options_prefix="u_corrector_"
        )


    def create_test_and_trail_functions(self):
        self.u_trail = ufl.TrialFunction(self.vector_space)
        self.u_test = ufl.TestFunction(self.vector_space)
        self.p_trail = ufl.TrialFunction(self.scalar_space)
        self.p_test = ufl.TestFunction(self.scalar_space)

    def set_pressure_bc(self, geometry_fn, value):
        """Set a Dirichlet boundary condtion for pressure.
        geometry_fn: A function that returns True given a point where the boundary condtion
        will be placed.
        fn: The function (or constant) to be imposed."""
        dofs = fem.locate_dofs_geometrical(self.scalar_space, geometry_fn)
        self.p_bcs.append(fem.dirichletbc(dolfinx.default_scalar_type(value), dofs, self.scalar_space))

    def set_velocity_bc(self, geometry_fn, value):
        """Set a Dirichlet boundary condtion for velocity.
        geometry_fn: A function that returns True given a point where the boundary condtion
        will be placed.
        fn: The function (or constant) to be imposed."""
        dofs = fem.locate_dofs_geometrical(self.vector_space, geometry_fn)
        self.u_bcs.append(fem.dirichletbc(dolfinx.default_scalar_type(value), dofs, self.vector_space))

    def set_x_velocity(self, geometry_fn, value):
        """Set a Dirichlet boundary condtion on the x-component of velocity.
        geometry_fn: A function that returns True given a point where the boundary condtion
        will be placed.
        fn: The function (or constant) to be imposed."""
        boundary_facets = locate_entities_boundary(self.domain.mesh, self.domain.mesh.topology.dim - 1, geometry_fn) 
        boundary_dofs_x = fem.locate_dofs_topological(self.vector_space.sub(0), self.domain.mesh.topology.dim - 1, boundary_facets)
        self.u_bcs.append(fem.dirichletbc(dolfinx.default_scalar_type(value), boundary_dofs_x, self.vector_space.sub(0)))

    def set_y_velocity(self, geometry_fn, value):
        """Set a Dirichlet boundary condtion on the y-component of velocity.
        geometry_fn: A function that returns True given a point where the boundary condtion
        will be placed.
        fn: The function (or constant) to be imposed."""
        boundary_facets = locate_entities_boundary(self.domain.mesh, self.domain.mesh.topology.dim - 1, geometry_fn) 
        boundary_dofs_x = fem.locate_dofs_topological(self.vector_space.sub(1), self.domain.mesh.topology.dim - 1, boundary_facets)
        self.u_bcs.append(fem.dirichletbc(dolfinx.default_scalar_type(value), boundary_dofs_x, self.vector_space.sub(1)))

    def set_density_as_const(self, rho_c):
        """Set the density of the fluid as a constant value."""
        self.rho.x.array[:] = rho_c * np.ones(self.rho.x.array.shape)
        self.rho_last.x.array[:] = rho_c * np.ones(self.rho_last.x.array.shape)

    def set_viscosity_as_const(self, mu_c):
        """Set the viscosity of the fluid as a constant value."""
        self.mu.x.array[:] = mu_c * np.ones(self.mu.x.array.shape)


    def compute_vorticity(self):
        space = dolfinx.fem.functionspace(self.mesh, self.p_element)
        omega = fem.Function(space)
        u, v = fem.Function(space), fem.Function(space)
        u.interpolate(fem.Expression(self.u[0], space.element.interpolation_points()))
        v.interpolate(fem.Expression(self.u[1], space.element.interpolation_points()))
        du, dv = fem.Function(self.velocity_space), fem.Function(self.velocity_space)
        du.interpolate(fem.Expression(grad(u), self.velocity_space.element.interpolation_points()))
        dv.interpolate(fem.Expression(grad(v), self.velocity_space.element.interpolation_points()))
        dudy, dvdx = fem.Function(self.density_space), fem.Function(self.density_space)
        dudy.interpolate(fem.Expression(du[1], self.density_space.element.interpolation_points()))
        dvdx.interpolate(fem.Expression(dv[0], self.density_space.element.interpolation_points()))
        omega.interpolate(fem.Expression(ufl.sqrt((dvdx - dudy)**2), self.density_space.element.interpolation_points()))
        return omega
    

    def time_step(self):
        """Perform a single timestep of the defined problem."""
        # solve for intermediate step, store in self.u
        self.predictor_problem.solve()
        # solver for pressure
        self.pressure_problem.solve()
        # solver for corrected u, overwrite the previous u
        self.correction_problem.solve()
        self.t += self.dt
        self.p_last.x.array[:] = self.p.x.array[:]
        self.u_last.x.array[:] = self.u.x.array

    def to_steady_state(self, tol):
        max_iters = 10000
        last_u = np.copy(self.u.x.array)
        last_p = np.copy(self.p.x.array)
        for _ in range(max_iters):
            self.time_step()
            if (e := np.linalg.norm(last_u - self.u.x.array)) < tol:
                break
            print(f"resdiual = {e:.4e}")
            last_u[:] = self.u.x.array[:]
            last_p[:] = self.p.x.array[:]



# from common import *
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotx
# from visualise import *
# from common import *
# from scipy.integrate import simpson

# def poiseuille_flow_test():
#     def compute_error(h):
#         n = math.ceil(1 / h)
#         mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
#         solver = IncompressibleNavierStokesSolver(mesh, h / 250)
#         exact = dolfinx.fem.Function(solver.velocity_space)
#         exact.interpolate(lambda x: (1/2 * x[1] * (1 - x[1]), 0 * x[0]))
#         solver.set_velocity_bc(y_equals(1), constant((0, 0), mesh, solver.velocity_space))
#         solver.set_velocity_bc(y_equals(0), constant((0, 0), mesh, solver.velocity_space))
#         solver.set_velocity_bc(x_equals(1), exact)
#         solver.set_velocity_bc(x_equals(0), exact)
#         # solver.set_pressure_bc(x_equals(0), constant(1, mesh, solver.pressure_space))
#         # solver.set_pressure_bc(x_equals(1), constant(0, mesh, solver.pressure_space))
#         solver.reset()
#         solver.to_steady_state(1e-8)
#         y = np.linspace(0, 1, 250)
#         x = np.full(250, 0.5)
#         x, y, u, _ = fem_vector_func_at_given_points(solver.u, mesh, dolfinx.geometry.bb_tree(mesh, 2), x, y)
#         axes = plt.axes()
#         u_e = 1/2 * y * (1 - y)
#         return simpson(y=np.abs(u - u_e), x=y)
#     compute_convergence(compute_error, [2, 4, 8, 16])

       

# def couette_flow_test():
#     def compute_error(h):
#         n = math.ceil(1 / h)
#         mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
#         solver = IncompressibleNavierStokesSolver(mesh, h / 500)
#         exact = dolfinx.fem.Function(solver.velocity_space)
#         exact.interpolate(lambda x: (1/2 * x[1] * (1 - x[1]), 0 * x[0]))
#         solver.set_velocity_bc(y_equals(1), constant((1, 0), mesh, solver.velocity_space))
#         solver.set_velocity_bc(y_equals(0), constant((0, 0), mesh, solver.velocity_space))
#         solver.set_y_velocity(x_equals(0), dolfinx.default_scalar_type(0))
#         solver.set_y_velocity(x_equals(1), dolfinx.default_scalar_type(0))
#         solver.reset()
#         T = 0.1
#         step_until(T, solver, lambda s: s.time_step())
#         y = np.linspace(0, 1, 250)
#         x = np.full(250, 0.5)
#         x, y, u, _ = fem_vector_func_at_given_points(solver.u, mesh, dolfinx.geometry.bb_tree(mesh, 2), x, y)
#         u_e = y - 2 / np.pi * sum(1/n * np.exp(-n**2*np.pi**2*T) * np.sin(n*np.pi*(1 - y)) for n in range(1, 100))
#         return simpson(y=np.abs(u - u_e), x=y)
#     compute_convergence(compute_error, [2, 3, 4, 5, 6, 7, 8])
