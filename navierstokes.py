from petsc4py import PETSc
import ufl
from ufl import inner, dot, grad, dx, nabla_grad, div, ds
import numpy as np
from dolfinx import fem, default_scalar_type
from dolfinx.mesh import locate_entities_boundary
import dolfinx.fem.petsc as petsc
import dolfinx
import mpi4py
import csv
import shutil
import pathlib

def u_dot_grad(u):
    return dot(u, ufl.nabla_grad(u))

class IncompressibleNavierStokesSolver:
    """Class for solving the incompressable Navier-Stokes equations using the fenicsx framework."""

    def __init__(self, mesh, timestep, p=2, density_viscosity_order=1, kinematic=False) -> None:
        """Creates a new solver object. Initalises variational forms used as well as solution
        fields for velocity, pressure, dencity and viscosity. Must be given the approprite
        function space objects the the soltuion will be defined over."""
        self.mesh = mesh
        # Taylor-Hood Elements
        self.pressure_space = fem.FunctionSpace(mesh, ("CG", p - 1))
        self.velosity_space = fem.VectorFunctionSpace(mesh, ("CG", p))
        self.density_space = fem.FunctionSpace(mesh, ("CG", density_viscosity_order))
        self.F_space = fem.VectorFunctionSpace(mesh, ("CG", 2))
        # Test and trail functions
        u_t = ufl.TrialFunction(self.velosity_space)
        v = ufl.TestFunction(self.velosity_space)
        p_t = ufl.TrialFunction(self.pressure_space)
        q = ufl.TestFunction(self.pressure_space)
        # Solution fields
        self.u = fem.Function(self.velosity_space)
        self.p = fem.Function(self.pressure_space)
        self.p_prev = fem.Function(self.pressure_space)
        self.F = fem.Function(self.F_space)
        self.rho = fem.Function(self.density_space)
        self.rho_prev = fem.Function(self.density_space)

        # Constants
        self.mu = fem.Function(self.density_space)
        dt = fem.Constant(mesh, PETSc.ScalarType(timestep))

        γ = lambda u: grad(u) + grad(u).T
        n = ufl.FacetNormal(mesh)
        
        σ = lambda u, p: -p * ufl.Identity(2) + self.mu * γ(u)
        
        int_step_form = (
            dot((self.rho * u_t - self.rho_prev * self.u) / dt, v) * dx
            + inner(σ((self.u + u_t)/2, self.p_prev), nabla_grad(v)) * dx
            + dot(self.p_prev * n, v) * ds 
            - dot(self.mu * nabla_grad((self.u + u_t)/2) * n, v) * ds
            - dot(self.F, v) * dx
        )
        if kinematic:
            int_step_form += - inner(dot(self.u, grad(v)), self.rho * u_t) * dx
        else:
            int_step_form += self.rho * dot(dot(u_t, nabla_grad(self.u)), v) * dx
        # Pressure step forms
        pressure_from = (
            (1 / dt) * q * div(self.u) * dx
            + inner((1 / self.rho)*grad(q), grad(p_t - self.p_prev)) * dx
        )
        # Correction step forms 
        correct_form = (
            (1 / dt) * inner(u_t - self.u, v) * dx
            + inner(grad(self.p - self.p_prev), (1 / self.rho) * v) * dx
        )
            
        # sort out the bilinear and linear forms
        self.int_step_a = fem.form(ufl.lhs(int_step_form))
        self.int_step_L = fem.form(ufl.rhs(int_step_form))
        self.pressure_step_a = fem.form(ufl.lhs(pressure_from))
        self.pressure_step_L = fem.form(ufl.rhs(pressure_from))
        self.correction_step_a = fem.form(ufl.lhs(correct_form))
        self.correction_step_L = fem.form(ufl.rhs(correct_form))
        # Create the velocity boundary condtions
        self.u_bcs = []  
        # Create the pressure boundary condtions
        self.p_bcs = []
        self.t = 0
        self.dt = timestep
        self.set_density_as_const(1)
        self.set_viscosity_as_const(1)
        self.reset()

    def set_pressure_bc(self, geometry_fn, fn):
        """Set a Dirichlet boundary condtion for pressure.
        geometry_fn: A function that returns True given a point where the boundary condtion
        will be placed.
        fn: The function (or constant) to be imposed."""
        p_boundary = fem.Function(self.pressure_space)
        p_boundary.interpolate(fn)
        dofs = fem.locate_dofs_geometrical(self.pressure_space, geometry_fn)
        self.p_bcs.append(fem.dirichletbc(p_boundary, dofs))

    def set_velosity_bc(self, geometry_fn, fn):
        """Set a Dirichlet boundary condtion for velosity.
        geometry_fn: A function that returns True given a point where the boundary condtion
        will be placed.
        fn: The function (or constant) to be imposed."""
        u_boundary = fem.Function(self.velosity_space)
        u_boundary.interpolate(fn)
        dofs = fem.locate_dofs_geometrical(self.velosity_space, geometry_fn)
        self.u_bcs.append(fem.dirichletbc(u_boundary, dofs))

    def set_x_velocity(self, geometry_fn, fn):
        """Set a Dirichlet boundary condtion on the x-component of velosity.
        geometry_fn: A function that returns True given a point where the boundary condtion
        will be placed.
        fn: The function (or constant) to be imposed."""
        boundary_facets = locate_entities_boundary(self.mesh, self.mesh.topology.dim - 1, geometry_fn) 
        boundary_dofs_x = fem.locate_dofs_topological(self.velosity_space.sub(0), self.mesh.topology.dim - 1, boundary_facets)
        self.u_bcs.append(fem.dirichletbc(fn, boundary_dofs_x, self.velosity_space.sub(0)))

    def set_y_velocity(self, geometry_fn, fn):
        """Set a Dirichlet boundary condtion on the y-component of velosity.
        geometry_fn: A function that returns True given a point where the boundary condtion
        will be placed.
        fn: The function (or constant) to be imposed."""
        boundary_facets = locate_entities_boundary(self.mesh, self.mesh.topology.dim - 1, geometry_fn) 
        boundary_dofs_x = fem.locate_dofs_topological(self.velosity_space.sub(1), self.mesh.topology.dim - 1, boundary_facets)
        self.u_bcs.append(fem.dirichletbc(fn, boundary_dofs_x, self.velosity_space.sub(1)))

    def set_density_as_const(self, rho_c):
        """Set the density of the fluid as a constant value."""
        self.rho.x.array[:] = rho_c * np.ones(self.rho.x.array.shape)
        self.rho_prev.x.array[:] = rho_c * np.ones(self.rho_prev.x.array.shape)

    def set_viscosity_as_const(self, mu_c):
        """Set the viscosity of the fluid as a constant value."""
        self.mu.x.array[:] = mu_c * np.ones(self.mu.x.array.shape)


    def compute_vorticity(self):
        space = fem.FunctionSpace(self.mesh, ("CG", 2))
        omega = fem.Function(self.density_space)
        u, v = fem.Function(space), fem.Function(space)
        u.interpolate(fem.Expression(self.u[0], space.element.interpolation_points()))
        v.interpolate(fem.Expression(self.u[1], space.element.interpolation_points()))
        du, dv = fem.Function(self.velosity_space), fem.Function(self.velosity_space)
        du.interpolate(fem.Expression(grad(u), self.velosity_space.element.interpolation_points()))
        dv.interpolate(fem.Expression(grad(v), self.velosity_space.element.interpolation_points()))
        dudy, dvdx = fem.Function(self.density_space), fem.Function(self.density_space)
        dudy.interpolate(fem.Expression(du[1], self.density_space.element.interpolation_points()))
        dvdx.interpolate(fem.Expression(dv[0], self.density_space.element.interpolation_points()))
        omega.interpolate(fem.Expression(ufl.sqrt((dvdx - dudy)**2), self.density_space.element.interpolation_points()))
        return omega

    def time_step(self):
        """Perform a single timestep of the defined problem."""
        # solve for intermediate step, store in self.u
        self.int_step_system.solve()
        # solver for pressure
        self.pressure_step_system.solve()
        # solver for corrected u, overwrite the previous u
        self.correction_step_system.solve()
        self.t += self.dt
        self.p_prev.x.array[:] = self.p.x.array[:]

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

    def reset(self):
        self.int_step_system = petsc.LinearProblem(self.int_step_a, self.int_step_L, self.u_bcs, self.u, petsc_options={"ksp_type": "bcgs", "pc_type": "jacobi"})
        self.pressure_step_system = petsc.LinearProblem(self.pressure_step_a, self.pressure_step_L, self.p_bcs, self.p, petsc_options={"ksp_type": "minres", "pc_type": "hypre"})
        self.correction_step_system = petsc.LinearProblem(self.correction_step_a, self.correction_step_L, self.u_bcs, self.u, petsc_options={"ksp_type": "cg", "pc_type": "sor"})


from common import *
import numpy as np
import matplotlib.pyplot as plt
import matplotx
from visualise import *
from common import *
from scipy.integrate import simpson

def poiseuille_flow_test():
    def compute_error(h):
        n = math.ceil(1 / h)
        mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
        solver = IncompressibleNavierStokesSolver(mesh, h / 250)
        exact = dolfinx.fem.Function(solver.velosity_space)
        exact.interpolate(lambda x: (1/2 * x[1] * (1 - x[1]), 0 * x[0]))
        solver.set_velosity_bc(y_equals(1), constant((0, 0), mesh, solver.velosity_space))
        solver.set_velosity_bc(y_equals(0), constant((0, 0), mesh, solver.velosity_space))
        solver.set_velosity_bc(x_equals(1), exact)
        solver.set_velosity_bc(x_equals(0), exact)
        # solver.set_pressure_bc(x_equals(0), constant(1, mesh, solver.pressure_space))
        # solver.set_pressure_bc(x_equals(1), constant(0, mesh, solver.pressure_space))
        solver.reset()
        solver.to_steady_state(1e-8)
        y = np.linspace(0, 1, 250)
        x = np.full(250, 0.5)
        x, y, u, _ = fem_vector_func_at_given_points(solver.u, mesh, dolfinx.geometry.bb_tree(mesh, 2), x, y)
        axes = plt.axes()
        u_e = 1/2 * y * (1 - y)
        return simpson(y=np.abs(u - u_e), x=y)
    compute_convergence(compute_error, [2, 4, 8, 16])

       

def couette_flow_test():
    def compute_error(h):
        n = math.ceil(1 / h)
        mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
        solver = IncompressibleNavierStokesSolver(mesh, h / 500)
        exact = dolfinx.fem.Function(solver.velosity_space)
        exact.interpolate(lambda x: (1/2 * x[1] * (1 - x[1]), 0 * x[0]))
        solver.set_velosity_bc(y_equals(1), constant((1, 0), mesh, solver.velosity_space))
        solver.set_velosity_bc(y_equals(0), constant((0, 0), mesh, solver.velosity_space))
        solver.set_y_velocity(x_equals(0), dolfinx.default_scalar_type(0))
        solver.set_y_velocity(x_equals(1), dolfinx.default_scalar_type(0))
        solver.reset()
        T = 0.1
        step_until(T, solver, lambda s: s.time_step())
        y = np.linspace(0, 1, 250)
        x = np.full(250, 0.5)
        x, y, u, _ = fem_vector_func_at_given_points(solver.u, mesh, dolfinx.geometry.bb_tree(mesh, 2), x, y)
        u_e = y - 2 / np.pi * sum(1/n * np.exp(-n**2*np.pi**2*T) * np.sin(n*np.pi*(1 - y)) for n in range(1, 100))
        return simpson(y=np.abs(u - u_e), x=y)
    compute_convergence(compute_error, 4)
