import numpy as np
from dolfinx import fem, geometry, default_scalar_type
from dolfinx.mesh import exterior_facet_indices
from petsc4py import PETSc
import ufl
from ufl import inner, dot, grad, dx, div

from lib.solver import *
from lib.bc import *

def make_marker_function(cx, cy, r, fluid_level, eps):
    def phi(x, y):
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if fluid_level is not None:
            return 1 / (1 + np.exp((dist - r) / eps)) + 1 / (1 + np.exp(-(y - fluid_level) / eps))
        return 1 / (1 + np.exp((dist - r) / eps))
    return phi

class NormalSolver:

    def __init__(self, mesh, space, f) -> None:
        self.mesh = mesh
        self.space = space
        self.f = f

        n_trail = ufl.TrialFunction(space)
        n_test = ufl.TestFunction(space)
        grad_f = grad(self.f)
        grad_len = ufl.sqrt(grad_f[0]**2 + grad_f[1]**2)
        self.a = fem.form(inner(n_trail, n_test) * dx)
        self.L = fem.form(inner(grad_f / grad_len, n_test) * dx)
        
        self.solver, self.b = create_solver(mesh, self.a, self.L, [])

    def compute_normals_of(self, f, n):
        self.f.x.array[:] = f.x.array[:]
        #self.solver, self.b = create_solver(self.mesh, self.a, self.L, [])
        update_solver(self.solver, self.a, self.L, self.b, [], n)

# class NormalSolver: # "naive" way

#     def __init__(self, mesh, space, f) -> None:
#         self.mesh = mesh
#         self.space = space
#         self.f = f
#         self.grad = fem.Function(space)
#         self.grad_solver = GradientSolver(mesh, space, f)

#     def compute_normals_of(self, f, n):
#         self.grad_solver.compute_gradient_of(f, self.grad)
#         length = ufl.sqrt(self.grad[0]**2 + self.grad[1]**2)
#         n_exp = fem.Expression(self.grad / length, self.space.element.interpolation_points())
#         n.interpolate(n_exp)



class GradientSolver:

    def __init__(self, mesh, space, f) -> None:
        self.mesh = mesh
        self.space = space
        self.f = f

        grad_trail = ufl.TrialFunction(space)
        grad_test = ufl.TestFunction(space)
        grad_f = grad(self.f)
        self.a = fem.form(inner(grad_trail, grad_test) * dx)
        self.L = fem.form(inner(grad_f, grad_test) * dx)
        
        self.solver, self.b = create_solver(mesh, self.a, self.L, [])

    def compute_gradient_of(self, f, g):
        self.f.x.array[:] = f.x.array[:]
        #self.solver, self.b = create_solver(self.mesh, self.a, self.L, [])
        update_solver(self.solver, self.a, self.L, self.b, [], g)

class CurvatureSolver:

    def __init__(self, mesh, h, k_space, grad_space, f) -> None:
        self.mesh = mesh
        self.space = k_space
        self.f = f
        self.grad_solver = GradientSolver(mesh, grad_space, f)
        self.grad = fem.Function(grad_space)

        k_trail = ufl.TrialFunction(k_space)
        k_test = ufl.TestFunction(k_space)

        grad_len = ufl.sqrt(self.grad[0]**2 + self.grad[1]**2)
        k_form = (
            k_trail * k_test * dx
            - inner(self.grad / grad_len, grad(k_test)) * dx
            #+ (h / 10) * inner(grad(k_trail), grad(k_test)) * dx
        )

        self.a = fem.form(ufl.lhs(k_form))
        self.L = fem.form(ufl.rhs(k_form))

        self.solver, self.b = create_solver(mesh, self.a, self.L, [])

    def compute_curvature_of(self, f, k):
        self.f.x.array[:] = f.x.array[:]
        self.grad_solver.compute_gradient_of(f, self.grad)
        update_solver(self.solver, self.a, self.L, self.b, [], k)
        
# class CurvatureSolver: # "naive" way

#     def __init__(self, mesh, h, k_space, grad_space, f) -> None:
#         self.mesh = mesh
#         self.space = k_space
#         self.f = f
#         self.normal_solver = NormalSolver(mesh, grad_space, f)
#         self.grad_solver = GradientSolver(mesh, grad_space, f)
#         self.n = fem.Function(grad_space)
#         self.grad = fem.Function(grad_space)

#     def compute_curvature_of(self, f, k):
#         self.normal_solver.compute_normals_of(f, self.n)
#         self.grad = self.normal_solver.grad
#         kappa = fem.Expression(-ufl.div(self.n), self.space.element.interpolation_points())
#         k.interpolate(kappa)


class LevelSetReinitializer:

    def __init__(self, mesh, k_tau, epsilon, phi_space, normal_space) -> None:
        self.mesh = mesh
        self.phi_space = phi_space
        self.normal_space = normal_space
        # Marker function for previous and currect time step
        self.phi_prev = fem.Function(self.phi_space)
        self.phi = fem.Function(self.phi_space)
        # Marker trial function
        phi_trial = ufl.TrialFunction(self.phi_space)
        # Marker test function
        phi_test = ufl.TestFunction(self.phi_space)
        # Constants
        dtau = fem.Constant(mesh, PETSc.ScalarType(k_tau))
        eps = fem.Constant(mesh, PETSc.ScalarType(epsilon))
        self.eps = eps
        self.bc = []

        self.n = fem.Function(self.normal_space)

        # reinit_form = inner(phi_trial - self.phi_prev, phi_test)*dx \
        #             - dtau*inner(phi_trial*(1 - self.phi_prev), dot(grad(phi_test), self.n))*dx \
        #             + (eps*dtau/2)*dot(grad(phi_trial + self.phi_prev), self.n)*dot(grad(phi_test), self.n)*dx

        reinit_form = (
            inner(phi_trial - self.phi_prev, phi_test) * dx
            - dtau * (0.5*(self.phi_prev + phi_trial) - self.phi_prev*phi_trial) * inner(self.n, grad(phi_test)) * dx
            + dtau * eps * inner(grad(0.5*(self.phi_prev + phi_trial)), self.n) * inner(self.n, grad(phi_test)) * dx
        )
  
        self.reinit_rhs = fem.form(ufl.rhs(reinit_form))
        self.reinit_lhs = fem.form(ufl.lhs(reinit_form))
        
        self.k_tau = k_tau
        self.tau = 0
        self.epsilon = epsilon

        

    def set_phi(self, phi_values):
        self.phi_prev.x.array[:] = phi_values

    def set_normals(self, n_values):
        self.n.x.array[:] = n_values
        self.reinit_solver, self.reinit_rhs_vec = create_solver(self.mesh, self.reinit_lhs, self.reinit_rhs, self.bc)

    def reinitialization(self, tol, show_ouput=False):
        max_iters = 1000
        tau = 0
        for i in range(max_iters):
            update_solver(self.reinit_solver, self.reinit_lhs, self.reinit_rhs, self.reinit_rhs_vec, self.bc, self.phi)
            tau += self.k_tau
            if (res := np.linalg.norm(self.phi.x.array - self.phi_prev.x.array)) < tol:
                return tau, i + 1
            if np.isclose(np.linalg.norm(self.phi.x.array), 0):
                raise RuntimeError("The level set blew up :(")
            if show_ouput:
                print(f"res = {res:.4e}")
            self.phi_prev.x.array[:] = self.phi.x.array[:]
        raise RuntimeError("The marker function didn't reinitialize :(")
        


class LevelSetSolver:

    def __init__(self, mesh, phi_init, _dx, _dt, _dtau, epsilon, phi_space, normal_space, velocity_space, reinit_tol=1e-4, output=False) -> None:
        # BB Tree of the mesh
        self.mesh = mesh
        # Function spaces
        self.phi_space = phi_space
        self.normal_space = normal_space
        self.velocity_space = velocity_space
        # Marker function
        self.phi = fem.Function(phi_space)
        # Velocity field
        self.u = fem.Function(velocity_space)
        # Marker trial function
        phi_trial = ufl.TrialFunction(phi_space)
        # Marker test function
        phi_test = ufl.TestFunction(phi_space)
        # Marker function inital condtion
        self.phi.interpolate(lambda x: phi_init(x[0], x[1]))

        # Normal forms
        self.grad_phi = fem.Function(normal_space)
        # grad_trail = ufl.TrialFunction(normal_space)
        # grad_test = ufl.TestFunction(normal_space)
        self.n = fem.Function(normal_space)
        self.kappa = fem.Function(phi_space)
        self.eps = epsilon
        self.bc = []
        self.boundary_mask = None
     

        # Advection forms
        dt = fem.Constant(mesh, default_scalar_type(_dt))
        h = fem.Constant(mesh, default_scalar_type(_dx))
        eps = fem.Constant(mesh, default_scalar_type(epsilon))
        # Fully explict
        self.advection_lhs = fem.form(inner(phi_trial, phi_test)*dx)
        self.advection_rhs = fem.form(inner(self.phi, phi_test)*dx + dt*inner(self.phi * self.u, grad(phi_test))*dx)


        self.t = 0
        self.k = _dt
        # Solver for reinitialation equations
        self.reinitializer = LevelSetReinitializer(mesh, _dtau, epsilon, self.phi_space, self.normal_space)
        self.output = output
        self.reinit_tol = reinit_tol
        self.normal_solver = NormalSolver(mesh, normal_space, self.phi)
        self.curvature_solver = CurvatureSolver(mesh, _dx, phi_space, normal_space, self.phi)

    def set_velosity(self, values):
        self.u.x.array[:] = values
        self.solver, self.advection_rhs_vec = create_solver(self.mesh, self.advection_lhs, self.advection_rhs, self.bc)

    def use_boundary_mask(self, s, xbounds, ybounds):
        self.boundary_mask = fem.Function(self.phi_space)
        S = lambda x, a: 0.5 * (1 + np.tanh((x - a) / s))
        a, b = xbounds
        c, d = ybounds
        self.boundary_mask.interpolate(
            lambda x: S(x[0], a) - S(x[0], b) + S(x[1], c) - S(x[1], d) - 1
        )
        

    def compute_normals(self):
        #update_solver(self.normal_solver, self.normal_lhs, self.normal_rhs, self.normal_rhs_vec, [], self.grad_phi)
        #mag = ufl.sqrt(self.grad_phi[0]**2 + self.grad_phi[1]**2)
        #self.grad_mag.interpolate(fem.Expression(mag, self.phi_space.element.interpolation_points()))
        #self.n.interpolate(fem.Expression(self.grad_phi / self.grad_mag, self.normal_space.element.interpolation_points()))
        self.normal_solver.compute_normals_of(self.phi, self.n)
        if self.boundary_mask is not None:
            self.n.interpolate(
                fem.Expression(self.boundary_mask * self.n, self.normal_space.element.interpolation_points())
            )

    def compute_gradient(self):
        self.grad_solver.compute_gradient_of(self.phi, self.grad_phi)

    def compute_curvature(self):
        self.curvature_solver.compute_curvature_of(self.phi, self.kappa)
        if self.boundary_mask is not None:
            self.kappa.interpolate(
                fem.Expression(self.boundary_mask * self.kappa, self.normal_space.element.interpolation_points())
            )

    def advection_step(self, reinit=True):
        # Advect one step
        update_solver(self.solver, self.advection_lhs, self.advection_rhs, self.advection_rhs_vec, self.bc, self.phi)
        self.t += self.k
        # Set phi in the reinitializer
        self.reinitializer.set_phi(self.phi.x.array[:])
        last = self.phi.x.array.copy()
        iters = 0
        if reinit:
            # Compute the normals of advected phi once, tau = 0
            self.compute_normals()
            #self.normal_solver.compute_normals_of(self.phi, self.n)
            self.reinitializer.set_normals(self.n.x.array[:])
            # reinitialize the level set
            tau, iters = self.reinitializer.reinitialization(self.reinit_tol)
            # save the reinitialized phi
            self.phi.x.array[:] = self.reinitializer.phi_prev.x.array[:]
        if self.output:
            print(f"Advection step: t = {self.t:.4f}, iterations: {iters}")

    def reinitalise(self):
        self.reinitializer.set_phi(self.phi.x.array[:])
        self.compute_normals()
        #self.normal_solver.compute_normals_of(self.phi, self.n)
        self.reinitializer.set_normals(self.n.x.array[:])
        # reinitialize the level set
        tau, iters = self.reinitializer.reinitialization(self.reinit_tol)
        # save the reinitialized phi
        self.phi.x.array[:] = self.reinitializer.phi_prev.x.array[:]

    def set_zero_bc(self):
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        boundary_facets = exterior_facet_indices(self.mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(self.phi_space, fdim, boundary_facets)

        no_slip = fem.Function(self.phi_space)
        no_slip.interpolate(constant(0, self.mesh, self.phi_space))

        bc = fem.dirichletbc(no_slip, boundary_dofs)
        self.bc.append(bc)
        self.reinitializer.bc.append(bc)