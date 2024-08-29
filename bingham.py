from navierstokes import NSSolver
from petsc4py import PETSc
import ufl
from ufl import inner, dot, grad, dx, nabla_grad, div, ds
from solver import *
import numpy as np
from dolfinx import default_scalar_type, fem
from dolfinx.mesh import locate_entities_boundary
import dolfinx.fem.petsc as petsc


class BinghamSolver(NSSolver):

    def __init__(self, mesh, dt, velocity_space, pressure_space, density_space, epsilon, tau_Y, kinematic=False) -> None:
        super().__init__(mesh, dt, velocity_space, pressure_space, density_space)
        self.epsilon = fem.Constant(mesh, default_scalar_type(epsilon))
        self.tau_Y = fem.Constant(mesh, PETSc.ScalarType(tau_Y))
        self.gamma_dot = fem.Function(self.density_space)
        self.stress = fem.Function(self.density_space)

        u_t = ufl.TrialFunction(self.velosity_space)
        v = ufl.TestFunction(self.velosity_space)
        p_t = ufl.TrialFunction(self.pressure_space)
        q = ufl.TestFunction(self.pressure_space)

        n = ufl.FacetNormal(mesh)
        γ = lambda u: nabla_grad(u) + nabla_grad(u).T
        η = self.mu + self.tau_Y / (self.gamma_dot + self.epsilon)
        σ = lambda u, p: -p * ufl.Identity(2) + η * γ(u)
        U = (self.u + u_t) / 2
        
        int_step_form = (
            dot((self.rho * u_t - self.rho_prev * self.u) / dt, v) * dx
            + inner(σ((self.u + u_t)/2, self.p_prev), nabla_grad(v)) * dx
            + dot(self.p_prev * n, v) * ds 
            - dot(η * nabla_grad((self.u + u_t)/2) * n, v) * ds
            - dot(self.F, v) * dx
        )
        if kinematic:
            int_step_form += - inner(dot(self.u, grad(v)), self.rho * u_t) * dx
        else:
            int_step_form += self.rho * dot(dot(u_t, nabla_grad(self.u)), v) * dx
        # Pressure step form
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

    def compute_gamma_dot(self):
        γ = lambda u: nabla_grad(u) + nabla_grad(u).T
        gamma_abs = lambda u: ufl.sqrt(0.5 * ufl.inner(γ(u), γ(u)))
        self.gamma_dot.interpolate(fem.Expression(gamma_abs(self.u), self.density_space.element.interpolation_points()))

    def compute_stress(self):
        η = self.mu + self.tau_Y / (self.gamma_dot + self.epsilon)
        γ = lambda u: nabla_grad(u) + nabla_grad(u).T
        self.compute_gamma_dot()
        τ = η * γ(self.u)
        self.stress.interpolate(fem.Expression(ufl.sqrt(0.5 * ufl.inner(τ, τ)), self.density_space.element.interpolation_points()))

    def time_step(self):
        self.compute_gamma_dot()
        super().time_step()