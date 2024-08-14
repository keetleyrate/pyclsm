from lib.twophaseflow import *
from lib.navierstokes import NSSolver
from petsc4py import PETSc
import ufl
from ufl import inner, dot, grad, dx, nabla_grad, div, ds
from lib.solver import *
import numpy as np
from dolfinx import default_scalar_type, fem


class ViscoPlasticTwoPhaseSolver(TwoPhaseFlowSolver):

    def __init__(self, mesh, dx, dt, rho1, rho2, mu1, mu2, sigma, tau_Y, initial_phi_args, xbounds, ybounds, g=0, reinit_tol=1, output=True, d=0.1, kinematic=False, epsilon=1e-3) -> None:
        """Note: rho2 / mu2 fluid must be the viscoplasic fluid."""
        super().__init__(mesh, dx, dt, rho1, rho2, mu1, mu2, sigma, initial_phi_args, xbounds, ybounds, g, reinit_tol, output, d, kinematic)
        self.epsilon = fem.Constant(mesh, default_scalar_type(epsilon))
        self.tau_Y = fem.Constant(mesh, PETSc.ScalarType(tau_Y))
        self.gamma_dot = fem.Function(self.density_space)

    def compute_gamma_dot(self):
        γ = lambda u: nabla_grad(u) + nabla_grad(u).T
        gamma_abs = lambda u: ufl.sqrt(0.5 * ufl.inner(γ(u), γ(u)))
        u = self.fluid_solver.u
        self.gamma_dot.interpolate(fem.Expression(gamma_abs(u), self.density_space.element.interpolation_points()))

    def compute_viscosity(self):
        self.compute_gamma_dot()
        η = self.mu2 + self.tau_Y / (self.gamma_dot + self.epsilon)
        phi = self.level_set_solver.phi
        mu = fem.Expression(
            self.mu1 + (η - self.mu1) * phi,
            self.density_space.element.interpolation_points()
        )
        self.fluid_solver.mu.interpolate(mu)

    def time_step(self, steps=1):
        for _ in range(steps):
            # Update the advection problem
            self.level_set_solver.set_velosity(self.fluid_solver.u.x.array[:])
             # Advect interface
            self.level_set_solver.advection_step()
            # Solve for fluid
            self.compute_dencity()
            self.compute_viscosity()
            self.compute_surface_tension()
            self.compute_forces()
            self.fluid_solver.reset()
            self.fluid_solver.time_step()
            self.t += self.dt
