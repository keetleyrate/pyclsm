from clevelset import ConservativeLevelSet, circular_level_set, box_phi
from twophase import IncompressibleTwoPhaseFlowSolver
from common import *
from plotter import *
import pathlib
import csv
import ufl

class ViscoPlasticTwoPhaseSolver(IncompressibleTwoPhaseFlowSolver):

    def __init__(self, mesh, h, dt, rho1, rho2, mu1, mu2, sigma, g, phi0, tau_Y, p_phi=1, which=0, d=0.1, kinematic=True, c_kappa=20, c_normal=1, epsilon=1e-3) -> None:
        super().__init__(mesh, h, dt, rho1, rho2, mu1, mu2, sigma, g, phi0, p_phi, d, kinematic, c_kappa, c_normal)
        self.epsilon = fem.Constant(mesh, dolfinx.default_scalar_type(epsilon))
        self.tau_Y = fem.Constant(mesh, dolfinx.default_scalar_type(tau_Y))
        self.gamma_dot = fem.Function(self.density_space)
        self.nu = fem.Function(self.density_space)
        self.which = which

    def compute_gamma_dot(self):
        γ = lambda u: ufl.nabla_grad(u) + ufl.nabla_grad(u).T
        gamma_abs = lambda u: ufl.sqrt(0.5 * ufl.inner(γ(u), γ(u)))
        self.gamma_dot.interpolate(fem.Expression(gamma_abs(self.u), self.density_space.element.interpolation_points()))

    def set_viscosity(self):
        self.compute_gamma_dot()
        phi = self.level_set.phi
        if self.which == 0: # newtonian fluid has phi=0
            η = self.mu2 + self.tau_Y / (self.gamma_dot + self.epsilon)
            mu = fem.Expression(
                self.mu1 + (η - self.mu1) * phi,
                self.density_space.element.interpolation_points()
            )
        else: # newtonian fluid has phi=1
            η = self.mu1 + self.tau_Y / (self.gamma_dot + self.epsilon)
            mu = fem.Expression(
                η + (self.mu2 - η) * phi,
                self.density_space.element.interpolation_points()
            )
        self.nu.interpolate(fem.Expression(η, self.density_space.element.interpolation_points()))
        self.mu.interpolate(mu)

    def time_step(self, steps=1):
        for _ in range(steps):
            self.level_set.transport(self.u)
            self.set_dencity()
            self.set_viscosity()
            self.set_body_forces()
            self.reset()
            super().compute_u()