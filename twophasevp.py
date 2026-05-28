from twophaseflow import IncompressibleTwoPhaseFlowSolver
import dolfinx
from common.common import *
import ufl
from ufl import dot, inner, nabla_grad, dx, ds, grad

class RegBinghamTwoPhaseSolver(IncompressibleTwoPhaseFlowSolver):

    def __init__(self, ls_solver, dt, rho0, rho1, mu0, mu1, sigma, yield_stress, kinematic_bc=False, epsilon=1e-3) -> None:
        super().__init__(ls_solver, dt, rho0, rho1, mu0, mu1, sigma, kinematic_bc)
        self.ϵ = dolfinx.fem.Constant(ls_solver.domain.mesh, dolfinx.default_scalar_type(epsilon))
        self.τ = dolfinx.fem.Constant(ls_solver.domain.mesh, dolfinx.default_scalar_type(yield_stress))
        self.gamma_dot = dolfinx.fem.Function(self.scalar_space)
        self.η = dolfinx.fem.Function(self.scalar_space)

    def compute_mu(self):
        ϕ = self.ls.ϕ
        γ = ufl.nabla_grad(self.u_last) + ufl.nabla_grad(self.u_last).T
        gamma_abs = ufl.sqrt(0.5 * ufl.inner(γ, γ))
        η = self.mu1 + self.τ / (gamma_abs + self.ϵ)
        μ_eff = self.mu0 * (1 - ϕ) + η * ϕ
        interpolate_expression(μ_eff, self.mu)


    
    def build_predictor_problem(self, f):
        ϕ = self.ls.ϕ
        γ = ufl.nabla_grad(self.u_last) + ufl.nabla_grad(self.u_last).T
        gamma_abs = ufl.sqrt(0.5 * ufl.inner(γ, γ))
        η = self.mu1 + self.τ / (gamma_abs + self.ϵ)
        μ_eff = self.mu0 * (1 - ϕ) + η * ϕ
        def σ(u, p):
            return -p * ufl.Identity(2) + μ_eff * (ufl.grad(u) + ufl.grad(u).T)
        trail, test = self.u_trail, self.u_test
        n = ufl.FacetNormal(self.domain.mesh)
        dt = self.dt
        form = (
            dot((self.rho * trail - self.rho_last * self.u_last) / dt, test) * dx
            + inner(σ((self.u_last + trail)/2, self.p_last), nabla_grad(test)) * dx
            + dot(self.p_last * n, test) * ds 
            - dot(μ_eff * nabla_grad((self.u_last + trail)/2) * n, test) * ds
            - dot(f, test) * dx
        )
        if self.kinematic:
            form += - inner(dot(self.u_last, grad(test)), self.rho * trail) * dx
        else:
            form += self.rho * dot(dot(trail, nabla_grad(self.u_last)), test) * dx
        self.predictor_problem = dolfinx.fem.petsc.LinearProblem(
            dolfinx.fem.form(ufl.lhs(form)),
            dolfinx.fem.form(ufl.rhs(form)),
            bcs=self.u_bcs,
            u=self.u_star,
            petsc_options={"ksp_type": "bcgs", "pc_type": "jacobi"},
            petsc_options_prefix="u_predictor_"
        )
        
