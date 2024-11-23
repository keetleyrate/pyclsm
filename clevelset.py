import dolfinx
import dolfinx.fem.petsc
import ufl
from ufl import inner, dx, grad
from normal import NormalProjector
from ellipticproject import EllipticProjector
from scipy.integrate import simpson

def circular_level_set(cx, cy, r, eps):
    def phi(w):
        x, y = w[0], w[1]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        return 1 / (1 + np.exp((dist - r)/eps))
    return phi

def phi(x, a, eps):
    return 1 / (1 + np.exp((a - x) / eps))

def box_phi(x1, y1, x2, y2, eps):
    def f(w):
        x, y = w[0], w[1]
        return 1 - (phi(x, x1, eps) - phi(x, x2, eps)) * (1 - phi(y, y2, eps)) * phi(y, y1, eps)
    return f

def line_phi(u, v, eps):
    def f(w):
        x, y = w[0], w[1]
        delta = (u * x + v * y) / (u**2 + v**2)
        dist = np.sqrt((x - delta * u)**2 + (y - delta * v)**2)
        return 1 / (1 + np.exp(dist / eps))
    return f


class ConservativeLevelSet:

    def __init__(self, mesh, h, dt, phi0, p=1, d=0.1, tol=1, c_normal=2, c_kappa=1, max_reinit_iters=1000) -> None:
        V_phi = dolfinx.fem.functionspace(mesh, ("CG", p))
        V_n = dolfinx.fem.VectorFunctionSpace(mesh, ("P", p))
        self.psi = ufl.TestFunction(V_phi)
        self.phi_t = ufl.TrialFunction(V_phi)
        self.phi = dolfinx.fem.Function(V_phi)
        self.n = dolfinx.fem.Function(V_n)
        self.phi.interpolate(phi0)
        self.h = h
        self.dt = dt
        self.dtau = h**(1 + d) / 2
        self.eps = h**(1 - d) / 2
        self.max_iters = max_reinit_iters
        self.tol = tol
        self.advection_lhs = ufl.inner(self.phi_t, self.psi) * ufl.dx
        self.normal_projector = NormalProjector(V_n, h, c_e=c_normal)
        self.projector = EllipticProjector(V_phi, h, c_kappa)
        self.kappa = dolfinx.fem.Function(V_phi)


    def set_u(self, u):
        self.u = u
        self.advection_rhs = ufl.inner(self.phi, self.psi)*ufl.dx + self.dt*ufl.inner(self.phi * self.u, ufl.grad(self.psi))*ufl.dx
        self.advection_solver = dolfinx.fem.petsc.LinearProblem(
            dolfinx.fem.form(self.advection_lhs),
            dolfinx.fem.form(self.advection_rhs),
            [],
            self.phi,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"}
        )

    def advect(self):
        self.advection_solver.solve()

    def prepare_reinit(self):
        self.normal_projector.set_function(self.phi)
        self.n = self.normal_projector.compute_normals()
        self.reinit_form = (
            inner(self.phi_t - self.phi, self.psi) * dx
            - self.dtau * (0.5*(self.phi + self.phi_t) - self.phi*self.phi_t) * inner(self.n, grad(self.psi)) * dx
            + self.dtau * self.eps * inner(grad(0.5*(self.phi + self.phi_t)), self.n) * inner(self.n, grad(self.psi)) * dx
        )
        self.reinit_rhs = dolfinx.fem.form(ufl.rhs(self.reinit_form))
        self.reinit_lhs = dolfinx.fem.form(ufl.lhs(self.reinit_form))
        self.reinit_solver = dolfinx.fem.petsc.LinearProblem(
            dolfinx.fem.form(self.reinit_lhs),
            dolfinx.fem.form(self.reinit_rhs),
            [],
            self.phi,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"}
        )

    def reinitalise(self):
        tol = self.dtau * self.tol
        for i in range(self.max_iters):
            prev = np.copy(self.phi.x.array)
            self.reinit_solver.solve()
            if np.linalg.norm(self.phi.x.array - prev) < tol:
                return i + 1
            if np.isclose(np.linalg.norm(self.phi.x.array), 0):
                raise RuntimeError("The level set blew up :(")
        raise RuntimeError("The marker function didn't reinitialize :(")
    
    def transport(self, u):
        self.set_u(u)
        self.advect()
        self.prepare_reinit()
        self.reinitalise()

    def compute_curvature(self):
        self.normal_projector.set_function(self.phi)
        self.n = self.normal_projector.compute_normals()
        self.projector.set_projected_function(-ufl.div(self.n))
        self.kappa = self.projector.project()
        return self.kappa

    


from mesh2d import rectangle
import numpy as np
import matplotlib.pyplot as plt
import matplotx
from visualise import *
from common import *
plt.style.use(matplotx.styles.ayu["dark"])

def constant_test():
    hs = []
    es = []
    for n in [2, 4, 8, 16, 32]:
        h = 1 / n
        mesh, _ = rectangle((0, -0.5), (2, 0.5), h)
        V_u = dolfinx.fem.VectorFunctionSpace(mesh, ("P", 2))
        u = dolfinx.fem.Function(V_u)
        u.interpolate(constant((1, 0), mesh, V_u))
        d = 0.1
        solver = ConservativeLevelSet(mesh, h, h / 10, circular_level_set(0.5, 0, 0.25, h**(1 - d) / 2), tol=1, p=1)
        T = 1
        step_until(T, solver, lambda s: s.transport(u))

        phi_exact = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1)))
        phi_exact.interpolate(circular_level_set(1.5, 0, 0.25, h**(1 - d) / 2))

        error = ufl.sqrt(ufl.inner(solver.phi - phi_exact, solver.phi - phi_exact)) * ufl.dx
        es.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(error)))
        hs.append(h)

    h = np.array(hs)
    e = np.array(es)

    logh = np.log2(h)
    loge = np.log2(e)

    a, _ = np.polyfit(logh, loge, deg=1)
    print("Convergence: ", a)


def curvature_test():
    def kappa_error(h):
        mesh, tree = rectangle((-1, -1), (1, 1), h)
        V_u = dolfinx.fem.VectorFunctionSpace(mesh, ("P", 2))
        u = dolfinx.fem.Function(V_u)
        u.interpolate(lambda x: (-2 * np.pi * x[1], 2 * np.pi * x[0]))
        d = 0.1
        r = 0.25
        solver = ConservativeLevelSet(mesh, h, h / 25, circular_level_set(0.5, 0, r, h**(1 - d) / 2), tol=1, p=1, c_normal=10, c_kappa=20)
        T = 1
        step_until(T, solver, lambda s: s.transport(u))
        theta = np.linspace(0, 2 * np.pi, 600)
        kappa_h = solver.compute_curvature()
        x, y, kappa_circle = fem_scalar_func_at_given_points(kappa_h, mesh, tree, 0.5 + r * np.cos(theta), r * np.sin(theta))
        kappa_e = np.sqrt(simpson(np.square(kappa_circle.reshape((600,)) - 1 / r), theta))
        return kappa_e
    compute_convergence(kappa_error, [2, 6, 10, 14, 18, 22])


