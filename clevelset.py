import dolfinx
import dolfinx.fem.petsc
import ufl
from normal import NormalProjector

def circular_level_set(cx, cy, r, eps):
    def phi(w):
        x, y = w[0], w[1]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        return 1 / (1 + np.exp((dist - r) / eps))
    return phi

class ConservativeLevelSet:

    def __init__(self, mesh, h, dt, phi0, p=1) -> None:
        V_phi = dolfinx.fem.functionspace(mesh, ("P", p))
        self.psi = ufl.TestFunction(V_phi)
        phi_t = ufl.TrialFunction(V_phi)
        self.phi = dolfinx.fem.Function(V_phi)
        self.phi.interpolate(phi0)
        self.h = h
        self.dt = dt
    
        self.advection_lhs = ufl.inner(phi_t, self.psi) * ufl.dx

        self.normal_projector = NormalProjector(dolfinx.fem.VectorFunctionSpace(mesh, ("P", p)), h)

        

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


from mesh2d import rectangle
import numpy as np
import matplotlib.pyplot as plt
import matplotx
from visualise import *
from common import *
from tqdm import tqdm
import math

plt.style.use(matplotx.styles.ayu["dark"])

h = 0.05
mesh, tree = rectangle((0, -0.5), (2, 0.5), h)
V_u = dolfinx.fem.VectorFunctionSpace(mesh, ("P", 2))
u = dolfinx.fem.Function(V_u)
u.interpolate(lambda x: (x[0] / x[0], 0 * x[0]))
d = 0.05
solver = ConservativeLevelSet(mesh, h, h / 10, circular_level_set(0.5, 0, 0.25, h**(1 - d) / 2))
solver.set_u(u)
T = 1
step_until(T, solver, lambda s: s.advect())


fig, axes = plt.subplots()
fem_plot_contor_filled(fig, axes, solver.phi, mesh, tree, (0, 2), (-0.5, 0.5), 100, levels=100)
fem_plot_contor(fig, axes, solver.phi, mesh, tree, (0, 2), (-0.5, 0.5), 100, levels=[0.5], colors=["white"])
axes.set_aspect("equal")
plt.show()
