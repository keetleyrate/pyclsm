from mesh2d import unit_square
from plotter import Plotter
from twophase import IncompressibleTwoPhaseFlowSolver
from clevelset import box_phi, ConservativeLevelSet, circular_level_set
from bc import *
from visualise import *
from scipy import integrate
import sys
from common import *

def advection(solver):
    solver.set_u(u)
    solver.advect()


n = 32
h = 1 / n
d = 0.1
eps = h ** (1 - d) / 2
mesh, tree = unit_square(h)
a = 0.5
phi0 = circular_level_set(0, 0, 0.5, eps)

u_space = fem.VectorFunctionSpace(mesh, ("CG", 2))
c_normal = 0.1
U = 1
u = fem.Function(u_space)
u.interpolate(lambda x: (-U * x[0], U * x[1]))
solver = ConservativeLevelSet(mesh, h, h / 10, phi0, d=d, c_normal=c_normal) #use c = h/2

fig, axes = plt.subplots()
fem_plot_contor(fig, axes, solver.phi, mesh, tree, (0, 1), (0, 1), 250, levels=[0.5], colors=["white"])

T = 1
step_until(T / 2, solver, lambda s: s.transport(u))

fem_plot_contor(fig, axes, solver.phi, mesh, tree, (0, 1), (0, 1), 250, levels=[0.5], colors=["red"])

u.interpolate(lambda x: (U * x[0], -U * x[1]))
step_until(T / 2, solver, lambda s: s.transport(u))


fem_plot_contor(fig, axes, solver.phi, mesh, tree, (0, 1), (0, 1), 250, levels=[0.5], colors=["blue"])
plt.savefig(f"contact-line-c-{c_normal}.png", dpi=400)  