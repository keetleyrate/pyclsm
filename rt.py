from mesh2d import rectangle
from plotter import Plotter
from twophase import IncompressibleTwoPhaseFlowSolver
from clevelset import phi, circular_level_set
from bc import *
from visualise import *
from scipy import integrate
import sys
from common import step_until

def phi0(eps):
    def _phi0(w):
        x, y = w[0], w[1]
        blob = circular_level_set(0.5, 2, 0.1, eps)
        return 1 - (1 - phi(y, 2, eps)) *  (1 - blob(w))
    return _phi0

c_normal = 0.1



n = 128
h = 1 / n
d = 0.1
eps = h ** (1 - d) / 2
mesh, tree = rectangle((0, 0), (1, 3), h)
a = 0.75
box = phi0(eps)
Re = 1
We = 10
D = 10
M = 1
Fr = 1
solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 1, D, 1 / Re, M / Re, 1 / We, 1 / Fr**2, box, d=d, kinematic=True, c_kappa=1, c_normal=c_normal)

no_slip = constant((0, 0), mesh, solver.velosity_space)

solver.set_no_slip_everywhere()


path = f"sols/rt-Re-{Re}-We-{We}"
T = 50
solver.save_to_files(path, T, 5)
plotter = Plotter(solver, (0, 1), (0, 3), 0.2, filename=path, density_points=100, levels=100)
plotter.save_to_mp4(f"videos/rt-Re-{Re}-We-{We}.mp4")
