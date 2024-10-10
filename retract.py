from mesh2d import unit_square
from plotter import Plotter
from twophase import IncompressibleTwoPhaseFlowSolver
from clevelset import box_phi
from bc import *

n = 100
h = 1 / n
d = 0.05
eps = h ** (1 - d) / 2
mesh, tree = unit_square(h)
phi0 = box_phi(-0.5, -0.5, 0.5, 0.5, eps)
Re = 1
We = 1
D = 1
M = 0.01
solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 1, D, 1 / Re, M / Re, 1 / We, 0, phi0, d=d, c_kappa=20)

no_slip = constant((0, 0), mesh, solver.velosity_space)

solver.set_velosity_bc(y_equals(1), no_slip)
solver.set_velosity_bc(x_equals(1), no_slip)

solver.set_x_velocity(x_equals(0), default_scalar_type(0))
solver.set_y_velocity(y_equals(0), default_scalar_type(0))

path = "sols/retract"
solver.save_to_files(path, 1, 5)
plotter = Plotter(solver, (0, 1), (0, 1), 0.2, interface_points=0, phi_points=250, filename=path, contor_color="lightblue")
# plotter.plot_from_solver()
# plotter.show()
plotter.save_to_mp4("videos/retract.mp4")
