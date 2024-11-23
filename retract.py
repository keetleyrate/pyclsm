from mesh2d import unit_square
from plotter import Plotter
from twophase import IncompressibleTwoPhaseFlowSolver
from clevelset import box_phi
from bc import *
from visualise import *
from scipy import integrate
import sys


def compute_aspect(phi, n):
    x = np.linspace(0, 1, n)
    y = np.zeros(len(x))
    _, _, phi_along_h = fem_scalar_func_at_given_points(phi, mesh, tree, x, y)
    _, _, phi_along_v = fem_scalar_func_at_given_points(phi, mesh, tree, y, x)
    a = 1 - integrate.trapz(phi_along_h.reshape(n), x)
    b = 1 - integrate.trapz(phi_along_v.reshape(n), x)
    return a, b


def write_aspect(solver):
    a, b = compute_aspect(solver.level_set.phi, 2 * n)
    aspects.append(a / b)


    
Re = int(sys.argv[1])

print(Re)

n = 32
h = 1 / n
d = 0.1
eps = h ** (1 - d) / 2
mesh, tree = unit_square(h)
a = 0.75
box = box_phi(-0.5, -0.5, a, a/4, eps)
We = 1
D = 1
M = 0.01
solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 1, D, 1 / Re, M / Re, 1 / We, 0, box, d=d, kinematic=False, c_normal=0.1)

no_slip = constant((0, 0), mesh, solver.velosity_space)

solver.set_velosity_bc(y_equals(1), no_slip)
solver.set_velosity_bc(x_equals(1), no_slip)

solver.set_x_velocity(x_equals(0), default_scalar_type(0))
solver.set_y_velocity(y_equals(0), default_scalar_type(0))

aspects = []

#solver.set_time_step_proc(write_aspect)

path = f"sols/retract-{Re}"
solver.save_to_files(path, 5, 5)

#with open(f"records/aspect-{Re}.txt", "w") as infile:
 #   infile.write("\n".join(map(str, aspects)))


plotter = Plotter(solver, (0, 1), (0, 1), 0.2, filename=path, visc_points=200, contor_color="lightblue")
#plotter.plot_from_solver()
# plotter.show()
plotter.save_to_mp4(f"videos/retract-{Re}.mp4")
