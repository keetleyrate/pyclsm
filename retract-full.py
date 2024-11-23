from mesh2d import rectangle
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


    
Re = 1#int(sys.argv[1])
c_normal = 0.1#float(sys.argv[1])

print(c_normal)

n = 32
h = 1 / n
d = 0.1
eps = h ** (1 - d) / 2
mesh, tree = rectangle((-1, -1), (1, 1), h)
a = 0.75
box = box_phi(-a, -a/4, a, a/4, eps)
We = 1
D = 1
M = 0.01
solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 1, D, 1 / Re, M / Re, 1 / We, 0, box, d=d, kinematic=True, c_kappa=1, c_normal=c_normal)

no_slip = constant((0, 0), mesh, solver.velosity_space)

solver.set_no_slip_everywhere()

aspects = []

#solver.set_time_step_proc(write_aspect)

path = f"sols/retract-full-{Re}"
T = 3
solver.time_step()
#solver.save_to_files(path, T, 5)

#with open(f"records/aspect-{Re}.txt", "w") as infile:
 #   infile.write("\n".join(map(str, aspects)))


plotter = Plotter(solver, (-1, 1), (-1, 1), 0.2, filename=path, visc_points=100, levels=100, contor_color="lightblue", colorbar=True)
#plotter.save_to_mp4(f"videos/retract-full-c-{c_normal}.mp4")
plotter.plot_from_solver()
plotter.show()
