from mesh2d import rectangle
from plotter import Plotter
from twophase import IncompressibleTwoPhaseFlowSolver
from clevelset import box_phi, line_phi
from bc import *
from visualise import *
from scipy import integrate
import sys



    
Re = 1#int(sys.argv[1])
c_normal = 0.1

print(c_normal)

n = 32
h = 1 / n
d = 0.1
eps = h ** (1 - d) / 2
mesh, tree = rectangle((-1, -1), (1, 1), h)
a = 0.75
box = line_phi(1, 1, eps)
We = 1
D = 1
M = 0.01
solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 1, D, 1 / Re, M / Re, 1 / We, 0, box, d=d, kinematic=True, c_kappa=1, c_normal=c_normal)

no_slip = constant((0, 0), mesh, solver.velosity_space)

solver.set_no_slip_everywhere()

aspects = []

#solver.set_time_step_proc(write_aspect)

path = f"sols/retract-full-{Re}"

#with open(f"records/aspect-{Re}.txt", "w") as infile:
 #   infile.write("\n".join(map(str, aspects)))


plotter = Plotter(solver, (-1, 1), (-1, 1), 0.2, filename=path, phi_points=100, contor_color="lightblue", colorbar=True)
plotter.plot_from_solver()
plt.show()