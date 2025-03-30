from twophaseflow import IncompressibleTwoPhaseFlowSolver
from common.domains import *
from common.visualize import *
from plotter import *
from scipy import integrate
from tqdm import tqdm
import numpy as np


def e_phi(w):
    x, y = w[0], w[1]
    dist = np.sqrt((x/L0)**2 + (y/B0)**2) - 1
    return 1 / (1 + np.exp(dist / solver.level_set.eps))

def compute_aspect(phi, n):
    x = np.linspace(-3, 3, n)
    y = np.zeros(len(x))
    _, _, phi_along_h = fem_scalar_func_at_given_points(phi, mesh, tree, x, y)
    _, _, phi_along_v = fem_scalar_func_at_given_points(phi, mesh, tree, y, x)
    a = integrate.trapz(phi_along_h.reshape(n), x) / 2
    b = integrate.trapz(phi_along_v.reshape(n), x) / 2
    return a, b

n = 32
h = 6 / n
L0 = 1.1055
B0 = 0.9045
D = 1
M = 1
Re = 1
We = 1

mu_drop  = 1 / Re
mu_matrix = M / Re
sigma = 1 / We

beta = mu_drop / mu_matrix
f = 40 * (beta + 1) / ((2*beta + 3) * (19*beta +16))
T = 8



mesh, tree = rectangular_domain(h, (-3, -3), (3, 3))
solver = IncompressibleTwoPhaseFlowSolver(
    mesh,
    h,
    0.1 * h,
    1,
    1,
    M/Re,
    1/Re,
    1/We,
    0,
    c_kappa=0.1,
    c_normal=0.1
)

solver.set_no_slip_everywhere()
solver.level_set.phi.interpolate(e_phi)

path = "relax-course"
t_sim = []
g_sim = []

reader = read_solution_files(solver, path)
for t, phi, u in reader:
    t_sim.append(t)
    solver.level_set.phi.x.array[:] = phi
    L, B = compute_aspect(solver.level_set.phi, 250)
    g_sim.append(L**2 - B**2)
    for _ in range(25):
        try:
            next(reader)
        except StopIteration:
            break

t = np.linspace(0, 10, 250)
g = (L0**2 - B0**2) * np.exp(-sigma / mu_matrix * f * t)



plt.plot(t, np.log(g)/(L0**2 - B0**2))
plt.plot(t_sim, np.log(g_sim)/(L0**2 - B0**2))
plt.savefig("aspect_log.png", dpi=400)

#s#olver.save_to_files(path, 12, 1)
# plotter = Plotter(solver, (-3, 3), (-3, 3), 0.5, visc_points=50, scale=None, filename=path)
# plotter.save_to_mp4("relax-course.mp4")
