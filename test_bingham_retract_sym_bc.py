from common.visualize import *
from common.bc import *
from twophaseflow import *
from levelset import *
import dolfinx
from mpi4py import MPI
from tqdm import tqdm
import sdf
import adios4dolfinx
import pathlib
from twophasevp import *

def com(domain, ϕ):
    x = ufl.SpatialCoordinate(domain.mesh)
    M = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * ufl.dx))
    x00 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 / M * x[0] * ϕ * ufl.dx))
    x01 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 / M * x[1] * ϕ * ufl.dx))
    return M, np.array([x00, x01])

def ar_eff(domain, ϕ):
    x = ufl.SpatialCoordinate(domain.mesh)
    M, x0 = com(domain, ϕ)
    print(x0)
    c00 = 1 / M * dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * (x[0] - x0[0])*(x[0] - x0[0]) * ufl.dx))
    c01 = 1 / M * dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * (x[0] - x0[0])*(x[1] - x0[1]) * ufl.dx))
    c10 = 1 / M * dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * (x[1] - x0[1])*(x[0] - x0[0]) * ufl.dx))
    c11 = 1 / M * dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * (x[1] - x0[1])*(x[1] - x0[1]) * ufl.dx))
    C = np.array([[c00, c01], [c10, c11]])
    l1, l2 = np.linalg.eig(C).eigenvalues
    l1, l2 = max(l1, l2), min(l1, l2)
    return np.sqrt(l1 / l2)

def plot_snapshot(ax, domain, phi, u, t):
    # top left
    fig = ax[0, 0].figure
    ax[0, 0].set_xlabel("$x$")
    ax[0, 0].set_ylabel("$y$")
    fem_plot_contor_filled(fig, ax[0, 0], phi, domain)
    # top right
    ax[0, 1].set_xlabel("$x$")
    ax[0, 1].set_ylabel("$y$")
    fem_plot_vectors(ax[0, 1], u, domain, n_points=35)
    fem_plot_contor(fig, ax[0, 1], phi, domain)
    # bottom left
    ax[1, 0].set_xlabel("$x$")
    ax[1, 0].set_ylabel("$u$")
    x, y, us, vs = fem_vector_func_at_given_points(u, domain.mesh, domain.tree, np.linspace(0, 3, 250), np.zeros(250))
    ax[1, 0].plot(x, us, label=f"$t={t:.2f}$")
    # bottom right
    ax[1, 1].set_xlabel("$y$")
    ax[1, 1].set_ylabel("$v$")
    x, y, us, vs = fem_vector_func_at_given_points(u, domain.mesh, domain.tree, np.zeros(250), np.linspace(0, 3, 250))
    ax[1, 1].plot(y, vs, label=f"$t={t:.2f}$")
    




class FixedDomain:

    def __init__(self, n, bl, tr):
        square = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [bl, tr], [n, n], cell_type=dolfinx.mesh.CellType.quadrilateral)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)

def f_ellipce(x):
    return x[0]**2 / a**2 + x[1]**2 / b **2 - 1

def g_ellipce(x):
    return np.array([2 * x[0] / a**2 , 2 * x[1] / b **2])

def inital_condtion(X, h):
    d = np.array([sdf.signed_distance_to_zero_level_set(f_ellipce, g_ellipce, (xi, yi)) for xi, yi in tqdm(zip(X[0], X[1]))])
    return 1 / (1 + np.exp(d / ( h)))


def problem():
    folder = pathlib.Path(f"retract_bingham.bp")
    n = 128
    L = 3
    h = L / n 
    dt = np.sqrt(h**3 / (4 * np.pi))
    domain = FixedDomain(n, [0, 0], [L, L])
    ls = ConservativeLevelSet(domain, h, dt, solver_options={"fix_interface": True}, c_kappa=0, sym_bcs=True)
    solver = RegBinghamTwoPhaseSolver(ls, dt, 1, 1, 1, 1, 1, 0.01, epsilon=1e-3)

    solver.set_x_velocity(x_equals(0), 0)
    solver.set_x_velocity(x_equals(L), 0)

    solver.set_y_velocity(y_equals(0), 0)
    solver.set_y_velocity(y_equals(L), 0)

    solver.ls.ϕ.interpolate(lambda x: inital_condtion(x, h))

    f = dolfinx.fem.Function(solver.vector_space)
    solver.create_test_and_trail_functions()
    solver.build_pressure_problem()
    solver.build_corrector_problem()
    solver.build_predictor_problem(f)
    solver.ls.build_problems(solver.u)
    solver.ls.reinit()


    adios4dolfinx.write_mesh(folder, domain.mesh)
    T = 1
    n_steps = math.ceil(T / solver.dt)
    for i in tqdm(range(n_steps)):
        t = solver.dt * i
        adios4dolfinx.write_function(folder, solver.ls.ϕ, time=i, name="phi")
        adios4dolfinx.write_function(folder, solver.u, time=i, name="u")
        adios4dolfinx.write_function(folder, solver.p, time=i, name="p")
        adios4dolfinx.write_function(folder, solver.ls.κ, time=i, name="k")
        print("saving solver state at t=", t)
        solver.ls.compute_gradient()
        solver.ls.normal_problem.solve()
        solver.ls.compute_curvature()
        interpolate_expression(solver.sigma * solver.ls.κ * solver.ls.grad_ϕ, f)
        solver.build_predictor_problem(f)
        solver.compute_u()
        solver.ls.advect(solver.u)

a = 1.1055
b = 0.9045

#problem()
L = 3





L = 3
n = 128
h = L / n
dt = np.sqrt(h**3 / (4 * np.pi))
n_steps = math.ceil(1 / dt)
folder = pathlib.Path(f"retract_bingham.bp")
in_mesh = adios4dolfinx.read_mesh(folder, MPI.COMM_WORLD)
domain = FixedDomain(n, [0, 0], [L, L])
domain.mesh = in_mesh
domain.tree = dolfinx.geometry.bb_tree(in_mesh, 2)
v_el = basix.ufl.element("Lagrange", in_mesh.topology.cell_name(), 2, shape=(in_mesh.geometry.dim,))
s_el = basix.ufl.element("Lagrange", in_mesh.topology.cell_name(), 1)
V = dolfinx.fem.functionspace(in_mesh, v_el)
S = dolfinx.fem.functionspace(in_mesh, s_el)
u = dolfinx.fem.Function(V)
phi = dolfinx.fem.Function(S)
mu = dolfinx.fem.Function(S)
kappa = dolfinx.fem.Function(S)
ar = []
for i in range(0, n_steps, 100):
    fig, ax = plt.subplots(2, 2)
    adios4dolfinx.read_function(folder, u, time=i, name="u")
    adios4dolfinx.read_function(folder, phi, time=i, name="phi")
    adios4dolfinx.read_function(folder, kappa, time=i, name="k")
    plot_snapshot(ax, domain, phi, u, i*dt)

  

    plt.show()


# # 
# TODO: Set up full versions of tests, compare time of full drop with AMR vs Quater drop with PP
# Maybe Full drop w/ AMR is better for cool non-symetric sims?

