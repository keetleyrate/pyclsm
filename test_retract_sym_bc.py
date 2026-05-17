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
import pickle

def com(domain, ϕ):
    x = ufl.SpatialCoordinate(domain.mesh)
    M = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * ufl.dx))
    x0 = dolfinx.fem.assemble_vector(dolfinx.fem.form(1 / M * x * ϕ * ufl.dx))
    return M, x0

def ar_eff(domain, ϕ):
    x = ufl.SpatialCoordinate(domain.mesh)
    M, x0 = com(domain, ϕ)
    c00 = 1 / M * dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * (x[0] - x0[0])*(x[0] - x0[0]) * ufl.dx))
    c01 = 1 / M * dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * (x[0] - x0[0])*(x[1] - x0[1]) * ufl.dx))
    c10 = 1 / M * dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * (x[1] - x0[1])*(x[0] - x0[0]) * ufl.dx))
    c11 = 1 / M * dolfinx.fem.assemble_scalar(dolfinx.fem.form(ϕ * (x[1] - x0[1])*(x[1] - x0[1]) * ufl.dx))
    C = np.array([c00, c01], [c10, c11])
    l1, l2 = np.linalg.eig(C).eigenvalues
    l1, l2 = max(l1, l2), min(l1, l2)
    return np.sqrt(l1 / l2)



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
    return 1 / (1 + np.exp(d / (2 * h)))


def problem(C):
    folder = pathlib.Path(f"test_quater_alt_kappa.bp")
    n = 128

    L = 3
    h = L / n 
    dt = 0.01#0.8 * np.sqrt(h**3 / (4 * np.pi))
    domain = FixedDomain(n, [0, 0], [L, L])
    ls = ConservativeLevelSet(domain, h, dt, solver_options={"fix_interface": True}, c_kappa=0, sym_bcs=True)
    solver = IncompressibleTwoPhaseFlowSolver(ls, dt, rho0=1, rho1=1, mu0=1, mu1=1, sigma=1, kinematic_bc=True)

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


    adios4dolfinx.write_mesh(folder, domain.mesh)
    T = 0.2
    for i in tqdm(range(math.ceil(T / solver.dt))):
        t = solver.dt * i
        adios4dolfinx.write_function(folder, solver.ls.ϕ, time=i, name="phi")
        adios4dolfinx.write_function(folder, solver.u, time=i, name="u")
        adios4dolfinx.write_function(folder, solver.p, time=i, name="p")
        solver.ls.compute_gradient()
        solver.ls.normal_problem.solve()
        solver.ls.compute_curvature()
        adios4dolfinx.write_function(folder, solver.ls.κ, time=i, name="k")
        interpolate_expression(solver.sigma * solver.ls.κ * solver.ls.grad_ϕ, f)
        solver.build_predictor_problem(f)
        solver.compute_u()
        solver.ls.advect(solver.u)

a = 1.1055
b = 0.9045

# problem(0)

fig, ax = plt.subplots(2, 2)
ax[0, 0].set_xlabel("$x$")
ax[0, 0].set_ylabel("$u$")
ax[0, 1].set_xlabel("$x$")
ax[0, 1].set_ylabel("$v$")
ax[1, 0].set_xlabel("$x$")
ax[1, 0].set_ylabel(r"$\phi$")
ax[1, 1].set_xlabel("$x$")
ax[1, 1].set_ylabel(r"$\kappa$")
# in_mesh = adios4dolfinx.read_mesh(folder, MPI.COMM_WORLD)
# domain = FixedDomain([0, 0], [L, L])
# domain.mesh = in_mesh
# domain.tree = dolfinx.geometry.bb_tree(in_mesh, 2)
# v_el = basix.ufl.element("Lagrange", in_mesh.topology.cell_name(), 2, shape=(in_mesh.geometry.dim,))
# s_el = basix.ufl.element("Lagrange", in_mesh.topology.cell_name(), 1)
# V = dolfinx.fem.functionspace(in_mesh, v_el)
# S = dolfinx.fem.functionspace(in_mesh, s_el)
# u = dolfinx.fem.Function(V)
# phi = dolfinx.fem.Function(S)
# kappa = dolfinx.fem.Function(S)
L = 3
n = 128
for C in [0, 16]:
    dt = 0.01
    folder = pathlib.Path(f"test_quater_{C}.bp")
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
    kappa = dolfinx.fem.Function(S)
    for i in [19]:
        adios4dolfinx.read_function(folder, u, time=i, name="u")
        adios4dolfinx.read_function(folder, phi, time=i, name="phi")
        adios4dolfinx.read_function(folder, kappa, time=i, name="k")
    
        x, y, us, vs = fem_vector_func_at_given_points(u, domain, np.linspace(0, 3, 250), np.zeros(250))
        ax[0, 0].plot(x, us, label=f"$t={(i + 1)*dt:.2f}, C={C}$")
        ax[0, 1].plot(x, vs, label=f"$t={(i + 1)*dt:.2f}, C={C}$")
        x, y, phis = fem_scalar_func_at_given_points(phi, domain, np.linspace(0, 3, 250), np.zeros(250))
        x, y, ks = fem_scalar_func_at_given_points(kappa, domain, np.linspace(0, 3, 250), np.zeros(250))
        ax[1, 0].plot(x, phis, label=f"$t={(i + 1)*dt}, C={C}$")
        ax[1, 1].plot(x, ks, label=f"$t={(i + 1)*dt}, C={C}$")

dt = 0.01
folder = pathlib.Path(f"test_quater_alt_kappa.bp")
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
kappa = dolfinx.fem.Function(S)
for i in [19]:
    adios4dolfinx.read_function(folder, u, time=i, name="u")
    adios4dolfinx.read_function(folder, phi, time=i, name="phi")
    adios4dolfinx.read_function(folder, kappa, time=i, name="k")

    x, y, us, vs = fem_vector_func_at_given_points(u, domain, np.linspace(0, 3, 250), np.zeros(250))
    ax[0, 0].plot(x, us, label=f"$t={(i + 1)*dt:.2f}, stab$")
    ax[0, 1].plot(x, vs, label=f"$t={(i + 1)*dt:.2f}, stab$")
    x, y, phis = fem_scalar_func_at_given_points(phi, domain, np.linspace(0, 3, 250), np.zeros(250))
    x, y, ks = fem_scalar_func_at_given_points(kappa, domain, np.linspace(0, 3, 250), np.zeros(250))
    ax[1, 0].plot(x, phis, label=f"$t={(i + 1)*dt}, stab$")
    ax[1, 1].plot(x, ks, label=f"$t={(i + 1)*dt}, stab$")

       # x, y, uss, vss = fem_vector_func_at_points(u, domain, 30)
        # data[i]["vecs"] = (x, y, uss, vss)
# with open("sym_data.pkl", "wb") as outfile:
#     pickle.dump(data, outfile) 
ax[0, 0].legend()
plt.show()




# # 
# TODO: Set up full versions of tests, compare time of full drop with AMR vs Quater drop with PP
# Maybe Full drop w/ AMR is better for cool non-symetric sims?

