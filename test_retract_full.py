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

def sdf_circle(x):
    x, y, _ = x
    r = 0.5
    return np.sqrt((x)**2 + (y)**2) - r

class FixedDomain:

    def __init__(self, n):
        #square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
        square = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[-2, -2], [2, 2]], [n, n], cell_type=dolfinx.mesh.CellType.quadrilateral)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)

def f_ellipce(x):
    return x[0]**2 + (1 + eps)**2*x[1]**2 - 1 - eps

def g_ellipce(x):
    return np.array([2 * x[0], 2*(1 + eps)**2*x[1]])

def inital_condtion(X):
    d = np.array([sdf.signed_distance_to_zero_level_set(f_ellipce, g_ellipce, (xi, yi)) for xi, yi in tqdm(zip(X[0], X[1]))])
    return 1 / (1 + np.exp(d / (3 * h)))



folder = pathlib.Path("test.bp")
n = 100
h = 4 / n
dt = 0.001
eps = 1
domain = FixedDomain(n)
ls = ConservativeLevelSet(domain, h, dt, solver_options={"fix_interface": False}, sym_bcs=False)
solver = IncompressibleTwoPhaseFlowSolver(ls, dt, rho0=1, rho1=1, mu0=1, mu1=1, sigma=10, kinematic_bc=True)

solver.set_x_velocity(x_equals(-2), 0)
solver.set_x_velocity(x_equals(2), 0)

solver.set_y_velocity(y_equals(-2), 0)
solver.set_y_velocity(y_equals(2), 0)

solver.ls.ϕ.interpolate(lambda x: inital_condtion(x))

f = dolfinx.fem.Function(solver.vector_space)
solver.create_test_and_trail_functions()
solver.build_pressure_problem()
solver.build_corrector_problem()
solver.build_predictor_problem(f)
solver.ls.build_problems(solver.u)

solver.ls.compute_gradient()
solver.ls.compute_curvature()

adios4dolfinx.write_mesh(folder, domain.mesh)
T = 1
for i in tqdm(range(math.ceil(T / solver.dt))):
    t = solver.dt * i
    adios4dolfinx.write_function(folder, solver.ls.ϕ, time=i, name="phi")
    adios4dolfinx.write_function(folder, solver.u, time=i, name="u")
    adios4dolfinx.write_function(folder, solver.p, time=i, name="p")
    solver.ls.compute_gradient()
    solver.ls.compute_curvature()
    adios4dolfinx.write_function(folder, solver.ls.κ, time=i, name="k")
    interpolate_expression(solver.sigma * solver.ls.κ * solver.ls.grad_ϕ, f)
    solver.build_predictor_problem(f)
    solver.compute_u()
    solver.ls.advect(solver.u)

ax = plt.axes()
fem_plot_contor(ax.figure, ax, solver.ls.ϕ, domain)
fem_plot_vectors(ax, solver.u, domain)
plt.show()

ax = plt.axes()
x, y, us, vs = fem_vector_func_at_given_points(solver.u, domain, np.linspace(0, 2, 250), np.zeros(250))
ax.plot(x, us)
plt.show()

    
fig, ax = plt.subplots(2, 2)
ax[0, 0].set_xlabel("$x$")
ax[0, 0].set_ylabel("$u$")
ax[0, 1].set_xlabel("$x$")
ax[0, 1].set_ylabel("$v$")
ax[1, 0].set_xlabel("$x$")
ax[1, 0].set_ylabel(r"$\phi$")
ax[1, 1].set_xlabel("$x$")
ax[1, 1].set_ylabel(r"$\kappa$")
in_mesh = adios4dolfinx.read_mesh(folder, MPI.COMM_WORLD)
domain = FixedDomain(n)
domain.mesh = in_mesh
domain.tree = dolfinx.geometry.bb_tree(in_mesh, 2)
v_el = basix.ufl.element("Lagrange", in_mesh.topology.cell_name(), 2, shape=(in_mesh.geometry.dim,))
s_el = basix.ufl.element("Lagrange", in_mesh.topology.cell_name(), 1)
V = dolfinx.fem.functionspace(in_mesh, v_el)
S = dolfinx.fem.functionspace(in_mesh, s_el)
u = dolfinx.fem.Function(V)
phi = dolfinx.fem.Function(S)
kappa = dolfinx.fem.Function(S)

data = {9 : {}, 49: {}, 99 : {}}

for i in [9, 49, 99]:
    adios4dolfinx.read_function(folder, u, time=i, name="u")
    adios4dolfinx.read_function(folder, phi, time=i, name="phi")
    adios4dolfinx.read_function(folder, kappa, time=i, name="k")
    x, y, us, vs = fem_vector_func_at_given_points(u, domain, np.linspace(0, 2, 250), np.zeros(250))
    ax[0, 0].plot(x, us, label=f"$t={(i + 1)*dt}$")
    ax[0, 1].plot(x, vs, label=f"$t={(i + 1)*dt}$")
    x, y, phis = fem_scalar_func_at_given_points(phi, domain, np.linspace(0, 2, 250), np.zeros(250))
    x, y, ks = fem_scalar_func_at_given_points(kappa, domain, np.linspace(0, 2, 250), np.zeros(250))
    ax[1, 0].plot(x, phis, label=f"$t={(i + 1)*dt}$")
    ax[1, 1].plot(x, ks, label=f"$t={(i + 1)*dt}$")
    data[i]["u"] = us
    data[i]["v"] = vs
    data[i]["phi"] = phis
    data[i]["k"] = ks
    x, y, uss, vss = fem_vector_func_at_points(u, domain, 60)
    data[i]["vecs"] = (x, y, uss, vss)
with open("full_data.pkl", "wb") as outfile:
    pickle.dump(data, outfile) 
ax[0, 0].legend()
plt.show()




# # 
# TODO: Set up full versions of tests, compare time of full drop with AMR vs Quater drop with PP
# Maybe Full drop w/ AMR is better for cool non-symetric sims?

