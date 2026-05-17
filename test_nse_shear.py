from common.visualize import *
from common.bc import *
from navierstokes import *
import dolfinx
from mpi4py import MPI
from tqdm import tqdm

class FixedDomain:

    def __init__(self, h):
        n = math.ceil(1 / h)
        square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.triangle)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)

h = 1 / 32
T = 0.01
dt = T / 150
Re = 2
mu = 1 / Re
rho = 1
domain = FixedDomain(h)
solver = IncompressibleNavierStokesSolver(domain, dt)
solver.set_density_as_const(rho)
solver.set_viscosity_as_const(mu)
solver.create_test_and_trail_functions()
solver.set_velocity_bc(y_equals(0), (1, 0))
solver.set_velocity_bc(y_equals(1), (0, 0))
solver.set_y_velocity(x_equals(0), 0)
solver.set_y_velocity(x_equals(1), 0)
f = dolfinx.fem.Constant(domain.mesh, dolfinx.default_scalar_type((0, 0)))
solver.build_predictor_problem(f)
solver.build_pressure_problem()
solver.build_corrector_problem()

for i in tqdm(range(math.ceil(T / solver.dt))):
    solver.time_step()

ax = plt.axes()
N = 100
x, y, u, v = fem_vector_func_at_given_points(solver.u, domain, 0.5 * np.ones(N), np.linspace(0, 1, N))
M = 10
nu = mu / rho
u_exact = (1 - y) - sum(2/(n * np.pi) * np.exp(-n**2*np.pi**2*nu*T)*np.sin(n*np.pi*y) for n in range(1, N + 1))
plt.plot(y, u_exact, color="red", linestyle="--")
plt.plot(y, u, color="blue")
plt.show()

# Try to use adams second order method + CN for predictor step.