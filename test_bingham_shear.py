from common.visualize import *
from common.bc import *
from twophasevp import *
from levelset import *
import dolfinx
from mpi4py import MPI
from tqdm import tqdm

class FixedDomain:

    def __init__(self, n, bl, tr):
        square = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [bl, tr], [n, n], cell_type=dolfinx.mesh.CellType.quadrilateral)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)


def u_exact(y):
    u = np.zeros_like(y)
    upper_inds = np.logical_and(Bn <= y, y <= 1)
    u[upper_inds] = (1 - y[upper_inds]) * (1 - 2*Bn + y[upper_inds])
    mid_inds = np.logical_and(-Bn <= y, y <= Bn)
    u[mid_inds] = (1 - Bn)**2 * np.ones(len(u[mid_inds]))
    lower_inds = np.logical_and(-1 <= y, y <= -Bn)
    u[lower_inds] = (1 + y[lower_inds]) * (1 - 2*Bn - y[lower_inds])
    return Re / 2 * u



n = 50
h = 2 / n
T = 1
dt = 0.005
Re = 1
Bn = 0.5    
mu = 1 / Re
rho = 1
domain = FixedDomain(n, [-1, -1], [1, 1])
ls = ConservativeLevelSet(domain, h, dt)
ls.ϕ.x.array[:] = 1
solver = RegBinghamTwoPhaseSolver(ls, dt, 1, 1, 1, 1, 0, Bn, epsilon=1e-6)
solver.create_test_and_trail_functions()
solver.set_velocity_bc(y_equals(-1), (0, 0))
solver.set_velocity_bc(y_equals(1), (0, 0))
solver.set_pressure_bc(x_equals(-1), 2)
solver.set_pressure_bc(x_equals(1), 0)
solver.set_y_velocity(x_equals(-1), 0)
solver.set_y_velocity(x_equals(1), 0)
f = dolfinx.fem.Constant(domain.mesh, dolfinx.default_scalar_type((0, 0)))
solver.build_predictor_problem(f)
solver.build_pressure_problem()
solver.build_corrector_problem()

u_last = dolfinx.fem.Function(solver.vector_space)
u_last.x.array[:] = solver.u.x.array
for i in range(1000):
    solver.compute_u()
    res = np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u_last - solver.u, u_last - solver.u) * ufl.dx)))
    if res < 1e-4:
        break
    print("SS res:", res)
    u_last.x.array[:] = solver.u.x.array

ax = plt.axes()
N = 100
x, y, u, v = fem_vector_func_at_given_points(solver.u, domain, np.zeros(N), np.linspace(-1, 1, N))
M = 10
nu = mu / rho
plt.plot(y, u, color="blue")
plt.plot(y, u_exact(y), color="red")
plt.show()

# Try to use adams second order method + CN for predictor step.