from common.visualize import *
from levelset import *
import matplotlib
import time
import os
import scipy
import copy
import dolfinx
import ufl

def read_solution(f):
    points = f['Mesh/mesh/geometry'][:]
    cells = f['Mesh/mesh/topology'][:]
    phi_values = f['Function/phi/0'][:]
    phi_values = phi_values.flatten()
    return points, cells, phi_values

class SolutionFile:

    def __init__(self, root_path):
        self.root_path = root_path
        self.h5_paths = sorted([path for path in os.listdir(root_path) if path.endswith(".h5")])
        self.T = float(self.h5_paths[-1][:-3])

    def read(self, t):
        index = min(math.floor(t / self.T * (len(self.h5_paths) - 1)), len(self.h5_paths) - 1)
        print(self.root_path + "/" + self.h5_paths[index])
        with h5py.File(self.root_path + "/" + self.h5_paths[index], "r") as h5_file:
            self.points, self.cells, self.values = read_solution(h5_file)

    def show_mesh(self):
        triangulation = matplotlib.tri.Triangulation(self.points[:, 0], self.points[:, 1], self.cells)
        ax = plt.axes()
        tpc = ax.tricontourf(triangulation, self.values, cmap='viridis', levels=100)
        #tpc = ax.tricontour(triangulation, self.values, cmap='viridis', levels=[0.05, 0.5, 0.95])
        ax.triplot(triangulation, color='white', lw=0.5, alpha=0.3)
        ax.set_aspect('equal')
        plt.show()


class FixedDomain:

    def __init__(self, h):
        n = math.ceil(1 / h)
        square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.triangle)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)



def sdf_circle(x):
    x, y, _ = x
    r = 0.5
    return np.sqrt((x)**2 + (y)**2) - r

h = 1 / 64
dt = 0.01
inital_epsilon = 0.05
target_epsilon = 0.01
reinit_iters = 5

fig, ax = plt.subplots(1, 4)
ax[0].set_title(r"Target $\phi$")
ax[1].set_title(r"Inital $\phi$")
ax[2].set_title(r"Reinitalised $\phi$")
ax[3].set_title(r"$\phi - \phi^T$")

domain = FixedDomain(h)
solver = ConservativeLevelSet(
    domain,
    h,
    dt,
    solver_options={"normal_method": "std", "fix_inteface": True}
)
solver.ϕ.interpolate(lambda x: 1/(1 + np.exp(sdf_circle(x) / inital_epsilon)))

fem_plot_contor_filled(fig, ax[1], solver.ϕ, domain)
fem_plot_contor(fig, ax[1], solver.ϕ, domain, colors=["black"])

solver.ϵ = dolfinx.fem.Constant(solver.scalar_space.mesh, target_epsilon)

target_phi = dolfinx.fem.Function(solver.scalar_space)
target_phi.interpolate(lambda x: 1/(1 + np.exp(sdf_circle(x) / target_epsilon)))

solver.create_test_and_trail_functions()
# solver.n.interpolate(lambda x: (
#     -x[0] / (np.sqrt(x[0]**2 + x[1]**2) + 1e-8),
#     -x[1] / (np.sqrt(x[0]**2 + x[1]**2) + 1e-8)
# ))
solver.compute_normals()
solver.build_reinit_problem(use_mesh_eps=False)

phi_m = dolfinx.fem.Function(solver.scalar_space)
phi_m.x.array[:] = solver.ϕ.x.array

for i in range(10000):
    solver.reinit_problem.solve()
    res = dolfinx.fem.form((phi_m - solver.ϕ) * (phi_m - solver.ϕ) * ufl.dx)
    res = dolfinx.fem.assemble_scalar(res)
    phi_m.x.array[:] = solver.ϕ.x.array
    if res < 1e-4:
        print(f"converged after {i + 1} iterations")
        break

I = dolfinx.fem.form((target_phi - solver.ϕ) * (target_phi - solver.ϕ) * ufl.dx)


error = dolfinx.fem.assemble_scalar(I)
print("Error:", error)


fem_plot_contor_filled(fig, ax[2], solver.ϕ, domain)
fem_plot_contor(fig, ax[2], solver.ϕ, domain, colors=["black"])
# fem_plot_vectors(ax[2], solver.n, domain)

fem_plot_contor_filled(fig, ax[0], target_phi, domain)
fem_plot_contor(fig, ax[0], target_phi, domain, colors=["black"])

x, y, phi = fem_scalar_func_at_points(solver.ϕ, domain)
x, y, target = fem_scalar_func_at_points(target_phi, domain)
ax[3].contourf(x, y, phi - target, levels=100)


for a in ax:
    a.set_aspect("equal")
plt.show()

