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


class FixedDomain:

    def __init__(self, h):
        n = math.ceil(1 / h)
        square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)



def sdf_circle(x):
    x, y, _ = x
    r = 0.5
    return np.sqrt((x)**2 + (y)**2) - r

h_max = 0.1
h_min = h_max / 16
print("nodes without amr:", int(1 / h_min)**2)
dt = 0.01
inital_epsilon = 4 * h_min
target_epsilon = 1.5 * h_min
reinit_iters = 5

fig, ax = plt.subplots(1, 4)
ax[0].set_title(r"Target $\phi$")
ax[1].set_title(r"Inital $\phi$")
ax[2].set_title(r"Reinitalised $\phi$")
ax[3].set_title(r"$\phi - \phi^T$")

domain = FixedDomain(h_min)
domain = AdaptiveLevelSetDomain(domain.mesh, h_min, 0.1)
solver = ConservativeLevelSet(
    domain,
    h_min,
    dt,
    solver_options={"normal_method": "std", "fix_inteface": True}
)
solver.ϕ.interpolate(lambda x: 1/(1 + np.exp(sdf_circle(x) / inital_epsilon)))
domain.remesh(solver, show=True)


# fem_plot_contor_filled(fig, ax[1], solver.ϕ, domain, (0, 1), (0, 1), 100, levels=100)
# fem_plot_contor(fig, ax[1], solver.ϕ, domain, (0, 1), (0, 1), 100, levels=[0.5], colors=["black"])

# solver.ϵ = dolfinx.fem.Constant(solver.scalar_space.mesh, target_epsilon)

# target_phi = dolfinx.fem.Function(solver.scalar_space)
# target_phi.interpolate(lambda x: 1/(1 + np.exp(sdf_circle(x) / target_epsilon)))

# solver.create_test_and_trail_functions()
# solver.compute_normals()
# solver.build_reinit_problem(use_mesh_eps=False)

# phi_m = dolfinx.fem.Function(solver.scalar_space)
# phi_m.x.array[:] = solver.ϕ.x.array

# for i in range(10000):
#     solver.reinit_problem.solve()
#     res = dolfinx.fem.form((phi_m - solver.ϕ) * (phi_m - solver.ϕ) * ufl.dx)
#     res = dolfinx.fem.assemble_scalar(res)
#     phi_m.x.array[:] = solver.ϕ.x.array
#     if res < 1e-4:
#         print(f"converged after {i + 1} iterations")
#         break

# I = dolfinx.fem.form((target_phi - solver.ϕ) * (target_phi - solver.ϕ) * ufl.dx)


# error = dolfinx.fem.assemble_scalar(I)
# print("Error:", error)


# fem_plot_contor_filled(fig, ax[2], solver.ϕ, domain, (0, 1), (0, 1), 100, levels=100)
# fem_plot_contor(fig, ax[2], solver.ϕ, domain, (0, 1), (0, 1), 100, levels=[0.5], colors=["black"])

# fem_plot_contor_filled(fig, ax[0], target_phi, domain, (0, 1), (0, 1), 100, levels=100)
# fem_plot_contor(fig, ax[0], target_phi, domain, (0, 1), (0, 1), 100, levels=[0.5], colors=["black"])

# x, y, phi = fem_scalar_func_at_points(solver.ϕ, domain, (0, 1), (0, 1), 100)
# x, y, target = fem_scalar_func_at_points(target_phi, domain, (0, 1), (0, 1), 100)
# ax[3].contourf(x, y, phi - target, levels=100)


# for a in ax:
#     a.set_aspect("equal")
# plt.show()

