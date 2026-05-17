from common.visualize import *
from levelset import *
import matplotlib
import time
import os
import scipy
import copy
import dolfinx
import ufl


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

def get_u(domain, t, T):
    u_space = dolfinx.fem.functionspace(domain.mesh, ("P", 2, (domain.mesh.topology.dim,)))
    u = dolfinx.fem.Function(u_space)
    if t >= T/2:
        u.interpolate(lambda x: (x[0], -x[1]))
    else:
        u.interpolate(lambda x: (-x[0], x[1]))
    return u

h_max = 0.1
h_min = h_max / 32
dt = 0.01
max_reinit_iters = 10

fig, ax = plt.subplots(1, 4)
ax[0].set_title(r"$t=0$")
ax[1].set_title(r"$t=0.5$")
ax[2].set_title(r"$t=1$")
ax[3].set_title(r"$|\phi(0) - \phi(1)|$")

domain = FixedDomain(h_min)
domain = AdaptiveLevelSetDomain(domain.mesh, h_min, h_max)
solver = ConservativeLevelSet(
    domain,
    h_min,
    dt,
    reinit_tol=1e-4,
    solver_options={"normal_method": "std", "fix_inteface": True}
)
solver.ϕ.interpolate(lambda x: 1/(1 + np.exp(sdf_circle(x) / (1.5 * h_min))))
fem_plot_contor_filled(fig, ax[0], solver.ϕ, domain)
fem_plot_contor(fig, ax[0], solver.ϕ, domain, colors=["black"])
x, y, inital_phi_grid = fem_scalar_func_at_points(solver.φ, domain)
T = 1
plotted = False
for i in range(math.ceil(T / solver.dt)):
    domain.remesh(solver)
    t = i * solver.dt
    print("t=", t)
    solver.advect(get_u(domain, t, T))
    if t >= T/2 and not plotted:
        fem_plot_contor_filled(fig, ax[1], solver.ϕ, domain)
        fem_plot_contor(fig, ax[1], solver.ϕ, domain, colors=["black"])
        plotted = True

fem_plot_contor_filled(fig, ax[2], solver.ϕ, domain)
fem_plot_contor(fig, ax[2], solver.ϕ, domain, colors=["black"])
x, y, final_phi_grid = fem_scalar_func_at_points(solver.φ, domain)
ax[3].contourf(x, y, np.abs(final_phi_grid - inital_phi_grid), levels=100)

error = 1 / np.sum(np.ones_like(inital_phi_grid)) * np.sqrt(np.sum(np.square(inital_phi_grid - final_phi_grid)))
print(error)
for a in ax:
    a.set_aspect("equal")
plt.show()

domain.remesh(solver, show=True)