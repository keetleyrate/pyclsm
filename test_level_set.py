from common.domains import *
from common.visualize import *

from levelset import *
from sdf import *
from common.bc import *
import h5py
import matplotlib
import time

vortex = lambda x, y: (
    np.sin(np.pi * x)*np.sin(np.pi * x)*np.sin(2*np.pi * y),
    -np.sin(np.pi * y)*np.sin(np.pi * y)*np.sin(2*np.pi * x)
)

vortex_rev = lambda x, y: (
    -np.sin(np.pi * x)*np.sin(np.pi * x)*np.sin(2*np.pi * y),
    np.sin(np.pi * y)*np.sin(np.pi * y)*np.sin(2*np.pi * x)
)

# # # # T = 0.1
# # # # for i in range(math.ceil(T / solver.dt)):
# # # #     t = i * solver.dt 
# # # #     domain.remesh(solver)
# # # #     u_space = dolfinx.fem.functionspace(domain.mesh, ("P", 2, (domain.mesh.topology.dim,)))
# # # #     u = dolfinx.fem.Function(u_space)
# # # #     if t <= T / 2:
# # # #         u.interpolate(lambda x: vortex (x[0], x[1]))
# # # #     else:
# # # #         u.interpolate(lambda x: vortex_rev (x[0], x[1]))
# # # #     solver.build_problems(u)
# # # #     solver.gls_advection_problem.solve()
# # # #     solver.normal_projector.compute_normals()
# # # #     for _ in range(7):
# # # #         phi_last = np.copy(solver.ϕ.x.array)
# # # #         solver.reinit_problem.solve()
# # # domain.mesh.comm.Barrier()


def cdist(x):
    x, y, _ = x
    r = 0.1
    return np.sqrt((x - 0.75)**2 + (y - 0.5)**2) - r

start = time.perf_counter()

h_max = 0.1
h_min = h_max/16
mesh, tree = rectangular_domain(h_min, (0, 0), (1, 1))
domain = AdaptiveLevelSetDomain(mesh, h_min, h_max)
solver = ConservativeLevelSet(domain, h_min, 0.01, p=1, reinit_iters=10, solver_options={"normal_method": "project", "cproj": 2, "fix_inteface": False})
eps = h_min
solver.ϕ.interpolate(lambda x: 1/(1 + np.exp(cdist(x)/(h_min/2))))
for _ in tqdm(range(1)):
    domain.remesh(solver)
    u_space = dolfinx.fem.functionspace(domain.mesh, ("P", 2, (domain.mesh.topology.dim,)))
    u = dolfinx.fem.Function(u_space)
    u.interpolate(lambda x: vortex (x[0], x[1]))
    solver.advect(u, show=True)



# TEST whats faster, using SDG from phi or gmsh..
    
# fig, ax = plt.subplots()
# x, y, phi = fem_scalar_func_at_points(solver.ϕ, domain, (0, 1), (0, 1), 100)
# p = 0.02
# d = np.abs(h_min / 2 * np.log(np.abs((1 - phi) / phi)))
# cts = ax.contourf(x, y, d, levels=100)
# #fem_plot_contor_filled(fig, ax, solver.ϕ, domain, (0, 1), (0, 1), 100, levels=100)
# #fem_plot_vectors(ax, u, domain, (0, 1), (0, 1), 40)
# plt.colorbar(cts)
# plt.show()





# for _ in range(10):
#     domain.remesh(solver, show=True)
#     u_space = dolfinx.fem.functionspace(domain.mesh, ("P", 2, (domain.mesh.topology.dim,)))
#     u = dolfinx.fem.Function(u_space)
#     u.interpolate(lambda x: vortex (x[0], x[1]))
#     solver.build_problems(u)
#     solver.gls_advection_problem.solve()

# end = time.perf_counter()
# if domain.mesh.comm.rank == 0:
#     print(end - start)

# xdmf = dolfinx.io.XDMFFile(domain.mesh.comm, "out.xdmf", "w")
# xdmf.write_mesh(domain.mesh)
# xdmf.write_function(solver.ϕ)



# with h5py.File("out.h5", "r") as f:
#     points = f['Mesh/mesh/geometry'][:]
#     x = points[:, 0]
#     y = points[:, 1]
#     cells = f['Mesh/mesh/topology'][:]
#     phi_values = f['Function/phi/0'][:]
#     if cells.shape[1] == 4:
#         cells = cells[:, 1:]
#     phi_values = phi_values.flatten()
#     triangulation = matplotlib.tri.Triangulation(x, y, cells)

#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Plot the filled contours of Phi
#     tpc = ax.tricontourf(triangulation, phi_values, cmap='viridis', levels=20)
#     fig.colorbar(tpc, label='$\phi$')

#     # Plot the mesh edges
#     ax.triplot(triangulation, color='white', lw=0.5, alpha=0.3)

#     ax.set_aspect('equal')
#     ax.set_title("Manual Plot from HDF5 Data")
#     plt.show()