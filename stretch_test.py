from ../common.domains import *
from ../common.visualize import *

from levelset import *
from sdf import *
from ../common.bc import *
import h5py
import matplotlib
import time
import os
import scipy

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
        square = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.triangle)
        square.name = "mesh"
        self.mesh = square
        self.tree = dolfinx.geometry.bb_tree(square, 2)






def cdist(x):
    x, y, _ = x
    r = 0.5
    return np.sqrt((x)**2 + (y)**2) - r

def get_u(domain, t, T):
    u_space = dolfinx.fem.functionspace(domain.mesh, ("P", 2, (domain.mesh.topology.dim,)))
    u = dolfinx.fem.Function(u_space)
    # if t >= T/2:
    #     u.interpolate(lambda x: (x[0], -x[1]))
    # else:
    u.interpolate(lambda x: (-x[0], x[1]))
    return u

def write_time_step(path, domain, solver, t):
    domain.mesh.name = "mesh"
    solver.ϕ.name = "phi"
    if not os.path.exists(path):
        os.makedirs(path)
    with dolfinx.io.XDMFFile(domain.mesh.comm, f"{path}/{t}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain.mesh)
        xdmf.write_function(solver.ϕ, t)



def make_dolfinx_solution(points, cells, phi_values):
    element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, element, points)
    scalar_element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1)
    scalar_space = dolfinx.fem.functionspace(mesh, scalar_element)
    phi = dolfinx.fem.Function(scalar_space)
    dof_coords = scalar_space.tabulate_dof_coordinates()
    tree = scipy.spatial.KDTree(points[:, :2])
    _, input_indices = tree.query(dof_coords[:, :2])
    phi.x.array[:] = phi_values[input_indices]
    return mesh, phi


def solve(path, h, dt, M):
    domain = FixedDomain(h)
    domain = AdaptiveLevelSetDomain(domain.mesh, 0.01, 0.1)
    solver = ConservativeLevelSet(domain, h, dt, reinit_iters=M, solver_options={"normal_method": "std", "fix_inteface": True})
    solver.ϕ.interpolate(lambda x: 1/(1 + np.exp(cdist(x)/(h/2))))
   # fem_plot_contor(fig, ax, solver.ϕ, domain, (0, 1), (0, 1), 100, levels=[0.5], colors=["blue"])
    T = 0.5
    for i in range(math.ceil(T / solver.dt)):
        domain.remesh(solver)
        t = i * solver.dt 
        write_time_step(path, domain, solver, t)
        u = get_u(domain, t, T)
        solver.advect(u)
    domain.remesh(solver, show=True)
    # fem_plot_contor(fig, ax, solver.ϕ, domain, (0, 1), (0, 1), 100, levels=[0.05, 0.5, 0.95])
    # fem_plot_vectors(ax, solver.n, domain, (0, 1), (0, 1), 60)
    

# fig, ax = plt.subplots()
dt = 0.01
h = 0.005
solve("test", h, dt, 1)
# plt.show()

# sol = SolutionFile("test")
# for t in [0.1]:
#     sol.read(t)
#     sol.show_mesh()

# TRY SCLSM WITH FIXED MESH and with p=2 elements for phi?


# fig, ax = plt.subplots(figsize=(10, 8))
# for root in root_paths:
#     paths = sorted([path for path in os.listdir(root) if path.endswith(".h5")])
#     xcoms, ycoms = [], []
#     areas = []
#     for path in tqdm(paths):
#         with h5py.File(root + "/" + path, "r") as f:
#             points, cells, phi_values = read_solution(f)
#             mesh, phi = make_dolfinx_solution(points, cells, phi_values)
#             x = ufl.SpatialCoordinate(mesh)
#             area = dolfinx.fem.form(phi * ufl.dx)
#             xform = dolfinx.fem.form(x[0] * phi * ufl.dx)
#             yform = dolfinx.fem.form(x[1] * phi * ufl.dx)
#             xcom, ycom = dolfinx.fem.assemble_scalar(xform) / dolfinx.fem.assemble_scalar(area), dolfinx.fem.assemble_scalar(yform) / dolfinx.fem.assemble_scalar(area)
#             xcoms.append(xcom)
#             ycoms.append(ycom)
#             areas.append(dolfinx.fem.assemble_scalar(area))

#     ax.plot(np.linspace(0, 1, len(xcoms)), xcoms, label=root)
#     #ax.plot(np.linspace(0, 1, len(areas)), np.abs(np.array(areas)[0] - np.array(areas)), label=root)
# plt.legend()
# plt.show()



        
      
        # ax.set_title("Manual Plot from HDF5 Data")
        # plt.pause(0.05)
