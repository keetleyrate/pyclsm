
import math
import dolfinx
import mpi4py
import numpy as np
import pyvista

class RectangularDomain:

    def __init__(self, xmin, xmax, ymin, ymax, h) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        num_x_cells = math.ceil(abs(xmax - xmin) / h)
        num_y_cells = math.ceil(abs(ymax - ymin) / h)
        self.mesh = dolfinx.mesh.create_rectangle(
            mpi4py.MPI.COMM_WORLD,
            [np.array([xmin, ymin]), np.array([xmax, ymax])],
            (num_x_cells, num_y_cells), 
            cell_type=dolfinx.mesh.CellType.quadrilateral
        )
        self.tree = dolfinx.geometry.bb_tree(self.mesh, 2)

    def plot_with_pyvista(self):
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(self.mesh, self.mesh.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.show()

    def get_dolfnix_dirichlet_bc(self, f, function_space, edge):
        boundary_func = dolfinx.fem.Function(function_space)
        assert edge in list("LRTB")
        match edge:
            case "L":
                geometry_func = lambda x: np.isclose(x[0], self.xmin)
            case "R":
                geometry_func = lambda x: np.isclose(x[0], self.xmax)
            case "T":
                geometry_func = lambda x: np.isclose(x[1], self.ymax)
            case "B":
                geometry_func = lambda x: np.isclose(x[1], self.ymin)
        boundary_func.interpolate(f)
        return dolfinx.fem.dirichletbc(boundary_func, dolfinx.fem.locate_dofs_geometrical(function_space, geometry_func))

