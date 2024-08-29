import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI
import pyvista
from dolfinx import plot, mesh
import dolfinx
from sympy import Polygon, Point
import math
import numpy as np
from bc import * 

def unit_square(dx, show=False):
    n = math.ceil(1 / dx)
    square = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.quadrilateral)
    if show:   
        topology, cell_types, geometry = plot.vtk_mesh(square, square.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True, color="white", edge_color="black", lighting=False)
        plotter.view_xy()
        plotter.show()
    return square, dolfinx.geometry.bb_tree(square, 2)

def rectangle(p1, p2, dx, show=False):
    x1, y1 = p1
    x2, y2 = p2
    nx = math.ceil(abs(x2 - x1) / dx)
    ny = math.ceil(abs(y2 - y1) / dx)
    print(f"x cells: {nx}, y cells: {ny}")
    rect = mesh.create_rectangle(MPI.COMM_WORLD, [np.array(p1), np.array(p2)], (nx, ny), cell_type=mesh.CellType.quadrilateral)
    if show:   
        topology, cell_types, geometry = plot.vtk_mesh(rect, rect.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.show()
    return rect, dolfinx.geometry.bb_tree(rect, 2)


class Mesh2D:

    def __init__(self, name: str, mesh_size: float) -> None:
        gmsh.initialize()
        gmsh.model.add(name)
        self.h = mesh_size
        self.border = None
        self.border_poly = None
        self.holes = []
        self.border_points = None

    def in_mesh(self, x, y):
        return self.border_poly.encloses_point(Point(x, y))

    def add_rectangle_border(self, points: list[tuple[float, float]]):
        a, b, c, d = points
        a = gmsh.model.occ.addPoint(*a, 0, meshSize=self.h)
        b = gmsh.model.occ.addPoint(*b, 0, meshSize=self.h)
        c = gmsh.model.occ.addPoint(*c, 0, meshSize=self.h)
        d = gmsh.model.occ.addPoint(*d, 0, meshSize=self.h)
        bottom = gmsh.model.occ.addLine(a, b)
        right = gmsh.model.occ.addLine(b, c)
        top = gmsh.model.occ.addLine(c, d)
        left = gmsh.model.occ.addLine(d, a)
        self.border = gmsh.model.occ.addCurveLoop([bottom, right, top, left])
        self.border_points = points
        self.border_poly = Polygon(*points)

    def add_border_from_points(self, points: list[tuple[float, float]]):
        n = len(points)
        point_tags = [gmsh.model.occ.addPoint(*p, 0, meshSize=self.h) for p in points]
        line_tags = [gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % n]) for i in range(n)]
        self.border = gmsh.model.occ.addCurveLoop(line_tags)
        self.border_poly = Polygon(*points)
        self.border_points = points

    def remove_circle(self, center: tuple[float, float], radius: float):
        if radius > 0:
            circle = gmsh.model.occ.addCircle(*center, 0, radius, xAxis=[1, 0, 0])
            self.holes.append(gmsh.model.occ.addCurveLoop([circle]))


    def remove_polygon(self, points: list[tuple[float, float]]):
        n = len(points)
        vert_tags = [gmsh.model.occ.addPoint(*p, 0, meshSize=self.h) for p in points]
        edge_tags = [gmsh.model.occ.addLine(vert_tags[i], vert_tags[(i + 1) % n]) for i in range(n)]
        self.holes.append(gmsh.model.occ.addCurveLoop(edge_tags))

    def generate(self, show=False):
        mesh = gmsh.model.occ.addPlaneSurface([self.border, *self.holes])
        gmsh.model.occ.synchronize()
        gdim = 2
        gmsh.model.addPhysicalGroup(gdim, [mesh], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin",self.h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax",self.h)
        #gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.model.mesh.generate(gdim)

        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

        if show:   
            topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=True, color="white", edge_color="black", lighting=False)
            plotter.view_xy()
            plotter.show()
        return domain, cell_markers, facet_markers
    
    def get_wall_function(self):
        pts = self.border_points
        return np.logical_or.reduce(tuple(axes_aligned_boundary((pts[i], pts[(i + 1) % len(pts)])) for i in range(len(pts))))

    