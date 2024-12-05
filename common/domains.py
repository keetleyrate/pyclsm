import math
import dolfinx
from dolfinx import mesh
import numpy as np
from mpi4py import MPI

def unit_square(dx):
    n = math.ceil(1 / dx)
    square = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.quadrilateral)
    return square, dolfinx.geometry.bb_tree(square, 2)

def rectangular_domain(dx, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    nx = math.ceil(abs(x2 - x1) / dx)
    ny = math.ceil(abs(y2 - y1) / dx)
    print(f"x cells: {nx}, y cells: {ny}")
    rect = mesh.create_rectangle(MPI.COMM_WORLD, [np.array(p1), np.array(p2)], (nx, ny), cell_type=mesh.CellType.quadrilateral)
    return rect, dolfinx.geometry.bb_tree(rect, 2)