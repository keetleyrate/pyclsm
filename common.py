from tqdm import tqdm
import math
import dolfinx
import numpy as np

def step_until(T, solver, method):
    for _ in tqdm(range(math.ceil(T / solver.dt))):
        method(solver)

def constant(value, mesh, space):
    return dolfinx.fem.Expression(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value)), space.element.interpolation_points())

def create_no_slip_bc(mesh, space):
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.folexterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(space, fdim, boundary_facets)
    no_slip = dolfinx.fem.Function(space)
    no_slip.interpolate(constant((0, 0), mesh, space))
    return dolfinx.fem.dirichletbc(no_slip, boundary_dofs)

def compute_convergence(find_error, points):
    hs = []
    es = []
    if type(points) == int:
        points = [2 ** i for i in range(1, points + 1)]
    for n in points:
        h  = 1 / n
        error = find_error(h)
        es.append(error)
        hs.append(h)
    h = np.array(hs)
    e = np.array(es)
    logh = np.log2(h)
    loge = np.log2(e)
    a, _ = np.polyfit(logh, loge, deg=1)
    print("Convergence: ", a)
    print(hs)
    print(es)

def x_equals(value):
    return lambda x: np.isclose(x[0], value)

def y_equals(value):
    return lambda x: np.isclose(x[1], value)