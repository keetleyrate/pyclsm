from tqdm import tqdm
import math
import dolfinx
import numpy as np

def step_until(T, solver, method):
    for _ in tqdm(range(math.ceil(T / solver.dt))):
        method(solver)

def constant(value, mesh, space):
    return dolfinx.fem.Expression(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value)), space.element.interpolation_points())

def compute_convergence(find_error, m):
    hs = []
    es = []
    for n in [2 ** i for i in range(1, m + 1)]:
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