import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, integrate
from tqdm import tqdm
import sympy as sp
import os
import math
from common.visualize import fem_scalar_func_at_given_points


def signed_distance_to_zero_level_set(f, g, point):
    px, py = point[0], point[1]
    def L(X):
        x, y, λ = X
        f_value = f((x, y))
        gradient = g((x, y))
        dfdx, dfdy = gradient[0], gradient[1]
        return np.array([
            x - px + λ*dfdx,
            y - py + λ*dfdy,
            f_value
        ])
    x_star, y_star, _ = optimize.root(L, np.array([*point, 0])).x
    return np.sign(f((x_star, y_star))) * math.dist(point, (x_star, y_star))

    

def fem_function_from_sdf(sdf, load_as=None):
    def phi(w):
        x, y = w[0], w[1]
        if load_as is not None:
            if os.path.exists(load_as):
                d = np.loadtxt(load_as)
            else:
                print("Computing SDF")
                d = np.array([sdf(np.array([xi, yi])) for xi, yi in tqdm(list(zip(x, y)))])
                np.savetxt(load_as, d)
        else:
            print("Computing SDF")
            d = np.array([sdf(np.array([xi, yi])) for xi, yi in tqdm(list(zip(x, y)))])
        return d
    return phi


def inside(r_func, x):
    r_point = np.sqrt(np.dot(x, x))
    theta = np.arctan2(x[1], x[0])
    return np.sign(r_point - r_func(theta))

def signed_distance(f, C, inside, jac, hess, p):
    result = optimize.minimize(
        f,
        np.arctan2(p[1], p[0]),
        args=tuple(p),
        tol=1e-10,
        jac=jac,
        hess=hess,
        method="Newton-CG"
    )
    return np.linalg.norm(C(*result.x) - p) * inside(p)


def make_sdf_ellipice(a, b):
    t, x0, y0 = sp.symbols("t x0 y0")
    x, y = a * sp.cos(t), b * sp.sin(t)
    sqrd_dist = (x - x0)**2 + (y - y0)**2
    F = sp.lambdify([t, x0, y0], sqrd_dist)
    jac = sp.lambdify([t, x0, y0], sqrd_dist.diff(t))
    hess = sp.lambdify([t, x0, y0], sqrd_dist.diff(t).diff(t))

    def inside(p):
        return np.sign(p[0]**2 / a**2 + p[1]**2 / b**2 - 1)
    
    def sdf(p):
        return signed_distance(F, lambda t: np.array([a * np.cos(t), b * np.sin(t)]), inside, jac, hess, p)
    
    return sdf

def make_sdf_cos(A, l):
    def sdf(p):
        xn = 0.5
        a, b = p
        D = lambda x: np.sqrt((x - a) ** 2 + (A * np.cos(l * x) - b) ** 2)

        x = optimize.minimize_scalar(D, bracket=(0, 1), ).x

        return np.sqrt((x - a) ** 2 + (A * np.cos(l * x) - b) ** 2) * np.sign(A * np.cos(l * x) - b)
    return sdf


def make_sdf_folium(D, k):
    t, x0, y0 = sp.symbols("t x0 y0")
    a = (D + sp.sqrt(6 - 2 * D**2)) / 3
    b = D - a
    r = a + b * sp.cos(k * t)
    x = r * sp.cos(t)
    y = r * sp.sin(t)
    squared_distance = 0.5 * ((x - x0)**2 + (y - y0)**2)
    r = sp.lambdify(t, r)
    F = sp.lambdify([t, x0, y0], squared_distance)
    jac = sp.lambdify([t, x0, y0], squared_distance.diff(t))
    hess = sp.lambdify([t, x0, y0], squared_distance.diff(t).diff(t))
    
    def sdf(p):
        return signed_distance(F, lambda t: np.array([r(t) * np.cos(t), r(t) * np.sin(t)]), lambda p: inside(r, p), jac, hess, p)
    return sdf


def roundness_folium(D, k):
    a = (D + np.sqrt(6 - 2 * D**2)) / 3
    b = D - a
    P = integrate.quad(lambda t: np.sqrt((a + b * np.cos(k * t))**2 + (b * k * np.sin(k * t))**2), 0, 2 * np.pi)[0]
    print(P)
    return (4 * np.pi ** 2) / P**2

def axis_given_roundness(R):
    def f(a):
        b = 1 / a
        P = integrate.quad(lambda t: np.sqrt(a**2 * np.sin(t)**2 + b**2 * np.cos(t)**2), 0, 2 * np.pi)[0]

        return 4 * np.pi ** 2 / P**2 - R
    a = abs(optimize.fsolve(f, 1)[0])
    if a < 1:
        return 1 / a, a
    return a, 1 / a

def d_given_roundness(R, k):
    def f(D):
        a = (D + np.sqrt(6 - 2 * D**2)) / 3
        b = D - a
        P = integrate.quad(lambda t: np.sqrt((a + b * np.cos(k * t))**2 + (b * k * np.sin(k * t))**2), 0, 2 * np.pi)[0]
        return 4 * np.pi ** 2 / P**2 - R
    D = optimize.fsolve(f, 1.1)[0]
    return D


def roundness_ellipce(a):
    b = 1 / a
    P = integrate.quad(lambda t: np.sqrt(a**2 * np.sin(t)**2 + b**2 * np.cos(t)**2), 0, 2 * np.pi)[0]
    return (4 * np.pi ** 2) / P**2

def curvature_folium(D, k): 
    a = (D + np.sqrt(6 - 2 * D**2)) / 3
    b = D - a
    t = sp.symbols("t")
    r = a + b * sp.cos(k * t)
    x = r * sp.cos(t)
    y = r * sp.sin(t)
    κ = (x.diff(t) * y.diff(t).diff(t) - y.diff(t) * x.diff(t).diff(t)) / (x.diff(t) ** 2+ y.diff(t)**2) ** (3/2)
    κ = sp.lambdify(t, κ)
    return κ

def curvature_ell(a): 
    b = 1 / a
    t = sp.symbols("t")
    x = a * sp.cos(t)
    y = b * sp.sin(t)
    κ = (x.diff(t) * y.diff(t).diff(t) - y.diff(t) * x.diff(t).diff(t)) / (x.diff(t) ** 2+ y.diff(t)**2) ** (3/2)
    κ = sp.lambdify(t, κ)
    return κ


def total_curvature(κ):
    K, _ = integrate.quad(lambda t: abs(κ(t)), 0, 2 * np.pi)
    return K


def D_given_max_curvature(k, kappa_max):
    def max_folium_curvature(D):
        κ = curvature_folium(D.item(), k)
        return max(abs(κ(0)), abs(κ(np.pi / k))) - kappa_max
    return optimize.fsolve(max_folium_curvature, 1).item()

def D_given_total_curvature(k, total):
    def mean_folium_curvature(D):
        s = total_curvature(curvature_folium(D.item(), k))
        return s - total
    return optimize.fsolve(mean_folium_curvature, np.random.random_sample(1)).item()


