import numpy as np
from dolfinx import geometry
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random

def eval_sol(u, points, mesh, tree, forse_eval=False):
    cells = []
    points_on_proc = []
    cell_candidates = geometry.compute_collisions_points(tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
    out_of_bounds_points = np.full(len(points.T), False)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
        elif forse_eval:
            points_on_proc.append(point)
            cells.append(0)
            out_of_bounds_points[i] = True
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    res = u.eval(points_on_proc, cells)
    res[out_of_bounds_points] = np.zeros(res[out_of_bounds_points].shape)
    return res

def fem_scalar_func_at_points(u_func, mesh, tree, xbounds, ybounds, npoints, forse_eval=False):
    x = np.linspace(*xbounds, npoints)
    y = np.linspace(*ybounds, npoints)
    x, y = np.meshgrid(x, y, indexing="ij")
    points = np.zeros((3, npoints ** 2))
    points[0] = x.flatten()
    points[1] = y.flatten()
    u = eval_sol(u_func, points, mesh, tree, forse_eval)
    return x.reshape((npoints, npoints)), y.reshape((npoints, npoints)), u.reshape((npoints, npoints))

def fem_scalar_func_at_given_points(u_func, mesh, tree, x, y, forse_eval=False):
    points = np.zeros((3, len(x)))
    points[0] = np.array(x)
    points[1] = np.array(y)
    u = eval_sol(u_func, points, mesh, tree, forse_eval=forse_eval)
    return x, y, u

def fem_vector_func_at_points(u_func, mesh, tree, xbounds, ybounds, npoints, forse_eval=False):
    x = np.linspace(*xbounds, npoints)
    y = np.linspace(*ybounds, npoints)
    x, y = np.meshgrid(x, y, indexing="ij")
    points = np.zeros((3, npoints ** 2))
    points[0] = x.flatten()
    points[1] = y.flatten()
    uvs = eval_sol(u_func, points, mesh, tree, forse_eval=forse_eval)
    u = np.array(list(uv[0] for uv in uvs))
    v = np.array(list(uv[1] for uv in uvs))
    return x.reshape((npoints, npoints)), y.reshape((npoints, npoints)), u.reshape((npoints, npoints)), v.reshape((npoints, npoints))

def fem_vector_func_at_given_points(u_func, mesh, tree, x, y, forse_eval=False):
    points = np.zeros((3, len(x)))
    points[0] = np.array(x)
    points[1] = np.array(y)
    uvs = eval_sol(u_func, points, mesh, tree, forse_eval=forse_eval)
    u = np.array(list(uv[0] for uv in uvs))
    v = np.array(list(uv[1] for uv in uvs))
    return x, y, u, v

def vector_to_rgb(magitute):
    return plt.cm.viridis(magitute)

def fem_plot_vectors(axes, u, mesh, tree, xbounds, ybounds, n_points, force=False, scale=None, random=False):
    if random:
        xrand = (xbounds[1] - xbounds[0]) * np.random.random(n_points) + xbounds[0]
        yrand = (ybounds[1] - ybounds[0]) * np.random.random(n_points) + ybounds[0]
        x, y, u, v = fem_vector_func_at_given_points(u, mesh, tree, xrand, yrand, forse_eval=force)
        lengths = np.sqrt(np.square(u) + np.square(v))
        to_remove = lengths < 1e-2 * 2
        if sum(to_remove) != len(x):
            x = np.delete(x, to_remove)
            y = np.delete(y, to_remove)
            u = np.delete(u, to_remove)
            v = np.delete(v, to_remove)
    else:
        x, y, u, v = fem_vector_func_at_points(u, mesh, tree, xbounds, ybounds, n_points, forse_eval=force)
    lengths = np.sqrt(np.square(u) + np.square(v))
    max_abs = np.max(lengths)
    c = np.array(list(map(plt.cm.cividis, lengths.flatten() / max_abs)))
    axes.quiver(x, y, u, v, color=c, scale=scale)

def fem_plot_streamlines(axes, u, mesh, tree, xbounds, ybounds, n_points, colors=True, seeds=None):
    x, y, u, v = fem_vector_func_at_points(u, mesh, tree, xbounds, ybounds, n_points)
    lengths = np.sqrt(np.square(u.T) + np.square(v.T))
    axes.streamplot(y, x, u.T, v.T, color=lengths, cmap="cividis", density=1.5, start_points=seeds)
    axes.set_xlabel("$x$", fontsize=14)
    axes.set_ylabel("$y$", fontsize=14)
   
def fem_plot_contor_filled(fig, axes, u, mesh, tree, xbounds, ybounds, n_points, levels=None, force=False, colorbar=False, norm=None, label=""):
    x, y, u = fem_scalar_func_at_points(u, mesh, tree, xbounds, ybounds, n_points, forse_eval=force)
    conts = axes.contourf(x, y, u, cmap="cividis", levels=levels, norm=norm)
    if colorbar:
        fig.colorbar(conts, label=label)

def fem_plot_contor(fig, axes, u, mesh, tree, xbounds, ybounds, n_points, levels=None, force=False, linewidths=None, colors=None):
    x, y, u = fem_scalar_func_at_points(u, mesh, tree, xbounds, ybounds, n_points, forse_eval=force)
    if levels is None:
        conts = axes.contour(x, y, u)
        fig.colorbar(conts)
    else:
        conts = axes.contour(x, y, u, levels=levels, linewidths=linewidths, colors=colors)
        #fig.colorbar(conts)
    

def read_solution_files(solver, foldername):
    solver.read_files(foldername)
    for phi_row, u_row in zip(solver.phi_reader, solver.u_reader):
        yield (
            float(phi_row[-1]),
            np.array(phi_row[:-1], dtype=np.float32),
            np.array(u_row[:-1], dtype=np.float32),
        )
    solver.close_files()

def read_solution_files_fluid(solver, foldername):
    solver.read_files(foldername)
    for u_row, p_row in zip(solver.u_reader, solver.p_reader):
        yield (
            float(u_row[-1]),
            np.array(u_row[:-1], dtype=np.float32),
            np.array(p_row[:-1], dtype=np.float32)
        )
    solver.close_files()


def update(frame, fig, axes, solver, reader, skip_every, plot_fluid, plot_mags, plot_phi, forse_eval, plot_k, plot_interface, extra_draw, scale):
    axes.clear()
    t, phi, u = frame
    for _ in range(skip_every):
        _ = next(reader)
    print(f"Writing t = {t:.2f}")
    solver.level_set_solver.phi.x.array[:] = phi
    if not plot_phi and plot_interface:
        solver.plot_interface(axes, 200, forse_eval=forse_eval)
    else:
        solver.plot_phi(axes, 200, threshold=True, forse_eval=forse_eval)
    if plot_fluid:
        solver.fluid_solver.u.x.array[:] = u
        solver.plot_fluid(axes, 40, forse_eval=forse_eval, scale=scale)
    if plot_mags:
        solver.fluid_solver.u.x.array[:] = u
        solver.plot_fluid_mags(axes, 4 * math.ceil(1 / solver.dx))
    if plot_k:
        solver.level_set_solver.kappa.x.array[:] = k
        solver.plot_curvature(None, axes, 2 * math.ceil(1 / solver.dx), forse_eval=forse_eval)
    if extra_draw is not None:
        extra_draw(axes, solver)
    axes.set_xlabel("$x$", fontsize=12)
    axes.set_ylabel("$y$", fontsize=12)
    axes.set_title(f"$t={t:.2f}$")
    axes.set_aspect("equal")

def gif_from_file(filename, gif_filename, extra_draws_fn=None, skip=0, plot_contour=True, plot_vecs=True):
    fig, axes = plt.subplots()
    reader = read_solution_files(filename)
    ani = animation.FuncAnimation(fig, update, frames=reader, repeat=False, fargs=(fig, axes, solver))
    ani.save(gif_filename, writer="Pillow", fps=30)   

                
def mp4_from_file(filename, mp4_filename, solver, fps=30, skip_every=0, plot_fluid=False, plot_mags=False, plot_phi=False, forse_eval=False, plot_k=False, plot_interface=True, extra_draw=None, scale=None):
    fig, axes = plt.subplots()
    reader = read_solution_files(solver, filename)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.FuncAnimation(fig, update, frames=reader, repeat=False, fargs=(fig, axes, solver, reader, skip_every, plot_fluid, plot_mags, plot_phi, forse_eval, plot_k, plot_interface, extra_draw, scale))
    ani.save(mp4_filename, writer=writer)

def gif_from_file(filename, gif_filename, solver, fps=30, skip_every=0, plot_fluid=False, plot_mags=False, plot_phi=False, forse_eval=False, plot_k=False, plot_interface=True, extra_draw=None):
    fig, axes = plt.subplots()
    reader = read_solution_files(solver, filename)
    ani = animation.FuncAnimation(fig, update, frames=reader, repeat=False, fargs=(fig, axes, solver, reader, skip_every, plot_fluid, plot_mags, plot_phi, forse_eval, plot_k, plot_interface, extra_draw))
    ani.save(gif_filename, writer="Pillow", fps=30)

       
       