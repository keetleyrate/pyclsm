import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from common.visualize import *
import scienceplots
from dolfinx import fem, geometry

plt.style.use(['science', 'no-latex', 'grid'])
plt.rcParams["figure.figsize"] = (6, 6)

def animation_update(frame, plotter, reader, skip_every):
    plotter.axes.clear()
    t, phi, u = frame
    print(f"Writing t: {t:.2f}")
    for _ in range(skip_every):
        _ = next(reader)
    plotter.solver.level_set.phi.x.array[:] = phi
    plotter.solver.u.x.array[:] = u
    plotter.axes.set_title(f"$t={t:.2f}$")
    plotter.plot_from_solver()


class Plotter:

    def __init__(self, solver, xlims, ylims, tickdelta, norm=None, xlabel="x", ylabel="y", force_plot=False, filename="", fluid_points=0, phi_points=0, mag_points=0, levels=100, interface_points=100, colorbar=False, contour_levels=[0.5], scale=20, linewidth=0.75, contor_color="darkslateblue", kappa_points=0, random_vects=False, mask_with_phi=False, mask_one_minus=False, visc_points=0, density_points=0) -> None:
        self.fig, self.axes = plt.subplots()
        self.xlims = xlims
        self.ylims = ylims
        self.tickdelta = tickdelta
        self.force = force_plot
        self.xlabel = f"${xlabel}$"
        self.ylabel = f"${ylabel}$"
        self.fsize = 14
        self.mesh = solver.mesh
        self.tree = geometry.bb_tree(solver.mesh, 2)
        self.norm = norm
        self.fluid_points = fluid_points
        self.phi_points = phi_points
        self.mag_points = mag_points
        self.kappa_points = kappa_points
        self.visc_points = visc_points
        self.density_points = density_points
        self.levels = levels
        self.interface_points = interface_points
        self.scale = scale
        self.solver = solver
        self.colorbar = colorbar
        self.contour_levels = contour_levels
        self.filename = filename
        self.linewidths = [linewidth for _ in range(len(contour_levels))]
        self.contor_color = contor_color
        self.random_vec_points = random_vects
        self.mask_with_phi = mask_with_phi
        self.one_minus = mask_one_minus
            

    def set_ticks(self):
        rounder = lambda v: round(v, 1)
        xticks = list(map(rounder, np.arange(self.xlims[0], self.xlims[1] + self.tickdelta, self.tickdelta)))
        yticks = list(map(rounder, np.arange(self.ylims[0], self.ylims[1] + self.tickdelta, self.tickdelta)))
        self.axes.set_xticks(xticks)
        self.axes.set_xticklabels(xticks)
        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels(yticks)

    def set_labels(self):
        self.axes.set_xlabel(self.xlabel, fontsize=self.fsize)
        self.axes.set_ylabel(self.ylabel, fontsize=self.fsize)

    def set_lims(self):
        self.axes.set_xlim(self.xlims)
        self.axes.set_ylim(self.ylims)

    def plot(self, x, y, label=None, color=None, linestyle=None):
        self.axes.plot(x, y, label=label, color=color, linestyle=linestyle)

    def plot_vector_field(self, U, n_vecs, random_points):
        fem_plot_vectors(self.axes, U, self.mesh, self.tree, self.xlims, self.ylims, n_vecs, force=self.force, scale=self.scale, random=random_points)


    def plot_scalar_field(self, S, n_points, levels, colorbar=False):
        fem_plot_contor_filled(self.fig, self.axes, S, self.mesh, self.tree, self.xlims, self.ylims, n_points, levels, force=self.force, colorbar=colorbar, norm=self.norm)

    def plot_contors(self, S, n_points, levels, colorbar=False):
        fem_plot_contor(self.fig, self.axes, S, self.mesh, self.tree, self.xlims, self.ylims, n_points, levels, force=self.force, linewidths=self.linewidths, colors=[self.contor_color])

    def plot_from_solver(self):
        if self.fluid_points > 0:
            phi = self.solver.level_set.phi
            if self.mask_with_phi:
                if self.one_minus:
                    phi = 1 - phi
                self.solver.u.interpolate(
                    fem.Expression(self.solver.u * phi, self.solver.velocity_space.element.interpolation_points())
                )
            self.plot_vector_field(self.solver.u, self.fluid_points, self.random_vec_points)
        if self.interface_points > 0:
            self.plot_contors(self.solver.level_set.phi, self.interface_points, self.contour_levels, colorbar=self.colorbar)
        if self.visc_points > 0:
            self.solver.set_viscosity()
            self.plot_scalar_field(self.solver.mu, self.visc_points, self.levels, colorbar=self.colorbar)
        if self.density_points > 0:
            self.solver.set_dencity()
            self.plot_scalar_field(self.solver.rho, self.density_points, self.levels, colorbar=self.colorbar)
        if self.phi_points > 0:
            self.plot_scalar_field(self.solver.level_set.phi, self.phi_points, self.levels, colorbar=self.colorbar)
        if self.mag_points > 0:
            self.plot_scalar_field(self.solver.fluid_mags_as_function(), self.mag_points, self.levels, colorbar=self.colorbar)
        if self.kappa_points > 0:
            self.solver.level_set_solver.compute_curvature()
            self.plot_scalar_field(self.solver.level_set.kappa, self.kappa_points, self.levels, colorbar=self.colorbar)
        self.axes.set_aspect("equal")
        self.set_ticks()
        self.set_lims()

    def plot_time_step(self, t):
        self.solver.read_files(self.filename)
        self.solver.load_time_step(t)
        if self.visc_points > 0:
            self.solver.compute_viscosity()
        if self.density_points > 0:
            self.solver.compute_dencity()
        self.plot_from_solver()
        self.solver.close_files()

    def save_to_mp4(self, mp4_filename, skip_every=0, fps=30):
        reader = read_solution_files(self.solver, self.filename)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani = animation.FuncAnimation(self.fig, animation_update, frames=reader, repeat=False, fargs=(self, reader, skip_every))
        ani.save(mp4_filename, writer=writer)

    def save(self, fname, dpi=400):
        self.axes.set_aspect("equal")
        self.set_ticks()
        self.set_labels()
        self.set_lims()
        plt.savefig(fname, dpi=dpi)
    
    def show(self):
        self.axes.set_aspect("equal")
        self.set_ticks()
        self.set_lims()
        plt.show()

    def clear(self):
        self.axes.clear()
        plt.cla()
        plt.clf()
        self.fig, self.axes = plt.subplots()

