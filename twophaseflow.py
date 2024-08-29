
from navierstokes import NSSolver
from levelset import LevelSetSolver, make_marker_function
from visualise import *
from dolfinx import geometry, fem
from dolfinx.mesh import exterior_facet_indices
import ufl
import math
import matplotlib.pyplot as plt
from bc import *
import pathlib
from mpi4py import MPI
import csv
import shutil

class TwoPhaseFlowSolver:

    def __init__(self, mesh, dx, dt, rho1, rho2, mu1, mu2, sigma, initial_phi_args, xbounds, ybounds, g=0, reinit_tol=1e-4, output=False, d=0.1, kinematic=False, adaptive_time_step=False, level_set_order=1) -> None:
        self.mesh = mesh
        self.tree = geometry.bb_tree(mesh, 2)
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.adaptive_time_step = adaptive_time_step
    

        # Conditons on interface thickness based on mesh size
        epsilon = dx**(1 - d) / 2
        self.eps = epsilon
        self.dx = dx
        dtau = dx**(1 + d) / 2
 

        # Required function spaces for each solver
        self.phi_space = fem.FunctionSpace(mesh, ("CG", level_set_order))
        self.normal_space = fem.FunctionSpace(mesh, ufl.VectorElement("CG", mesh.ufl_cell(), level_set_order))
        self.velocity_space = fem.FunctionSpace(mesh, ufl.VectorElement("CG", mesh.ufl_cell(), 2))
        self.pressure_space = fem.FunctionSpace(mesh, ("CG", 1))
        self.density_space = fem.FunctionSpace(mesh, ("CG", 1))

        self.fluid_solver = NSSolver(mesh, dt, self.velocity_space, self.pressure_space, self.density_space, kinematic)
        self.level_set_solver = LevelSetSolver(mesh, make_marker_function(*initial_phi_args, epsilon), dx, dt, dtau, epsilon, self.phi_space, self.normal_space, self.velocity_space, dtau * reinit_tol, output)
        self.dt = dt
        self.t = 0
        self.plotting_args = self.mesh, self.tree, self.xbounds, self.ybounds
        self.g = 0
        self.u = fem.Function(self.density_space)
        self.v = fem.Function(self.density_space)

        self.rho1 = fem.Constant(mesh, default_scalar_type(rho1))
        self.rho2 = fem.Constant(mesh, default_scalar_type(rho2))
        self.mu1 = fem.Constant(mesh, default_scalar_type(mu1))
        self.mu2 = fem.Constant(mesh, default_scalar_type(mu2))
        self.sigma = fem.Constant(mesh, default_scalar_type(sigma))
        self.g = fem.Constant(mesh, default_scalar_type((0, -g)))
        self.surf_ten = fem.Function(self.velocity_space)

    def compute_dencity(self):
        # Move dencity from previous time step into rho^n
        self.fluid_solver.rho_prev.x.array[:] = self.fluid_solver.rho.x.array[:] 
        # Compute new density using phi
        rho = fem.Expression(self.rho1 + (self.rho2 - self.rho1) * self.level_set_solver.phi, self.density_space.element.interpolation_points())
        self.fluid_solver.rho.interpolate(rho)

    def compute_viscosity(self):
        # Compute new viscosity using phi
        mu = fem.Expression(self.mu1 + (self.mu2 - self.mu1) * self.level_set_solver.phi, self.density_space.element.interpolation_points())
        self.fluid_solver.mu.interpolate(mu)

    def compute_surface_tension(self):
        self.level_set_solver.compute_curvature()
        surf_ten =  self.level_set_solver.kappa * self.level_set_solver.curvature_solver.grad
        self.surf_ten.interpolate(fem.Expression(surf_ten, self.velocity_space.element.interpolation_points()))
    
    def compute_forces(self):
        F = fem.Expression(self.fluid_solver.rho * self.g + self.sigma * self.surf_ten, self.fluid_solver.F_space.element.interpolation_points())
        self.fluid_solver.F.interpolate(F)
    

    def time_step(self, steps=1):
        for _ in range(steps):
            # Update the advection problem
            self.level_set_solver.set_velosity(self.fluid_solver.u.x.array[:])
             # Advect interface
            self.level_set_solver.advection_step()
            # Solve for fluid
            self.compute_dencity()
            self.compute_viscosity()
            self.compute_surface_tension()
            self.compute_forces()
            self.fluid_solver.reset()
            self.fluid_solver.time_step()
            if self.adaptive_time_step:
                self.update_dt()
            self.t += self.dt
            self.level_set_solver.t = self.t

    def reinitialise(self):
        self.level_set_solver.reinitalise()

    def to_steady_state(self, tol, max_iters=1000):
        for _ in range(max_iters):
            last = self.level_set_solver.phi.x.array.copy()
            self.time_step()
            if (res := np.linalg.norm(self.level_set_solver.phi.x.array - last) > tol):
                print(f"res: {res:.4e}")
            else:
                break


    
    def set_u_bc(self, geom_fn, fn):
        self.fluid_solver.set_velosity_bc(geom_fn, fn)

    def set_p_bc(self, geom_fn, fn):
        self.fluid_solver.set_pressure_bc(geom_fn, fn)

    def set_x_velocity(self, geometry_fn, fn):
        self.fluid_solver.set_x_velocity(geometry_fn, fn)
    
    def set_y_velocity(self, geometry_fn, fn):
        self.fluid_solver.set_y_velocity(geometry_fn, fn)

    def set_no_slip_everywhere(self):
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        boundary_facets = exterior_facet_indices(self.mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(self.velocity_space, fdim, boundary_facets)

        no_slip = fem.Function(self.velocity_space)
        no_slip.interpolate(constant((0, 0), self.mesh, self.velocity_space))

        bc = fem.dirichletbc(no_slip, boundary_dofs)
        self.fluid_solver.u_bcs.append(bc)




    def compute_area(self):
        I = fem.form(self.level_set_solver.phi * ufl.dx)
        area = fem.assemble_scalar(I)
        return area
    
    def compute_perimeter(self):
        self.level_set_solver.compute_curvature()
        k = self.level_set_solver.kappa
        phi = self.level_set_solver.phi
        I = fem.form(k * phi * ufl.dx)
        return fem.assemble_scalar(I)
    
    def com(self):
        x = ufl.SpatialCoordinate(self.mesh)
        phi = self.level_set_solver.phi
        com_x = fem.form(x[0] * phi * ufl.dx)
        com_y = fem.form(x[1] * phi * ufl.dx)
        dem = fem.form(phi * ufl.dx)
        return fem.assemble_scalar(com_x) / fem.assemble_scalar(dem), fem.assemble_scalar(com_y) / fem.assemble_scalar(dem)

    def average_velosity(self):
        phi = self.level_set_solver.phi
        u = self.fluid_solver.u
        U_x = fem.form(u[0] * phi * ufl.dx)
        U_y = fem.form(u[1] * phi * ufl.dx)
        dem = fem.form(phi * ufl.dx)
        return fem.assemble_scalar(U_x) / fem.assemble_scalar(dem), fem.assemble_scalar(U_y) / fem.assemble_scalar(dem)
    
    def circularity(self):
        self.level_set_solver.compute_curvature()
        phi = self.level_set_solver.phi
        kappa = self.level_set_solver.kappa
        num = fem.form(4 * np.pi * phi * ufl.dx)
        dem = fem.form(phi * kappa * ufl.dx)
        return np.sqrt(fem.assemble_scalar(num)) / fem.assemble_scalar(dem)


    def plot_fluid(self, axes, n_points, forse_eval=False, scale=None, sigmoid=False):
        x, y, u, v = fem_vector_func_at_points(self.fluid_solver.u, *self.plotting_args, n_points, forse_eval=forse_eval)
        lengths = np.sqrt(np.square(u) + np.square(v))
        max_abs = np.max(lengths)
        c = np.array(list(map(plt.cm.viridis, lengths.flatten() / max_abs)))
        if sigmoid:
            u = np.tanh(u)
            v = np.tanh(v)
        axes.quiver(x, y, u, v, color=c, scale=scale)

    def plot_phi(self, axes, n_points, threshold=False, forse_eval=False):
        _, _, phi = fem_scalar_func_at_points(self.level_set_solver.phi, *self.plotting_args, n_points, forse_eval=forse_eval)
        if threshold:
            inside = phi >= 0.5
            phi[inside] = np.ones(phi[inside].shape)
            phi[np.logical_not(inside)] = np.zeros(phi[np.logical_not(inside)].shape)
        axes.imshow(np.transpose(np.flip(phi, 1)), extent=[self.xbounds[0], self.xbounds[1], self.ybounds[0], self.ybounds[1]])

    def plot_interface(self, axes, n_points, forse_eval=False, colors=None, width=None):
        x, y, phi = fem_scalar_func_at_points(self.level_set_solver.phi, *self.plotting_args, n_points, forse_eval=forse_eval)
       #axes.contour(x, y, phi, levels=[0.1, 0.5, 0.9])
        axes.contour(x, y, phi, levels=[0.5], colors=colors, linewidths=width)

    def plot_interface_normals(self, axes, n_vectors):
        x, y, u, v = fem_vector_func_at_points(self.level_set_solver.n, *self.plotting_args, n_vectors)
        axes.quiver(x, y, u, v, color="cadetblue")

    def plot_F(self, axes, n_vectors):
        x, y, u, v = fem_vector_func_at_points(self.fluid_solver.F, *self.plotting_args, n_vectors)
        axes.quiver(x, y, u, v, color="cadetblue")

    def plot_Fs(self, axes, n_vectors):
        x, y, u, v = fem_vector_func_at_points(self.surf_ten, *self.plotting_args, n_vectors)
        axes.quiver(x, y, u, v, color="cadetblue")

    def plot_phi_grad(self, axes, n_vectors):
        x, y, u, v = fem_vector_func_at_points(self.level_set_solver.curvature_solver.grad, *self.plotting_args, n_vectors)
        axes.quiver(x, y, u, v, color="cadetblue")


    def plot_density(self, fig, axes, n_points):
        #self.fluid_solver.compute_dencity(self.level_set_solver.phi)
        x, y, rho = fem_scalar_func_at_points(self.fluid_solver.rho, *self.plotting_args, n_points)
        cts = axes.imshow(np.transpose(np.flip(rho, 1)), extent=[self.xbounds[0], self.xbounds[1], self.ybounds[0], self.ybounds[1]])
        cbar = fig.colorbar(cts)
        cbar.set_label(r"$\rho(x)$")

    def plot_viscosity(self, fig, axes, n_points, forse_eval=False):
        #self.fluid_solver.compute_dencity(self.level_set_solver.phi)
        x, y, rho = fem_scalar_func_at_points(self.fluid_solver.mu, *self.plotting_args, n_points, forse_eval=forse_eval)
        cts = axes.imshow(np.transpose(np.flip(rho, 1)), extent=[self.xbounds[0], self.xbounds[1], self.ybounds[0], self.ybounds[1]])
        cbar = fig.colorbar(cts)
        cbar.set_label(r"$\rho(x)$")

    def plot_curvature(self, fig, axes, n_points):
        #self.fluid_solver.compute_dencity(self.level_set_solver.phi)
        x, y, rho = fem_scalar_func_at_points(self.level_set_solver.kappa, *self.plotting_args, n_points)
        cts = axes.imshow(np.transpose(np.flip(rho, 1)), extent=[self.xbounds[0], self.xbounds[1], self.ybounds[0], self.ybounds[1]])
        cbar = fig.colorbar(cts)
        cbar.set_label(r"$\kappa(x)$")

    def fluid_mags_as_function(self):
        u = self.fluid_solver.u
        length = fem.Function(self.pressure_space)
        length.interpolate(fem.Expression(ufl.sqrt(u[0]**2 + u[1]**2), self.pressure_space.element.interpolation_points()))
        return length



    def save_flow_attributes(self, T, filename):
        # t, A, C, x_com, y_com, Ux, Uy
        with open(filename, "w") as outfile:
            for _ in range(math.ceil(T / self.dt)):
                A = self.compute_area()
                C = self.circularity()
                com = self.com()
                U = self.average_velosity()
                outfile.write(f",".join(list(map(str, [self.t, A, C, *com, *U]))) + "\n")
                self.time_step()
    
    def save_to_files(self, foldername, T, steps=1):
        results_folder = pathlib.Path(foldername)
        results_folder.mkdir(exist_ok=True, parents=True)

        phi_file = open(foldername + "/phi.csv", "w")
        phi_writer = csv.writer(phi_file)
        u_file = open(foldername + "/u.csv", "w")
        u_writer = csv.writer(u_file)

        for _ in range(math.ceil(T / self.dt / steps)):
            t = str(self.t)
            phi = list(np.array(self.level_set_solver.phi.x.array.copy(), dtype=np.float32))
            kappa = list(np.array(self.level_set_solver.kappa.x.array.copy(), dtype=np.float32))
            u = list(np.array(self.fluid_solver.u.x.array.copy(), dtype=np.float32))
            p = list(np.array(self.fluid_solver.p.x.array.copy(), dtype=np.float32))
            phi.append(t)
            kappa.append(t)
            u.append(t)
            p.append(t)
            phi_writer.writerow(phi)
            u_writer.writerow(u)
            self.time_step(steps)

        phi_file.close()
        u_file.close()
        shutil.make_archive(foldername, "zip", foldername)


    def load_time_step(self, T):
        for phi_row, u_row, in zip(self.phi_reader, self.u_reader):
            t = float(phi_row[-1])
            if t >= T:
                self.level_set_solver.phi.x.array[:] = np.array(phi_row[:-1])
                self.fluid_solver.u.x.array[:] = np.array(u_row[:-1])
                return
        self.level_set_solver.phi.x.array[:] = np.array(phi_row[:-1])
        self.fluid_solver.u.x.array[:] = np.array(u_row[:-1])

    def read_files(self, foldername):
        self.phi_file = open(foldername + "/phi.csv")
        self.phi_reader = csv.reader(self.phi_file)
        self.u_file = open(foldername + "/u.csv")
        self.u_reader = csv.reader(self.u_file)

    def close_files(self):
        self.phi_file.close()
        self.u_file.close()




