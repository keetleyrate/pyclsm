from clevelset import ConservativeLevelSet, circular_level_set, box_phi
from navierstokes import IncompressibleNavierStokesSolver
from common import *
from plotter import *
import pathlib
import csv

class IncompressibleTwoPhaseFlowSolver(IncompressibleNavierStokesSolver):

    def __init__(self, mesh, h, dt, rho1, rho2, mu1, mu2, sigma, g, phi0, p_phi=1, d=0.1, kinematic=True, c_kappa=20, c_normal=2) -> None:
        super().__init__(mesh, dt, kinematic=kinematic)
        self.mesh = mesh
        self.level_set = ConservativeLevelSet(mesh, h, dt, phi0, p=p_phi, d=d, c_kappa=c_kappa, c_normal=c_normal)
        self.rho1 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(rho1))
        self.rho2 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(rho2))
        self.mu1 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(mu1))
        self.mu2 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(mu2))
        self.has_surface_tension = sigma > 0
        self.sigma = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sigma))
        self.g = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0, -g)))
        self.step_proc = None

    def set_dencity_and_viscosity(self):
        self.rho_prev.x.array[:] = self.rho.x.array[:] 
        rho = dolfinx.fem.Expression(
            self.rho1 + (self.rho2 - self.rho1) * self.level_set.phi, self.density_space.element.interpolation_points()
        )
        self.rho.interpolate(rho)
        mu = dolfinx.fem.Expression(
            self.mu1 + (self.mu2 - self.mu1) * self.level_set.phi, self.density_space.element.interpolation_points()
        )
        self.mu.interpolate(mu)

    def set_body_forces(self):
        if self.has_surface_tension:
            kappa = self.level_set.compute_curvature()
            grad_phi = self.level_set.normal_projector.nabla_f
            self.F.interpolate(
                dolfinx.fem.Expression(
                    self.rho * self.g + self.sigma * kappa * grad_phi,
                    self.F_space.element.interpolation_points()
                )
            )
        else:
            self.F.interpolate(
                dolfinx.fem.Expression(
                    self.rho * self.g,
                    self.F_space.element.interpolation_points()
                )
            )

    def set_time_step_proc(self, proc):
        self.step_proc = proc
    
    def time_step(self, steps=1):
        for _ in range(steps):
            self.level_set.transport(self.u)
            self.set_dencity_and_viscosity()
            self.set_body_forces()
            self.reset()
            super().time_step()
            if self.step_proc is not None:
                self.step_proc(self)

    def set_no_slip_everywhere(self):
        self.u_bcs.append(create_no_slip_bc(self.mesh, self.velosity_space))

    def save_to_files(self, foldername, T, steps=1):
        results_folder = pathlib.Path(foldername)
        results_folder.mkdir(exist_ok=True, parents=True)

        self.phi_file = open(foldername + "/phi.csv", "w")
        phi_writer = csv.writer(self.phi_file)
        self.u_file = open(foldername + "/u.csv", "w")
        u_writer = csv.writer(self.u_file)

        for _ in tqdm(range(math.ceil(T / self.dt / steps))):
            t = str(self.t)
            phi = list(np.array(self.level_set.phi.x.array.copy(), dtype=np.float32))
            u = list(np.array(self.u.x.array.copy(), dtype=np.float32))
            phi.append(t)
            u.append(t)
            phi_writer.writerow(phi)
            u_writer.writerow(u)
            self.time_step(steps)
        self.phi_file.close()
        self.u_file.close()

    def read_files(self, foldername):
        self.phi_file = open(foldername + "/phi.csv")
        self.phi_reader = csv.reader(self.phi_file)
        self.u_file = open(foldername + "/u.csv")
        self.u_reader = csv.reader(self.u_file)

    def close_files(self):
        self.phi_file.close()
        self.u_file.close()


from common import *
import numpy as np
import matplotlib.pyplot as plt
import matplotx
import mpi4py
from visualise import *
from common import *
from scipy.integrate import simpson

def couette_flow_test():
    def compute_error(h):
        n = math.ceil(1 / h)
        mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
        solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 500, 1, 1, 1, 1, 0, 0, circular_level_set(0, 0, 0, 0))
        exact = dolfinx.fem.Function(solver.velosity_space)
        exact.interpolate(lambda x: (1/2 * x[1] * (1 - x[1]), 0 * x[0]))
        solver.set_velosity_bc(y_equals(1), constant((1, 0), mesh, solver.velosity_space))
        solver.set_velosity_bc(y_equals(0), constant((0, 0), mesh, solver.velosity_space))
        solver.set_y_velocity(x_equals(0), dolfinx.default_scalar_type(0))
        solver.set_y_velocity(x_equals(1), dolfinx.default_scalar_type(0))
        solver.reset()
        T = 0.1
        step_until(T, solver, lambda s: s.time_step())
        y = np.linspace(0, 1, 250)
        x = np.full(250, 0.5)
        x, y, u, _ = fem_vector_func_at_given_points(solver.u, mesh, dolfinx.geometry.bb_tree(mesh, 2), x, y)
        u_e = y - 2 / np.pi * sum(1/n * np.exp(-n**2*np.pi**2*T) * np.sin(n*np.pi*(1 - y)) for n in range(1, 100))
        return simpson(y=np.abs(u - u_e), x=y)
    compute_convergence(compute_error, 4)

def shear_test():
    n = 32
    h = 1 / n
    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
    d = 0.1
    solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 2, 1, 2, 1, 0.1, 0, circular_level_set(0.5, 0.5, 0.15, h ** (1 - d) / 2), )
    solver.set_velosity_bc(y_equals(1), constant((1, 0), mesh, solver.velosity_space))
    solver.set_velosity_bc(y_equals(0), constant((-1, 0), mesh, solver.velosity_space))
    solver.set_y_velocity(x_equals(0), dolfinx.default_scalar_type(0))
    solver.set_y_velocity(x_equals(1), dolfinx.default_scalar_type(0))
    solver.save_to_files("sols/shear", 3, steps=5)
    plotter = Plotter(solver, (0, 1), (0, 1), 0.1, density_points=200, levels=100, interface_points=0, filename="sols/shear")
    plotter.save_to_mp4("videos/shear.mp4")

   
from mesh2d import rectangle

def surface_tension():
    n = 16
    h = 1 / n
    mesh, tree = rectangle((0, 0), (2, 2), h)
    #mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)

    solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 0.1, 1, 0.1, 1, 10, 0, circular_level_set(0, 0, 0.2, 0.1), d=0.05)
    solver.set_no_slip_everywhere()
    solver.level_set.phi.interpolate(box_phi(0.5, 0.5, 1.5, 1.5, solver.level_set.eps))
    plotter = Plotter(solver, (0, 2), (0, 2), 0.1, contor_color="white", filename="sols/tens")
    # solver.time_step()
    # plotter.plot_from_solver()
    # plotter.show()
    #solver.save_to_files("sols/tens", 2, steps=5)
    plotter.save_to_mp4("videos/tens.mp4")
   

