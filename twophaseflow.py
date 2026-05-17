from navierstokes import IncompressibleNavierStokesSolver
from clevelset import ConservativeLevelSet, circular_level_set, box_phi
from common.bc import *
from common.common import *
import pathlib
import csv
import ufl
import dolfinx

class IncompressibleTwoPhaseFlowSolver(IncompressibleNavierStokesSolver):

    def __init__(self, ls_solver: ConservativeLevelSet, dt, rho0, rho1, mu0, mu1, sigma, kinematic_bc=True) -> None:
        super().__init__(ls_solver.domain, dt, kinematic_bc=kinematic_bc)
        self.ls = ls_solver
        self.rho0 = rho0
        self.rho1 = rho1
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma

    def set_rho(self):
        self.rho_last.x.array[:] = self.rho.x.array[:] 
        rho = dolfinx.fem.Expression(
            self.rho0 + (self.rho1 - self.rho0) * self.ls.ϕ, self.scalar_space.element.interpolation_points
        )
        self.rho.interpolate(rho)
      

    def set_mu(self):
        mu = dolfinx.fem.Expression(
            self.mu0 + (self.mu1 - self.mu0) * self.ls.ϕ, self.scalar_space.element.interpolation_points
        )
        self.mu.interpolate(mu)

    
    def time_step(self, steps=1):
        for _ in range(steps):
            self.ls.advect(self.u)
            self.set_rho()
            self.set_mu()
            self.set_body_forces()
            self.compute_u()


    
    def velocity_magniute(self):
        mags = dolfinx.fem.Function(self.density_space)
        mags.interpolate(
            dolfinx.fem.Expression(
                ufl.sqrt(self.u[0]**2 + self.u[1]**2),
                self.density_space.element.interpolation_points()
            )
        )
        return mags


    def compute_u(self):
        super().time_step()

    def set_no_slip_everywhere(self):
        self.u_bcs.append(create_no_slip_bc(self.mesh, self.velocity_space))

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


# from common import *
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotx
# import mpi4py
# from visualise import *
# from common import *
# from scipy.integrate import simpson

# def couette_flow_test():
#     def compute_error(h):
#         n = math.ceil(1 / h)
#         mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
#         solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 500, 1, 1, 1, 1, 0, 0, circular_level_set(0, 0, 0, 0))
#         exact = dolfinx.fem.Function(solver.velosity_space)
#         exact.interpolate(lambda x: (1/2 * x[1] * (1 - x[1]), 0 * x[0]))
#         solver.set_velosity_bc(y_equals(1), constant((1, 0), mesh, solver.velosity_space))
#         solver.set_velosity_bc(y_equals(0), constant((0, 0), mesh, solver.velosity_space))
#         solver.set_y_velocity(x_equals(0), dolfinx.default_scalar_type(0))
#         solver.set_y_velocity(x_equals(1), dolfinx.default_scalar_type(0))
#         solver.reset()
#         T = 0.1
#         step_until(T, solver, lambda s: s.time_step())
#         y = np.linspace(0, 1, 250)
#         x = np.full(250, 0.5)
#         x, y, u, _ = fem_vector_func_at_given_points(solver.u, mesh, dolfinx.geometry.bb_tree(mesh, 2), x, y)
#         u_e = y - 2 / np.pi * sum(1/n * np.exp(-n**2*np.pi**2*T) * np.sin(n*np.pi*(1 - y)) for n in range(1, 100))
#         return simpson(y=np.abs(u - u_e), x=y)
#     compute_convergence(compute_error, 4)

# def shear_test():
#     n = 32
#     h = 1 / n
#     mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
#     d = 0.1
#     solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 2, 1, 2, 1, 0.1, 0, circular_level_set(0.5, 0.5, 0.15, h ** (1 - d) / 2), )
#     solver.set_velosity_bc(y_equals(1), constant((1, 0), mesh, solver.velosity_space))
#     solver.set_velosity_bc(y_equals(0), constant((-1, 0), mesh, solver.velosity_space))
#     solver.set_y_velocity(x_equals(0), dolfinx.default_scalar_type(0))
#     solver.set_y_velocity(x_equals(1), dolfinx.default_scalar_type(0))
#     solver.save_to_files("sols/shear", 3, steps=5)
#     plotter = Plotter(solver, (0, 1), (0, 1), 0.1, density_points=200, levels=100, interface_points=0, filename="sols/shear")
#     plotter.save_to_mp4("videos/shear.mp4")

   
# from mesh2d import rectangle

# def surface_tension():
#     n = 16
#     h = 1 / n
#     mesh, tree = rectangle((0, 0), (2, 2), h)
#     #mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)

#     solver = IncompressibleTwoPhaseFlowSolver(mesh, h, h / 10, 0.1, 1, 0.1, 1, 10, 0, circular_level_set(0, 0, 0.2, 0.1), d=0.05)
#     solver.set_no_slip_everywhere()
#     solver.level_set.phi.interpolate(box_phi(0.5, 0.5, 1.5, 1.5, solver.level_set.eps))
#     plotter = Plotter(solver, (0, 2), (0, 2), 0.1, contor_color="white", filename="sols/tens")
#     # solver.time_step()
#     # plotter.plot_from_solver()
#     # plotter.show()
#     #solver.save_to_files("sols/tens", 2, steps=5)
#     plotter.save_to_mp4("videos/tens.mp4")
   

