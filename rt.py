from domain import *
from visualise import *
from bc import *
from navierstokes import IncompressibleNavierStokesSolver



square = RectangularDomain(0, 1, 0, 1, 0.05)
solver = IncompressibleNavierStokesSolver(square, 0.01, kinematic=True)
solver.set_no_slip("L")
solver.set_no_slip("R")
solver.set_no_slip("B")
solver.set_const_velocity_bc((1, 0), "T")
solver.to_steady_state(1e-4)

axes = plt.axes()
fem_plot_vectors(axes, solver.u, square.mesh, square.tree, (0, 1), (0, 1), 50)
axes.set_aspect("equal")
plt.show()
