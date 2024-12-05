from navierstokes import IncompressibleNavierStokesSolver
from common import *
from visualise import *

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotx

# plt.style.use(matplotx.styles.dracula)

def update(frame, axes, solver):
    for ax in axes:
        ax.clear()
    fem_plot_contor_filled(fig, axes[0], solver.u.sub(0), mesh, tree, (0, 1), (0, 1), 150, levels=250)
    x, y, u, v = fem_vector_func_at_given_points(solver.u, mesh, tree, 0.5 * np.ones(150), np.linspace(0, 1, 150))
    n = 20
    t = solver.t
    u_exact = U * y / h - 2 * U / np.pi * sum(1 / j * np.exp(-j**2*np.pi**2*nu*t/h**2) * np.sin(j*np.pi*(1 - y/h)) for j in range(1, n+1))
    axes[1].plot(u, y, label="Numerical")
    axes[1].plot(u_exact, y, linestyle="--", label="Exact")
    axes[1].grid(True)
    axes[1].legend(loc="best")
    axes[1].set_xlabel("$u$")
    axes[1].set_ylabel("$y$")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_aspect("equal")
    axes[1].set_aspect("equal")
    axes[0].set_title(f"$t={solver.t:.2f}$")
    axes[1].set_title(f"$t={solver.t:.2f}$")
    print(f"{solver.t:.4f}")
    plt.tight_layout()
    solver.time_step()




dx = 0.05
mesh, tree = unit_square(dx)
rectangle()
U = 1
h = 1
rho = 1
mu = 1
nu = mu / rho
solver = IncompressibleNavierStokesSolver(mesh, 0.0005, kinematic=False)
solver.set_density_as_const(1)
solver.set_viscosity_as_const(1)
solver.set_velocity_bc(y_equals(0), (0, 0))
solver.set_velocity_bc(y_equals(1), (1, 0))
solver.set_y_velocity(x_equals(0), 0)
solver.set_y_velocity(x_equals(1), 0)
solver.reset()
T = 0.25

fig, axes = plt.subplots(1, 2)
ani = animation.FuncAnimation(fig, update, math.ceil(T / solver.dt), fargs=(axes, solver))
ani.save("couette-tutorial.gif", writer="Pillow", fps=30)
#plt.show()
#plt.savefig("couette-tutorial-map.png", dpi=400)